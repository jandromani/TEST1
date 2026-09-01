from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np

from rtmdet_r_postprocess_adapter import RTMDetRSharedPostprocessAdapter


ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)
PROTOCOL = ROOT / "cycle17c_apache_postprocess_protocol.json"
MMR = pathlib.Path(os.environ["MMROTATE_DIR"]).resolve()
CFG = MMR / "configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py"
CKPT = pathlib.Path(os.environ["RTMDET_CKPT"]).resolve()
IMG = pathlib.Path(os.environ["RTMDET_IMAGE"]).resolve()
ONNX_PATH = pathlib.Path(os.environ["C17B_ONNX"]).resolve()
EXPECTED_COMMIT = "3ff004eb21ea040455b5585db229edba4037f1bf"
EXPECTED_CKPT_SHA = "9d821076f9d3f9bbe5a709524e6b0cf907ad58a7fb92615321f351cd389e51a3"
EXPECTED_ONNX_SHA = "5336b5a5a54bfcdfca4e0dde3dc3710498da7f5885de18237c40c7f405808699"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_input() -> np.ndarray:
    image = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read input image: {IMG}")
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mean = np.asarray([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1))[None, ...], dtype=np.float32)


def flatten_tensors(value: Any) -> tuple[Any, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in flatten_tensors(item))
    raise TypeError(f"unsupported raw model output type: {type(value)!r}")


def rows(records: Sequence[Any]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def _signed_polygon_area(polygon: Sequence[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )


def _box_corners(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    cx, cy = float(geometry["cx"]), float(geometry["cy"])
    width, height = float(geometry["width"]), float(geometry["height"])
    angle = float(geometry["angle_rad"])
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        (cx + x * cosine - y * sine, cy + x * sine + y * cosine)
        for x, y in (
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        )
    ]


def _edge_cross(a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def _line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-15:
        return p2
    factor = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    return x1 + factor * (x2 - x1), y1 + factor * (y2 - y1)


def _convex_intersection(
    subject: Sequence[tuple[float, float]],
    clipper: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = list(subject)
    orientation = 1.0 if _signed_polygon_area(clipper) >= 0 else -1.0
    for index, edge_start in enumerate(clipper):
        edge_end = clipper[(index + 1) % len(clipper)]
        input_polygon, output = output, []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        previous_inside = _edge_cross(edge_start, edge_end, previous) * orientation >= -1e-12
        for current in input_polygon:
            current_inside = _edge_cross(edge_start, edge_end, current) * orientation >= -1e-12
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous, previous_inside = current, current_inside
    return output


def rotated_iou_float64(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_polygon, right_polygon = _box_corners(left), _box_corners(right)
    left_area = abs(_signed_polygon_area(left_polygon))
    right_area = abs(_signed_polygon_area(right_polygon))
    intersection_area = abs(_signed_polygon_area(_convex_intersection(left_polygon, right_polygon)))
    union = left_area + right_area - intersection_area
    return intersection_area / union if union > 0 else 0.0


def match_detections(pt_rows: list[dict[str, Any]], ort_rows: list[dict[str, Any]], iou_floor: float) -> dict[str, Any]:
    if not pt_rows or not ort_rows:
        matched = 0
        return {
            "matched": matched,
            "precision": 1.0 if not ort_rows else 0.0,
            "recall": 1.0 if not pt_rows else 0.0,
            "min_rotated_iou": 1.0 if not pt_rows and not ort_rows else 0.0,
            "max_score_abs_diff": 0.0,
            "max_center_abs_delta_px": 0.0,
            "max_size_abs_delta_px": 0.0,
            "max_angle_abs_delta_rad": 0.0,
            "pairs": [],
        }

    pairs = []
    for i, (left, right) in enumerate(zip(pt_rows, ort_rows)):
        if left["asset_class"] != right["asset_class"]:
            continue
        iou = rotated_iou_float64(left["geometry"], right["geometry"])
        if iou < iou_floor:
            continue
        lg, rg = left["geometry"], right["geometry"]
        pairs.append({
            "pt_index": int(i),
            "onnx_index": int(i),
            "asset_class": left["asset_class"],
            "rotated_iou": iou,
            "score_abs_diff": abs(float(left["confidence"]) - float(right["confidence"])),
            "center_abs_delta_px": max(abs(float(lg["cx"]) - float(rg["cx"])), abs(float(lg["cy"]) - float(rg["cy"]))),
            "size_abs_delta_px": max(abs(float(lg["width"]) - float(rg["width"])), abs(float(lg["height"]) - float(rg["height"]))),
            "angle_abs_delta_rad": angle_delta(float(lg["angle_rad"]), float(rg["angle_rad"])),
        })
    def maximum(key: str) -> float:
        return max((float(pair[key]) for pair in pairs), default=0.0)

    matched = len(pairs)
    return {
        "matched": matched,
        "precision": matched / len(ort_rows),
        "recall": matched / len(pt_rows),
        "min_rotated_iou": min((float(pair["rotated_iou"]) for pair in pairs), default=0.0),
        "max_score_abs_diff": maximum("score_abs_diff"),
        "max_center_abs_delta_px": maximum("center_abs_delta_px"),
        "max_size_abs_delta_px": maximum("size_abs_delta_px"),
        "max_angle_abs_delta_rad": maximum("angle_abs_delta_rad"),
        "pairs": pairs,
    }


def main() -> None:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text())

    import torch
    import torchvision
    import mmcv
    import mmdet
    import mmengine
    import mmrotate
    import onnxruntime as ort
    from mmdet.apis import init_detector
    from mmrotate.utils import register_all_modules

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MMR, text=True).strip()
    if source_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"source commit mismatch: {source_commit}")
    if sha256(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    if sha256(ONNX_PATH) != EXPECTED_ONNX_SHA:
        raise RuntimeError("Cycle 17B ONNX SHA-256 mismatch")

    register_all_modules()
    model = init_detector(str(CFG), str(CKPT), palette="dota", device="cpu")
    model.eval()
    input_np = build_input()
    input_tensor = torch.from_numpy(input_np)

    with torch.inference_mode():
        pt_outputs = flatten_tensors(model.bbox_head(model.extract_feat(input_tensor)))
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    ort_outputs = tuple(session.run(output_names, {"images": input_np}))
    ort_repeat = tuple(session.run(output_names, {"images": input_np}))

    class_names = tuple(model.dataset_meta.get("classes") or ())
    if len(class_names) != 15:
        raise RuntimeError(f"expected DOTA 15-class taxonomy, got {class_names!r}")
    adapter = RTMDetRSharedPostprocessAdapter(model.bbox_head, class_names, angle_version="le90")
    pt_records = adapter.adapt(pt_outputs, backend="pytorch-native")
    ort_records = adapter.adapt(ort_outputs, backend="onnxruntime-cpu")
    ort_repeat_records = adapter.adapt(ort_repeat, backend="onnxruntime-cpu")
    pt_rows, ort_rows, ort_repeat_rows = rows(pt_records), rows(ort_records), rows(ort_repeat_records)

    pt_payload = {"schema": "assetgraph/raw-detections-v1", "backend": "pytorch-native", "detections": pt_rows}
    ort_payload = {"schema": "assetgraph/raw-detections-v1", "backend": "onnxruntime-cpu", "detections": ort_rows}
    pt_path = OUT / "cycle17c_pytorch_raw_detections.json"
    ort_path = OUT / "cycle17c_onnx_raw_detections.json"
    pt_path.write_text(json.dumps(pt_payload, indent=2))
    ort_path.write_text(json.dumps(ort_payload, indent=2))

    gate_cfg = protocol["parity_gate"]
    parity = match_detections(pt_rows, ort_rows, float(gate_cfg["match_iou_floor"]))
    pt_counts = dict(sorted(Counter(row["asset_class"] for row in pt_rows).items()))
    ort_counts = dict(sorted(Counter(row["asset_class"] for row in ort_rows).items()))
    repeat_exact = canonical_digest(ort_rows) == canonical_digest(ort_repeat_rows)
    gates = {
        "source_commit_exact": source_commit == EXPECTED_COMMIT,
        "checkpoint_hash_exact": sha256(CKPT) == EXPECTED_CKPT_SHA,
        "c17b_onnx_hash_exact": sha256(ONNX_PATH) == EXPECTED_ONNX_SHA,
        "final_detection_count_exact": len(pt_rows) == len(ort_rows),
        "per_class_count_exact": pt_counts == ort_counts,
        "matched_precision_pass": parity["precision"] >= float(gate_cfg["matched_precision_min"]),
        "matched_recall_pass": parity["recall"] >= float(gate_cfg["matched_recall_min"]),
        "rotated_iou_pass": parity["min_rotated_iou"] >= float(gate_cfg["matched_rotated_iou_min"]),
        "score_delta_pass": parity["max_score_abs_diff"] <= float(gate_cfg["score_abs_diff_max"]),
        "center_delta_pass": parity["max_center_abs_delta_px"] <= float(gate_cfg["center_abs_delta_px_max"]),
        "size_delta_pass": parity["max_size_abs_delta_px"] <= float(gate_cfg["size_abs_delta_px_max"]),
        "angle_delta_pass": parity["max_angle_abs_delta_rad"] <= float(gate_cfg["angle_abs_delta_rad_max"]),
        "onnx_repeat_digest_exact": repeat_exact,
        "anti_leakage_pass": protocol["anti_leakage"]["transset_access"] is False and protocol["anti_leakage"]["uavobb_access"] is False,
    }
    gates["final_detection_parity_pass"] = all(gates.values())

    report = {
        "schema": "assetgraph-evidence/apache-final-detection-parity-v1",
        "cycle": "17C",
        "protocol_sha256": sha256(PROTOCOL),
        "source": {"repository": "open-mmlab/mmrotate", "commit": source_commit, "license_detected": "Apache-2.0"},
        "model": {"name": protocol["model"]["name"], "checkpoint_sha256": sha256(CKPT)},
        "transport": {"source_cycle": "17B", "onnx_sha256": sha256(ONNX_PATH), "output_names": output_names},
        "input": {"image": IMG.name, "image_sha256": sha256(IMG), "network_input_sha256": hashlib.sha256(input_np.tobytes()).hexdigest()},
        "adapter": protocol["shared_adapter"],
        "measurement_repair": protocol.get("measurement_repair"),
        "final_detections": {
            "pytorch_count": len(pt_rows),
            "onnx_count": len(ort_rows),
            "pytorch_per_class": pt_counts,
            "onnx_per_class": ort_counts,
            "pytorch_payload_sha256": sha256(pt_path),
            "onnx_payload_sha256": sha256(ort_path),
            "onnx_repeat_digest_exact": repeat_exact,
        },
        "parity": parity,
        "gates": gates,
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__, "mmcv": mmcv.__version__,
            "mmdet": mmdet.__version__, "mmengine": mmengine.__version__, "mmrotate": mmrotate.__version__,
            "onnxruntime": ort.__version__,
        },
        "postprocess_included": True,
        "nms_included": True,
        "transset_accessed": False,
        "uavobb_accessed": False,
        "training_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    evidence_path = OUT / "cycle17c_apache_final_detection_parity.json"
    evidence_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "pytorch_count": len(pt_rows), "onnx_count": len(ort_rows),
        "parity": {key: value for key, value in parity.items() if key != "pairs"},
        "gates": gates,
    }, indent=2))
    if not gates["final_detection_parity_pass"]:
        raise SystemExit("Cycle 17C final-detection parity gate failed")


if __name__ == "__main__":
    main()
