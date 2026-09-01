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


def match_detections(pt_rows: list[dict[str, Any]], ort_rows: list[dict[str, Any]], iou_floor: float) -> dict[str, Any]:
    import torch
    from mmcv.ops import box_iou_rotated

    def boxes(items: list[dict[str, Any]]):
        return torch.tensor([[
            row["geometry"]["cx"], row["geometry"]["cy"],
            row["geometry"]["width"], row["geometry"]["height"],
            row["geometry"]["angle_rad"],
        ] for row in items], dtype=torch.float32)

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

    ious = box_iou_rotated(boxes(pt_rows), boxes(ort_rows)).detach().cpu().numpy()
    for i, left in enumerate(pt_rows):
        for j, right in enumerate(ort_rows):
            if left["asset_class"] != right["asset_class"]:
                ious[i, j] = -1.0

    work = ious.copy()
    pairs = []
    while work.size:
        flat = int(np.argmax(work))
        i, j = np.unravel_index(flat, work.shape)
        iou = float(work[i, j])
        if iou < iou_floor:
            break
        left, right = pt_rows[i], ort_rows[j]
        lg, rg = left["geometry"], right["geometry"]
        pairs.append({
            "pt_index": int(i),
            "onnx_index": int(j),
            "asset_class": left["asset_class"],
            "rotated_iou": iou,
            "score_abs_diff": abs(float(left["confidence"]) - float(right["confidence"])),
            "center_abs_delta_px": max(abs(float(lg["cx"]) - float(rg["cx"])), abs(float(lg["cy"]) - float(rg["cy"]))),
            "size_abs_delta_px": max(abs(float(lg["width"]) - float(rg["width"])), abs(float(lg["height"]) - float(rg["height"]))),
            "angle_abs_delta_rad": angle_delta(float(lg["angle_rad"]), float(rg["angle_rad"])),
        })
        work[i, :] = -1.0
        work[:, j] = -1.0

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
