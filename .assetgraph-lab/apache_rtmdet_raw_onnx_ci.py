from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "evidence"
OUT.mkdir(exist_ok=True)
PROTOCOL = ROOT / "cycle17b_apache_onnx_protocol.json"
MMR = pathlib.Path(os.environ["MMROTATE_DIR"]).resolve()
CFG = MMR / "configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py"
CKPT = pathlib.Path(os.environ["RTMDET_CKPT"]).resolve()
IMG = pathlib.Path(os.environ["RTMDET_IMAGE"]).resolve()
ONNX_PATH = OUT / "rtmdet_r_tiny_raw_1024.onnx"
EXPECTED_COMMIT = "3ff004eb21ea040455b5585db229edba4037f1bf"
EXPECTED_CKPT_SHA = "081a74b2c84407347c7b62b45a8647c70a816452027aaf2e0d6cefae3a2b6e9d"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MMR, text=True).strip()


def flatten_tensors(value: Any) -> tuple:
    import torch
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(flatten_tensors(item))
        return tuple(out)
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(flatten_tensors(value[key]))
        return tuple(out)
    raise TypeError(f"unsupported raw model output type: {type(value)!r}")


def build_input() -> np.ndarray:
    image = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read input image: {IMG}")
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mean = np.asarray([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(image, dtype=np.float32)


def stats_for(pt: np.ndarray, ort: np.ndarray) -> dict[str, Any]:
    if pt.shape != ort.shape:
        return {"shape_exact": False, "pt_shape": list(pt.shape), "ort_shape": list(ort.shape)}
    a = pt.astype(np.float64, copy=False).ravel()
    b = ort.astype(np.float64, copy=False).ravel()
    diff = np.abs(a - b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom else (1.0 if np.allclose(a, b) else 0.0)
    return {
        "shape_exact": True,
        "shape": list(pt.shape),
        "elements": int(a.size),
        "cosine": cosine,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "p99_abs_diff": float(np.quantile(diff, 0.99)) if diff.size else 0.0,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "pt_min": float(a.min()) if a.size else 0.0,
        "pt_max": float(a.max()) if a.size else 0.0,
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
    import onnx
    import onnxruntime as ort
    from mmdet.apis import init_detector
    from mmrotate.utils import register_all_modules

    source_commit = git_head()
    if source_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"source commit mismatch: {source_commit}")
    if sha256(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError("checkpoint SHA-256 mismatch")

    register_all_modules()
    model = init_detector(str(CFG), str(CKPT), palette="dota", device="cpu")
    model.eval()

    class RawWrapper(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, images):
            feats = self.detector.extract_feat(images)
            raw = self.detector.bbox_head(feats)
            return flatten_tensors(raw)

    wrapper = RawWrapper(model).eval()
    input_np = build_input()
    input_tensor = torch.from_numpy(input_np)

    with torch.inference_mode():
        pt_outputs = wrapper(input_tensor)
    if not isinstance(pt_outputs, tuple) or not pt_outputs:
        raise RuntimeError("raw wrapper did not expose tensor outputs")

    output_names = [f"raw_{i}" for i in range(len(pt_outputs))]
    torch.onnx.export(
        wrapper,
        input_tensor,
        str(ONNX_PATH),
        export_params=True,
        opset_version=int(protocol["export"]["opset"]),
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(output_names, {"images": input_np})

    per_output = []
    for name, pt, ov in zip(output_names, pt_outputs, ort_outputs):
        row = {"name": name, **stats_for(pt.detach().cpu().numpy(), ov)}
        per_output.append(row)

    all_shapes = all(row.get("shape_exact") is True for row in per_output)
    cosine_min = min(row.get("cosine", -1.0) for row in per_output)
    mean_max = max(row.get("mean_abs_diff", float("inf")) for row in per_output)
    p99_max = max(row.get("p99_abs_diff", float("inf")) for row in per_output)
    abs_max = max(row.get("max_abs_diff", float("inf")) for row in per_output)

    gate_cfg = protocol["parity_gate"]
    gates = {
        "source_commit_exact": source_commit == EXPECTED_COMMIT,
        "checkpoint_hash_exact": sha256(CKPT) == EXPECTED_CKPT_SHA,
        "all_output_shapes_exact": all_shapes,
        "cosine_gate": cosine_min >= float(gate_cfg["cosine_min"]),
        "mean_abs_diff_gate": mean_max <= float(gate_cfg["mean_abs_diff_max"]),
        "p99_abs_diff_gate": p99_max <= float(gate_cfg["p99_abs_diff_max"]),
        "max_abs_diff_gate": abs_max <= float(gate_cfg["max_abs_diff_max"]),
        "onnx_checker_pass": True,
        "onnxruntime_cpu_pass": True,
        "anti_leakage_pass": protocol["anti_leakage"]["transset_access"] is False and protocol["anti_leakage"]["uavobb_training"] is False,
    }
    gates["raw_onnx_parity_pass"] = all(gates.values())

    report = {
        "schema": "assetgraph-evidence/apache-raw-onnx-parity-v1",
        "cycle": "17B",
        "protocol_sha256": sha256(PROTOCOL),
        "source": {
            "repository": "open-mmlab/mmrotate",
            "commit": source_commit,
            "license_detected": "Apache-2.0",
        },
        "model": {
            "name": protocol["model"]["name"],
            "checkpoint_sha256": sha256(CKPT),
            "checkpoint_bytes": CKPT.stat().st_size,
        },
        "input": {
            "image": IMG.name,
            "image_sha256": sha256(IMG),
            "network_input_shape": list(input_np.shape),
            "network_input_sha256": hashlib.sha256(input_np.tobytes()).hexdigest(),
        },
        "onnx": {
            "path": ONNX_PATH.name,
            "sha256": sha256(ONNX_PATH),
            "bytes": ONNX_PATH.stat().st_size,
            "opset": protocol["export"]["opset"],
            "output_count": len(output_names),
        },
        "parity": {
            "cosine_min": cosine_min,
            "mean_abs_diff_worst": mean_max,
            "p99_abs_diff_worst": p99_max,
            "max_abs_diff_worst": abs_max,
            "per_output": per_output,
        },
        "gates": gates,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "mmcv": mmcv.__version__,
            "mmdet": mmdet.__version__,
            "mmengine": mmengine.__version__,
            "mmrotate": mmrotate.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "postprocess_included": False,
        "nms_included": False,
        "transset_accessed": False,
        "uavobb_training_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    evidence_path = OUT / "cycle17b_apache_raw_onnx_parity.json"
    evidence_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "onnx_bytes": report["onnx"]["bytes"],
        "outputs": len(output_names),
        "cosine_min": cosine_min,
        "p99_abs_diff_worst": p99_max,
        "max_abs_diff_worst": abs_max,
        "gates": gates,
    }, indent=2))

    if not gates["raw_onnx_parity_pass"]:
        raise SystemExit("Cycle 17B raw ONNX parity gate failed")


if __name__ == "__main__":
    main()
