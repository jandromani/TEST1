from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class CanonicalRawDetection:
    asset_class: str
    confidence: float
    geometry: Mapping[str, Any]
    subtype_hypothesis: str | None
    subtype_confidence: float | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RTMDetRSharedPostprocessAdapter:
    """Shared final-detection boundary for native and transported tensors.

    The detector head remains the frozen Apache-2.0 MMRotate implementation for
    Cycle 17C. Both PyTorch and ONNX Runtime outputs enter this exact same code
    path, preventing backend-specific thresholds, decoders or NMS from hiding a
    deployment regression.
    """

    def __init__(self, head: Any, class_names: Sequence[str], *, angle_version: str = "le90") -> None:
        self.head = head
        self.class_names = tuple(str(name) for name in class_names)
        self.angle_version = angle_version

    @staticmethod
    def _as_tensor(value: Any):
        import numpy as np
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value)).cpu()
        raise TypeError(f"unsupported RTMDet-R tensor type: {type(value)!r}")

    def adapt(self, raw_outputs: Sequence[Any], *, backend: str) -> list[CanonicalRawDetection]:
        import torch

        if len(raw_outputs) != 9:
            raise ValueError(f"RTMDet-R Tiny adapter requires nine raw tensors, got {len(raw_outputs)}")
        tensors = tuple(self._as_tensor(value) for value in raw_outputs)
        cls_scores = list(tensors[0:3])
        bbox_preds = list(tensors[3:6])
        angle_preds = list(tensors[6:9])
        for level, (cls_score, bbox_pred, angle_pred) in enumerate(zip(cls_scores, bbox_preds, angle_preds)):
            if cls_score.ndim != 4 or bbox_pred.ndim != 4 or angle_pred.ndim != 4:
                raise ValueError(f"level {level} tensors must be NCHW")
            if cls_score.shape[0] != 1 or bbox_pred.shape[0] != 1 or angle_pred.shape[0] != 1:
                raise ValueError("Cycle 17C contract supports batch=1")
            if cls_score.shape[-2:] != bbox_pred.shape[-2:] or cls_score.shape[-2:] != angle_pred.shape[-2:]:
                raise ValueError(f"level {level} spatial shapes disagree")

        img_meta = {
            "img_shape": (1024, 1024, 3),
            "ori_shape": (1024, 1024, 3),
            "pad_shape": (1024, 1024, 3),
            "scale_factor": (1.0, 1.0),
        }
        with torch.inference_mode():
            instances = self.head.predict_by_feat(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                angle_preds=angle_preds,
                batch_img_metas=[img_meta],
                cfg=None,
                rescale=False,
                with_nms=True,
            )[0]

        boxes = instances.bboxes.tensor.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        records: list[CanonicalRawDetection] = []
        for box, score, label in zip(boxes, scores, labels):
            class_index = int(label)
            if not 0 <= class_index < len(self.class_names):
                raise ValueError(f"class index outside declared taxonomy: {class_index}")
            cx, cy, width, height, angle = (float(v) for v in box)
            values = (cx, cy, width, height, angle, float(score))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("non-finite final detection")
            if width <= 0 or height <= 0:
                raise ValueError("non-positive final oriented box")
            asset_class = self.class_names[class_index]
            records.append(CanonicalRawDetection(
                asset_class=asset_class,
                confidence=float(score),
                geometry={
                    "type": "oriented_box",
                    "format": "cx_cy_width_height_angle_rad",
                    "angle_version": self.angle_version,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                    "angle_rad": angle,
                },
                subtype_hypothesis=asset_class,
                subtype_confidence=float(score),
                metadata={
                    "backend": backend,
                    "family": "rotated-rtmdet",
                    "postprocess": "mmrotate-shared-cycle17c",
                    "class_index": class_index,
                },
            ))
        records.sort(key=lambda row: (
            row.asset_class,
            row.geometry["cx"],
            row.geometry["cy"],
            row.geometry["width"],
            row.geometry["height"],
            row.geometry["angle_rad"],
            -row.confidence,
        ))
        return records
