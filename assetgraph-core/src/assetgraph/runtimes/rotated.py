from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .base import RawDetection, RuntimeInfo


RTMDET_R_TINY_C17C_INFO = RuntimeInfo(
    runtime_id="rtmdet-r-tiny-apache-onnx-c17c",
    family="RTMDet-R Tiny",
    version="cycle17c",
    backend="onnxruntime-cpu+shared-rotated-postprocess",
    license="Apache-2.0",
    product_status="TRANSPORT_VALIDATED_DOMAIN_UNVALIDATED",
    model_sha256="5336b5a5a54bfcdfca4e0dde3dc3710498da7f5885de18237c40c7f405808699",
    source_commit="3ff004eb21ea040455b5585db229edba4037f1bf",
    metadata={
        "input_shape": [1, 3, 1024, 1024],
        "output_count": 9,
        "raw_parity_evidence": "Cycle17B/cycle17b_apache_raw_onnx_parity.json",
        "final_parity_evidence": "Cycle17C/cycle17c_apache_final_detection_parity.json",
        "domain_accuracy_gate": False,
        "weights_status": "DOTA_EVALUATION_ONLY_NOT_PRODUCT_WEIGHTS",
    },
)


@dataclass(frozen=True, slots=True)
class CanonicalOrientedBox:
    cx: float
    cy: float
    width: float
    height: float
    angle_rad: float
    angle_version: str = "le90"

    def __post_init__(self) -> None:
        values = (self.cx, self.cy, self.width, self.height, self.angle_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("oriented box values must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("oriented box width and height must be positive")
        if self.angle_version != "le90":
            raise ValueError("Cycle 17C contract requires le90 angles")

    def to_geometry(self) -> dict[str, Any]:
        return {
            "type": "oriented_box",
            "format": "cx_cy_width_height_angle_rad",
            "angle_version": self.angle_version,
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": float(self.width),
            "height": float(self.height),
            "angle_rad": float(self.angle_rad),
        }


def validate_rtmdet_r_tiny_raw_outputs(outputs: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    """Validate the exact fixed-shape tensor boundary proven by Cycle 17B."""
    if len(outputs) != 9:
        raise ValueError(f"expected nine RTMDet-R raw outputs, got {len(outputs)}")
    expected_channels = (15, 15, 15, 4, 4, 4, 1, 1, 1)
    expected_spatial = ((128, 128), (64, 64), (32, 32)) * 3
    shapes: list[tuple[int, ...]] = []
    for index, (output, channels, spatial) in enumerate(zip(outputs, expected_channels, expected_spatial)):
        shape = tuple(int(value) for value in getattr(output, "shape", ()))
        if len(shape) != 4:
            raise ValueError(f"output {index} must be NCHW, got {shape!r}")
        if shape != (1, channels, *spatial):
            raise ValueError(f"output {index} violates the frozen Cycle 17B shape: {shape!r}")
        shapes.append(shape)
    return tuple(shapes)


def canonicalize_rtmdet_r_detection(record: Mapping[str, Any]) -> RawDetection:
    """Convert a Cycle 17C final record into the stable AssetGraph contract."""
    geometry = record.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("final detection requires geometry")
    if geometry.get("type") != "oriented_box":
        raise ValueError("final detection geometry must be oriented_box")
    if geometry.get("format") != "cx_cy_width_height_angle_rad":
        raise ValueError("unsupported oriented box format")
    box = CanonicalOrientedBox(
        cx=float(geometry["cx"]),
        cy=float(geometry["cy"]),
        width=float(geometry["width"]),
        height=float(geometry["height"]),
        angle_rad=float(geometry["angle_rad"]),
        angle_version=str(geometry.get("angle_version", "le90")),
    )
    asset_class = str(record.get("asset_class") or "")
    if not asset_class:
        raise ValueError("final detection requires asset_class")
    confidence = float(record["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    subtype_confidence = record.get("subtype_confidence")
    if subtype_confidence is not None:
        subtype_confidence = float(subtype_confidence)
        if not 0.0 <= subtype_confidence <= 1.0:
            raise ValueError("subtype_confidence must be in [0, 1]")
    return RawDetection(
        asset_class=asset_class,
        confidence=confidence,
        geometry=box.to_geometry(),
        subtype_hypothesis=record.get("subtype_hypothesis"),
        subtype_confidence=subtype_confidence,
        metadata=dict(record.get("metadata") or {}),
    )
