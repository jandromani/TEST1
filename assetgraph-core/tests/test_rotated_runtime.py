from __future__ import annotations

from dataclasses import dataclass

import pytest

from assetgraph.runtimes.rotated import (
    RTMDET_R_TINY_C17C_INFO,
    canonicalize_rtmdet_r_detection,
    validate_rtmdet_r_tiny_raw_outputs,
)


@dataclass
class ShapeOnly:
    shape: tuple[int, ...]


def test_cycle17b_raw_tensor_contract_is_explicit():
    shapes = [
        (1, 15, 128, 128), (1, 15, 64, 64), (1, 15, 32, 32),
        (1, 4, 128, 128), (1, 4, 64, 64), (1, 4, 32, 32),
        (1, 1, 128, 128), (1, 1, 64, 64), (1, 1, 32, 32),
    ]
    assert validate_rtmdet_r_tiny_raw_outputs([ShapeOnly(shape) for shape in shapes]) == tuple(shapes)
    with pytest.raises(ValueError):
        validate_rtmdet_r_tiny_raw_outputs([ShapeOnly((1, 15, 128, 128))])


def test_cycle17c_final_detection_maps_to_core_contract():
    raw = canonicalize_rtmdet_r_detection({
        "asset_class": "plane",
        "confidence": 0.91,
        "geometry": {
            "type": "oriented_box",
            "format": "cx_cy_width_height_angle_rad",
            "angle_version": "le90",
            "cx": 100.0,
            "cy": 120.0,
            "width": 40.0,
            "height": 20.0,
            "angle_rad": 0.25,
        },
        "subtype_hypothesis": "plane",
        "subtype_confidence": 0.91,
        "metadata": {"backend": "onnxruntime-cpu"},
    })
    assert raw.asset_class == "plane"
    assert raw.geometry["angle_version"] == "le90"
    assert raw.metadata["backend"] == "onnxruntime-cpu"


def test_c17c_profile_cannot_claim_product_accuracy():
    assert RTMDET_R_TINY_C17C_INFO.license == "Apache-2.0"
    assert RTMDET_R_TINY_C17C_INFO.product_status == "TRANSPORT_VALIDATED_DOMAIN_UNVALIDATED"
    assert RTMDET_R_TINY_C17C_INFO.metadata["domain_accuracy_gate"] is False
