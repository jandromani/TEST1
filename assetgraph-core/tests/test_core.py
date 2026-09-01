from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from assetgraph.domain.models import AssetState
from assetgraph.evidence.integrity import ReplayManifest, sha256_file
from assetgraph.pipeline import AssetGraphPipeline, MissionState
from assetgraph.runtimes.base import RawDetection, RuntimeInfo
from assetgraph.runtimes.registry import RuntimeRegistration, RuntimeRegistry


class FakeRuntime:
    def __init__(self) -> None:
        self._calls = 0
        self._info = RuntimeInfo(
            runtime_id="fake-test-runtime",
            family="deterministic-test",
            version="1",
            backend="python",
            license="Apache-2.0",
            product_status="PRODUCT_CANDIDATE",
            model_sha256="0" * 64,
        )

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def healthcheck(self):
        return {"ok": True}

    def infer(self, source, *, context=None):
        self._calls += 1
        x = 10.0 if self._calls == 1 else 25.0
        return [RawDetection(
            asset_class="vehicle",
            confidence=0.95,
            geometry={"centroid": [x, 10.0]},
            subtype_hypothesis="car",
            subtype_confidence=0.80,
        )]


def test_vertical_slice_produces_persistent_asset_event_and_decision(tmp_path: Path):
    image = tmp_path / "frame.bin"
    image.write_bytes(b"real-source-bytes-for-test")
    runtime = FakeRuntime()
    pipe = AssetGraphPipeline(runtime)
    state = MissionState("M-TEST")

    pipe.ingest_image(state, image, source_id="F1", observed_at="2026-09-01T10:00:00+00:00", sensor="rgb")
    pipe.ingest_image(state, image, source_id="F2", observed_at="2026-09-01T10:00:01+00:00", sensor="rgb")

    assert len(state.assets) == 1
    asset = next(iter(state.assets.values()))
    assert len(asset.observation_ids) == 2
    assert asset.state == AssetState.MOVED
    assert [e.event_type for e in state.events] == ["asset_moved"]

    decision = pipe.compile_decision_object(state).to_dict()
    assert decision["schema"] == "assetgraph/decision-object-v1"
    assert decision["decision_object_id"] == "DO-M-TEST"
    assert len(decision["assets"]) == 1
    assert decision["review"]["status"] == "machine_hypothesis"
    assert decision["claims"][0]["evidence"][0]["sha256"] == sha256_file(image)


def test_runtime_registry_blocks_unpromoted_runtime():
    reg = RuntimeRegistry()
    runtime = FakeRuntime()
    info = RuntimeInfo(
        runtime_id="blocked",
        family="lab",
        version="1",
        backend="python",
        license="REVIEW_REQUIRED",
        product_status="LAB_ONLY",
    )
    reg.register(RuntimeRegistration(info=info, factory=lambda: runtime, evidence_ids=("E1",), promotion_gates={"license": False}))
    with pytest.raises(RuntimeError):
        reg.load("blocked", require_product_candidate=True)


def test_runtime_registry_allows_fully_gated_candidate():
    reg = RuntimeRegistry()
    runtime = FakeRuntime()
    reg.register(RuntimeRegistration(
        info=runtime.info,
        factory=lambda: runtime,
        evidence_ids=("E17A",),
        promotion_gates={"license": True, "runtime": True, "accuracy": True},
    ))
    assert reg.load("fake-test-runtime", require_product_candidate=True).healthcheck()["ok"] is True


def test_replay_manifest_rejects_tampering():
    manifest = ReplayManifest(
        run_id="RUN-1",
        inputs={"image": "a" * 64},
        components={"runtime": "b" * 64},
        parameters={"conf": 0.3},
        outputs={"decision": "c" * 64},
    )
    manifest.verify_inputs({"image": "a" * 64})
    with pytest.raises(ValueError):
        manifest.verify_inputs({"image": "d" * 64})
