from __future__ import annotations

from assetgraph.domain.models import AssetHypothesis, AssetState, DecisionObject, EvidenceRef, Event, Observation
from assetgraph.storage import SQLiteAssetGraphRepository


def test_sqlite_repository_persists_and_deduplicates(tmp_path):
    db = tmp_path / "assetgraph.db"
    ev = EvidenceRef("E1", "frame-1", "a" * 64, observed_at="2026-09-01T10:00:00+00:00")
    obs = Observation(
        observation_id="OBS-1",
        mission_id="M-1",
        observed_at="2026-09-01T10:00:00+00:00",
        sensor="rgb",
        source_id="frame-1",
        sha256="a" * 64,
        asset_class="vehicle",
        confidence=0.91,
        geometry={"centroid": [10.0, 20.0]},
        evidence=(ev,),
    )
    asset = AssetHypothesis(
        asset_id="ASSET-1",
        asset_class="vehicle",
        state=AssetState.ACTIVE,
        identity_confidence=0.88,
        observation_ids=["OBS-1"],
        first_seen=obs.observed_at,
        last_seen=obs.observed_at,
    )
    event = Event(
        event_id="EVT-1",
        event_type="asset_observed",
        asset_ids=("ASSET-1",),
        confidence=0.88,
        started_at=obs.observed_at,
        evidence=(ev,),
    )
    decision = DecisionObject(
        decision_object_id="DO-1",
        mission_id="M-1",
        assets=[asset],
        events=[event],
        claims=[],
        provenance={"runtime": "test"},
    )

    with SQLiteAssetGraphRepository(db) as repo:
        assert repo.put_observation(obs) is True
        assert repo.put_observation(obs) is False
        repo.put_asset("M-1", asset)
        assert repo.append_event("M-1", event) is True
        assert repo.append_event("M-1", event) is False
        repo.put_decision_object(decision)

    with SQLiteAssetGraphRepository(db) as repo:
        got_obs = repo.get_observation("OBS-1")
        assert got_obs is not None and got_obs.evidence[0].sha256 == "a" * 64
        got_asset = repo.get_asset("ASSET-1")
        assert got_asset is not None and got_asset.state == AssetState.ACTIVE
        assert len(repo.list_assets("M-1")) == 1
        assert repo.list_events("M-1")[0].event_type == "asset_observed"
        got_decision = repo.get_decision_object("DO-1")
        assert got_decision is not None
        assert got_decision["schema"] == "assetgraph/decision-object-v1"
        assert got_decision["review"]["status"] == "machine_hypothesis"
