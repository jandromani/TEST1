from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping

from assetgraph.domain.models import AssetHypothesis, Claim, DecisionObject, EvidenceRef, Observation
from assetgraph.identity.resolver import IdentityResolver, normalized_motion_scorer
from assetgraph.runtimes.base import PerceptionRuntime
from assetgraph.temporal.engine import TemporalEngine


@dataclass(slots=True)
class MissionState:
    mission_id: str
    assets: dict[str, AssetHypothesis] = field(default_factory=dict)
    observations: dict[str, Observation] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)


class AssetGraphPipeline:
    def __init__(
        self,
        runtime: PerceptionRuntime,
        *,
        identity: IdentityResolver | None = None,
        temporal: TemporalEngine | None = None,
    ) -> None:
        self.runtime = runtime
        self.identity = identity or IdentityResolver()
        self.temporal = temporal or TemporalEngine()

    def ingest_image(
        self,
        state: MissionState,
        source: str | Path | bytes,
        *,
        source_id: str,
        observed_at: str,
        sensor: str,
        telemetry: Mapping[str, Any] | None = None,
    ) -> list[Observation]:
        raw_bytes = source if isinstance(source, bytes) else Path(source).read_bytes()
        source_sha = hashlib.sha256(raw_bytes).hexdigest()
        detections = self.runtime.infer(source, context={"mission_id": state.mission_id, "sensor": sensor})
        observations: list[Observation] = []
        for index, det in enumerate(detections):
            oid = f"OBS-{state.mission_id}-{source_id}-{index:04d}"
            evidence = EvidenceRef(
                evidence_id=f"EVID-{oid}",
                source_id=source_id,
                sha256=source_sha,
                observed_at=observed_at,
            )
            obs = Observation(
                observation_id=oid,
                mission_id=state.mission_id,
                observed_at=observed_at,
                sensor=sensor,
                source_id=source_id,
                sha256=source_sha,
                asset_class=det.asset_class,
                confidence=det.confidence,
                geometry=dict(det.geometry),
                telemetry=dict(telemetry or {}),
                subtype_hypothesis=det.subtype_hypothesis,
                subtype_confidence=det.subtype_confidence,
                model={
                    "runtime_id": self.runtime.info.runtime_id,
                    "version": self.runtime.info.version,
                    "model_sha256": self.runtime.info.model_sha256,
                },
                evidence=(evidence,),
            )
            state.observations[oid] = obs
            observations.append(obs)
            self._resolve_observation(state, obs)
        return observations

    def _resolve_observation(self, state: MissionState, obs: Observation) -> AssetHypothesis | None:
        decision = self.identity.resolve(obs, list(state.assets.values()), scorer=normalized_motion_scorer)
        if decision.action == "abstain":
            return None
        if decision.action == "new_asset":
            aid = f"ASSET-{state.mission_id}-{len(state.assets)+1:06d}"
            asset = AssetHypothesis(aid, obs.asset_class)
            asset.attach(obs, max(0.50, obs.confidence))
            if obs.geometry.get("centroid") is not None:
                asset.attributes["last_centroid"] = obs.geometry["centroid"]
            state.assets[aid] = asset
            return asset
        asset = state.assets[decision.selected_asset_id]  # type: ignore[index]
        events = self.temporal.apply_observation(asset, obs)
        asset.attach(obs, decision.confidence)
        state.events.extend(events)
        return asset

    def compile_decision_object(self, state: MissionState) -> DecisionObject:
        claims: list[Claim] = []
        for event in state.events:
            claims.append(Claim(
                claim_id=f"CLM-{event.event_id}",
                text=f"AssetGraph observed event: {event.event_type}",
                confidence=event.confidence,
                evidence=event.evidence,
                scope=state.mission_id,
                limitations=("machine-generated claim; analyst review may be required",),
            ))
        return DecisionObject(
            decision_object_id=f"DO-{state.mission_id}",
            mission_id=state.mission_id,
            assets=list(state.assets.values()),
            events=list(state.events),
            claims=claims,
            provenance={"runtime": self.runtime.info.runtime_id, "runtime_version": self.runtime.info.version},
        )
