from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from assetgraph.domain.models import AssetHypothesis, AssetState, Event, EvidenceRef, Observation


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass(frozen=True, slots=True)
class TemporalPolicy:
    absent_after_seconds: float = 30.0
    moved_distance: float = 10.0
    reacquire_after_seconds: float = 2.0


class TemporalEngine:
    def __init__(self, policy: TemporalPolicy | None = None) -> None:
        self.policy = policy or TemporalPolicy()

    def apply_observation(self, asset: AssetHypothesis, observation: Observation) -> list[Event]:
        events: list[Event] = []
        prior_last = asset.last_seen
        prior_state = asset.state
        prior_centroid = asset.attributes.get("last_centroid")
        current_centroid = observation.geometry.get("centroid")

        if prior_last is not None:
            gap = (_parse(observation.observed_at) - _parse(prior_last)).total_seconds()
            if gap >= self.policy.reacquire_after_seconds and prior_state in {AssetState.ABSENT, AssetState.OBSERVED, AssetState.ACTIVE}:
                events.append(Event(
                    event_id=f"EVT-{asset.asset_id}-reacquired-{observation.observation_id}",
                    event_type="asset_reacquired",
                    asset_ids=(asset.asset_id,),
                    confidence=asset.identity_confidence,
                    started_at=observation.observed_at,
                    evidence=observation.evidence,
                    attributes={"gap_seconds": gap},
                ))
                asset.state = AssetState.REACQUIRED

        if prior_centroid is not None and current_centroid is not None:
            try:
                dx = float(current_centroid[0]) - float(prior_centroid[0])
                dy = float(current_centroid[1]) - float(prior_centroid[1])
                distance = (dx * dx + dy * dy) ** 0.5
            except Exception:
                distance = 0.0
            if distance >= self.policy.moved_distance:
                events.append(Event(
                    event_id=f"EVT-{asset.asset_id}-moved-{observation.observation_id}",
                    event_type="asset_moved",
                    asset_ids=(asset.asset_id,),
                    confidence=min(observation.confidence, asset.identity_confidence),
                    started_at=observation.observed_at,
                    evidence=observation.evidence,
                    attributes={"distance": distance, "from": prior_centroid, "to": current_centroid},
                ))
                asset.state = AssetState.MOVED

        if current_centroid is not None:
            asset.attributes["last_centroid"] = current_centroid
        return events

    def mark_absent(self, asset: AssetHypothesis, *, now: str) -> Event | None:
        if not asset.last_seen:
            return None
        gap = (_parse(now) - _parse(asset.last_seen)).total_seconds()
        if gap < self.policy.absent_after_seconds or asset.state == AssetState.ABSENT:
            return None
        asset.state = AssetState.ABSENT
        return Event(
            event_id=f"EVT-{asset.asset_id}-absent-{int(_parse(now).timestamp())}",
            event_type="asset_absent",
            asset_ids=(asset.asset_id,),
            confidence=asset.identity_confidence,
            started_at=now,
            attributes={"gap_seconds": gap},
        )
