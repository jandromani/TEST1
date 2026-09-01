from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStatus(str, Enum):
    MACHINE_HYPOTHESIS = "machine_hypothesis"
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class AssetState(str, Enum):
    UNKNOWN = "unknown"
    OBSERVED = "observed"
    ACTIVE = "active"
    MOVED = "moved"
    ABSENT = "absent"
    REACQUIRED = "reacquired"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    evidence_id: str
    source_id: str
    sha256: str
    observed_at: str | None = None
    frame_id: str | None = None
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    mission_id: str
    observed_at: str
    sensor: str
    source_id: str
    sha256: str
    asset_class: str
    confidence: float
    geometry: Mapping[str, Any] = field(default_factory=dict)
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    subtype_hypothesis: str | None = None
    subtype_confidence: float | None = None
    model: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.subtype_confidence is not None and not 0.0 <= self.subtype_confidence <= 1.0:
            raise ValueError("subtype_confidence must be in [0, 1]")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hex digest")


@dataclass(slots=True)
class AssetHypothesis:
    asset_id: str
    asset_class: str
    state: AssetState = AssetState.UNKNOWN
    identity_confidence: float = 0.0
    observation_ids: list[str] = field(default_factory=list)
    first_seen: str | None = None
    last_seen: str | None = None
    subtype_hypothesis: str | None = None
    subtype_confidence: float | None = None
    identity_alternatives: list[Mapping[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def attach(self, observation: Observation, identity_confidence: float) -> None:
        if observation.asset_class != self.asset_class:
            raise ValueError("coarse asset class mismatch")
        if not 0.0 <= identity_confidence <= 1.0:
            raise ValueError("identity_confidence must be in [0, 1]")
        if observation.observation_id not in self.observation_ids:
            self.observation_ids.append(observation.observation_id)
        self.identity_confidence = identity_confidence
        self.first_seen = self.first_seen or observation.observed_at
        self.last_seen = observation.observed_at
        self.state = AssetState.OBSERVED if self.state == AssetState.UNKNOWN else AssetState.ACTIVE
        if observation.subtype_hypothesis:
            self.subtype_hypothesis = observation.subtype_hypothesis
            self.subtype_confidence = observation.subtype_confidence


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    event_type: str
    asset_ids: tuple[str, ...]
    confidence: float
    started_at: str
    ended_at: str | None = None
    evidence: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str
    confidence: float
    evidence: tuple[EvidenceRef, ...]
    scope: str
    status: str = "machine_hypothesis"
    limitations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class DecisionObject:
    decision_object_id: str
    mission_id: str
    assets: list[AssetHypothesis]
    events: list[Event]
    claims: list[Claim]
    provenance: Mapping[str, Any]
    review_status: ReviewStatus = ReviewStatus.MACHINE_HYPOTHESIS
    analyst_id: str | None = None
    generated_at: str = field(default_factory=utc_now_iso)
    schema: str = "assetgraph/decision-object-v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["review"] = {
            "status": self.review_status.value,
            "analyst": self.analyst_id,
        }
        payload.pop("review_status", None)
        payload.pop("analyst_id", None)
        payload["mission"] = {"mission_id": self.mission_id, "type": "asset_intelligence"}
        payload.pop("mission_id", None)
        for asset in payload["assets"]:
            asset["asset_hypothesis_id"] = asset.pop("asset_id")
            asset["confidence"] = asset.pop("identity_confidence")
            if isinstance(asset.get("state"), Enum):
                asset["state"] = asset["state"].value
        return payload
