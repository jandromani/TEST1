from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

DecisionState = Literal["CONFIRMED", "CANDIDATE", "UNKNOWN", "NEW"]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evidence_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _norm(v: Sequence[float] | None) -> list[float] | None:
    if v is None:
        return None
    x = [float(a) for a in v]
    n = math.sqrt(sum(a * a for a in x))
    if n <= 1e-12:
        return [0.0 for _ in x]
    return [a / n for a in x]


def cosine01(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    na, nb = _norm(a), _norm(b)
    assert na is not None and nb is not None
    return max(0.0, min(1.0, (sum(x * y for x, y in zip(na, nb)) + 1.0) / 2.0))


def geometry_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    """Compare [log_area, log_aspect, norm_x, norm_y].

    This deliberately stays sensor-agnostic. A deployment may replace it with
    geospatial/pose-aware scoring while retaining the Decision Object contract.
    """
    if a is None or b is None or len(a) < 4 or len(b) < 4:
        return None
    ds = math.exp(-abs(float(a[0]) - float(b[0])))
    ar = math.exp(-abs(float(a[1]) - float(b[1])))
    pd = math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))
    pos = math.exp(-2.0 * pd)
    return max(0.0, min(1.0, 0.5 * ds + 0.3 * ar + 0.2 * pos))


@dataclass(frozen=True)
class Provenance:
    source_uri: str | None = None
    source_sha256: str | None = None
    detector_name: str | None = None
    detector_version: str | None = None
    detector_sha256: str | None = None
    embedding_model: str | None = None
    embedding_model_version: str | None = None
    embedding_model_sha256: str | None = None
    sensor_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    timestamp: str
    class_name: str
    appearance_embedding: list[float] | None = None
    secondary_embedding: list[float] | None = None
    context_embedding: list[float] | None = None
    geometry_signature: list[float] | None = None
    geo: dict[str, float] | None = None
    confidence: float | None = None
    evidence_refs: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    attributes: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return evidence_hash(asdict(self))


@dataclass(frozen=True)
class AssetMemory:
    asset_id: str
    class_name: str
    appearance_embedding: list[float] | None = None
    secondary_embedding: list[float] | None = None
    context_embedding: list[float] | None = None
    geometry_signature: list[float] | None = None
    first_seen: str | None = None
    last_seen: str | None = None
    observation_count: int = 0
    evidence_refs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentityPolicy:
    policy_version: str = "assetgraph-identity-policy/v1"
    appearance_weight: float = 0.68
    secondary_weight: float = 0.10
    geometry_weight: float = 0.08
    context_weight: float = 0.14
    confirm_min_score: float = 1.10
    confirm_min_margin: float = 1.10
    candidate_min_score: float = 0.70
    new_max_score: float = 0.45
    class_must_match: bool = True
    require_two_candidates_for_margin: bool = False
    max_candidates: int = 10
    calibration_ref: str = "UNSET_FAIL_CLOSED"

    def validate(self) -> None:
        weights = [self.appearance_weight, self.secondary_weight, self.geometry_weight, self.context_weight]
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("IdentityPolicy weights must be non-negative and sum to > 0")
        for name in ("confirm_min_score", "candidate_min_score", "new_max_score"):
            v = float(getattr(self, name))
            if not 0 <= v <= 1.10:
                raise ValueError(f"{name} must be in [0, 1.10]")
        if self.confirm_min_margin < 0:
            raise ValueError("confirm_min_margin must be >= 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")


@dataclass(frozen=True)
class CandidateScore:
    asset_id: str
    class_name: str
    score: float
    components: dict[str, float | None]
    evidence_refs: list[str]
    observation_count: int
    last_seen: str | None


@dataclass(frozen=True)
class IdentityDecisionObject:
    schema: str
    decision_id: str
    observation_id: str
    decision: DecisionState
    resolved_asset_id: str | None
    confidence: float
    top1_margin: float | None
    candidates: list[CandidateScore]
    reason_codes: list[str]
    counterevidence: list[str]
    evidence_refs: list[str]
    observation_fingerprint: str
    policy_version: str
    policy_calibration_ref: str
    model_versions: dict[str, str | None]
    created_at: str
    decision_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["decision_fingerprint"]:
            d["decision_fingerprint"] = evidence_hash({k: v for k, v in d.items() if k not in {"decision_fingerprint", "created_at"}})
        return d


class IdentityResolver:
    """Model-agnostic persistent identity resolver.

    Safety contract:
    - no GT fields exist in runtime inputs;
    - confirmation is impossible with the default fail-closed policy;
    - ambiguous evidence returns CANDIDATE/UNKNOWN rather than forced identity;
    - NEW is reserved for observations with no plausible compatible memory.
    """

    def __init__(self, policy: IdentityPolicy | None = None):
        self.policy = policy or IdentityPolicy()
        self.policy.validate()

    def _candidate(self, obs: Observation, mem: AssetMemory) -> CandidateScore | None:
        if self.policy.class_must_match and obs.class_name.lower() != mem.class_name.lower():
            return None
        comp = {
            "appearance": cosine01(obs.appearance_embedding, mem.appearance_embedding),
            "secondary": cosine01(obs.secondary_embedding, mem.secondary_embedding),
            "geometry": geometry_similarity(obs.geometry_signature, mem.geometry_signature),
            "context": cosine01(obs.context_embedding, mem.context_embedding),
        }
        raw_weights = {
            "appearance": self.policy.appearance_weight,
            "secondary": self.policy.secondary_weight,
            "geometry": self.policy.geometry_weight,
            "context": self.policy.context_weight,
        }
        available = {k: w for k, w in raw_weights.items() if comp[k] is not None and w > 0}
        if not available:
            return None
        denom = sum(available.values())
        score = sum(float(comp[k]) * w for k, w in available.items()) / denom
        return CandidateScore(
            asset_id=mem.asset_id,
            class_name=mem.class_name,
            score=float(score),
            components=comp,
            evidence_refs=list(mem.evidence_refs),
            observation_count=int(mem.observation_count),
            last_seen=mem.last_seen,
        )

    def resolve(self, obs: Observation, memories: Iterable[AssetMemory]) -> IdentityDecisionObject:
        ranked = [c for m in memories if (c := self._candidate(obs, m)) is not None]
        ranked.sort(key=lambda c: (-c.score, c.asset_id))
        ranked = ranked[: self.policy.max_candidates]
        reasons: list[str] = []
        counter: list[str] = []
        resolved: str | None = None
        margin: float | None = None

        if not ranked:
            decision: DecisionState = "NEW"
            confidence = 1.0
            reasons.append("NO_CLASS_COMPATIBLE_MEMORY")
        else:
            top = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None
            margin = top.score - second.score if second else (None if self.policy.require_two_candidates_for_margin else top.score)
            confidence = top.score
            confirm_margin_ok = margin is not None and margin >= self.policy.confirm_min_margin
            if top.score >= self.policy.confirm_min_score and confirm_margin_ok:
                decision = "CONFIRMED"
                resolved = top.asset_id
                reasons += ["SCORE_ABOVE_CONFIRM_THRESHOLD", "MARGIN_ABOVE_CONFIRM_THRESHOLD"]
            elif top.score < self.policy.new_max_score:
                decision = "NEW"
                reasons.append("ALL_MEMORY_BELOW_NEW_THRESHOLD")
            elif top.score >= self.policy.candidate_min_score:
                decision = "CANDIDATE"
                reasons.append("PLAUSIBLE_IDENTITY_NOT_SAFE_TO_CONFIRM")
                if top.score >= self.policy.confirm_min_score and not confirm_margin_ok:
                    counter.append("INSUFFICIENT_TOP1_MARGIN")
                elif top.score < self.policy.confirm_min_score:
                    counter.append("BELOW_CONFIRM_SCORE")
            else:
                decision = "UNKNOWN"
                reasons.append("EVIDENCE_INSUFFICIENT")
                counter.append("TOP_SCORE_BETWEEN_NEW_AND_CANDIDATE_THRESHOLDS")

            if second and margin is not None and margin < self.policy.confirm_min_margin:
                counter.append(f"NEAR_COMPETING_CANDIDATE:{second.asset_id}")

        prov = obs.provenance
        model_versions = {
            "detector": prov.detector_version,
            "embedding_model": prov.embedding_model_version,
        }
        evidence = list(dict.fromkeys(obs.evidence_refs + [r for c in ranked[:3] for r in c.evidence_refs]))
        decision_id = "iddec_" + hashlib.sha256(
            f"{obs.observation_id}|{obs.fingerprint()}|{self.policy.policy_version}|{self.policy.calibration_ref}".encode()
        ).hexdigest()[:20]
        obj = IdentityDecisionObject(
            schema="assetgraph-identity-decision-object/v1",
            decision_id=decision_id,
            observation_id=obs.observation_id,
            decision=decision,
            resolved_asset_id=resolved,
            confidence=float(confidence),
            top1_margin=margin,
            candidates=ranked,
            reason_codes=reasons,
            counterevidence=list(dict.fromkeys(counter)),
            evidence_refs=evidence,
            observation_fingerprint=obs.fingerprint(),
            policy_version=self.policy.policy_version,
            policy_calibration_ref=self.policy.calibration_ref,
            model_versions=model_versions,
            created_at=utcnow(),
        )
        d = obj.to_dict()
        return IdentityDecisionObject(**{**asdict(obj), "decision_fingerprint": d["decision_fingerprint"]})


class SQLiteIdentityLedger:
    """Small append-only evidence ledger for offline/on-prem pilots."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations(
              observation_id TEXT PRIMARY KEY,
              fingerprint TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions(
              decision_id TEXT PRIMARY KEY,
              observation_id TEXT NOT NULL,
              decision TEXT NOT NULL,
              resolved_asset_id TEXT,
              fingerprint TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback(
              feedback_id TEXT PRIMARY KEY,
              decision_id TEXT NOT NULL,
              disposition TEXT NOT NULL,
              corrected_asset_id TEXT,
              note TEXT,
              created_at TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def append_observation(self, obs: Observation) -> str:
        payload = asdict(obs)
        fp = obs.fingerprint()
        self.db.execute(
            "INSERT OR IGNORE INTO observations VALUES(?,?,?,?)",
            (obs.observation_id, fp, canonical_json(payload), utcnow()),
        )
        self.db.commit()
        return fp

    def append_decision(self, decision: IdentityDecisionObject) -> str:
        payload = decision.to_dict()
        fp = payload["decision_fingerprint"]
        self.db.execute(
            "INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?,?,?)",
            (
                decision.decision_id,
                decision.observation_id,
                decision.decision,
                decision.resolved_asset_id,
                fp,
                canonical_json(payload),
                decision.created_at,
            ),
        )
        self.db.commit()
        return fp

    def append_feedback(self, decision_id: str, disposition: str, corrected_asset_id: str | None = None, note: str | None = None) -> str:
        payload = {"decision_id": decision_id, "disposition": disposition, "corrected_asset_id": corrected_asset_id, "note": note}
        fid = "feedback_" + evidence_hash(payload)[:20]
        self.db.execute(
            "INSERT OR IGNORE INTO feedback VALUES(?,?,?,?,?,?)",
            (fid, decision_id, disposition, corrected_asset_id, note, utcnow()),
        )
        self.db.commit()
        return fid

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT payload_json FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.db.close()


def new_observation_id(prefix: str = "obs") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
