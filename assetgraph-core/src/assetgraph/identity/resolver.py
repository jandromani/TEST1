from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

from assetgraph.domain.models import AssetHypothesis, Observation


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    asset_id: str
    score: float
    reasons: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    observation_id: str
    selected_asset_id: str | None
    confidence: float
    candidates: tuple[IdentityCandidate, ...]
    action: str  # attach | new_asset | abstain


class IdentityResolver:
    """Resolve observations to persistent assets without depending on a detector backend.

    Production resolvers may combine motion, geometry, appearance embeddings, sensor
    metadata, graph priors and analyst feedback. This baseline intentionally supports
    abstention rather than forcing an identity match.
    """

    def __init__(self, *, accept_threshold: float = 0.75, abstain_threshold: float = 0.45) -> None:
        if not 0 <= abstain_threshold <= accept_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= abstain <= accept <= 1")
        self.accept_threshold = accept_threshold
        self.abstain_threshold = abstain_threshold

    def resolve(
        self,
        observation: Observation,
        assets: Sequence[AssetHypothesis],
        *,
        scorer: Callable[[Observation, AssetHypothesis], IdentityCandidate],
    ) -> IdentityDecision:
        compatible = [a for a in assets if a.asset_class == observation.asset_class]
        candidates = tuple(sorted((scorer(observation, a) for a in compatible), key=lambda c: c.score, reverse=True))
        if not candidates:
            return IdentityDecision(observation.observation_id, None, 0.0, (), "new_asset")
        best = candidates[0]
        if best.score >= self.accept_threshold:
            return IdentityDecision(observation.observation_id, best.asset_id, best.score, candidates, "attach")
        if best.score < self.abstain_threshold:
            return IdentityDecision(observation.observation_id, None, 1.0 - best.score, candidates, "new_asset")
        return IdentityDecision(observation.observation_id, None, best.score, candidates, "abstain")


def normalized_motion_scorer(observation: Observation, asset: AssetHypothesis) -> IdentityCandidate:
    """Simple deterministic scorer used for tests/demo, not promoted as a universal ReID model."""
    current = observation.geometry.get("centroid")
    previous = asset.attributes.get("last_centroid")
    if not current or not previous:
        score = 0.50
        distance = float("inf")
    else:
        distance = math.dist(tuple(current), tuple(previous))
        max_distance = float(asset.attributes.get("identity_radius", 100.0))
        score = max(0.0, 1.0 - distance / max(max_distance, 1e-9))
    subtype_bonus = 0.0
    if observation.subtype_hypothesis and observation.subtype_hypothesis == asset.subtype_hypothesis:
        subtype_bonus = 0.05
    score = min(1.0, score + subtype_bonus)
    return IdentityCandidate(asset.asset_id, score, {"motion": score - subtype_bonus, "subtype": subtype_bonus, "distance": distance})
