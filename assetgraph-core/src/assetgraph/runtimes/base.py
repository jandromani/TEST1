from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    runtime_id: str
    family: str
    version: str
    backend: str
    license: str
    product_status: str
    model_sha256: str | None = None
    source_commit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawDetection:
    asset_class: str
    confidence: float
    geometry: Mapping[str, Any]
    subtype_hypothesis: str | None = None
    subtype_confidence: float | None = None
    embedding: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PerceptionRuntime(Protocol):
    @property
    def info(self) -> RuntimeInfo: ...

    def infer(
        self,
        source: str | Path | bytes,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Sequence[RawDetection]: ...

    def healthcheck(self) -> Mapping[str, Any]: ...
