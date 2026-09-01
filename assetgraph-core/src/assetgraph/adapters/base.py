from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SourceEnvelope:
    source_id: str
    sensor: str
    observed_at: str
    uri: str
    media_type: str
    geometry: Mapping[str, Any] = field(default_factory=dict)
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    provider: str | None = None
    collection: str | None = None
    source_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SourceAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def normalize(self, source: Any) -> Iterable[SourceEnvelope]: ...
