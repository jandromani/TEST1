from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(raw)


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    run_id: str
    inputs: Mapping[str, str]
    components: Mapping[str, str]
    parameters: Mapping[str, Any]
    outputs: Mapping[str, str]

    def digest(self) -> str:
        return canonical_json_sha256(asdict(self))

    def verify_inputs(self, current: Mapping[str, str]) -> None:
        if dict(current) != dict(self.inputs):
            raise ValueError("replay input hash mismatch")

    def verify_components(self, current: Mapping[str, str]) -> None:
        if dict(current) != dict(self.components):
            raise ValueError("replay component hash mismatch")
