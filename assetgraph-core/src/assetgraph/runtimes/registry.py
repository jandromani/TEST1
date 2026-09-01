from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from .base import PerceptionRuntime, RuntimeInfo


@dataclass(slots=True)
class RuntimeRegistration:
    info: RuntimeInfo
    factory: Callable[[], PerceptionRuntime]
    evidence_ids: tuple[str, ...] = ()
    promotion_gates: Mapping[str, bool] = field(default_factory=dict)

    @property
    def promotable(self) -> bool:
        return (
            self.info.product_status == "PRODUCT_CANDIDATE"
            and bool(self.promotion_gates)
            and all(self.promotion_gates.values())
        )


class RuntimeRegistry:
    def __init__(self) -> None:
        self._items: dict[str, RuntimeRegistration] = {}

    def register(self, registration: RuntimeRegistration, *, replace: bool = False) -> None:
        rid = registration.info.runtime_id
        if rid in self._items and not replace:
            raise KeyError(f"runtime already registered: {rid}")
        self._items[rid] = registration

    def get(self, runtime_id: str) -> RuntimeRegistration:
        try:
            return self._items[runtime_id]
        except KeyError as exc:
            raise KeyError(f"unknown runtime: {runtime_id}") from exc

    def load(self, runtime_id: str, *, require_product_candidate: bool = False) -> PerceptionRuntime:
        reg = self.get(runtime_id)
        if require_product_candidate and not reg.promotable:
            raise RuntimeError(f"runtime {runtime_id} has not cleared product promotion gates")
        return reg.factory()

    def snapshot(self) -> list[dict[str, object]]:
        out = []
        for rid in sorted(self._items):
            reg = self._items[rid]
            out.append({
                "runtime_id": rid,
                "family": reg.info.family,
                "version": reg.info.version,
                "backend": reg.info.backend,
                "license": reg.info.license,
                "product_status": reg.info.product_status,
                "promotable": reg.promotable,
                "evidence_ids": list(reg.evidence_ids),
                "promotion_gates": dict(reg.promotion_gates),
            })
        return out
