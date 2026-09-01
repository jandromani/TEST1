from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from assetgraph_identity_runtime import (
    AssetMemory,
    IdentityPolicy,
    IdentityResolver,
    Observation,
    Provenance,
    SQLiteIdentityLedger,
)

app = FastAPI(
    title="AssetGraph Identity Resolution Runtime",
    version="0.1.0",
    description=(
        "Model-agnostic persistent identity middleware. Runtime inputs contain no ground truth. "
        "The resolver returns CONFIRMED, CANDIDATE, UNKNOWN or NEW plus ranked evidence."
    ),
)

LEDGER_PATH = Path(os.environ.get("ASSETGRAPH_LEDGER_PATH", "/tmp/assetgraph_identity_runtime.db"))


class ResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: dict[str, Any]
    memories: list[dict[str, Any]] = Field(default_factory=list)
    policy: dict[str, Any] | None = None
    persist: bool = True


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str
    disposition: str
    corrected_asset_id: str | None = None
    note: str | None = None


def _filter(cls, raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown {cls.__name__} fields: {unknown}")
    return {k: v for k, v in raw.items() if k in allowed}


def _observation(raw: dict[str, Any]) -> Observation:
    d = _filter(Observation, raw)
    p = d.get("provenance")
    if isinstance(p, dict):
        d["provenance"] = Provenance(**_filter(Provenance, p))
    return Observation(**d)


def _memory(raw: dict[str, Any]) -> AssetMemory:
    return AssetMemory(**_filter(AssetMemory, raw))


def _policy(raw: dict[str, Any] | None) -> IdentityPolicy:
    if raw is None:
        return IdentityPolicy()
    return IdentityPolicy(**_filter(IdentityPolicy, raw))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "assetgraph-identity-runtime",
        "decision_schema": "assetgraph-identity-decision-object/v1",
        "default_policy": "FAIL_CLOSED",
    }


@app.post("/v1/identity/resolve")
def resolve(req: ResolveRequest) -> dict[str, Any]:
    try:
        observation = _observation(req.observation)
        memories = [_memory(x) for x in req.memories]
        policy = _policy(req.policy)
        decision = IdentityResolver(policy).resolve(observation, memories)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if req.persist:
        ledger = SQLiteIdentityLedger(LEDGER_PATH)
        try:
            ledger.append_observation(observation)
            ledger.append_decision(decision)
        finally:
            ledger.close()
    return decision.to_dict()


@app.post("/v1/identity/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    ledger = SQLiteIdentityLedger(LEDGER_PATH)
    try:
        existing = ledger.get_decision(req.decision_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Decision not found in this ledger")
        feedback_id = ledger.append_feedback(req.decision_id, req.disposition, req.corrected_asset_id, req.note)
    finally:
        ledger.close()
    return {"feedback_id": feedback_id, "decision_id": req.decision_id, "status": "recorded"}


@app.get("/v1/identity/decisions/{decision_id}")
def get_decision(decision_id: str) -> dict[str, Any]:
    ledger = SQLiteIdentityLedger(LEDGER_PATH)
    try:
        decision = ledger.get_decision(decision_id)
    finally:
        ledger.close()
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision
