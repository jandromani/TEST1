from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from assetgraph.domain.models import AssetHypothesis, AssetState, DecisionObject, EvidenceRef, Event, Observation


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _dump(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _evidence(rows: list[dict[str, Any]] | None) -> tuple[EvidenceRef, ...]:
    return tuple(EvidenceRef(**row) for row in (rows or []))


def _observation(payload: Mapping[str, Any]) -> Observation:
    data = dict(payload)
    data["evidence"] = _evidence(data.get("evidence"))
    return Observation(**data)


def _asset(payload: Mapping[str, Any]) -> AssetHypothesis:
    data = dict(payload)
    data["state"] = AssetState(data.get("state", AssetState.UNKNOWN.value))
    return AssetHypothesis(**data)


def _event(payload: Mapping[str, Any]) -> Event:
    data = dict(payload)
    data["asset_ids"] = tuple(data.get("asset_ids", []))
    data["evidence"] = _evidence(data.get("evidence"))
    return Event(**data)


class SQLiteAssetGraphRepository:
    """Small deterministic reference store.

    This is the local/default backend, not the enterprise scale target. It establishes
    persistence semantics and idempotency before Postgres/object-store adapters arrive.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations(
              observation_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_observations_mission ON observations(mission_id);

            CREATE TABLE IF NOT EXISTS assets(
              asset_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_assets_mission ON assets(mission_id);

            CREATE TABLE IF NOT EXISTS events(
              event_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_mission ON events(mission_id);

            CREATE TABLE IF NOT EXISTS decision_objects(
              decision_object_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL,
              payload TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def put_observation(self, observation: Observation) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO observations(observation_id,mission_id,payload) VALUES(?,?,?)",
            (observation.observation_id, observation.mission_id, _dump(observation)),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_observation(self, observation_id: str) -> Observation | None:
        row = self.conn.execute("SELECT payload FROM observations WHERE observation_id=?", (observation_id,)).fetchone()
        return _observation(json.loads(row["payload"])) if row else None

    def put_asset(self, mission_id: str, asset: AssetHypothesis) -> None:
        self.conn.execute(
            "INSERT INTO assets(asset_id,mission_id,payload) VALUES(?,?,?) "
            "ON CONFLICT(asset_id) DO UPDATE SET mission_id=excluded.mission_id,payload=excluded.payload",
            (asset.asset_id, mission_id, _dump(asset)),
        )
        self.conn.commit()

    def get_asset(self, asset_id: str) -> AssetHypothesis | None:
        row = self.conn.execute("SELECT payload FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return _asset(json.loads(row["payload"])) if row else None

    def list_assets(self, mission_id: str) -> list[AssetHypothesis]:
        rows = self.conn.execute("SELECT payload FROM assets WHERE mission_id=? ORDER BY asset_id", (mission_id,)).fetchall()
        return [_asset(json.loads(row["payload"])) for row in rows]

    def append_event(self, mission_id: str, event: Event) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO events(event_id,mission_id,payload) VALUES(?,?,?)",
            (event.event_id, mission_id, _dump(event)),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def list_events(self, mission_id: str) -> list[Event]:
        rows = self.conn.execute("SELECT payload FROM events WHERE mission_id=? ORDER BY event_id", (mission_id,)).fetchall()
        return [_event(json.loads(row["payload"])) for row in rows]

    def put_decision_object(self, decision: DecisionObject) -> None:
        payload = decision.to_dict()
        self.conn.execute(
            "INSERT INTO decision_objects(decision_object_id,mission_id,payload) VALUES(?,?,?) "
            "ON CONFLICT(decision_object_id) DO UPDATE SET mission_id=excluded.mission_id,payload=excluded.payload",
            (decision.decision_object_id, decision.mission_id, _dump(payload)),
        )
        self.conn.commit()

    def get_decision_object(self, decision_object_id: str) -> Mapping[str, Any] | None:
        row = self.conn.execute("SELECT payload FROM decision_objects WHERE decision_object_id=?", (decision_object_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
