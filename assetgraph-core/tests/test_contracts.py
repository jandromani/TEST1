from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from assetgraph.domain.models import AssetHypothesis, DecisionObject


def test_decision_object_matches_shipped_schema():
    schema = json.loads(Path("schemas/decision-object-v1.schema.json").read_text())
    decision = DecisionObject(
        decision_object_id="DO-SCHEMA-1",
        mission_id="M-SCHEMA-1",
        assets=[AssetHypothesis("ASSET-1", "vehicle", identity_confidence=0.8)],
        events=[],
        claims=[],
        provenance={"runtime": "contract-test"},
    ).to_dict()
    jsonschema.Draft202012Validator(schema).validate(decision)


def test_openapi_contract_is_shipped():
    text = Path("api/openapi.yaml").read_text()
    assert "openapi: 3.1.0" in text
    assert "/v1/missions" in text
    assert "/v1/decision-objects/{decision_object_id}" in text
