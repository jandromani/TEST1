from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import identity_api


class IdentityApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        identity_api.LEDGER_PATH = Path(self.tmp.name) / "identity-api.db"
        self.client = TestClient(identity_api.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["default_policy"], "FAIL_CLOSED")

    def test_resolve_persist_and_feedback(self):
        req = {
            "observation": {
                "observation_id": "api-o1",
                "timestamp": "2026-09-01T00:00:00Z",
                "class_name": "swimmer",
                "appearance_embedding": [1, 0],
                "evidence_refs": ["sha256:frame"],
                "provenance": {"detector_name": "BYO", "detector_version": "x"}
            },
            "memories": [
                {"asset_id": "asset-a", "class_name": "swimmer", "appearance_embedding": [1, 0], "evidence_refs": ["sha256:a"]},
                {"asset_id": "asset-b", "class_name": "swimmer", "appearance_embedding": [0, 1], "evidence_refs": ["sha256:b"]}
            ],
            "policy": {
                "policy_version": "test-policy",
                "confirm_min_score": 0.9,
                "confirm_min_margin": 0.08,
                "candidate_min_score": 0.65,
                "new_max_score": 0.4,
                "calibration_ref": "api-unit"
            },
            "persist": True
        }
        r = self.client.post("/v1/identity/resolve", json=req)
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["decision"], "CONFIRMED")
        self.assertEqual(d["resolved_asset_id"], "asset-a")
        self.assertTrue(d["decision_fingerprint"])

        rr = self.client.get(f"/v1/identity/decisions/{d['decision_id']}")
        self.assertEqual(rr.status_code, 200)
        self.assertEqual(rr.json()["decision_id"], d["decision_id"])

        fb = self.client.post("/v1/identity/feedback", json={"decision_id": d["decision_id"], "disposition": "ACCEPT"})
        self.assertEqual(fb.status_code, 200)
        self.assertTrue(fb.json()["feedback_id"].startswith("feedback_"))

    def test_default_policy_never_silently_confirms(self):
        r = self.client.post("/v1/identity/resolve", json={
            "observation": {"observation_id": "api-o2", "timestamp": "2026-09-01T00:00:00Z", "class_name": "car", "appearance_embedding": [1, 0]},
            "memories": [{"asset_id": "a", "class_name": "car", "appearance_embedding": [1, 0]}],
            "persist": False
        })
        self.assertEqual(r.status_code, 200)
        self.assertNotEqual(r.json()["decision"], "CONFIRMED")
        self.assertIsNone(r.json()["resolved_asset_id"])

    def test_unknown_input_field_is_rejected(self):
        r = self.client.post("/v1/identity/resolve", json={
            "observation": {"observation_id": "api-o3", "timestamp": "2026-09-01T00:00:00Z", "class_name": "car", "secret_gt_track": 123},
            "memories": [],
            "persist": False
        })
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
