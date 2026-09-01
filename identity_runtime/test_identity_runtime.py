from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assetgraph_identity_runtime import (
    AssetMemory,
    IdentityPolicy,
    IdentityResolver,
    Observation,
    Provenance,
    SQLiteIdentityLedger,
    evidence_hash,
)


def mem(asset_id, emb, *, cls="swimmer", ctx=None, geo=None):
    return AssetMemory(
        asset_id=asset_id,
        class_name=cls,
        appearance_embedding=emb,
        context_embedding=ctx,
        geometry_signature=geo,
        observation_count=5,
        evidence_refs=[f"sha256:{asset_id.lower()}"]
    )


def obs(obs_id, emb, *, cls="swimmer", ctx=None, geo=None):
    return Observation(
        observation_id=obs_id,
        timestamp="2026-09-01T00:00:00Z",
        class_name=cls,
        appearance_embedding=emb,
        context_embedding=ctx,
        geometry_signature=geo,
        evidence_refs=["sha256:frame"],
        provenance=Provenance(detector_name="BYO", detector_version="1", embedding_model="DINOv2", embedding_model_version="vits14")
    )


class IdentityRuntimeTests(unittest.TestCase):
    def test_default_policy_fails_closed(self):
        r = IdentityResolver()
        d = r.resolve(obs("o1", [1, 0]), [mem("A", [1, 0])])
        self.assertNotEqual(d.decision, "CONFIRMED")
        self.assertIsNone(d.resolved_asset_id)

    def test_confirmed_requires_score_and_margin(self):
        p = IdentityPolicy(confirm_min_score=.90, confirm_min_margin=.08, candidate_min_score=.65, new_max_score=.40, calibration_ref="unit")
        r = IdentityResolver(p)
        d = r.resolve(obs("o2", [1, 0]), [mem("A", [1, 0]), mem("B", [0, 1])])
        self.assertEqual(d.decision, "CONFIRMED")
        self.assertEqual(d.resolved_asset_id, "A")
        self.assertGreaterEqual(d.top1_margin or 0, .08)

    def test_ambiguous_identity_abstains(self):
        p = IdentityPolicy(confirm_min_score=.90, confirm_min_margin=.08, candidate_min_score=.65, new_max_score=.40, calibration_ref="unit")
        r = IdentityResolver(p)
        d = r.resolve(obs("o3", [1, 0]), [mem("A", [1, 0]), mem("B", [.999, .01])])
        self.assertEqual(d.decision, "CANDIDATE")
        self.assertIsNone(d.resolved_asset_id)
        self.assertTrue(any("MARGIN" in x or "COMPETING" in x for x in d.counterevidence))

    def test_no_compatible_memory_is_new(self):
        p = IdentityPolicy(confirm_min_score=.9, confirm_min_margin=.1, candidate_min_score=.7, new_max_score=.4, calibration_ref="unit")
        d = IdentityResolver(p).resolve(obs("o4", [1, 0], cls="boat"), [mem("A", [1, 0], cls="swimmer")])
        self.assertEqual(d.decision, "NEW")
        self.assertEqual(d.reason_codes, ["NO_CLASS_COMPATIBLE_MEMORY"])

    def test_low_but_nonzero_evidence_can_be_unknown(self):
        p = IdentityPolicy(confirm_min_score=.95, confirm_min_margin=.1, candidate_min_score=.75, new_max_score=.45, calibration_ref="unit")
        # Cosine01 of orthogonal vectors = .5, which is above NEW and below CANDIDATE.
        d = IdentityResolver(p).resolve(obs("o5", [1, 0]), [mem("A", [0, 1])])
        self.assertEqual(d.decision, "UNKNOWN")
        self.assertIsNone(d.resolved_asset_id)

    def test_component_scores_and_evidence_are_exposed(self):
        p = IdentityPolicy(confirm_min_score=.8, confirm_min_margin=.01, candidate_min_score=.6, new_max_score=.3, calibration_ref="unit")
        d = IdentityResolver(p).resolve(
            obs("o6", [1, 0], ctx=[1, 1], geo=[-3, 0, .2, .3]),
            [mem("A", [1, 0], ctx=[1, 1], geo=[-3.1, .05, .21, .31]), mem("B", [0, 1], ctx=[-1, -1], geo=[-1, 1, .8, .8])],
        )
        payload = d.to_dict()
        self.assertIn("components", payload["candidates"][0])
        self.assertIn("sha256:frame", payload["evidence_refs"])
        self.assertEqual(len(payload["decision_fingerprint"]), 64)

    def test_ledger_is_append_only_and_retrievable(self):
        p = IdentityPolicy(confirm_min_score=.9, confirm_min_margin=.05, candidate_min_score=.7, new_max_score=.4, calibration_ref="unit")
        o = obs("o7", [1, 0])
        d = IdentityResolver(p).resolve(o, [mem("A", [1, 0]), mem("B", [0, 1])])
        with tempfile.TemporaryDirectory() as td:
            ledger = SQLiteIdentityLedger(Path(td) / "identity.db")
            ofp = ledger.append_observation(o)
            dfp = ledger.append_decision(d)
            got = ledger.get_decision(d.decision_id)
            self.assertEqual(ofp, o.fingerprint())
            self.assertEqual(dfp, d.to_dict()["decision_fingerprint"])
            self.assertEqual(got["decision_id"], d.decision_id)
            feedback_id = ledger.append_feedback(d.decision_id, "ACCEPT")
            self.assertTrue(feedback_id.startswith("feedback_"))
            ledger.close()

    def test_hash_is_deterministic(self):
        self.assertEqual(evidence_hash({"b": 2, "a": 1}), evidence_hash({"a": 1, "b": 2}))

    def test_schema_file_has_four_decision_states(self):
        schema = json.loads(Path(__file__).with_name("identity_decision_object_v1.schema.json").read_text())
        self.assertEqual(set(schema["properties"]["decision"]["enum"]), {"CONFIRMED", "CANDIDATE", "UNKNOWN", "NEW"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
