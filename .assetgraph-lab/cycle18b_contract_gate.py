from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "cycle18b_frozen_intake_protocol.json"
SOURCES = ROOT / "cycle18b_sources.json"
OUT = ROOT / "evidence" / "cycle18b" / "contract_gate.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = load(PROTOCOL)
    s = load(SOURCES)
    rows = {x["id"]: x for x in s["sources"]}
    approved = set(p["approved_candidate_ids"])
    quarantined = set(p["quarantined_candidate_ids"])

    required_ids_present = approved | quarantined == set(rows)
    approved_not_quarantined = approved.isdisjoint(quarantined)
    auair_locked = rows["au-air-2019-license-conflict"]["state"] == "QUARANTINED_NO_DOWNLOAD"
    uav_v1_locked = rows["uav-obb-mendeley-v1"]["version"] == 1
    hit_snapshot_conflict_explicit = (
        rows["hit-uav-main-6692596"]["annotation_commit_license"] == "CC-BY-4.0"
        and rows["hit-uav-main-6692596"]["payload_release"]["release_license_expected"] == "CC0-1.0"
        and rows["hit-uav-main-6692596"]["snapshot_reconciliation"]["required"] is True
    )
    seadronessee_block_explicit = rows["seadronessee-main-8062b5d"]["state"] == "BLOCKED_PENDING_EXACT_OFFICIAL_ODV2_ASSET_PIN"
    mirrors_not_authoritative = rows["seadronessee-main-8062b5d"]["third_party_reference_only"]["authoritative"] is False
    seals_held = (
        p["hard_boundary"]["TRANSSET-H01"] == "SEALED"
        and p["hard_boundary"]["TRANSSET-UAVOBB"] == "SEALED"
        and p["hard_boundary"]["external_test_scoring_allowed"] is False
    )
    training_locked = p["hard_boundary"]["training_allowed"] is False and p["hard_boundary"]["model_selection_allowed"] is False

    gates = {
        "required_source_ids_exact": required_ids_present,
        "approved_and_quarantined_disjoint": approved_not_quarantined,
        "auair_quarantine_held": auair_locked,
        "uavobb_version1_held": uav_v1_locked,
        "hit_snapshot_license_transition_explicit": hit_snapshot_conflict_explicit,
        "seadronessee_odv2_exact_asset_blocker_explicit": seadronessee_block_explicit,
        "third_party_seadronessee_mirror_not_authoritative": mirrors_not_authoritative,
        "external_gates_sealed": seals_held,
        "training_and_model_selection_locked": training_locked,
    }
    gates["contract_gate_pass"] = all(gates.values())

    evidence = {
        "schema": "assetgraph-evidence/cycle18b-contract-gate-v1",
        "cycle": "18B",
        "protocol_sha256": sha(PROTOCOL),
        "sources_sha256": sha(SOURCES),
        "gates": gates,
        "dataset_states": {k: v["state"] for k, v in rows.items()},
        "partial_byte_freeze_ready": [
            k for k in approved if rows[k]["state"].startswith("READY_FOR_BYTE_FREEZE")
        ],
        "cycle18b_complete": False,
        "training_allowed": False,
        "blocking_reason": "SeaDronesSee Object Detection v2 exact official asset is not yet immutably pinned; UAV-OBB and HIT-UAV may be byte-frozen independently without authorizing training.",
        "external_gates_accessed": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if not gates["contract_gate_pass"]:
        raise SystemExit("Cycle 18B contract gate failed")


if __name__ == "__main__":
    main()
