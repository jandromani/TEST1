from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "cycle18a_dataset_intake_protocol.json"
REGISTRY_PATH = ROOT / "cycle18a_dataset_registry.json"
EVIDENCE_PATH = ROOT / "evidence" / "cycle18a_lawful_intake.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def main() -> None:
    protocol = load(PROTOCOL_PATH)
    registry = load(REGISTRY_PATH)
    allowed = set(protocol["policy"]["allowed_license_ids"])
    candidates = registry["candidates"]
    by_id = {row["id"]: row for row in candidates}

    unique_ids = len(by_id) == len(candidates)
    required_ids = set(protocol["gate"]["required_candidate_ids"])
    required_present = required_ids == set(by_id)
    expected_download = set(protocol["gate"]["expected_download_queue"])
    expected_training = set(protocol["gate"]["expected_training_queue"])
    actual_download = {row["id"] for row in candidates if row["approval"]["download"]}
    actual_training = {row["id"] for row in candidates if row["approval"]["training"]}

    source_and_license_evidence_pinned = all(
        bool(row["source"].get("version_pinned"))
        and bool(row["source"].get("canonical_uri"))
        and is_sha256(row["license"]["evidence"].get("sha256"))
        for row in candidates
        if row["approval"]["download"]
    )
    download_licenses_allowed = all(
        row["license"]["id"] in allowed
        and row["license"]["commercial_policy_eligible"] is True
        and (
            row["license"]["id"] == "CC0-1.0"
            or row["license"].get("attribution_required") is True
        )
        for row in candidates
        if row["approval"]["download"]
    )
    quarantined_do_not_leak = all(
        not row["approval"]["download"] and not row["approval"]["training"]
        for row in candidates
        if row["status"].startswith("QUARANTINED_")
    )
    auair = by_id.get("au-air-2019-license-conflict", {})
    embedded = set(auair.get("license", {}).get("embedded_annotation_claims", []))
    auair_conflict_explicit = (
        auair.get("license", {}).get("id") == "CONFLICTING"
        and auair.get("license", {}).get("commercial_policy_eligible") is False
        and {"CC-BY-NC-SA-2.0", "CC-BY-NC-2.0"}.issubset(embedded)
    )
    training_locked_until_c18b = not actual_training
    seals_held = (
        protocol["seals"]["TRANSSET-H01"] == "SEALED"
        and protocol["seals"]["TRANSSET-UAVOBB"] == "SEALED"
        and protocol["seals"]["external_test_scoring"] is False
    )
    no_data_or_training = (
        protocol["execution"]["dataset_bytes_downloaded"] is False
        and protocol["execution"]["training_performed"] is False
        and protocol["execution"]["model_selected"] is False
    )

    gates = {
        "unique_candidate_ids": unique_ids,
        "required_candidates_present_exactly": required_present,
        "download_queue_exact": actual_download == expected_download,
        "training_queue_exact": actual_training == expected_training,
        "approved_sources_and_license_evidence_pinned": source_and_license_evidence_pinned,
        "approved_download_licenses_allowed": download_licenses_allowed,
        "quarantined_candidates_do_not_leak": quarantined_do_not_leak,
        "auair_license_conflict_explicit": auair_conflict_explicit,
        "training_locked_until_cycle18b": training_locked_until_c18b,
        "external_gates_remain_sealed": seals_held,
        "no_data_download_or_training": no_data_or_training,
    }
    gates["cycle18a_intake_pass"] = all(gates.values())

    decisions = [
        {
            "id": row["id"],
            "status": row["status"],
            "license_id": row["license"]["id"],
            "download_approved": row["approval"]["download"],
            "training_approved": row["approval"]["training"],
        }
        for row in candidates
    ]
    evidence = {
        "schema": "assetgraph-evidence/lawful-multidomain-intake-v1",
        "cycle": "18A",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "registry_sha256": sha256(REGISTRY_PATH),
        "decisions": decisions,
        "download_queue": sorted(actual_download),
        "training_queue": sorted(actual_training),
        "h01_accessed": False,
        "transset_uavobb_accessed": False,
        "dataset_bytes_downloaded": False,
        "training_performed": False,
        "gates": gates,
        "next": protocol["next_if_pass"],
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    if not gates["cycle18a_intake_pass"]:
        raise SystemExit("Cycle 18A intake gate failed")


if __name__ == "__main__":
    main()
