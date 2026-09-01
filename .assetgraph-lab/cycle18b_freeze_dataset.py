from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "cycle18b_sources.json").read_text(encoding="utf-8"))
EVIDENCE = ROOT / "evidence" / "cycle18b"
EVIDENCE.mkdir(parents=True, exist_ok=True)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DOC_IMAGE_MARKERS = ("readme", "sample", "result", "figure", "fig_")


def source(dataset_id: str) -> dict[str, Any]:
    return next(x for x in SOURCES["sources"] if x["id"] == dataset_id)


def sha256_stream(stream, sink=None) -> tuple[str, int]:
    h = hashlib.sha256(); n = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk: break
        h.update(chunk); n += len(chunk)
        if sink is not None: sink.write(chunk)
    return h.hexdigest(), n


def download(urls: Iterable[str], out: Path, accept: str = "application/octet-stream,*/*") -> dict[str, Any]:
    errors = []
    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 AssetGraph-C18B/2.0", "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=180) as r, out.open("wb") as f:
                digest, size = sha256_stream(r, f)
                return {"requested_url": url, "final_url": r.geturl(), "status": getattr(r, "status", 200), "content_type": r.headers.get("Content-Type"), "sha256": digest, "bytes": size}
        except Exception as exc:
            errors.append({"url": url, "error": repr(exc)}); out.unlink(missing_ok=True)
    raise RuntimeError(json.dumps({"download_failed": errors}, indent=2))


def safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts


def classify_uavobb(name: str) -> tuple[str | None, str | None]:
    n = "/" + name.lower().replace("\\", "/").strip("/") + "/"
    split = next((s for s in ("train", "valid", "test") if f"/{s}/" in n), None)
    kind = "image" if "/images/" in n else ("label" if "/labels/" in n else None)
    return split, kind


def validate_uavobb_label(text: str) -> tuple[bool, str | None, int]:
    rows = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        rows += 1; parts = line.split()
        if len(parts) != 9: return False, f"line {lineno}: expected 9 fields, got {len(parts)}", rows
        try:
            cls = int(parts[0]); coords = [float(x) for x in parts[1:]]
        except ValueError: return False, f"line {lineno}: non-numeric field", rows
        if not 0 <= cls <= 5: return False, f"line {lineno}: class outside 0..5", rows
        if any(x < 0.0 or x > 1.0 for x in coords): return False, f"line {lineno}: coordinate outside [0,1]", rows
    return True, None, rows


def inspect_zip(path: Path, dataset_id: str) -> dict[str, Any]:
    manifest = []; unsafe = []; image_records = []; annotation_stems = set(); license_hits = []
    image_hash_splits: dict[str, set[str]] = defaultdict(set)
    uav_counts = {s: {"image": 0, "label": 0} for s in ("train", "valid", "test")}
    invalid_labels = []; label_rows = 0; json_count = 0; xml_count = 0
    image_groups: dict[str, list[str]] = defaultdict(list)

    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir(): continue
            name = info.filename
            if not safe_member(name): unsafe.append(name); continue
            suffix = Path(name).suffix.lower(); h = hashlib.sha256()
            buf = bytearray() if suffix in {".txt", ".json", ".xml"} and info.file_size <= 8_000_000 else None
            with zf.open(info) as r:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk: break
                    h.update(chunk)
                    if buf is not None: buf.extend(chunk)
            digest = h.hexdigest()
            manifest.append({"path": name, "bytes": info.file_size, "crc32": f"{info.CRC:08x}", "sha256": digest})
            if suffix in IMAGE_EXT:
                parent = str(PurePosixPath(name.replace("\\", "/")).parent)
                image_records.append({"path": name, "parent": parent, "stem": Path(name).stem, "sha256": digest})
                image_groups[parent].append(digest)
            elif suffix == ".json": json_count += 1; annotation_stems.add(Path(name).stem)
            elif suffix == ".xml": xml_count += 1; annotation_stems.add(Path(name).stem)
            if "license" in Path(name).name.lower() and buf is not None:
                text = bytes(buf).decode("utf-8", "replace")
                marker = "CC0-1.0" if "CC0 1.0 Universal" in text else ("CC-BY-4.0" if "Attribution 4.0 International" in text else "UNRECOGNIZED")
                license_hits.append({"path": name, "detected": marker, "sha256": digest})
            if dataset_id == "uav-obb-mendeley-v1":
                split, kind = classify_uavobb(name)
                if split and kind: uav_counts[split][kind] += 1
                if suffix in IMAGE_EXT and split: image_hash_splits[digest].add(split)
                if kind == "label" and suffix == ".txt" and buf is not None:
                    ok, reason, rows = validate_uavobb_label(bytes(buf).decode("utf-8", "replace")); label_rows += rows
                    if not ok and len(invalid_labels) < 25: invalid_labels.append({"path": name, "reason": reason})

    primary = [r for r in image_records if not any(m in r["path"].lower() for m in DOC_IMAGE_MARKERS)]
    unique_primary = {r["sha256"] for r in primary}
    group_rows = []
    for parent, hashes in sorted(image_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        signature = hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()
        group_rows.append({"parent": parent, "count": len(hashes), "unique_hashes": len(set(hashes)), "hashset_signature": signature})
    candidate_2898 = [r for r in group_rows if r["count"] == 2898]
    candidate_signatures = sorted({r["hashset_signature"] for r in candidate_2898})
    canonical_stems = {r["stem"] for r in primary}
    cross_split = [{"sha256": h, "splits": sorted(s)} for h, s in image_hash_splits.items() if len(s) > 1]
    multiplicity = Counter(r["sha256"] for r in primary)

    return {
        "members": manifest,
        "member_count": len(manifest),
        "unsafe_members": unsafe,
        "image_count": len(image_records),
        "primary_image_members": len(primary),
        "unique_primary_image_hashes": len(unique_primary),
        "primary_hash_multiplicity": dict(sorted(Counter(multiplicity.values()).items())),
        "image_groups": group_rows[:80],
        "candidate_2898_groups": candidate_2898,
        "candidate_2898_hashset_signature_count": len(candidate_signatures),
        "annotation_stem_coverage": (len(canonical_stems & annotation_stems) / len(canonical_stems)) if canonical_stems else 0.0,
        "json_count": json_count,
        "xml_count": xml_count,
        "license_hits": license_hits,
        "uavobb_split_counts": uav_counts if dataset_id == "uav-obb-mendeley-v1" else None,
        "uavobb_label_rows": label_rows if dataset_id == "uav-obb-mendeley-v1" else None,
        "uavobb_invalid_labels": invalid_labels if dataset_id == "uav-obb-mendeley-v1" else None,
        "cross_split_exact_duplicates": cross_split if dataset_id == "uav-obb-mendeley-v1" else None,
    }


def find_nested_zip(archive: Path, work: Path) -> Path | None:
    with zipfile.ZipFile(archive) as zf:
        candidates = sorted((x for x in zf.infolist() if not x.is_dir() and x.filename.lower().endswith(".zip")), key=lambda x: x.file_size, reverse=True)
        if not candidates: return None
        info = candidates[0]
        if info.file_size > 2_500_000_000: raise RuntimeError("nested zip exceeds safety cap")
        target = work / "nested.zip"
        with zf.open(info) as src, target.open("wb") as dst: shutil.copyfileobj(src, dst, 1024 * 1024)
        return target


def freeze_release_sidecar(meta: dict[str, Any], work: Path, key: str) -> dict[str, Any]:
    asset = meta[key]; p = work / asset["name"]
    dl = download([asset["uri"]], p, "text/plain,application/octet-stream,*/*")
    text = p.read_text(encoding="utf-8", errors="replace")
    return {"asset": asset, "download": dl, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "expected_marker_present": asset.get("expected_marker") in text if asset.get("expected_marker") else True}


def dataset_gate(dataset_id: str, src: dict[str, Any], audit: dict[str, Any], sidecars: dict[str, Any]) -> dict[str, Any]:
    base = {"archive_path_safe": not audit["unsafe_members"], "member_manifest_present": audit["member_count"] > 0}
    if dataset_id == "uav-obb-mendeley-v1":
        expected = src["declared"]["splits"]; counts = audit["uavobb_split_counts"] or {}
        base.update({
            "image_counts_match": all(counts[s]["image"] == expected[s] for s in expected),
            "label_counts_match_images": all(counts[s]["label"] == counts[s]["image"] for s in expected),
            "obb_labels_structurally_valid": not audit["uavobb_invalid_labels"],
            "no_exact_cross_split_duplicates": not audit["cross_split_exact_duplicates"],
        })
    elif dataset_id == "hit-uav-main-6692596":
        expected = src["declared"]["images"]
        logical_count_proven = audit["unique_primary_image_hashes"] == expected or (
            bool(audit["candidate_2898_groups"]) and audit["candidate_2898_hashset_signature_count"] == 1
        )
        base.update({
            "logical_image_count_proven": logical_count_proven,
            "annotations_present": (audit["json_count"] + audit["xml_count"]) > 0,
            "annotation_stem_coverage_nonzero": audit["annotation_stem_coverage"] > 0.0,
            "release_license_asset_frozen": sidecars.get("license", {}).get("expected_marker_present") is True,
            "release_license_same_release_id": src["payload_release"]["release_id"] == 92134997,
            "snapshot_reconciliation_required_for_external_mix": True,
        })
    base["byte_freeze_pass"] = all(base.values())
    return base


def write_blocked(dataset_id: str, src: dict[str, Any], reason: str) -> None:
    report = {"schema": "assetgraph-evidence/cycle18b-byte-freeze-v2", "cycle": "18B", "dataset_id": dataset_id, "status": "BLOCKED", "reason": reason, "source_state": src["state"], "training_allowed": False, "external_gates_accessed": False}
    out = EVIDENCE / f"{dataset_id}.json"; out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def run(dataset_id: str) -> None:
    src = source(dataset_id)
    if dataset_id == "uav-obb-mendeley-v1":
        resolved = src.get("resolved_public_acquisition_uris") or []
        if not resolved:
            write_blocked(dataset_id, src, "No unauthenticated immutable Version 1 acquisition URI is frozen yet; OAuth-only API endpoints are not sufficient for reproducible public CI.")
            return
        urls = resolved
    elif dataset_id == "hit-uav-main-6692596": urls = [src["payload_release"]["uri"]]
    else: raise SystemExit(dataset_id)

    with tempfile.TemporaryDirectory(prefix="assetgraph-c18b-") as td:
        work = Path(td); archive = work / "source.zip"; dl = download(urls, archive, "application/zip,application/octet-stream,*/*")
        if not zipfile.is_zipfile(archive): raise RuntimeError(f"download is not ZIP: {dl}")
        audit = inspect_zip(archive, dataset_id); nested_meta = None
        if audit["image_count"] < 100:
            nested = find_nested_zip(archive, work)
            if nested is not None and zipfile.is_zipfile(nested):
                nested_meta = {"sha256": hashlib.sha256(nested.read_bytes()).hexdigest(), "bytes": nested.stat().st_size}; audit = inspect_zip(nested, dataset_id)
        sidecars = {}
        if dataset_id == "hit-uav-main-6692596":
            sidecars["license"] = freeze_release_sidecar(src["payload_release"], work, "license_asset")
            sidecars["citation"] = freeze_release_sidecar(src["payload_release"], work, "citation_asset")
        gates = dataset_gate(dataset_id, src, audit, sidecars)
        member_manifest_sha = hashlib.sha256(json.dumps(audit["members"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report = {"schema": "assetgraph-evidence/cycle18b-byte-freeze-v2", "cycle": "18B", "dataset_id": dataset_id, "status": "PASS" if gates["byte_freeze_pass"] else "FAIL", "source_state": src["state"], "download": dl, "nested_payload": nested_meta, "sidecars": sidecars, "member_manifest_sha256": member_manifest_sha, "audit": audit, "gates": gates, "training_allowed": False, "external_gates_accessed": False}
        out = EVIDENCE / f"{dataset_id}.json"; out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"dataset_id": dataset_id, "status": report["status"], "archive_sha256": dl["sha256"], "archive_bytes": dl["bytes"], "member_count": audit["member_count"], "image_members": audit["image_count"], "unique_primary_image_hashes": audit["unique_primary_image_hashes"], "candidate_2898_groups": audit["candidate_2898_groups"], "member_manifest_sha256": member_manifest_sha, "gates": gates, "evidence": str(out)}, indent=2))
        if not gates["byte_freeze_pass"]: raise SystemExit(f"{dataset_id} byte-freeze gate failed")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("dataset_id", choices=["uav-obb-mendeley-v1", "hit-uav-main-6692596"]); run(ap.parse_args().dataset_id)


if __name__ == "__main__": main()
