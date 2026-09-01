from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "cycle18b_sources.json").read_text(encoding="utf-8"))
EVIDENCE = ROOT / "evidence" / "cycle18b"
EVIDENCE.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def source(dataset_id: str) -> dict[str, Any]:
    for row in SOURCES["sources"]:
        if row["id"] == dataset_id:
            return row
    raise KeyError(dataset_id)


def sha256_stream(stream, sink=None) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        h.update(chunk)
        n += len(chunk)
        if sink is not None:
            sink.write(chunk)
    return h.hexdigest(), n


def download(urls: Iterable[str], out: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    for url in urls:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 AssetGraph-C18B/1.0",
                "Accept": "application/zip,application/octet-stream,*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r, out.open("wb") as f:
                digest, size = sha256_stream(r, f)
                return {
                    "requested_url": url,
                    "final_url": r.geturl(),
                    "status": getattr(r, "status", 200),
                    "content_type": r.headers.get("Content-Type"),
                    "sha256": digest,
                    "bytes": size,
                }
        except Exception as exc:
            errors.append({"url": url, "error": repr(exc)})
            out.unlink(missing_ok=True)
    raise RuntimeError(json.dumps({"download_failed": errors}, indent=2))


def safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts


def classify_uavobb(name: str) -> tuple[str | None, str | None]:
    n = "/" + name.lower().replace("\\", "/").strip("/") + "/"
    split = None
    for candidate in ("train", "valid", "test"):
        if f"/{candidate}/" in n:
            split = candidate
            break
    kind = None
    if "/images/" in n:
        kind = "image"
    elif "/labels/" in n:
        kind = "label"
    return split, kind


def validate_uavobb_label(text: str) -> tuple[bool, str | None, int]:
    rows = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rows += 1
        parts = line.split()
        if len(parts) != 9:
            return False, f"line {lineno}: expected 9 fields, got {len(parts)}", rows
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            return False, f"line {lineno}: non-numeric field", rows
        if not (0 <= cls <= 5):
            return False, f"line {lineno}: class {cls} outside 0..5", rows
        if any((x < 0.0 or x > 1.0) for x in coords):
            return False, f"line {lineno}: normalized coordinate outside [0,1]", rows
    return True, None, rows


def inspect_zip(path: Path, dataset_id: str) -> dict[str, Any]:
    manifest: list[dict[str, Any]] = []
    unsafe: list[str] = []
    image_hash_splits: dict[str, set[str]] = {}
    uav_counts = {s: {"image": 0, "label": 0} for s in ("train", "valid", "test")}
    invalid_labels: list[dict[str, Any]] = []
    label_rows = 0
    image_count = 0
    json_count = 0
    xml_count = 0
    license_hits: list[dict[str, Any]] = []

    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            if not safe_member(name):
                unsafe.append(name)
                continue
            suffix = Path(name).suffix.lower()
            h = hashlib.sha256()
            buf = bytearray() if (suffix in {".txt", ".json", ".xml"} and info.file_size <= 4_000_000) else None
            with zf.open(info) as r:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
                    if buf is not None:
                        buf.extend(chunk)
            digest = h.hexdigest()
            manifest.append({
                "path": name,
                "bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "sha256": digest,
            })

            lname = name.lower()
            if suffix in IMAGE_EXT:
                image_count += 1
            elif suffix == ".json":
                json_count += 1
            elif suffix == ".xml":
                xml_count += 1

            if "license" in Path(name).name.lower() and buf is not None:
                text = bytes(buf).decode("utf-8", "replace")
                marker = "CC0-1.0" if "CC0 1.0 Universal" in text else ("CC-BY-4.0" if "Attribution 4.0 International" in text else "UNRECOGNIZED")
                license_hits.append({"path": name, "detected": marker, "sha256": digest})

            if dataset_id == "uav-obb-mendeley-v1":
                split, kind = classify_uavobb(name)
                if split and kind:
                    uav_counts[split][kind] += 1
                if suffix in IMAGE_EXT and split:
                    image_hash_splits.setdefault(digest, set()).add(split)
                if kind == "label" and suffix == ".txt" and buf is not None:
                    ok, reason, rows = validate_uavobb_label(bytes(buf).decode("utf-8", "replace"))
                    label_rows += rows
                    if not ok and len(invalid_labels) < 25:
                        invalid_labels.append({"path": name, "reason": reason})

    cross_split_exact_duplicates = [
        {"sha256": digest, "splits": sorted(splits)}
        for digest, splits in image_hash_splits.items()
        if len(splits) > 1
    ]
    return {
        "members": manifest,
        "member_count": len(manifest),
        "unsafe_members": unsafe,
        "image_count": image_count,
        "json_count": json_count,
        "xml_count": xml_count,
        "license_hits": license_hits,
        "uavobb_split_counts": uav_counts if dataset_id == "uav-obb-mendeley-v1" else None,
        "uavobb_label_rows": label_rows if dataset_id == "uav-obb-mendeley-v1" else None,
        "uavobb_invalid_labels": invalid_labels if dataset_id == "uav-obb-mendeley-v1" else None,
        "cross_split_exact_duplicates": cross_split_exact_duplicates if dataset_id == "uav-obb-mendeley-v1" else None,
    }


def find_nested_zip(archive: Path, work: Path) -> Path | None:
    with zipfile.ZipFile(archive) as zf:
        candidates = [x for x in zf.infolist() if not x.is_dir() and x.filename.lower().endswith(".zip")]
        candidates.sort(key=lambda x: x.file_size, reverse=True)
        if not candidates:
            return None
        info = candidates[0]
        if info.file_size > 2_500_000_000:
            raise RuntimeError("nested zip exceeds C18B safety cap")
        target = work / "nested.zip"
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        return target


def dataset_gate(dataset_id: str, src: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    base = {
        "archive_path_safe": not audit["unsafe_members"],
        "member_manifest_present": audit["member_count"] > 0,
    }
    if dataset_id == "uav-obb-mendeley-v1":
        expected = src["declared"]["splits"]
        counts = audit["uavobb_split_counts"] or {}
        base.update({
            "image_counts_match": all(counts[s]["image"] == expected[s] for s in expected),
            "label_counts_match_images": all(counts[s]["label"] == counts[s]["image"] for s in expected),
            "obb_labels_structurally_valid": not audit["uavobb_invalid_labels"],
            "no_exact_cross_split_duplicates": not audit["cross_split_exact_duplicates"],
        })
    elif dataset_id == "hit-uav-main-6692596":
        base.update({
            "image_count_matches": audit["image_count"] == src["declared"]["images"],
            "annotations_present": (audit["json_count"] + audit["xml_count"]) > 0,
            "release_license_bound_to_payload": any(x["detected"] == "CC0-1.0" for x in audit["license_hits"]),
            "snapshot_reconciliation_required_for_any_external_annotation_mix": True,
        })
    base["byte_freeze_pass"] = all(base.values())
    return base


def run(dataset_id: str) -> None:
    src = source(dataset_id)
    if dataset_id == "uav-obb-mendeley-v1":
        urls = src["acquisition_uris"]
    elif dataset_id == "hit-uav-main-6692596":
        urls = [src["payload_release"]["uri"]]
    else:
        raise SystemExit(f"C18B byte-freeze currently supports only UAV-OBB and HIT-UAV, got {dataset_id}")

    with tempfile.TemporaryDirectory(prefix="assetgraph-c18b-") as td:
        work = Path(td)
        archive = work / "source.zip"
        dl = download(urls, archive)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(f"download is not a ZIP: {dl}")
        audit = inspect_zip(archive, dataset_id)

        # Some repository systems wrap an uploaded dataset ZIP inside a download-all ZIP.
        # If the expected image payload is not visible, inspect the largest nested ZIP as the authoritative member payload.
        needs_nested = (
            dataset_id == "uav-obb-mendeley-v1" and audit["image_count"] < 100
        ) or (
            dataset_id == "hit-uav-main-6692596" and audit["image_count"] < 100
        )
        nested_meta = None
        if needs_nested:
            nested = find_nested_zip(archive, work)
            if nested is not None and zipfile.is_zipfile(nested):
                nested_sha = hashlib.sha256(nested.read_bytes()).hexdigest()
                nested_meta = {"sha256": nested_sha, "bytes": nested.stat().st_size}
                audit = inspect_zip(nested, dataset_id)

        gates = dataset_gate(dataset_id, src, audit)
        member_manifest_sha = hashlib.sha256(
            json.dumps(audit["members"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        report = {
            "schema": "assetgraph-evidence/cycle18b-byte-freeze-v1",
            "cycle": "18B",
            "dataset_id": dataset_id,
            "source_state": src["state"],
            "download": dl,
            "nested_payload": nested_meta,
            "member_manifest_sha256": member_manifest_sha,
            "audit": audit,
            "gates": gates,
            "training_allowed": False,
            "external_gates_accessed": False,
        }
        out = EVIDENCE / f"{dataset_id}.json"
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "dataset_id": dataset_id,
            "archive_sha256": dl["sha256"],
            "archive_bytes": dl["bytes"],
            "nested_payload": nested_meta,
            "member_count": audit["member_count"],
            "image_count": audit["image_count"],
            "member_manifest_sha256": member_manifest_sha,
            "gates": gates,
            "evidence": str(out),
        }, indent=2))
        if not gates["byte_freeze_pass"]:
            raise SystemExit(f"{dataset_id} byte-freeze gate failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id", choices=["uav-obb-mendeley-v1", "hit-uav-main-6692596"])
    args = ap.parse_args()
    run(args.dataset_id)


if __name__ == "__main__":
    main()
