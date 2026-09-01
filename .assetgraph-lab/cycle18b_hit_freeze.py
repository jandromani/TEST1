from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import cycle18b_freeze_dataset as core

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LAYOUTS = {
    "normal_json": "HIT-UAV/normal_json/",
    "rotate_json": "HIT-UAV/rotate_json/",
    "normal_xml": "HIT-UAV/normal_xml/JPEGImages/",
    "rotate_xml": "HIT-UAV/rotate_xml/JPEGImages/",
}


def logical_layouts(audit):
    out = []
    for name, prefix in LAYOUTS.items():
        hashes = [
            row["sha256"] for row in audit["members"]
            if row["path"].startswith(prefix)
            and Path(row["path"]).suffix.lower() in IMAGE_EXT
            and not row["path"].startswith("__MACOSX/")
        ]
        unique = set(hashes)
        signature = hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest()
        out.append({"layout": name, "prefix": prefix, "count": len(hashes), "unique_hashes": len(unique), "hashset_signature": signature})
    return out


_original_gate = core.dataset_gate


def strict_hit_gate(dataset_id, src, audit, sidecars):
    gates = _original_gate(dataset_id, src, audit, sidecars)
    if dataset_id != "hit-uav-main-6692596":
        return gates
    expected = src["declared"]["images"]
    layouts = logical_layouts(audit)
    eligible = [x for x in layouts if x["count"] == expected and x["unique_hashes"] == expected]
    votes = Counter(x["hashset_signature"] for x in eligible)
    winner, votes_n = votes.most_common(1)[0] if votes else (None, 0)
    canonical = [x["layout"] for x in eligible if x["hashset_signature"] == winner]
    outlier = [x["layout"] for x in eligible if x["hashset_signature"] != winner]
    audit["hit_logical_layouts"] = layouts
    audit["hit_canonical_hashset_signature"] = winner
    audit["hit_canonical_layouts"] = canonical
    audit["hit_outlier_layouts"] = outlier
    audit["hit_canonical_vote_count"] = votes_n
    gates["logical_image_count_proven"] = votes_n >= 3 and len(canonical) >= 3
    gates["canonical_corpus_majority_byte_identical"] = votes_n >= 3
    gates["rotate_json_uses_canonical_image_bytes"] = "rotate_json" in canonical
    gates["normal_json_uses_canonical_image_bytes"] = "normal_json" in canonical
    gates["normal_xml_uses_canonical_image_bytes"] = "normal_xml" in canonical
    gates["derived_or_outlier_layout_is_explicit"] = len(outlier) == 1
    gates["byte_freeze_pass"] = all(v for k, v in gates.items() if k != "byte_freeze_pass")
    return gates


core.dataset_gate = strict_hit_gate
core.run("hit-uav-main-6692596")
