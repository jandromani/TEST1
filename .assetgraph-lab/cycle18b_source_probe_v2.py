from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "evidence" / "cycle18b" / "source_probe_v2.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 AssetGraph-C18B-SourceProbe/2.0"


def request(url: str, max_bytes: int = 5_000_000, method: str = "GET", data: bytes | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": UA, "Accept": "text/html,application/json,application/javascript,text/javascript,application/octet-stream,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read(max_bytes)
            return {"ok": True, "status": getattr(r, "status", 200), "requested_url": url, "final_url": r.geturl(), "content_type": r.headers.get("Content-Type"), "content_disposition": r.headers.get("Content-Disposition"), "content_length": r.headers.get("Content-Length"), "bytes_read": len(body), "sha256": hashlib.sha256(body).hexdigest(), "text": body.decode("utf-8", "replace")}
    except Exception as exc:
        return {"ok": False, "requested_url": url, "error": repr(exc), "text": ""}


def links(text: str, base: str) -> list[str]:
    out = set()
    for raw in re.findall(r'''(?:href|src|data-href|data-url)\s*=\s*["']([^"']+)["']''', text, flags=re.I): out.add(urllib.parse.urljoin(base, html.unescape(raw)))
    for raw in re.findall(r'https?://[^\s"\'<>\\]+', text): out.add(html.unescape(raw).rstrip('),.;'))
    return sorted(out)


def contexts(text: str, pats: list[str], radius: int = 500, limit: int = 60):
    out = []; lower = text.lower()
    for pat in pats:
        pos = 0; p = pat.lower()
        while len(out) < limit:
            i = lower.find(p, pos)
            if i < 0: break
            out.append({"pattern": pat, "context": text[max(0, i-radius):min(len(text), i+radius)]}); pos = i + len(p)
    return out


def compact(r: dict[str, Any], snippet: int = 1200):
    text = r.pop("text", "")
    r["snippet"] = text[:snippet]
    return r


def mendeley():
    landing = "https://data.mendeley.com/datasets/6snrjwcpkh/1"
    page = request(landing); text = page.pop("text", "")
    uuids = sorted(set(re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', text)))
    uuid_contexts = {u: contexts(text, [u], 900, 3) for u in uuids}
    bundle_urls = [u for u in links(text, landing) if "bundle.js" in u]
    bundles = []
    for u in bundle_urls[:3]:
        r = request(u, 8_000_000); t = r.pop("text", "")
        bundles.append({"meta": r, "contexts": contexts(t, ["public-files", "file_downloaded", "/files", "dataset-files", "file-tree"], 900, 50)})
    candidate_responses = {}
    for u in uuids:
        url = f"https://data.mendeley.com/public-files/datasets/6snrjwcpkh/files/{u}/file_downloaded?version=1"
        candidate_responses[url] = compact(request(url, 65536))
    listing_urls = [
        "https://data.mendeley.com/public-files/datasets/6snrjwcpkh/files?version=1",
        "https://data.mendeley.com/public-files/datasets/6snrjwcpkh/files",
        "https://data.mendeley.com/public-files/datasets/6snrjwcpkh/versions/1/files",
        "https://api.data.mendeley.com/datasets/6snrjwcpkh/files?version=1",
        "https://api.data.mendeley.com/datasets/6snrjwcpkh?version=1",
    ]
    listing = {u: compact(request(u, 250000)) for u in listing_urls}
    return {"landing": page, "uuids": uuids, "uuid_contexts": uuid_contexts, "bundle_reports": bundles, "candidate_public_file_responses": candidate_responses, "listing_probes": listing}


def sea():
    base = "https://seadronessee.cs.uni-tuebingen.de"
    redirecters = ["OD_redirecter", "ODv2_redirecter", "SOT_redirecter", "MOT_redirecter"]
    probes = {}
    for name in redirecters:
        u = f"{base}/dataset/{name}"
        probes[name] = compact(request(u, 65536, method="POST", data=b""))
    page = request(base + "/dataset"); text = page.pop("text", "")
    return {"landing": page, "redirecter_post_probes": probes, "odv2_contexts": contexts(text, ["ODv2_redirecter", "Object Detection v2"], 700, 10)}


def main():
    report = {"schema": "assetgraph-evidence/cycle18b-source-probe-v2", "cycle": "18B", "mendeley_uavobb_v1": mendeley(), "seadronessee": sea(), "training_allowed": False, "external_gates_accessed": False}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    sea_summary = {k: {"status": v.get("status"), "final_url": v.get("final_url"), "content_type": v.get("content_type")} for k, v in report["seadronessee"]["redirecter_post_probes"].items()}
    m_summary = {u: {"status": v.get("status"), "content_type": v.get("content_type"), "content_disposition": v.get("content_disposition"), "snippet": v.get("snippet", "")[:240]} for u, v in report["mendeley_uavobb_v1"]["candidate_public_file_responses"].items()}
    print(json.dumps({"sea_redirects": sea_summary, "mendeley_candidate_responses": m_summary, "evidence": str(OUT)}, indent=2))


if __name__ == "__main__": main()
