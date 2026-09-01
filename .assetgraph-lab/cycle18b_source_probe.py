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
OUT = ROOT / "evidence" / "cycle18b" / "source_probe.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 AssetGraph-C18B-SourceProbe/1.0"


def get(url: str, max_bytes: int = 4_000_000, headers: dict[str, str] | None = None) -> dict[str, Any]:
    h = {"User-Agent": UA, "Accept": "text/html,application/json,application/javascript,text/javascript,*/*"}; h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read(max_bytes)
            return {"ok": True, "status": getattr(r, "status", 200), "requested_url": url, "final_url": r.geturl(), "content_type": r.headers.get("Content-Type"), "bytes_read": len(body), "sha256": hashlib.sha256(body).hexdigest(), "text": body.decode("utf-8", "replace")}
    except Exception as exc:
        return {"ok": False, "requested_url": url, "error": repr(exc), "text": ""}


def urls(text: str, base: str) -> list[str]:
    found = set()
    for raw in re.findall(r'''(?:href|src|data-href|data-url)\s*=\s*["']([^"']+)["']''', text, flags=re.I):
        found.add(urllib.parse.urljoin(base, html.unescape(raw)))
    for raw in re.findall(r'https?://[^\s"\'<>\\]+', text):
        found.add(html.unescape(raw).rstrip('),.;'))
    return sorted(found)


def contexts(text: str, patterns: list[str], radius: int = 700, limit: int = 30) -> list[dict[str, str]]:
    out = []
    lower = text.lower()
    for pat in patterns:
        start = 0; p = pat.lower()
        while len(out) < limit:
            i = lower.find(p, start)
            if i < 0: break
            out.append({"pattern": pat, "context": text[max(0, i-radius):min(len(text), i+radius)]})
            start = i + max(1, len(p))
    return out


def probe_range(url: str) -> dict[str, Any]:
    r = get(url, 1024, {"Range": "bytes=0-1023", "Accept": "application/octet-stream,*/*"})
    r.pop("text", None)
    return r


def mendeley() -> dict[str, Any]:
    landing = "https://data.mendeley.com/datasets/6snrjwcpkh/1"
    page = get(landing); text = page.pop("text", "")
    uuids = sorted(set(re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', text)))
    all_urls = urls(text, landing)
    interesting = [u for u in all_urls if any(k in u.lower() for k in ("download", "file", "zip", "dataset", "6snrjwcpkh"))][:200]
    candidates = []
    for u in uuids[:30]:
        candidates.extend([
            f"https://api.data.mendeley.com/datasets/6snrjwcpkh/files/{u}/file_downloaded?version=1",
            f"https://api.mendeley.com/datasets/6snrjwcpkh/files/{u}/file_downloaded?version=1",
            f"https://data.mendeley.com/public-files/datasets/6snrjwcpkh/files/{u}/file_downloaded?version=1",
        ])
    probes = {u: probe_range(u) for u in candidates}
    public_2xx = [u for u, r in probes.items() if r.get("ok") and int(r.get("status", 0)) in (200, 206)]
    return {"landing": page, "uuid_count": len(uuids), "uuids": uuids[:100], "interesting_urls": interesting, "contexts": contexts(text, ["Download All", "file_downloaded", "6snrjwcpkh", ".zip", "files"], limit=40), "candidate_file_probes": probes, "public_2xx_candidates": public_2xx}


def seadronessee() -> dict[str, Any]:
    landing = "https://seadronessee.cs.uni-tuebingen.de/dataset"
    page = get(landing); text = page.pop("text", "")
    all_urls = urls(text, landing)
    first_party_assets = [u for u in all_urls if urllib.parse.urlparse(u).netloc == "seadronessee.cs.uni-tuebingen.de" and any(x in u.lower() for x in (".js", ".json"))][:40]
    asset_reports = []
    discovered = set(all_urls)
    for u in first_party_assets:
        r = get(u, 3_000_000); t = r.pop("text", ""); discovered.update(urls(t, u))
        if any(k in t.lower() for k in ("compressedversion", "cloud.cs.uni-tuebingen.de", "object detection v2", ".zip")):
            asset_reports.append({"asset": r, "contexts": contexts(t, ["CompressedVersion", "cloud.cs.uni-tuebingen.de", "Object Detection v2", ".zip"], limit=30)})
    interesting = sorted(u for u in discovered if any(k in u.lower() for k in ("cloud.cs.uni-tuebingen.de", "download", ".zip", "compressed", "dataset")))
    odv2_contexts = contexts(text, ["Object Detection v2", "CompressedVersion", "8930", "1547", "cloud.cs.uni-tuebingen.de"], limit=50)
    return {"landing": page, "interesting_urls": interesting[:300], "landing_contexts": odv2_contexts, "first_party_asset_reports": asset_reports}


def main() -> None:
    report = {"schema": "assetgraph-evidence/cycle18b-source-probe-v1", "cycle": "18B", "mendeley_uavobb_v1": mendeley(), "seadronessee_odv2": seadronessee(), "training_allowed": False, "external_gates_accessed": False}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mendeley": {"uuid_count": report["mendeley_uavobb_v1"]["uuid_count"], "public_2xx_candidates": report["mendeley_uavobb_v1"]["public_2xx_candidates"]}, "seadronessee_interesting_urls": report["seadronessee_odv2"]["interesting_urls"], "evidence": str(OUT)}, indent=2))


if __name__ == "__main__": main()
