from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import requests
from PIL import Image, ImageChops, ImageStat
from remotezip import RemoteZip

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / ".assetgraph-lab" / "console_v2_work"
DIST = ROOT / "dist"
WORK.mkdir(parents=True, exist_ok=True)
DIST.mkdir(parents=True, exist_ok=True)

OUT_HTML = DIST / "ASSETGRAPH_INTELLIGENCE_CONSOLE_v2_REAL_IMAGERY.html"
OUT_MANIFEST = DIST / "assetgraph_console_v2_manifest.json"

HIT_VIDEOS = [
    {
        "id": "hit-60m-30-1",
        "name": "60m-30_1.mov",
        "url": "https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/main/video_sample/60m-30_1.mov",
        "altitude_m": 60,
        "camera_angle_deg": 30,
        "source": "HIT-UAV official GitHub video_sample",
    },
    {
        "id": "hit-70m-90-1",
        "name": "70m-90_1.mov",
        "url": "https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/main/video_sample/70m-90_1.mov",
        "altitude_m": 70,
        "camera_angle_deg": 90,
        "source": "HIT-UAV official GitHub video_sample",
    },
]
HIT_RESULT_URL = "https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/main/0_readme_images/fig_sample_result.jpg"

SEA_ANN_URL = "https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/annotations/instances_val_objects_in_water.json?download=true"
SEA_ZIP_URL = "https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/SeaDronesSee_MOT_jpg_compressed.zip?download=true"
SEA_AUTHORITY = "SeaDronesSee / University of Tuebingen"
SEA_TRANSPORT = "ObjEarth/ObjEarth-Data transport mirror used by prior AssetGraph MOT experiments"

KNOWN_LEDGER = {
    "project_branch": "assetgraph-cycle18b-frozen-intake",
    "baseline_commit": "8050ef89e231fc731680a3b60de786cba7f2eea4",
    "hit_uav_archive_sha256": "335416598c7debdb4316179b57aadcc4a5928a315a9d88cdb0f32f6d20622960",
    "hit_uav_member_manifest_sha256": "edcbe5b782a4c82e2b2578c215a4bc33afc019ce8a1f84c1205ac83227c36c71",
    "hit_uav_canonical_images": 2898,
    "hit_uav_freeze": "BYTE_FREEZE_PASS",
    "uav_obb_v1": "BLOCKED_PENDING_ANONYMOUS_V1_FILE_TREE_RESOLUTION",
    "au_air": "QUARANTINED_LICENSE_CONFLICT",
    "transset_h01": "SEALED",
    "training": "LOCKED_UNTIL_C18B_CLOSES",
    "rtmdet_r_onnx": "VERIFIED_RUNTIME_AND_POSTPROCESS_PARITY",
    "persistent_identity": "EXPERIMENTAL_NOT_FROZEN",
    "intelligence_memory": "RESEARCH_VALIDATED_NOT_PRODUCT_FROZEN",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, path: pathlib.Path) -> pathlib.Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"download: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path


def ffprobe_duration(path: pathlib.Path) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(0.5, float(p.stdout.strip()))


def extract_video_frames(video: pathlib.Path, out_dir: pathlib.Path, count: int = 8) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) == count:
        return existing
    for p in existing:
        p.unlink()
    duration = ffprobe_duration(video)
    times = np.linspace(duration * 0.06, duration * 0.94, count)
    out = []
    for i, t in enumerate(times):
        dst = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.4f}", "-i", str(video),
                "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", str(dst),
            ],
            check=True,
        )
        out.append(dst)
    return out


def motion_evidence(frame_paths: list[pathlib.Path]) -> list[dict]:
    rows = []
    prev = None
    for idx, p in enumerate(frame_paths):
        im = Image.open(p).convert("L").resize((640, 360))
        arr = np.asarray(im, dtype=np.float32)
        if prev is None:
            rows.append({"score": 0.0, "centroid": [0.5, 0.5], "bbox": [0.4, 0.4, 0.2, 0.2]})
        else:
            diff = np.abs(arr - prev)
            threshold = max(14.0, float(np.percentile(diff, 94)))
            mask = diff >= threshold
            yy, xx = np.where(mask)
            score = float(diff.mean() / 255.0)
            if len(xx) > 30:
                x1, x2 = np.percentile(xx, [10, 90])
                y1, y2 = np.percentile(yy, [10, 90])
                cx = float(xx.mean() / arr.shape[1])
                cy = float(yy.mean() / arr.shape[0])
                bbox = [float(x1 / arr.shape[1]), float(y1 / arr.shape[0]), float((x2-x1)/arr.shape[1]), float((y2-y1)/arr.shape[0])]
            else:
                cx, cy, bbox = 0.5, 0.5, [0.4, 0.4, 0.2, 0.2]
            rows.append({"score": score, "centroid": [cx, cy], "bbox": bbox})
        prev = arr
    return rows


def compress_jpeg_bytes(path: pathlib.Path, max_w: int = 1500, quality: int = 82) -> tuple[bytes, int, int]:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.Resampling.LANCZOS)
    import io
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue(), im.width, im.height


def data_uri_jpeg_bytes(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def frame_payload(path: pathlib.Path, source: dict, boxes: list[dict] | None = None, frame_id: str | None = None, extra: dict | None = None) -> dict:
    raw = path.read_bytes()
    packed, w, h = compress_jpeg_bytes(path)
    row = {
        "frame_id": frame_id or path.stem,
        "image": data_uri_jpeg_bytes(packed),
        "width": w,
        "height": h,
        "raw_sha256": sha256_bytes(raw),
        "packed_sha256": sha256_bytes(packed),
        "source": source,
        "boxes": boxes or [],
    }
    if extra:
        row.update(extra)
    return row


def crop_payload(img_path: pathlib.Path, bbox_xyxy: list[float], pad: float = 0.18) -> str:
    im = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
    x1 = max(0, int(x1 - bw * pad)); x2 = min(im.width, int(x2 + bw * pad))
    y1 = max(0, int(y1 - bh * pad)); y2 = min(im.height, int(y2 + bh * pad))
    c = im.crop((x1, y1, max(x1+1, x2), max(y1+1, y2)))
    c.thumbnail((360, 240), Image.Resampling.LANCZOS)
    import io
    b = io.BytesIO(); c.save(b, "JPEG", quality=84, optimize=True)
    return data_uri_jpeg_bytes(b.getvalue())


def xywh_to_xyxy(b: list[float]) -> list[float]:
    x, y, w, h = map(float, b)
    return [x, y, x+w, y+h]


def sea_annotations() -> tuple[dict, dict[int, list[dict]], dict[int, str], pathlib.Path]:
    ann_path = download(SEA_ANN_URL, WORK / "sea_instances_val.json")
    data = json.loads(ann_path.read_text())
    cats = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    ignored = {k for k, v in cats.items() if "ignore" in v.lower()}
    anns: dict[int, list[dict]] = defaultdict(list)
    for a in data["annotations"]:
        if int(a.get("category_id", -999)) in ignored:
            continue
        anns[int(a["image_id"])].append({
            "bbox": xywh_to_xyxy(a["bbox"]),
            "track_id": int(a["track_id"]),
            "category_id": int(a.get("category_id", -1)),
            "class_name": cats.get(int(a.get("category_id", -1)), str(a.get("category_id"))),
        })
    return data, anns, cats, ann_path


def sea_windows(data: dict, anns: dict[int, list[dict]], window: int = 120, stride: int = 30) -> list[dict]:
    byvid: dict[int, list[dict]] = defaultdict(list)
    for im in data["images"]:
        byvid[int(im["video_id"])].append(im)
    out = []
    for vid, ims in byvid.items():
        ims = sorted(ims, key=lambda x: x.get("frame_index", x["id"]))
        if len(ims) < window:
            continue
        for st in range(0, len(ims)-window+1, stride):
            w = ims[st:st+window]
            ids = [int(x["id"]) for x in w]
            counts = [len(anns.get(i, [])) for i in ids]
            tracks = {a["track_id"] for i in ids for a in anns.get(i, [])}
            score = 10*(max(counts) if counts else 0) + 3*len(tracks) + 2*(sum(counts)/max(len(counts),1)) + sum(c>0 for c in counts)/max(len(counts),1)
            out.append({"video_id": vid, "start": st, "images": w, "tracks": tracks, "score": float(score), "max_concurrent": max(counts) if counts else 0})
    return out


def pick_windows(candidates: list[dict]) -> tuple[dict, dict, dict, dict]:
    # stress: highest structural challenge
    stress = max(candidates, key=lambda x: (x["score"], len(x["tracks"])))

    # identity pair: same video, non-overlapping or substantially separated windows, maximum shared GT track IDs.
    best = None
    same = defaultdict(list)
    for c in candidates:
        same[c["video_id"]].append(c)
    for vid, rows in same.items():
        rows = sorted(rows, key=lambda x: x["start"])
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                gap = b["start"] - a["start"]
                if gap < 90:
                    continue
                shared = a["tracks"] & b["tracks"]
                if not shared:
                    continue
                score = len(shared)*100 + min(gap, 360)*0.2 + a["score"]*0.02 + b["score"]*0.02
                if best is None or score > best[0]:
                    best = (score, a, b, shared)
    if best is None:
        # fallback permits some overlap but still demands shared identity.
        for vid, rows in same.items():
            rows = sorted(rows, key=lambda x: x["start"])
            for i, a in enumerate(rows):
                for b in rows[i+1:]:
                    if b["start"] - a["start"] < 30:
                        continue
                    shared = a["tracks"] & b["tracks"]
                    if shared:
                        score = len(shared)*100 + (b["start"]-a["start"])*0.1
                        if best is None or score > best[0]:
                            best = (score, a, b, shared)
    if best is None:
        raise RuntimeError("Could not find a GT-backed cross-window identity pair")
    _, identity_a, identity_b, shared = best
    shared_track = sorted(shared)[0]
    identity_a = {**identity_a, "primary_track": shared_track}
    identity_b = {**identity_b, "primary_track": shared_track}

    # pattern window: best structurally different video if possible
    pattern_pool = [c for c in candidates if c["video_id"] not in {stress["video_id"], identity_a["video_id"]}]
    pattern = max(pattern_pool or candidates, key=lambda x: (len(x["tracks"]), x["score"]))
    return stress, identity_a, identity_b, pattern


def sample_window(window: dict, count: int = 8) -> list[dict]:
    ims = window["images"]
    idx = np.linspace(0, len(ims)-1, count).round().astype(int)
    return [ims[int(i)] for i in idx]


def acquire_sea_frames(windows: dict[str, dict], anns: dict[int, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    rz = RemoteZip(SEA_ZIP_URL)
    names = set(rz.namelist())
    out: dict[str, list[dict]] = {}
    evidence = {"archive_transport": SEA_ZIP_URL, "members": {}}
    try:
        for key, window in windows.items():
            sampled = sample_window(window, 8)
            rows = []
            for j, im in enumerate(sampled):
                image_id = int(im["id"])
                member = f"Compressed/val/{image_id}.jpg"
                if member not in names:
                    raise RuntimeError(f"SeaDronesSee member missing: {member}")
                b = rz.read(member)
                dst = WORK / "sea" / key / f"{j:02d}_{image_id}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(b)
                with Image.open(dst) as chk:
                    chk.verify()
                boxes = []
                for a in anns.get(image_id, []):
                    boxes.append({
                        "xyxy": a["bbox"],
                        "track_id": a["track_id"],
                        "class_name": a["class_name"],
                        "category_id": a["category_id"],
                        "evidence": "GROUND_TRUTH",
                        "confidence": 1.0,
                    })
                rows.append({"path": dst, "im": im, "boxes": boxes, "member": member, "raw_sha256": sha256_bytes(b)})
                evidence["members"][member] = {"sha256": sha256_bytes(b), "bytes": len(b), "image_id": image_id}
            out[key] = rows
    finally:
        rz.close()
    return out, evidence


def primary_track_for_rows(rows: list[dict], requested: int | None = None) -> int | None:
    if requested is not None:
        return requested
    counts = defaultdict(int); areas = defaultdict(float)
    for r in rows:
        for b in r["boxes"]:
            tid = int(b["track_id"]); counts[tid] += 1
            x1,y1,x2,y2 = b["xyxy"]; areas[tid] += max(1, (x2-x1)*(y2-y1))
    if not counts:
        return None
    return max(counts, key=lambda t: (counts[t], areas[t]))


def pack_sea_mission(mission_id: str, title: str, subtitle: str, rows: list[dict], window: dict, primary_track: int | None, mode: str, narrative: str) -> dict:
    primary_track = primary_track_for_rows(rows, primary_track)
    source = {"authority": SEA_AUTHORITY, "transport": SEA_TRANSPORT, "license": "CC0-1.0", "track": "MOT validation reproduction"}
    frames = []
    crops = []
    trajectory = []
    events = []
    for idx, r in enumerate(rows):
        payload = frame_payload(r["path"], source, r["boxes"], str(r["im"]["id"]), {"member": r["member"], "video_id": int(r["im"]["video_id"]), "frame_index": r["im"].get("frame_index")})
        frames.append(payload)
        for b in r["boxes"]:
            if primary_track is not None and int(b["track_id"]) == int(primary_track):
                x1,y1,x2,y2 = b["xyxy"]
                trajectory.append({"frame": idx, "x": (x1+x2)/2/payload["width"], "y": (y1+y2)/2/payload["height"], "track_id": primary_track})
                crops.append({"frame": idx, "image": crop_payload(r["path"], b["xyxy"]), "track_id": primary_track, "class_name": b["class_name"]})
    if trajectory:
        events.append({"type": "TRACK_PRESENT", "severity": "info", "text": f"GT track {primary_track} observed in {len(trajectory)}/{len(frames)} sampled frames."})
        if len(trajectory) >= 2:
            dx = trajectory[-1]["x"] - trajectory[0]["x"]; dy = trajectory[-1]["y"] - trajectory[0]["y"]
            events.append({"type": "DISPLACEMENT", "severity": "info", "text": f"Normalized displacement dx={dx:+.3f}, dy={dy:+.3f}."})
    return {
        "id": mission_id,
        "title": title,
        "subtitle": subtitle,
        "sensor": "UAV RGB / maritime",
        "evidence_status": "VERIFIED_GT",
        "mode": mode,
        "narrative": narrative,
        "frames": frames,
        "primary_asset": f"GT-{primary_track}" if primary_track is not None else None,
        "primary_track": primary_track,
        "crops": crops,
        "trajectory": trajectory,
        "events": events,
        "window": {"video_id": window["video_id"], "start": window["start"], "score": window["score"], "track_count": len(window["tracks"]), "max_concurrent": window["max_concurrent"]},
    }


def hit_mission(video_meta: dict, mission_id: str, title: str, subtitle: str, frame_paths: list[pathlib.Path], narrative: str, official_result_uri: str | None = None) -> dict:
    motion = motion_evidence(frame_paths)
    source = {"authority": "HIT-UAV official GitHub repository", "transport": "raw.githubusercontent.com", "license_note": "sample video in official repository; release-byte freeze tracked separately", "video": video_meta["name"]}
    frames = []
    trajectory = []
    for i, p in enumerate(frame_paths):
        m = motion[i]
        packed, w, h = compress_jpeg_bytes(p)
        x,y,bw,bh = m["bbox"]
        # Motion ROI is analytical evidence, not a semantic detection.
        box = {
            "xyxy": [x*w, y*h, (x+bw)*w, (y+bh)*h],
            "track_id": "MOTION-ROI",
            "class_name": "motion region",
            "category_id": -1,
            "confidence": round(min(0.99, 0.45 + m["score"]*5), 3),
            "evidence": "MOTION_ANALYTICS_EXPERIMENTAL",
        }
        row = {
            "frame_id": f"{video_meta['name']}:{i:02d}",
            "image": data_uri_jpeg_bytes(packed),
            "width": w, "height": h,
            "raw_sha256": sha256_file(p), "packed_sha256": sha256_bytes(packed),
            "source": source,
            "boxes": [box],
            "motion_score": m["score"],
        }
        frames.append(row)
        trajectory.append({"frame": i, "x": m["centroid"][0], "y": m["centroid"][1], "track_id": "MOTION-ROI"})
    return {
        "id": mission_id,
        "title": title,
        "subtitle": subtitle,
        "sensor": "UAV thermal / infrared",
        "evidence_status": "REAL_IMAGERY_EXPERIMENTAL_ANALYTICS",
        "mode": "THERMAL_MOTION",
        "narrative": narrative,
        "frames": frames,
        "primary_asset": "MOTION-ROI",
        "primary_track": "MOTION-ROI",
        "crops": [],
        "trajectory": trajectory,
        "events": [
            {"type": "REAL_SEQUENCE", "severity": "verified", "text": f"8 frames extracted from official HIT-UAV sample video {video_meta['name']}."},
            {"type": "ANALYTICS_BOUNDARY", "severity": "warn", "text": "Highlighted ROI is frame-difference motion evidence, not a semantic detector output."},
        ],
        "video_meta": {**video_meta, "sha256": sha256_file(WORK / video_meta["name"])},
        "official_detection_result": official_result_uri,
    }


def js_safe(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_html(missions: list[dict], manifest: dict) -> str:
    data_json = js_safe({"missions": missions, "manifest": manifest, "ledger": KNOWN_LEDGER})
    css = r'''
:root{--bg:#061014;--panel:#0a171c;--panel2:#0d2027;--line:#17333d;--ink:#ecf7f8;--muted:#86a5ad;--cyan:#51e5ff;--lime:#8bffb7;--amber:#ffc857;--red:#ff6b75;--violet:#b69cff}
*{box-sizing:border-box}html,body{margin:0;background:radial-gradient(circle at 55% -20%,#15333c 0,#071318 34%,#03090c 100%);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;height:100%}button{font:inherit}.app{min-height:100vh;display:grid;grid-template-rows:auto 1fr}.top{display:flex;align-items:center;gap:16px;padding:13px 18px;border-bottom:1px solid var(--line);background:#061116e8;backdrop-filter:blur(16px);position:sticky;top:0;z-index:40}.brand{font-weight:800;letter-spacing:.13em;font-size:14px}.brand b{color:var(--cyan)}.tag{font-size:10px;border:1px solid #26505d;color:#9fc8d1;border-radius:999px;padding:5px 8px}.top .spacer{flex:1}.btn{border:1px solid #244b57;background:#0a1d24;color:#dff8fb;border-radius:9px;padding:8px 11px;cursor:pointer}.btn:hover{border-color:var(--cyan);box-shadow:0 0 22px #51e5ff19}.btn.primary{background:linear-gradient(135deg,#173e46,#0e2730);border-color:#3cc9df}.shell{display:grid;grid-template-columns:255px minmax(0,1fr) 330px;min-height:0}.left,.right{background:#071319d9;min-height:0;overflow:auto}.left{border-right:1px solid var(--line)}.right{border-left:1px solid var(--line)}.sectionTitle{font-size:10px;letter-spacing:.18em;color:#71929a;padding:16px 14px 7px}.mission{margin:5px 8px;padding:10px;border:1px solid transparent;border-radius:10px;cursor:pointer;transition:.18s}.mission:hover{background:#0d222a}.mission.active{border-color:#245a68;background:#0c222a;box-shadow:inset 3px 0 var(--cyan)}.mission .id{font:700 10px ui-monospace,SFMono-Regular,monospace;color:var(--cyan)}.mission .name{font-weight:700;margin:3px 0;font-size:13px}.mission .sub{font-size:10px;color:var(--muted);line-height:1.35}.status{display:inline-block;font:700 9px ui-monospace,monospace;padding:3px 5px;border-radius:4px;margin-top:6px}.status.verified{color:#8bffb7;background:#123225}.status.exp{color:#ffc857;background:#382b13}.main{min-width:0;padding:12px;overflow:auto}.hero{position:relative;background:#020708;border:1px solid var(--line);border-radius:13px;overflow:hidden;min-height:520px;box-shadow:0 30px 80px #0009}.hero img{width:100%;height:min(64vh,680px);display:block;object-fit:contain;background:#020607}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.box{fill:none;stroke:var(--lime);stroke-width:2;vector-effect:non-scaling-stroke}.box.motion{stroke:var(--amber);stroke-dasharray:7 5}.boxLabel{fill:#021014;stroke:none}.labelText{fill:var(--lime);font:700 12px ui-monospace,monospace}.motionText{fill:var(--amber)}.hud{position:absolute;left:12px;top:12px;background:#041013d9;border:1px solid #28505b;border-radius:8px;padding:8px 10px;font:11px ui-monospace,monospace;backdrop-filter:blur(8px)}.hud strong{color:var(--cyan)}.framebar{display:flex;gap:7px;padding:9px 2px;overflow:auto}.thumb{width:108px;min-width:108px;height:66px;border-radius:7px;overflow:hidden;border:1px solid #1d3b44;cursor:pointer;position:relative;background:#09161b}.thumb.active{border-color:var(--cyan);box-shadow:0 0 0 1px var(--cyan)}.thumb img{width:100%;height:100%;object-fit:cover}.thumb span{position:absolute;bottom:2px;right:3px;background:#000b;padding:2px 4px;border-radius:3px;font:9px ui-monospace,monospace}.panel{margin:10px;border:1px solid var(--line);border-radius:11px;background:#09171c;overflow:hidden}.panel h3{font-size:11px;letter-spacing:.12em;margin:0;padding:10px 11px;border-bottom:1px solid var(--line);color:#9dc0c8}.panel .body{padding:10px}.metricgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.metric{background:#0b2027;border:1px solid #163740;padding:8px;border-radius:8px}.metric b{font:700 16px ui-monospace,monospace;color:#e5fbff}.metric small{display:block;color:var(--muted);font-size:9px;margin-top:3px}.timeline{font-size:11px}.event{padding:7px 0;border-bottom:1px solid #122a31}.event:last-child{border:0}.event .t{color:var(--cyan);font:10px ui-monospace,monospace}.event.warn .t{color:var(--amber)}.event.verified .t{color:var(--lime)}.cropstrip{display:flex;gap:6px;overflow:auto;padding-bottom:5px}.cropstrip img{width:92px;height:66px;object-fit:contain;background:#020708;border:1px solid #20424c;border-radius:7px}.tabs{display:flex;gap:4px;padding:10px}.tab{flex:1;padding:7px 4px;border:1px solid #1b3942;background:#08171c;color:#8db0b8;border-radius:7px;font-size:10px;cursor:pointer}.tab.active{color:#e8fbff;border-color:#41b9ce;background:#102831}.tabpage{display:none}.tabpage.active{display:block}.evidence{font:10px ui-monospace,monospace;line-height:1.55;word-break:break-all;color:#9fc0c7}.evidence b{color:#e6fbff}.graph{height:240px;width:100%;background:radial-gradient(circle at 50% 50%,#10252d,#061015 70%)}.node{fill:#0b2630;stroke:#47d6eb;stroke-width:1.5}.node.primary{fill:#17362c;stroke:#8bffb7;stroke-width:2}.edge{stroke:#315963;stroke-width:1.2}.nodeText{fill:#dff8fb;font:9px ui-monospace,monospace;text-anchor:middle}.footerNote{font-size:9px;color:#64848b;padding:10px 12px;line-height:1.45}.fusion{border-color:#8bffb7!important;box-shadow:0 0 38px #8bffb71a!important}.memoryPulse{animation:pulse 1.4s infinite}@keyframes pulse{50%{filter:drop-shadow(0 0 8px #8bffb7)}}.modal{position:fixed;inset:0;background:#000c;z-index:100;display:none;align-items:center;justify-content:center;padding:20px}.modal.open{display:flex}.modalCard{width:min(980px,96vw);max-height:90vh;overflow:auto;background:#07171d;border:1px solid #2d5c68;border-radius:14px;box-shadow:0 30px 100px #000}.modalHead{display:flex;align-items:center;padding:12px;border-bottom:1px solid var(--line)}.modalHead b{flex:1}.modalBody{padding:14px}.close{background:none;border:0;color:#bcd4da;font-size:22px;cursor:pointer}.liveDrop{border:1px dashed #39707e;padding:30px;text-align:center;border-radius:12px;color:#9ebec5}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.small{font-size:10px;color:var(--muted)}.pill{display:inline-flex;gap:5px;align-items:center;border:1px solid #234953;border-radius:999px;padding:4px 7px;font-size:9px;margin:2px}.pill.ok{color:var(--lime)}.pill.warn{color:var(--amber)}.techrow{display:grid;grid-template-columns:1fr auto;gap:8px;border-bottom:1px solid #143039;padding:7px 0;font-size:10px}.techrow:last-child{border:0}@media(max-width:1050px){.shell{grid-template-columns:210px minmax(0,1fr)}.right{display:none}}@media(max-width:720px){.shell{display:block}.left{display:flex;overflow:auto;border:0;border-bottom:1px solid var(--line);position:sticky;top:54px;z-index:20;background:#071319}.sectionTitle{display:none}.mission{min-width:180px}.main{padding:7px}.hero{min-height:350px}.hero img{height:55vh}.top .tag{display:none}.brand{font-size:11px}}
'''
    js = r'''
const DATA=__DATA__;
let missionIndex=0, frameIndex=0, activeTab='intel', autoplay=null;
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function statusClass(m){return m.evidence_status.includes('VERIFIED')?'verified':'exp'}
function renderMissionList(){const host=$('#missions');host.innerHTML='';DATA.missions.forEach((m,i)=>{const d=document.createElement('div');d.className='mission '+(i===missionIndex?'active':'');d.innerHTML=`<div class="id">${m.id}</div><div class="name">${esc(m.title)}</div><div class="sub">${esc(m.subtitle)}</div><span class="status ${statusClass(m)}">${esc(m.evidence_status)}</span>`;d.onclick=()=>selectMission(i);host.appendChild(d)})}
function selectMission(i){missionIndex=i;frameIndex=0;renderMissionList();render();if(DATA.missions[i].id==='M07') setTimeout(()=>openFusion(),350)}
function selectFrame(i){frameIndex=i;renderFrame()}
function render(){const m=DATA.missions[missionIndex];$('#missionTitle').textContent=`${m.id} · ${m.title}`;$('#missionSub').textContent=m.subtitle;$('#sensor').textContent=m.sensor;$('#evidenceStatus').textContent=m.evidence_status;$('#narrative').textContent=m.narrative;$('#hero').classList.toggle('fusion',m.id==='M07');renderFrame();renderPanels();}
function renderFrame(){const m=DATA.missions[missionIndex],f=m.frames[frameIndex]||m.frames[0];$('#mainImage').src=f.image;$('#frameId').textContent=f.frame_id;$('#frameHash').textContent=f.raw_sha256.slice(0,16)+'…';$('#frameCount').textContent=`${frameIndex+1}/${m.frames.length}`;const ov=$('#overlay');ov.setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);ov.innerHTML='';(f.boxes||[]).forEach((b,k)=>{const [x1,y1,x2,y2]=b.xyxy;const g=document.createElementNS('http://www.w3.org/2000/svg','g');const motion=String(b.evidence).includes('MOTION');g.innerHTML=`<rect class="box ${motion?'motion':''}" x="${x1}" y="${y1}" width="${Math.max(1,x2-x1)}" height="${Math.max(1,y2-y1)}"/><rect class="boxLabel" x="${x1}" y="${Math.max(0,y1-18)}" width="${Math.max(110,(x2-x1))}" height="18"/><text class="labelText ${motion?'motionText':''}" x="${x1+4}" y="${Math.max(12,y1-5)}">${esc(b.class_name)} · ${esc(b.track_id)}</text>`;ov.appendChild(g)});const tb=$('#frames');tb.innerHTML='';m.frames.forEach((x,i)=>{const d=document.createElement('div');d.className='thumb '+(i===frameIndex?'active':'');d.innerHTML=`<img src="${x.image}"><span>${i+1}</span>`;d.onclick=()=>selectFrame(i);tb.appendChild(d)});renderFrameEvidence(f)}
function renderFrameEvidence(f){$('#frameEvidence').innerHTML=`<b>FRAME</b> ${esc(f.frame_id)}<br><b>RAW SHA-256</b> ${esc(f.raw_sha256)}<br><b>PACKED SHA-256</b> ${esc(f.packed_sha256)}<br><b>SOURCE</b> ${esc(JSON.stringify(f.source))}<br><b>OVERLAYS</b> ${(f.boxes||[]).map(b=>esc(`${b.evidence}:${b.track_id}:${b.class_name}`)).join('<br>')||'none'}`}
function renderPanels(){const m=DATA.missions[missionIndex];$('#assetId').textContent=m.primary_asset||'N/A';$('#obsCount').textContent=m.frames.length;$('#trackCount').textContent=m.window?.track_count??(m.primary_asset?1:0);$('#proof').textContent=m.evidence_status;$('#events').innerHTML=(m.events||[]).map((e,i)=>`<div class="event ${e.severity||''}"><div class="t">${String(i+1).padStart(2,'0')} · ${esc(e.type)}</div>${esc(e.text)}</div>`).join('');$('#crops').innerHTML=(m.crops||[]).length?m.crops.map(c=>`<img src="${c.image}" title="${esc(c.class_name)} · track ${c.track_id} · frame ${c.frame+1}">`).join(''):'<span class="small">No semantic GT crop available in this mission.</span>';renderGraph(m);renderTech();}
function renderGraph(m){const host=$('#graph');const tid=m.primary_track??m.primary_asset??'asset';const nodes=[['mission',m.id,90,55],['obs','OBSERVATIONS',250,55],['asset',String(tid),420,55],['memory','MEMORY',420,155],['event','EVENTS',250,155],['report','REPORT',90,155]];const edges=[[0,1],[1,2],[2,3],[3,4],[4,5],[2,4]];let svg=`<svg viewBox="0 0 510 220" width="100%" height="100%">`;edges.forEach(([a,b])=>{svg+=`<line class="edge" x1="${nodes[a][2]}" y1="${nodes[a][3]}" x2="${nodes[b][2]}" y2="${nodes[b][3]}"/>`});nodes.forEach((n,i)=>{svg+=`<g class="${(i===2||m.id==='M07'&&i===3)?'memoryPulse':''}"><circle class="node ${i===2?'primary':''}" cx="${n[2]}" cy="${n[3]}" r="30"/><text class="nodeText" x="${n[2]}" y="${n[3]+3}">${esc(n[1]).slice(0,16)}</text></g>`});svg+='</svg>';host.innerHTML=svg}
function renderTech(){const l=DATA.ledger;const rows=[['RTMDet-R ONNX',l.rtmdet_r_onnx,'ok'],['HIT-UAV frozen intake',l.hit_uav_freeze,'ok'],['Persistent identity',l.persistent_identity,'warn'],['Intelligence memory',l.intelligence_memory,'warn'],['UAV-OBB v1',l.uav_obb_v1,'warn'],['AU-AIR',l.au_air,'warn'],['TRANSSET-H01',l.transset_h01,'warn'],['Training',l.training,'warn']];$('#tech').innerHTML=rows.map(r=>`<div class="techrow"><span>${esc(r[0])}</span><span class="pill ${r[2]}">${esc(r[1])}</span></div>`).join('')}
function tab(name){activeTab=name;$$('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));$$('.tabpage').forEach(x=>x.classList.toggle('active',x.id==='tab-'+name))}
function openFusion(){const m=DATA.missions[missionIndex];if(m.id!=='M07')return;const msg=`Memory match confirmed by dataset ground truth. Track ${m.primary_track} is present in an earlier mission window and reappears in this later window. The console is not inferring identity here: it is demonstrating the product behavior against a GT-backed identity.`;$('#fusionText').textContent=msg;openModal('fusionModal')}
function runDemo(){if(autoplay){clearInterval(autoplay);autoplay=null;return}let mi=0,fi=0;selectMission(0);autoplay=setInterval(()=>{const m=DATA.missions[mi];if(fi<m.frames.length-1){fi++;frameIndex=fi;renderFrame()}else{mi++;fi=0;if(mi>=DATA.missions.length){clearInterval(autoplay);autoplay=null;return}selectMission(mi)}},850)}
function openModal(id){$('#'+id).classList.add('open')}function closeModal(id){$('#'+id).classList.remove('open')}
function report(){const m=DATA.missions[missionIndex];return {schema:'assetgraph-intelligence-report/v2',generated_at:new Date().toISOString(),mission:{id:m.id,title:m.title,sensor:m.sensor,evidence_status:m.evidence_status},assessment:m.narrative,primary_asset:m.primary_asset,events:m.events,observations:m.frames.map(f=>({frame_id:f.frame_id,sha256:f.raw_sha256,boxes:f.boxes?.map(b=>({track_id:b.track_id,class_name:b.class_name,evidence:b.evidence,confidence:b.confidence,xyxy:b.xyxy}))})),provenance:m.frames.map(f=>f.source),technology_ledger:DATA.ledger}}
function downloadText(name,text,type='application/json'){const b=new Blob([text],{type});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500)}
function exportReport(){downloadText(`assetgraph_${DATA.missions[missionIndex].id}_report.json`,JSON.stringify(report(),null,2))}
function exportEvidence(){downloadText('assetgraph_console_v2_evidence_manifest.json',JSON.stringify(DATA.manifest,null,2))}
async function handleFiles(files){const host=$('#liveResult');host.innerHTML='';for(const file of files){const buf=await file.arrayBuffer();const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',buf))].map(b=>b.toString(16).padStart(2,'0')).join('');let dims='';try{const u=URL.createObjectURL(file);const im=new Image();dims=await new Promise(res=>{im.onload=()=>{res(`${im.naturalWidth}×${im.naturalHeight}`);URL.revokeObjectURL(u)};im.onerror=()=>res('n/a');im.src=u})}catch(e){}host.innerHTML+=`<div class="panel"><div class="body mono small"><b>${esc(file.name)}</b><br>${esc(file.type||'unknown')} · ${file.size.toLocaleString()} bytes · ${esc(dims)}<br>SHA-256 ${hash}<br><span style="color:var(--amber)">Ingest + hash are real. Semantic inference is intentionally not claimed until the product model is packaged.</span></div></div>`}}
window.addEventListener('keydown',e=>{if(e.key==='Escape')$$('.modal.open').forEach(m=>m.classList.remove('open'));if(e.key==='ArrowRight'){const m=DATA.missions[missionIndex];selectFrame(Math.min(m.frames.length-1,frameIndex+1))}if(e.key==='ArrowLeft')selectFrame(Math.max(0,frameIndex-1))});
window.addEventListener('DOMContentLoaded',()=>{renderMissionList();render();tab('intel');$$('.modal').forEach(m=>m.addEventListener('click',e=>{if(e.target===m)m.classList.remove('open')}));const drop=$('#drop');drop.addEventListener('dragover',e=>{e.preventDefault();drop.style.borderColor='var(--cyan)'});drop.addEventListener('dragleave',()=>drop.style.borderColor='');drop.addEventListener('drop',e=>{e.preventDefault();handleFiles(e.dataTransfer.files)});$('#file').addEventListener('change',e=>handleFiles(e.target.files))});
'''.replace('__DATA__', data_json)
    body = r'''
<div class="app"><header class="top"><div class="brand"><b>ASSETGRAPH</b> INTELLIGENCE CONSOLE</div><span class="tag">v2 · REAL IMAGERY PACK</span><span class="tag">OFFLINE SINGLE FILE</span><div class="spacer"></div><button class="btn" onclick="openModal('liveModal')">LIVE INGEST</button><button class="btn" onclick="exportEvidence()">EVIDENCE PACK</button><button class="btn primary" onclick="runDemo()">▶ RUN DEMO</button></header><div class="shell"><aside class="left"><div class="sectionTitle">MISSION PACKS</div><div id="missions"></div><div class="footerNote">The interface separates VERIFIED ground-truth evidence, REAL imagery with experimental analytics, and capabilities that are not yet product-frozen.</div></aside><main class="main"><div style="display:flex;align-items:end;gap:10px;margin:3px 2px 10px"><div><div id="missionTitle" style="font-weight:800;font-size:18px"></div><div id="missionSub" class="small"></div></div><div class="spacer"></div><span class="pill" id="sensor"></span><span class="pill ok" id="evidenceStatus"></span></div><div id="hero" class="hero"><img id="mainImage" alt="mission frame"><svg id="overlay" class="overlay"></svg><div class="hud"><strong id="frameId"></strong><br><span id="frameCount"></span> · SHA <span id="frameHash"></span></div></div><div id="frames" class="framebar"></div><div class="panel"><h3>MISSION ASSESSMENT</h3><div class="body"><div id="narrative" style="line-height:1.6;font-size:13px"></div></div></div></main><aside class="right"><div class="tabs"><button class="tab active" data-tab="intel" onclick="tab('intel')">INTEL</button><button class="tab" data-tab="graph" onclick="tab('graph')">GRAPH</button><button class="tab" data-tab="evidence" onclick="tab('evidence')">EVIDENCE</button><button class="tab" data-tab="tech" onclick="tab('tech')">TRUST</button></div><div class="tabpage active" id="tab-intel"><div class="panel"><h3>ASSET DOSSIER</h3><div class="body"><div class="metricgrid"><div class="metric"><b id="assetId">—</b><small>PRIMARY ASSET</small></div><div class="metric"><b id="obsCount">—</b><small>SAMPLED OBSERVATIONS</small></div><div class="metric"><b id="trackCount">—</b><small>WINDOW TRACKS</small></div><div class="metric"><b id="proof">—</b><small>EVIDENCE CLASS</small></div></div></div></div><div class="panel"><h3>ASSET CROPS</h3><div class="body cropstrip" id="crops"></div></div><div class="panel"><h3>MISSION TIMELINE</h3><div class="body timeline" id="events"></div></div><div style="padding:0 10px 10px"><button class="btn primary" style="width:100%" onclick="exportReport()">EXPORT INTELLIGENCE REPORT</button></div></div><div class="tabpage" id="tab-graph"><div class="panel"><h3>ASSET GRAPH</h3><div class="body"><div id="graph" class="graph"></div><p class="small">Mission → observation → asset → memory → event → report. M07 lights memory because the same GT track exists in an earlier separated window.</p></div></div></div><div class="tabpage" id="tab-evidence"><div class="panel"><h3>FRAME PROVENANCE</h3><div class="body evidence" id="frameEvidence"></div></div><div class="panel"><h3>EVIDENCE POLICY</h3><div class="body small">Green overlays on SeaDronesSee are dataset ground truth. Amber overlays on HIT-UAV sample-video missions are motion regions computed from real frames and are explicitly not semantic detections. No UI element upgrades an experimental result to VERIFIED.</div></div></div><div class="tabpage" id="tab-tech"><div class="panel"><h3>TECHNOLOGY LEDGER</h3><div class="body" id="tech"></div></div></div></aside></div></div>
<div class="modal" id="fusionModal"><div class="modalCard"><div class="modalHead"><b>INTELLIGENCE FUSION · MEMORY HIT</b><button class="close" onclick="closeModal('fusionModal')">×</button></div><div class="modalBody"><div style="font-size:20px;font-weight:800;color:var(--lime);margin-bottom:12px">PRIOR OBSERVATION RECOVERED</div><p id="fusionText" style="line-height:1.7"></p><div class="panel"><div class="body"><b>WHY THIS MATTERS</b><p class="small">The commercial behavior is not “detect another object”. It is: use prior missions to enrich the new acquisition, surface historical evidence, relationships and uncertainty, then compile a defensible intelligence object.</p></div></div><button class="btn primary" onclick="closeModal('fusionModal');tab('graph')">ENTER ASSET GRAPH</button></div></div></div>
<div class="modal" id="liveModal"><div class="modalCard"><div class="modalHead"><b>LIVE INGEST · LOCAL BROWSER</b><button class="close" onclick="closeModal('liveModal')">×</button></div><div class="modalBody"><label id="drop" class="liveDrop" style="display:block">Drop new imagery here<br><span class="small">or click to choose files</span><input id="file" type="file" accept="image/*" multiple style="display:block;margin:14px auto"></label><div id="liveResult"></div></div></div></div>
'''
    return "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ASSETGRAPH Intelligence Console v2 · Real Imagery</title><style>" + css + "</style></head><body>" + body + "<script>" + js + "</script></body></html>"


def main() -> None:
    # 1) HIT-UAV official sample videos and official detection-result contact sheet.
    hit_frames = {}
    for meta in HIT_VIDEOS:
        video_path = download(meta["url"], WORK / meta["name"])
        frames = extract_video_frames(video_path, WORK / "hit_frames" / meta["id"], 8)
        hit_frames[meta["id"]] = frames
    hit_result_path = download(HIT_RESULT_URL, WORK / "hit_official_sample_result.jpg")
    hit_result_packed, _, _ = compress_jpeg_bytes(hit_result_path, max_w=1400, quality=83)
    hit_result_uri = data_uri_jpeg_bytes(hit_result_packed)

    # 2) SeaDronesSee real MOT frames + GT identity.
    data, anns, cats, ann_path = sea_annotations()
    candidates = sea_windows(data, anns)
    stress, identity_a, identity_b, pattern = pick_windows(candidates)
    windows = {"stress": stress, "identity_early": identity_a, "identity_late": identity_b, "pattern": pattern}
    sea_rows, sea_ev = acquire_sea_frames(windows, anns)

    shared_tid = int(identity_a["primary_track"])
    missions = [
        hit_mission(HIT_VIDEOS[0], "M01", "THERMAL URBAN WATCH", "Real HIT-UAV sequence · motion evidence", hit_frames[HIT_VIDEOS[0]["id"]], "A real thermal UAV sequence is ingested as a mission pack. The console computes frame-difference motion evidence and provenance locally; it does not mislabel that evidence as a semantic detector output.", None),
        hit_mission(HIT_VIDEOS[1], "M02", "THERMAL NIGHT OPS", "Sensor-domain contrast · official HIT-UAV sample evidence", hit_frames[HIT_VIDEOS[1]["id"]], "A second official HIT-UAV sample sequence demonstrates domain and viewpoint change. The official repository detection contact sheet is embedded as supporting source evidence while semantic product inference remains explicitly unfrozen.", hit_result_uri),
        pack_sea_mission("M03", "MARITIME SEARCH", "GT-backed high-challenge MOT window", sea_rows["stress"], stress, None, "GT_TRACKING", "A structurally difficult SeaDronesSee MOT validation window is selected without model output. Ground-truth boxes and track IDs make detection density, occlusion pressure and target continuity inspectable frame by frame."),
        pack_sea_mission("M04", "PERSISTENT IDENTITY", "Earlier separated window · same-video GT identity", sea_rows["identity_early"], identity_a, shared_tid, "GT_IDENTITY_EARLY", f"This earlier mission window contains GT track {shared_tid}. Its crops and trajectory become the memory record used later by Mission 07. Identity is grounded in dataset track ID, not guessed by the UI."),
        pack_sea_mission("M05", "PATTERN OF LIFE", "Trajectory aggregation over real maritime frames", sea_rows["pattern"], pattern, None, "GT_PATTERN", "AssetGraph converts repeated observations into trajectory and activity context: which tracks remain present, where they move, and how activity evolves through the selected acquisition window."),
        pack_sea_mission("M06", "EVENT & RELATIONSHIP INTELLIGENCE", "Events derived from GT observations", sea_rows["stress"], stress, None, "GT_EVENTS", "The same evidence can be lifted above boxes into events: presence, displacement, entrances/exits and co-occurrence. The console keeps the underlying frame, bbox and track provenance available for analyst inspection."),
        pack_sea_mission("M07", "INTELLIGENCE FUSION", "Later separated window · prior GT track reappears", sea_rows["identity_late"], identity_b, shared_tid, "GT_MEMORY_FUSION", f"A later acquisition window contains GT track {shared_tid}, which already existed in Mission 04. The console retrieves the prior dossier, crops and trajectory and exposes a memory hit before compiling the intelligence report. This is the commercial payoff: new imagery is enriched by accumulated evidence."),
    ]
    missions[1]["events"].append({"type": "OFFICIAL_SAMPLE_RESULT", "severity": "verified", "text": "Official HIT-UAV repository YOLOv4 detection contact sheet embedded in this standalone as source evidence."})

    manifest = {
        "schema": "assetgraph-console-real-imagery-manifest/v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder": ".assetgraph-lab/build_intelligence_console_v2.py",
        "ground_truth_policy": "SeaDronesSee overlays are direct GT from the validation MOT annotation transport used in prior AssetGraph experiments; HIT-UAV moving ROI is analytics-only.",
        "hit_uav": {
            "official_sample_videos": [{**m, "sha256": sha256_file(WORK / m["name"]), "bytes": (WORK / m["name"]).stat().st_size} for m in HIT_VIDEOS],
            "official_sample_result_sha256": sha256_file(hit_result_path),
            "release_freeze": {"archive_sha256": KNOWN_LEDGER["hit_uav_archive_sha256"], "member_manifest_sha256": KNOWN_LEDGER["hit_uav_member_manifest_sha256"], "canonical_images": 2898},
        },
        "seadronessee": {
            "authority": SEA_AUTHORITY,
            "transport": SEA_TRANSPORT,
            "annotation_sha256": sha256_file(ann_path),
            "selected_windows": {k: {"video_id": v["video_id"], "start": v["start"], "score": v["score"], "tracks": sorted(v["tracks"])} for k, v in windows.items()},
            "shared_identity_track": shared_tid,
            "frame_evidence": sea_ev,
            "categories": cats,
        },
        "technology_ledger": KNOWN_LEDGER,
    }

    html_text = build_html(missions, manifest)
    OUT_HTML.write_text(html_text, encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build guardrails.
    if html_text.count("data:image/jpeg;base64,") < 35:
        raise RuntimeError("Too few embedded real-image assets")
    if "PRIOR OBSERVATION RECOVERED" not in html_text:
        raise RuntimeError("Fusion payoff missing")
    if "MOTION_ANALYTICS_EXPERIMENTAL" not in html_text or "GROUND_TRUTH" not in html_text:
        raise RuntimeError("Evidence boundary labels missing")
    print(json.dumps({
        "html": str(OUT_HTML),
        "bytes": OUT_HTML.stat().st_size,
        "embedded_jpegs": html_text.count("data:image/jpeg;base64,"),
        "manifest": str(OUT_MANIFEST),
        "shared_gt_track": shared_tid,
        "sea_annotation_sha256": manifest["seadronessee"]["annotation_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
