from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import requests

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'
DIST.mkdir(parents=True, exist_ok=True)
OUT = DIST / 'c20a_multivideo_lockbox_plan.json'
ANN_URL = 'https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/annotations/instances_val_objects_in_water.json?download=true'
WINDOW = 120
STRIDE = 30
MIN_GAP = 120
SAMPLE_COUNT = 12

META_ALIASES = {
    'gps_latitude': ['gps_latitude', 'latitude', 'lat'],
    'gps_longitude': ['gps_longitude', 'longitude', 'lon', 'lng'],
    'altitude': ['altitude', 'altitude_m'],
    'gimbal_pitch': ['gimbal_pitch', 'gimbal_pitch_degree', 'pitch'],
    'gimbal_heading': ['gimbal_heading', 'gimbal_yaw', 'yaw'],
    'compass_heading': ['compass_heading', 'heading'],
    'speed': ['speed', 'ground_speed'],
    'xspeed': ['xspeed', 'x_speed'],
    'yspeed': ['yspeed', 'y_speed'],
    'zspeed': ['zspeed', 'z_speed'],
    'timestamp': ['timestamp', 'time', 'date_time'],
}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def choose_value(im: dict, aliases: list[str]):
    # SeaDronesSee metadata may be flat or nested under metadata/meta.
    sources = [im]
    for k in ('metadata', 'meta', 'drone_metadata'):
        if isinstance(im.get(k), dict):
            sources.append(im[k])
    for src in sources:
        for k in aliases:
            if k in src and src[k] not in (None, ''):
                return src[k]
    return None


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def sample_rows(rows: list[dict], n: int = SAMPLE_COUNT) -> list[dict]:
    if len(rows) <= n:
        return rows
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [rows[i] for i in idx]


def metadata_summary(rows: list[dict]) -> dict:
    out = {}
    for canonical, aliases in META_ALIASES.items():
        vals = [choose_value(x, aliases) for x in rows]
        vals = [x for x in vals if x not in (None, '')]
        if not vals:
            continue
        numeric = [to_float(x) for x in vals]
        numeric = [x for x in numeric if x is not None and math.isfinite(x)]
        if len(numeric) >= max(2, len(vals) // 2):
            out[canonical] = {
                'median': median(numeric),
                'min': min(numeric),
                'max': max(numeric),
                'count': len(numeric),
            }
        else:
            out[canonical] = {'sample': str(vals[0]), 'count': len(vals)}
    return out


def meta_delta(a: dict, b: dict) -> dict:
    out = {}
    for k in sorted(set(a) & set(b)):
        if 'median' in a[k] and 'median' in b[k]:
            out[k] = b[k]['median'] - a[k]['median']
    return out


def deterministic_partition(video_id: int) -> str:
    # Frozen before any model scores: ~65% DEV, ~35% LOCKBOX.
    x = int(hashlib.sha256(f'SeaDronesSee-video-{video_id}'.encode()).hexdigest()[:8], 16) % 100
    return 'DEV' if x < 65 else 'LOCKBOX'


def main():
    r = requests.get(ANN_URL, timeout=180)
    r.raise_for_status()
    raw = r.content
    data = r.json()
    cats = {int(c['id']): str(c['name']) for c in data.get('categories', [])}
    ignored = {k for k, v in cats.items() if 'ignore' in v.lower()}
    anns = defaultdict(list)
    for a in data.get('annotations', []):
        if int(a.get('category_id', -999)) in ignored:
            continue
        anns[int(a['image_id'])].append({
            'track_id': int(a['track_id']),
            'category_id': int(a.get('category_id', -1)),
            'class_name': cats.get(int(a.get('category_id', -1)), str(a.get('category_id'))),
        })

    byvid = defaultdict(list)
    all_keys = set()
    for im in data.get('images', []):
        byvid[int(im['video_id'])].append(im)
        all_keys.update(im.keys())
        for k in ('metadata', 'meta', 'drone_metadata'):
            if isinstance(im.get(k), dict):
                all_keys.update(f'{k}.{x}' for x in im[k])

    pairs = []
    for vid, ims in sorted(byvid.items()):
        ims = sorted(ims, key=lambda x: x.get('frame_index', x['id']))
        if len(ims) < WINDOW * 2:
            continue
        windows = []
        for st in range(0, len(ims) - WINDOW + 1, STRIDE):
            rows = ims[st:st + WINDOW]
            tracks = {a['track_id'] for im in rows for a in anns.get(int(im['id']), [])}
            if not tracks:
                continue
            windows.append({'start': st, 'rows': rows, 'tracks': tracks})
        best = None
        for i, a in enumerate(windows):
            for b in windows[i + 1:]:
                gap = b['start'] - a['start']
                if gap < MIN_GAP:
                    continue
                shared = a['tracks'] & b['tracks']
                if not shared:
                    continue
                absent = a['tracks'] - b['tracks']
                new = b['tracks'] - a['tracks']
                # Model-independent selection: reward many shared identities, open-set negatives/new arrivals and long gaps.
                score = len(shared) * 1000 + min(gap, 1200) * 2 + min(len(absent), 10) * 100 + min(len(new), 10) * 100
                cand = (score, gap, len(shared), len(absent) + len(new), a, b, shared, absent, new)
                if best is None or cand[:4] > best[:4]:
                    best = cand
        if best is None:
            continue
        _, gap, _, _, a, b, shared, absent, new = best
        ma = metadata_summary(a['rows'])
        mb = metadata_summary(b['rows'])
        pairs.append({
            'video_id': vid,
            'partition': deterministic_partition(vid),
            'early_start': a['start'],
            'late_start': b['start'],
            'window_size': WINDOW,
            'gap_frames': gap,
            'shared_track_ids': sorted(shared),
            'shared_track_count': len(shared),
            'absent_track_ids': sorted(absent),
            'absent_track_count': len(absent),
            'new_track_ids': sorted(new),
            'new_track_count': len(new),
            'early_sample_image_ids': [int(x['id']) for x in sample_rows(a['rows'])],
            'late_sample_image_ids': [int(x['id']) for x in sample_rows(b['rows'])],
            'early_metadata': ma,
            'late_metadata': mb,
            'metadata_delta': meta_delta(ma, mb),
        })

    # Ensure both partitions have enough independent videos without looking at model performance.
    dev = [p for p in pairs if p['partition'] == 'DEV']
    lock = [p for p in pairs if p['partition'] == 'LOCKBOX']
    if len(dev) < 3 or len(lock) < 2:
        # Deterministic fallback by eligible video order: every third video is LOCKBOX.
        for i, p in enumerate(sorted(pairs, key=lambda x: x['video_id'])):
            p['partition'] = 'LOCKBOX' if i % 3 == 2 else 'DEV'
        dev = [p for p in pairs if p['partition'] == 'DEV']
        lock = [p for p in pairs if p['partition'] == 'LOCKBOX']

    meta_fields_detected = sorted({k for p in pairs for side in ('early_metadata', 'late_metadata') for k in p[side]})
    plan = {
        'schema': 'assetgraph-c20a-multivideo-lockbox-plan/v1',
        'status': 'PLAN_FROZEN_BEFORE_MODEL_EVALUATION',
        'source': {
            'authority': 'SeaDronesSee / University of Tuebingen',
            'transport_url': ANN_URL,
            'annotation_sha256': sha_bytes(raw),
            'images': len(data.get('images', [])),
            'annotations': len(data.get('annotations', [])),
            'videos_seen': len(byvid),
            'image_keys_detected': sorted(all_keys),
            'metadata_fields_detected': meta_fields_detected,
        },
        'protocol': {
            'window_frames': WINDOW,
            'stride_frames': STRIDE,
            'minimum_start_gap_frames': MIN_GAP,
            'sample_images_per_window': SAMPLE_COUNT,
            'partition_rule': 'SHA256(video_id) deterministic split before model scores; fallback every third eligible video LOCKBOX only if cardinality insufficient',
            'dev_usage': 'engine selection, fusion weights, thresholds, metadata ablation',
            'lockbox_usage': 'one-shot evaluation only after policy artifact hash is frozen',
            'ground_truth_boundary': 'GT track IDs used only to construct/evaluate benchmark tasks. Runtime candidate ranking may not ingest GT.',
            'open_set_required': True,
        },
        'eligible_video_pairs': len(pairs),
        'dev_video_count': len(dev),
        'lockbox_video_count': len(lock),
        'dev_video_ids': sorted(p['video_id'] for p in dev),
        'lockbox_video_ids': sorted(p['video_id'] for p in lock),
        'pairs': sorted(pairs, key=lambda x: x['video_id']),
        'next_gate': {
            'name': 'C20A_MULTI_VIDEO_IDENTITY_LOCKBOX',
            'required_metrics': ['Recall@1', 'Recall@5', 'MRR', 'confirmed_precision', 'safe_auto_coverage', 'false_merge_rate_absent', 'open_set_safe_rate', 'performance_by_gap', 'performance_by_metadata_delta'],
            'ablation': ['VISUAL_ONLY', 'VISUAL_PLUS_GEOMETRY_CONTEXT', 'VISUAL_PLUS_SENSOR_METADATA'],
            'product_freeze_allowed': False,
        },
    }
    frozen = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    plan['plan_sha256'] = sha_bytes(frozen)
    OUT.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({
        'out': str(OUT),
        'plan_sha256': plan['plan_sha256'],
        'eligible_pairs': len(pairs),
        'dev_videos': plan['dev_video_ids'],
        'lockbox_videos': plan['lockbox_video_ids'],
        'metadata_fields': meta_fields_detected,
        'shared_identity_tasks_dev': sum(p['shared_track_count'] for p in dev),
        'shared_identity_tasks_lockbox': sum(p['shared_track_count'] for p in lock),
        'open_set_absent_tasks_lockbox': sum(p['absent_track_count'] for p in lock),
        'new_identity_tasks_lockbox': sum(p['new_track_count'] for p in lock),
    }, indent=2))


if __name__ == '__main__':
    main()
