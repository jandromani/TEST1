from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).with_name('build_intelligence_console_v2.py')
spec = importlib.util.spec_from_file_location('assetgraph_console_v2_base', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def strict_pick_windows(candidates: list[dict]):
    stress = max(candidates, key=lambda x: (x['score'], len(x['tracks'])))
    same = defaultdict(list)
    for c in candidates:
        same[c['video_id']].append(c)

    def choose(*, exclude_stress_video: bool, min_gap: int):
        best = None
        for vid, rows in same.items():
            if exclude_stress_video and vid == stress['video_id']:
                continue
            rows = sorted(rows, key=lambda x: x['start'])
            for i, a in enumerate(rows):
                for b in rows[i+1:]:
                    gap = b['start'] - a['start']
                    if gap < min_gap:
                        continue
                    shared = a['tracks'] & b['tracks']
                    if not shared:
                        continue
                    # Prefer a different acquisition/video from the stress demo, then
                    # true non-overlap, larger temporal separation and persistent shared IDs.
                    score = (
                        len(shared) * 1000
                        + min(gap, 900) * 1.5
                        + a['score'] * 0.02
                        + b['score'] * 0.02
                    )
                    if best is None or score > best[0]:
                        best = (score, a, b, shared, gap)
        return best

    # Frontier preference order:
    # A) different video from stress + no overlap (window=120)
    # B) any video + no overlap
    # C) different video + substantial separation (90)
    # D) original safe fallback (>=30)
    best = (
        choose(exclude_stress_video=True, min_gap=120)
        or choose(exclude_stress_video=False, min_gap=120)
        or choose(exclude_stress_video=True, min_gap=90)
        or choose(exclude_stress_video=False, min_gap=30)
    )
    if best is None:
        raise RuntimeError('Could not find a GT-backed cross-window identity pair')

    _, identity_a, identity_b, shared, gap = best
    shared_track = sorted(shared)[0]
    identity_a = {**identity_a, 'primary_track': shared_track, 'identity_gap_frames': gap}
    identity_b = {**identity_b, 'primary_track': shared_track, 'identity_gap_frames': gap}

    pattern_pool = [c for c in candidates if c['video_id'] not in {stress['video_id'], identity_a['video_id']}]
    pattern = max(pattern_pool or candidates, key=lambda x: (len(x['tracks']), x['score']))

    print({
        'strict_identity_selection': True,
        'stress_video': stress['video_id'],
        'identity_video': identity_a['video_id'],
        'identity_early_start': identity_a['start'],
        'identity_late_start': identity_b['start'],
        'identity_gap_frames': gap,
        'non_overlapping': gap >= 120,
        'shared_track_count': len(shared),
        'primary_track': shared_track,
    })
    return stress, identity_a, identity_b, pattern


mod.pick_windows = strict_pick_windows
mod.main()
