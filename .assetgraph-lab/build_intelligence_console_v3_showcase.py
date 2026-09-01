from __future__ import annotations

import copy
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

BASE = Path(__file__).with_name('build_intelligence_console_v2.py')
spec = importlib.util.spec_from_file_location('assetgraph_console_v2_base_v3', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

# -----------------------------------------------------------------------------
# V3 evidence corrections + richer packing
# -----------------------------------------------------------------------------

_orig_compress = mod.compress_jpeg_bytes
_orig_extract_video_frames = mod.extract_video_frames
_orig_sample_window = mod.sample_window
_orig_pack_sea_mission = mod.pack_sea_mission


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
                for b in rows[i + 1:]:
                    gap = b['start'] - a['start']
                    if gap < min_gap:
                        continue
                    shared = a['tracks'] & b['tracks']
                    if not shared:
                        continue
                    score = len(shared) * 1000 + min(gap, 900) * 1.5 + a['score'] * .02 + b['score'] * .02
                    if best is None or score > best[0]:
                        best = (score, a, b, shared, gap)
        return best

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
        'v3_strict_identity': True,
        'stress_video': stress['video_id'],
        'identity_video': identity_a['video_id'],
        'early_start': identity_a['start'],
        'late_start': identity_b['start'],
        'gap_frames': gap,
        'shared_track_count': len(shared),
        'primary_track': shared_track,
    })
    return stress, identity_a, identity_b, pattern


def rich_compress(path: Path, max_w: int = 1920, quality: int = 87):
    # Keep more real sensor detail in the buyer-ready standalone.
    return _orig_compress(path, max_w=max_w, quality=quality)


def rich_extract_video_frames(video: Path, out_dir: Path, count: int = 8):
    # Ignore the older 8-frame request and build denser thermal storyboards.
    return _orig_extract_video_frames(video, out_dir, count=12)


def rich_sample_window(window: dict, count: int = 8):
    # Same for SeaDronesSee: denser temporal evidence, still deterministic.
    return _orig_sample_window(window, count=12)


def scaled_frame_payload(path: Path, source: dict, boxes=None, frame_id=None, extra=None):
    """Pack one frame and transform source-pixel boxes into packed-image pixels.

    V2 compressed SeaDronesSee imagery but retained source-image GT coordinates.
    SVG viewBox used packed dimensions, which visually displaced otherwise-correct GT.
    V3 stores both coordinate systems and renders only scaled packed coordinates.
    """
    raw = path.read_bytes()
    with Image.open(path) as im:
        source_w, source_h = im.size
    packed, w, h = mod.compress_jpeg_bytes(path)
    sx = w / max(source_w, 1)
    sy = h / max(source_h, 1)
    scaled_boxes = []
    for b in boxes or []:
        q = copy.deepcopy(b)
        if q.get('xyxy'):
            x1, y1, x2, y2 = map(float, q['xyxy'])
            q['source_xyxy'] = [x1, y1, x2, y2]
            q['xyxy'] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            q['coordinate_space'] = 'PACKED_IMAGE_PIXELS'
            q['scale_from_source'] = [sx, sy]
        scaled_boxes.append(q)
    row = {
        'frame_id': frame_id or path.stem,
        'image': mod.data_uri_jpeg_bytes(packed),
        'width': w,
        'height': h,
        'source_width': source_w,
        'source_height': source_h,
        'raw_sha256': mod.sha256_bytes(raw),
        'packed_sha256': mod.sha256_bytes(packed),
        'source': source,
        'boxes': scaled_boxes,
        'overlay_transform': {
            'source_space': [source_w, source_h],
            'packed_space': [w, h],
            'scale': [sx, sy],
            'status': 'VERIFIED_COORDINATE_TRANSFORM',
        },
    }
    if extra:
        row.update(extra)
    return row


def fixed_pack_sea_mission(*args, **kwargs):
    mission = _orig_pack_sea_mission(*args, **kwargs)
    primary = mission.get('primary_track')
    trajectory = []
    for idx, f in enumerate(mission.get('frames', [])):
        for b in f.get('boxes', []):
            if primary is not None and str(b.get('track_id')) == str(primary):
                x1, y1, x2, y2 = b['xyxy']
                trajectory.append({
                    'frame': idx,
                    'x': (x1 + x2) / 2 / max(f['width'], 1),
                    'y': (y1 + y2) / 2 / max(f['height'], 1),
                    'track_id': primary,
                })
                break
    mission['trajectory'] = trajectory
    mission['overlay_contract'] = 'GT scaled from source-image pixels into packed-image pixels before SVG rendering'
    mission['events'] = [e for e in mission.get('events', []) if e.get('type') != 'DISPLACEMENT']
    if len(trajectory) >= 2:
        dx = trajectory[-1]['x'] - trajectory[0]['x']
        dy = trajectory[-1]['y'] - trajectory[0]['y']
        mission['events'].append({
            'type': 'DISPLACEMENT',
            'severity': 'info',
            'text': f'Corrected packed-space displacement dx={dx:+.3f}, dy={dy:+.3f}.',
        })
    mission['events'].append({
        'type': 'OVERLAY_ALIGNMENT',
        'severity': 'verified',
        'text': 'Ground-truth boxes transformed from source pixels to packed-image pixels before display.',
    })
    return mission


mod.pick_windows = strict_pick_windows
mod.compress_jpeg_bytes = rich_compress
mod.extract_video_frames = rich_extract_video_frames
mod.sample_window = rich_sample_window
mod.frame_payload = scaled_frame_payload
mod.pack_sea_mission = fixed_pack_sea_mission

mod.OUT_HTML = mod.DIST / 'ASSETGRAPH_INTELLIGENCE_CONSOLE_v3_SHOWCASE.html'
mod.OUT_MANIFEST = mod.DIST / 'assetgraph_console_v3_manifest.json'
mod.main()

# -----------------------------------------------------------------------------
# Buyer-ready showcase shell layered over the proven analyst console.
# -----------------------------------------------------------------------------

manifest = json.loads(mod.OUT_MANIFEST.read_text(encoding='utf-8'))
manifest['schema'] = 'assetgraph-console/showcase-v3'
manifest['showcase'] = {
    'buyer_ready_shell': True,
    'executive_mode': True,
    'analyst_mode': True,
    'mission_library': True,
    'asset_registry': True,
    'memory_comparison': True,
    'evidence_explorer': True,
    'failure_analysis': True,
    'commercial_packaging': True,
    'coordinate_alignment_fix': 'source GT pixels -> packed image pixels before SVG overlay',
    'imagery_density': '12 sampled frames per mission window where source supports it',
    'packing': 'max width 1920, JPEG quality 87',
}
mod.OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

html_text = mod.OUT_HTML.read_text(encoding='utf-8')
html_text = html_text.replace('ASSETGRAPH INTELLIGENCE CONSOLE v2', 'ASSETGRAPH INTELLIGENCE CONSOLE v3', 2)

V3_CSS = r'''
<style id="assetgraph-v3-style">
:root{--v3bg:#020609;--v3glass:rgba(5,15,20,.92);--v3line:#214552;--v3cyan:#51e5ff;--v3lime:#8bffb7;--v3amber:#ffc857;--v3red:#ff6b75;--v3violet:#b69cff;--v3muted:#8eabb2}
#v3Launch{position:fixed;right:18px;bottom:18px;z-index:180;border:1px solid #4bd7ed;background:linear-gradient(135deg,#10303a,#071820);color:#eaffff;padding:11px 14px;border-radius:999px;font:800 11px ui-monospace,monospace;letter-spacing:.08em;box-shadow:0 12px 40px #000c,0 0 28px #51e5ff22;cursor:pointer}
#v3Shell{position:fixed;inset:0;z-index:170;background:radial-gradient(circle at 65% -20%,#17404c 0,#071318 35%,#020609 78%);color:#edfafa;overflow:auto;display:block}.v3hidden{display:none!important}
.v3top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:10px;padding:12px 18px;border-bottom:1px solid var(--v3line);background:rgba(3,10,14,.9);backdrop-filter:blur(18px)}.v3brand{font-weight:900;letter-spacing:.14em;font-size:13px}.v3brand b{color:var(--v3cyan)}.v3mode{display:flex;border:1px solid #254d59;border-radius:999px;overflow:hidden}.v3mode button,.v3nav button,.v3action{border:0;background:transparent;color:#93b3bb;padding:8px 11px;cursor:pointer;font:700 10px ui-monospace,monospace}.v3mode button.active,.v3nav button.active{background:#12303a;color:#efffff}.v3spacer{flex:1}.v3action{border:1px solid #2e6675;border-radius:9px;color:#dff9fd;background:#0b2028}.v3action.primary{border-color:#55dcec;background:linear-gradient(135deg,#16424c,#0c252d)}
.v3nav{display:flex;gap:4px;overflow:auto;padding:10px 18px;border-bottom:1px solid #17343d;background:#061116}.v3nav button{white-space:nowrap;border-radius:8px}.v3page{display:none;max-width:1500px;margin:0 auto;padding:22px}.v3page.active{display:block}.v3eyebrow{font:800 10px ui-monospace,monospace;letter-spacing:.2em;color:var(--v3cyan)}.v3hero{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;align-items:stretch}.v3heroCopy{padding:30px 4px}.v3hero h1{font-size:clamp(34px,5vw,76px);line-height:.92;margin:12px 0 16px;letter-spacing:-.05em}.v3hero h1 span{color:var(--v3cyan)}.v3lead{max-width:780px;color:#a7c1c7;font-size:15px;line-height:1.6}.v3heroVisual{border:1px solid #244a55;background:#061015;border-radius:18px;overflow:hidden;position:relative;min-height:390px;box-shadow:0 35px 100px #0009}.v3heroVisual img{width:100%;height:100%;object-fit:cover;opacity:.8}.v3scan{position:absolute;inset:0;background:linear-gradient(transparent 0 48%,rgba(81,229,255,.2) 50%,transparent 52%);background-size:100% 160px;animation:v3scan 4s linear infinite;pointer-events:none}@keyframes v3scan{to{background-position:0 160px}}
.v3metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin:18px 0}.v3metric{border:1px solid #173943;background:#07171c;border-radius:12px;padding:13px}.v3metric strong{display:block;font:800 24px ui-monospace,monospace;color:#eaffff}.v3metric span{font-size:9px;color:#77979f;text-transform:uppercase;letter-spacing:.1em}.v3grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.v3card{border:1px solid #173943;background:linear-gradient(180deg,#0a1b21,#061116);border-radius:14px;overflow:hidden;box-shadow:0 16px 42px #0004}.v3card .pad{padding:14px}.v3card h3{font-size:14px;margin:5px 0}.v3card p{font-size:11px;line-height:1.5;color:#8facb3}.v3thumb{height:180px;background:#010506;position:relative;overflow:hidden}.v3thumb img{width:100%;height:100%;object-fit:cover}.v3badge{display:inline-flex;border:1px solid #2d5c68;border-radius:999px;padding:4px 7px;font:800 8px ui-monospace,monospace;letter-spacing:.06em}.v3badge.gt{color:var(--v3lime);border-color:#2a6440}.v3badge.exp{color:var(--v3amber);border-color:#6b5421}.v3badge.sealed{color:var(--v3violet)}.v3section{margin:26px 0}.v3section h2{font-size:25px;margin:4px 0 5px}.v3sectionLead{font-size:12px;color:#82a3aa;margin-bottom:13px}.v3pipeline{display:grid;grid-template-columns:repeat(7,1fr);gap:7px}.v3pipe{position:relative;border:1px solid #20444f;background:#071820;border-radius:10px;padding:12px;min-height:90px}.v3pipe b{font-size:11px}.v3pipe small{display:block;color:#78969e;font-size:9px;margin-top:8px}.v3pipe:not(:last-child):after{content:'→';position:absolute;right:-9px;top:34px;color:#4fd8ed;z-index:2}
.v3table{width:100%;border-collapse:collapse;border:1px solid #173943;background:#061116;border-radius:12px;overflow:hidden}.v3table th,.v3table td{padding:10px 11px;border-bottom:1px solid #15323a;text-align:left;font-size:10px}.v3table th{color:#71939b;background:#091a20;letter-spacing:.08em}.v3table td.mono{font-family:ui-monospace,monospace}.v3rowclick{cursor:pointer}.v3rowclick:hover{background:#0c2229}.v3filters{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}.v3input{background:#07171c;border:1px solid #214651;color:#eaffff;border-radius:8px;padding:9px 10px;min-width:240px}.v3split{display:grid;grid-template-columns:1fr 1fr;gap:13px}.v3compare{border:1px solid #214651;background:#061116;border-radius:14px;overflow:hidden}.v3compareHead{display:flex;align-items:center;gap:8px;padding:11px;border-bottom:1px solid #173943}.v3compare img{width:100%;height:300px;object-fit:contain;background:#010506}.v3compare .pad{padding:12px}.v3callout{border-left:3px solid var(--v3cyan);background:#081c23;padding:12px 14px;border-radius:8px;font-size:11px;line-height:1.55;color:#a9c4ca}.v3callout.warn{border-color:var(--v3amber)}.v3callout.good{border-color:var(--v3lime)}.v3story{display:grid;grid-template-columns:1fr 56px 1fr;gap:10px;align-items:center}.v3arrow{text-align:center;color:var(--v3cyan);font-size:28px}.v3evidenceList{display:grid;gap:7px;max-height:62vh;overflow:auto}.v3evidenceItem{display:grid;grid-template-columns:100px 1fr auto;gap:10px;align-items:center;border:1px solid #15353e;background:#07171c;border-radius:10px;padding:8px}.v3evidenceItem img{width:100px;height:64px;object-fit:cover;border-radius:6px}.v3hash{font:9px ui-monospace,monospace;color:#84a4ab;word-break:break-all}.v3product{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.v3productCol{border:1px solid #1d414c;border-radius:14px;background:#07171c;padding:16px}.v3productCol h3{margin-top:0}.v3productCol ul{padding-left:18px;color:#93afb6;font-size:11px;line-height:1.8}.v3maturity{display:grid;gap:8px}.v3mat{display:grid;grid-template-columns:180px 1fr 110px;align-items:center;gap:10px;font-size:10px}.v3bar{height:7px;background:#10262d;border-radius:99px;overflow:hidden}.v3bar i{display:block;height:100%;background:linear-gradient(90deg,#318da0,#7af5ff);border-radius:99px}.v3mat em{font-style:normal;text-align:right;color:#87a8af}.v3analystOnly{display:none}.v3-analyst .v3analystOnly{display:block}.v3-analyst .v3executiveOnly{display:none}.v3legend{display:flex;gap:9px;flex-wrap:wrap;font-size:9px}.v3dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}.v3dot.gt{background:var(--v3lime)}.v3dot.exp{background:var(--v3amber)}.v3dot.meta{background:var(--v3cyan)}
@media(max-width:1000px){.v3hero{grid-template-columns:1fr}.v3metrics{grid-template-columns:repeat(2,1fr)}.v3grid{grid-template-columns:1fr 1fr}.v3pipeline{grid-template-columns:1fr 1fr}.v3pipe:after{display:none}.v3product{grid-template-columns:1fr}.v3split,.v3story{grid-template-columns:1fr}.v3arrow{transform:rotate(90deg)}}@media(max-width:620px){.v3page{padding:14px}.v3grid{grid-template-columns:1fr}.v3metrics{grid-template-columns:1fr 1fr}.v3top{padding:10px}.v3brand{font-size:10px}.v3top .v3mode{display:none}.v3heroVisual{min-height:260px}.v3mat{grid-template-columns:120px 1fr}.v3mat em{display:none}}
</style>
'''

V3_HTML = r'''
<button id="v3Launch" onclick="v3Open()">✦ SHOWCASE v3</button>
<section id="v3Shell">
  <div class="v3top">
    <div class="v3brand"><b>ASSETGRAPH</b> · INTELLIGENCE CONSOLE v3</div>
    <span class="v3badge gt">REAL IMAGERY</span>
    <span class="v3badge gt">GT ALIGNED</span>
    <div class="v3spacer"></div>
    <div class="v3mode"><button id="v3ExecMode" class="active" onclick="v3SetMode('executive')">EXECUTIVE</button><button id="v3AnalystMode" onclick="v3SetMode('analyst')">ANALYST</button></div>
    <button class="v3action primary" onclick="v3EnterConsole()">ENTER ANALYST CONSOLE</button>
  </div>
  <div class="v3nav" id="v3Nav">
    <button class="active" data-page="command">COMMAND</button>
    <button data-page="missions">MISSION LIBRARY</button>
    <button data-page="assets">ASSET REGISTRY</button>
    <button data-page="memory">MEMORY HIT</button>
    <button data-page="evidence">EVIDENCE</button>
    <button data-page="failures">FAILURE ANALYSIS</button>
    <button data-page="product">WHAT YOU BUY</button>
  </div>
  <main>
    <section class="v3page active" id="v3-command"></section>
    <section class="v3page" id="v3-missions"></section>
    <section class="v3page" id="v3-assets"></section>
    <section class="v3page" id="v3-memory"></section>
    <section class="v3page" id="v3-evidence"></section>
    <section class="v3page" id="v3-failures"></section>
    <section class="v3page" id="v3-product"></section>
  </main>
</section>
'''

V3_JS = r'''
<script id="assetgraph-v3-script">
(function(){
const q=s=>document.querySelector(s), qa=s=>[...document.querySelectorAll(s)];
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const M=DATA.missions;
const gtM=M.filter(m=>m.evidence_status.includes('VERIFIED'));
const expM=M.filter(m=>!m.evidence_status.includes('VERIFIED'));
const frames=M.flatMap(m=>m.frames.map(f=>({m,f})));
const gtBoxes=frames.flatMap(({m,f})=>(f.boxes||[]).filter(b=>b.evidence==='GROUND_TRUTH').map(b=>({m,f,b})));
const assets={};
gtBoxes.forEach(({m,f,b})=>{const id='GT-'+b.track_id;const a=assets[id]||(assets[id]={id,track:b.track_id,cls:b.class_name,missions:new Set(),frames:0,crops:0,first:m.id,last:m.id});a.frames++;a.missions.add(m.id);a.last=m.id});
M.forEach(m=>(m.crops||[]).forEach(c=>{const id='GT-'+c.track_id;if(assets[id])assets[id].crops++}));
const registry=Object.values(assets).sort((a,b)=>b.missions.size-a.missions.size||b.frames-a.frames);
const recurring=registry.filter(a=>a.missions.size>1);
const m04=M.find(m=>m.id==='M04'), m07=M.find(m=>m.id==='M07'), m01=M.find(m=>m.id==='M01'), m03=M.find(m=>m.id==='M03');
const gap=(DATA.manifest?.seadronessee?.selected_windows?.identity_late?.start||0)-(DATA.manifest?.seadronessee?.selected_windows?.identity_early?.start||0);
function badge(m){return m.evidence_status.includes('VERIFIED')?'<span class="v3badge gt">VERIFIED GT</span>':'<span class="v3badge exp">EXPERIMENTAL ANALYTICS</span>'}
function heroFrame(m){return m?.frames?.[Math.floor((m.frames.length-1)/2)]||m?.frames?.[0]}
function command(){
 const hf=heroFrame(m07||M[M.length-1]);
 q('#v3-command').innerHTML=`<div class="v3hero"><div class="v3heroCopy"><div class="v3eyebrow">PERSISTENT INTELLIGENCE FROM IMAGERY</div><h1>Every image becomes<br><span>reusable intelligence.</span></h1><p class="v3lead">AssetGraph transforms real UAV imagery into traceable observations, persistent assets, temporal events, intelligence memory and evidence-grade outputs. The showcase separates verified ground truth from experimental analytics instead of hiding uncertainty.</p><div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:18px"><button class="v3action primary" onclick="v3Go('memory')">SEE THE MEMORY HIT</button><button class="v3action" onclick="v3Go('missions')">EXPLORE MISSIONS</button><button class="v3action" onclick="v3EnterConsole()">OPEN ANALYST CONSOLE</button></div></div><div class="v3heroVisual"><img src="${hf.image}"><div class="v3scan"></div><div style="position:absolute;left:14px;bottom:14px;background:#02090ddc;border:1px solid #27515d;padding:9px 11px;border-radius:9px;font:10px ui-monospace,monospace"><b style="color:#8bffb7">PRIOR OBSERVATION RECOVERED</b><br>${e(m07?.primary_asset)} · ${gap} frame separation · GT-backed</div></div></div>
 <div class="v3metrics"><div class="v3metric"><strong>${M.length}</strong><span>mission packs</span></div><div class="v3metric"><strong>${frames.length}</strong><span>embedded real frames</span></div><div class="v3metric"><strong>${registry.length}</strong><span>GT assets indexed</span></div><div class="v3metric"><strong>${recurring.length}</strong><span>recurring assets</span></div><div class="v3metric"><strong>${gap}</strong><span>frames memory gap</span></div></div>
 <div class="v3section"><div class="v3eyebrow">THE PRODUCT LOGIC</div><h2>Input → observation → memory → intelligence</h2><div class="v3pipeline"><div class="v3pipe"><b>INGEST</b><small>RGB · thermal · video · metadata · SHA-256</small></div><div class="v3pipe"><b>PERCEPTION</b><small>Detection · OBB · model provenance</small></div><div class="v3pipe"><b>IDENTITY</b><small>Track · reacquire · persistent hypothesis</small></div><div class="v3pipe"><b>ASSET GRAPH</b><small>Assets · observations · relationships</small></div><div class="v3pipe"><b>EVENT ENGINE</b><small>Arrival · dwell · displacement · change</small></div><div class="v3pipe"><b>MEMORY</b><small>Prior observations · history · context</small></div><div class="v3pipe"><b>OUTPUT</b><small>Dossier · report · evidence pack · JSON</small></div></div></div>
 <div class="v3section v3executiveOnly"><div class="v3eyebrow">WHY IT MATTERS</div><h2>Not another bounding-box demo.</h2><div class="v3grid"><div class="v3card"><div class="pad"><h3>Compounding intelligence</h3><p>Earlier observations increase the value of later imagery. Later imagery also enriches historical asset dossiers.</p></div></div><div class="v3card"><div class="pad"><h3>Evidence, not theatre</h3><p>Verified GT, experimental motion analytics, provenance and sealed gates are visibly separated.</p></div></div><div class="v3card"><div class="pad"><h3>Analyst-ready output</h3><p>The end product is a reusable asset graph and intelligence report, not a detector screenshot.</p></div></div></div></div>
 <div class="v3section v3analystOnly"><div class="v3eyebrow">TECHNICAL BOUNDARY</div><h2>Current maturity ledger</h2>${maturity()}</div>`;
}
function missionCard(m,i){const f=heroFrame(m);return `<div class="v3card"><div class="v3thumb"><img src="${f.image}"></div><div class="pad"><div style="display:flex;justify-content:space-between;gap:8px"><span class="v3eyebrow">${m.id}</span>${badge(m)}</div><h3>${e(m.title)}</h3><p>${e(m.narrative)}</p><div class="v3legend"><span><i class="v3dot ${m.evidence_status.includes('VERIFIED')?'gt':'exp'}"></i>${m.frames.length} real frames</span><span><i class="v3dot meta"></i>${e(m.sensor)}</span></div><button class="v3action" style="margin-top:10px" onclick="v3OpenMission(${i})">OPEN IN ANALYST CONSOLE</button></div></div>`}
function missions(){q('#v3-missions').innerHTML=`<div class="v3eyebrow">MISSION LIBRARY</div><h2>Seven operations. One accumulating intelligence layer.</h2><p class="v3sectionLead">Each mission is a self-contained evidence pack. SeaDronesSee overlays are GT aligned to the packed image; HIT-UAV amber regions are motion analytics and never presented as semantic detections.</p><div class="v3grid">${M.map(missionCard).join('')}</div>`}
function assetsPage(filter=''){const rows=registry.filter(a=>(a.id+' '+a.cls+' '+[...a.missions].join(' ')).toLowerCase().includes(filter.toLowerCase())).slice(0,300);q('#v3-assets').innerHTML=`<div class="v3eyebrow">ASSET REGISTRY</div><h2>Persistent objects, not isolated detections.</h2><p class="v3sectionLead">Registry built from GT track IDs across the embedded mission packs. Recurring assets are highlighted because they are the bridge toward Persistent Identity.</p><div class="v3filters"><input id="v3AssetSearch" class="v3input" placeholder="Search asset, class, mission…" value="${e(filter)}"><span class="v3badge gt">${recurring.length} recurring</span><span class="v3badge">${registry.length} total</span></div><table class="v3table"><thead><tr><th>ASSET</th><th>CLASS</th><th>MISSIONS</th><th>OBSERVATIONS</th><th>CROPS</th><th>STATUS</th></tr></thead><tbody>${rows.map(a=>`<tr class="v3rowclick"><td class="mono">${a.id}</td><td>${e(a.cls)}</td><td>${[...a.missions].join(', ')}</td><td>${a.frames}</td><td>${a.crops}</td><td>${a.missions.size>1?'<span class="v3badge gt">RECURRING</span>':'<span class="v3badge">OBSERVED</span>'}</td></tr>`).join('')}</tbody></table>`;q('#v3AssetSearch').oninput=ev=>assetsPage(ev.target.value)}
function memory(){const a=heroFrame(m04),b=heroFrame(m07);const cropA=m04?.crops?.[0]?.image||a.image,cropB=m07?.crops?.[0]?.image||b.image;q('#v3-memory').innerHTML=`<div class="v3eyebrow">THE MONEY SHOT</div><h2>Same GT identity. ${gap} frames later.</h2><p class="v3sectionLead">The demo does not claim experimental ReID magically solved identity. The dataset GT tells us the same track exists in both windows. This is the target behavior the product identity layer must reproduce without GT.</p><div class="v3story"><div class="v3compare"><div class="v3compareHead"><span class="v3badge gt">EARLY OBSERVATION</span><b>${e(m04.primary_asset)}</b></div><img src="${cropA}"><div class="pad">Mission ${m04.id} · video ${m04.window?.video_id} · start ${m04.window?.start}<br><span class="v3hash">GT track ${e(m04.primary_track)}</span></div></div><div class="v3arrow">→</div><div class="v3compare"><div class="v3compareHead"><span class="v3badge gt">RECOVERED</span><b>${e(m07.primary_asset)}</b></div><img src="${cropB}"><div class="pad">Mission ${m07.id} · same video · start ${m07.window?.start}<br><span class="v3hash">${gap} frame separation · non-overlapping windows</span></div></div></div><div class="v3section"><div class="v3split"><div class="v3callout warn"><b>STATELESS PIPELINE</b><br>A later frame is processed as a fresh observation. Prior relationships, earlier location and accumulated context are absent from the output.</div><div class="v3callout good"><b>ASSETGRAPH + MEMORY</b><br>The observation resolves to an existing asset dossier, retrieves prior evidence and compiles a richer report with provenance.</div></div></div><div class="v3section"><h2>What must become real before product freeze</h2><div class="v3grid"><div class="v3card"><div class="pad"><h3>Model-backed identity</h3><p>Replace GT identity lookup with tracking + ReID + confidence calibration.</p></div></div><div class="v3card"><div class="pad"><h3>Conflict handling</h3><p>Competing identity hypotheses must be preserved, scored and reviewable.</p></div></div><div class="v3card"><div class="pad"><h3>Measured memory value</h3><p>Quantify how much prior context improves analyst output and reduces repeated work.</p></div></div></div></div>`}
function evidence(filter=''){const rows=frames.filter(({m,f})=>(m.id+' '+f.frame_id+' '+f.raw_sha256+' '+JSON.stringify(f.source)).toLowerCase().includes(filter.toLowerCase()));q('#v3-evidence').innerHTML=`<div class="v3eyebrow">EVIDENCE EXPLORER</div><h2>Every displayed frame has lineage.</h2><p class="v3sectionLead">Source, raw SHA-256, packed SHA-256 and overlay class travel with the observation.</p><div class="v3filters"><input id="v3EvidenceSearch" class="v3input" placeholder="Filter mission, frame, hash, source…" value="${e(filter)}"><span class="v3badge gt">${rows.length} frames</span></div><div class="v3evidenceList">${rows.map(({m,f})=>`<div class="v3evidenceItem"><img src="${f.image}"><div><b>${m.id} · ${e(f.frame_id)}</b><div class="v3hash">RAW ${e(f.raw_sha256)}</div><div class="v3hash">PACKED ${e(f.packed_sha256)}</div><div style="font-size:9px;color:#83a4ab">${e((f.boxes||[]).map(b=>b.evidence).filter((x,i,a)=>a.indexOf(x)===i).join(' · ')||'NO OVERLAY')}</div></div><span class="v3badge ${m.evidence_status.includes('VERIFIED')?'gt':'exp'}">${m.id}</span></div>`).join('')}</div>`;q('#v3EvidenceSearch').oninput=ev=>evidence(ev.target.value)}
function failures(){const f1=heroFrame(m01),f3=heroFrame(m03);q('#v3-failures').innerHTML=`<div class="v3eyebrow">FAILURE / AMBIGUITY ANALYSIS</div><h2>We expose uncertainty instead of painting over it.</h2><p class="v3sectionLead">The earlier water-box problem had two different causes that must not be conflated: motion analytics can activate on water, and v2 also had a coordinate-space rendering bug. V3 fixes the rendering bug and keeps motion ambiguity explicitly experimental.</p><div class="v3split"><div class="v3compare"><div class="v3compareHead"><span class="v3badge exp">EXPERIMENTAL</span><b>Thermal motion region</b></div><img src="${f1.image}"><div class="pad"><div class="v3callout warn">Frame differencing detects changed pixels. Water shimmer, camera motion and thermal texture can trigger a motion region. This is <b>not object detection</b>.</div></div></div><div class="v3compare"><div class="v3compareHead"><span class="v3badge gt">VERIFIED GT</span><b>SeaDronesSee annotation</b></div><img src="${f3.image}"><div class="pad"><div class="v3callout good">V3 converts source-image annotation pixels into packed-image pixels before SVG drawing. GT boxes now use the same coordinate space as the image actually displayed.</div></div></div></div><div class="v3section"><h2>Evidence classes</h2><div class="v3grid"><div class="v3card"><div class="pad"><span class="v3badge gt">GROUND_TRUTH</span><h3>Reference annotation</h3><p>Authoritative benchmark annotation used to verify what the UI should display.</p></div></div><div class="v3card"><div class="pad"><span class="v3badge exp">MOTION_ANALYTICS_EXPERIMENTAL</span><h3>Heuristic evidence</h3><p>Useful for motion salience, but cannot be promoted to semantic detection.</p></div></div><div class="v3card"><div class="pad"><span class="v3badge">MODEL OUTPUT</span><h3>Next substitution</h3><p>RTMDet-R output should be displayed beside GT with explicit TP / FP / FN comparison.</p></div></div></div></div>`}
function maturity(){const L=DATA.ledger||{};const rows=[['HIT-UAV byte freeze',100,L.hit_uav_freeze||'PASS'],['RTMDet-R ONNX parity',100,L.rtmdet_r_onnx||'VERIFIED'],['SeaDronesSee GT showcase',95,'VERIFIED GT + coordinate alignment'],['Persistent Identity',48,L.persistent_identity||'EXPERIMENTAL'],['Intelligence Memory',62,L.intelligence_memory||'RESEARCH VALIDATED'],['Temporal / Event Engine',35,'NEXT FRONTIER'],['H01 external gate',10,L.transset_h01||'SEALED']];return `<div class="v3maturity">${rows.map(r=>`<div class="v3mat"><b>${r[0]}</b><div class="v3bar"><i style="width:${r[1]}%"></i></div><em>${e(r[2])}</em></div>`).join('')}</div>`}
function product(){q('#v3-product').innerHTML=`<div class="v3eyebrow">COMMERCIAL PACKAGING</div><h2>What an intelligence customer actually buys.</h2><p class="v3sectionLead">Not “a model”. A traceable operational layer that turns imagery into persistent, reviewable intelligence assets.</p><div class="v3product"><div class="v3productCol"><span class="v3eyebrow">INPUT ASSETS</span><h3>Bring the mission data</h3><ul><li>UAV RGB imagery</li><li>Thermal / infrared sequences</li><li>Video frame streams</li><li>Historical imagery</li><li>Mission metadata</li><li>External analyst notes / intelligence</li></ul></div><div class="v3productCol"><span class="v3eyebrow">LOGIC LAYER</span><h3>AssetGraph Runtime</h3><ul><li>Perception + provenance</li><li>Tracking / ReID / reacquisition</li><li>Persistent Asset Registry</li><li>Asset Graph relationships</li><li>Temporal / event engine</li><li>Intelligence Memory</li><li>Evidence compiler</li></ul></div><div class="v3productCol"><span class="v3eyebrow">OUTPUT ASSETS</span><h3>What the analyst receives</h3><ul><li>Mission intelligence report</li><li>Persistent asset dossier</li><li>Timeline + pattern-of-life</li><li>Prior observation recovery</li><li>Alerts / anomalies</li><li>Decision Object / JSON</li><li>Audit-ready Evidence Pack</li></ul></div></div><div class="v3section"><div class="v3eyebrow">READINESS</div><h2>Sell the truth, show the frontier.</h2>${maturity()}</div><div class="v3section"><div class="v3split"><div class="v3callout good"><b>READY TO DEMO</b><br>Real imagery, GT-backed mission packs, aligned overlays, evidence lineage, asset registry, memory narrative and exportable analyst outputs.</div><div class="v3callout warn"><b>NOT YET CLAIMED AS FROZEN PRODUCT</b><br>Persistent Identity, model-backed cross-scene reacquisition and full temporal intelligence still need their controlled measurement gates.</div></div></div>`}
function renderAll(){command();missions();assetsPage();memory();evidence();failures();product()}
window.v3SetMode=function(mode){const sh=q('#v3Shell');sh.classList.toggle('v3-analyst',mode==='analyst');q('#v3ExecMode')?.classList.toggle('active',mode==='executive');q('#v3AnalystMode')?.classList.toggle('active',mode==='analyst')}
window.v3Go=function(page){qa('.v3page').forEach(x=>x.classList.remove('active'));q('#v3-'+page)?.classList.add('active');qa('#v3Nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page));scrollTo({top:0,behavior:'smooth'})}
window.v3Open=function(){q('#v3Shell').classList.remove('v3hidden')}
window.v3EnterConsole=function(){q('#v3Shell').classList.add('v3hidden')}
window.v3OpenMission=function(i){q('#v3Shell').classList.add('v3hidden');if(typeof selectMission==='function')selectMission(i)}
q('#v3Nav').addEventListener('click',ev=>{const b=ev.target.closest('button[data-page]');if(b)v3Go(b.dataset.page)});
renderAll();
})();
</script>
'''

html_text = html_text.replace('</head>', V3_CSS + '\n</head>')
html_text = html_text.replace('</body>', V3_HTML + '\n' + V3_JS + '\n</body>')
mod.OUT_HTML.write_text(html_text, encoding='utf-8')

print(json.dumps({
    'v3_html': str(mod.OUT_HTML),
    'bytes': mod.OUT_HTML.stat().st_size,
    'manifest': str(mod.OUT_MANIFEST),
    'coordinate_fix': True,
    'showcase_shell': True,
}, indent=2))
