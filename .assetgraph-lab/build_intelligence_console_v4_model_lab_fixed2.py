from __future__ import annotations
import hashlib, importlib.util, json, re
from pathlib import Path

BASE = Path(__file__).with_name('build_intelligence_console_v4_model_lab.py')
spec = importlib.util.spec_from_file_location('assetgraph_v4_base2', BASE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

mod.mod.sha256_file = sha256_file


def safe_patch_html(s: str, data: dict, manifest: dict) -> str:
    # Replace only the deterministic DATA payload. Do not regex-rewrite working V3 functions.
    dm = re.search(r'const DATA=(.*?);\nlet missionIndex=', s, re.S)
    if not dm:
        raise RuntimeError('DATA payload not found while patching v4')
    packed = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    s = s[:dm.start(1)] + packed + s[dm.end(1):]
    s = s.replace('ASSETGRAPH INTELLIGENCE CONSOLE v3', 'ASSETGRAPH INTELLIGENCE CONSOLE v4', 2)
    s = s.replace('INTELLIGENCE CONSOLE v3', 'INTELLIGENCE CONSOLE v4', 2)
    s = s.replace('✦ SHOWCASE v3', '✦ SHOWCASE v4', 1)

    mem = data.get('model_memory_retrieval', {})
    sea = manifest['v4']['sea_model']
    rtm = manifest['v4']['rtmdet_domain_stress']
    sea_tp = sum(x['tp'] for x in sea['missions'].values())
    sea_fp = sum(x['fp'] for x in sea['missions'].values())
    sea_fn = sum(x['fn'] for x in sea['missions'].values())
    retrieval_status = 'PASS' if mem.get('pass') else 'FAIL'

    model_page = f'''
    <section class="v3page" id="v3-model">
      <div class="v3eyebrow">MODEL LAB · REAL INFERENCE · GT AS JUDGE</div>
      <div class="v3section">
        <h2>RAW → GT → MODEL → ERROR → TRACK → IDENTITY → MEMORY</h2>
        <div class="v3sectionLead">Ground Truth evaluates model output. It does not rank the M07 retrieval candidates.</div>
      </div>
      <div class="v3metrics">
        <div class="v3metric"><strong>{sea_tp}</strong><span>MODEL TP</span></div>
        <div class="v3metric"><strong>{sea_fp}</strong><span>MODEL FP</span></div>
        <div class="v3metric"><strong>{sea_fn}</strong><span>MODEL FN</span></div>
        <div class="v3metric"><strong>{float(mem.get('top1_similarity', 0) or 0):.3f}</strong><span>MEMORY SIMILARITY</span></div>
        <div class="v3metric"><strong>{retrieval_status}</strong><span>GT-386 RETRIEVAL</span></div>
      </div>
      <div class="v3split">
        <div class="v3card"><div class="pad">
          <span class="v3badge gt">DOMAIN-MATCHED BENCHMARK</span>
          <h3>SeaDronesSee YOLOv8n + ByteTrack</h3>
          <p>Real inference on the exact embedded maritime frames. AGPL-3.0 benchmark-only: evidence for capability development, not a product distribution candidate.</p>
          <pre class="mono" style="white-space:pre-wrap;max-height:420px;overflow:auto">{json.dumps(sea['missions'], indent=2)}</pre>
        </div></div>
        <div class="v3card"><div class="pad">
          <span class="v3badge sealed">OUT-OF-DOMAIN CONTROL</span>
          <h3>RTMDet-R tiny · DOTA taxonomy</h3>
          <p>Exact Apache MMRotate checkpoint run against thermal and maritime scenes. It is intentionally out of domain here: the console exposes what a technically valid model does when its taxonomy/sensor assumptions are wrong.</p>
          <pre class="mono" style="white-space:pre-wrap;max-height:420px;overflow:auto">{json.dumps(rtm['frames'], indent=2)}</pre>
        </div></div>
      </div>
      <div class="v3section">
        <h2>Model-only memory retrieval</h2>
        <div class="v3callout {'good' if mem.get('pass') else 'warn'}">
          M04 enrollment is benchmark-aligned to GT-386 so we know which model crop defines the query. M07 ranking itself uses appearance similarity only. GT is consulted only after ranking to score the result.
        </div>
        <div class="v3card" style="margin-top:12px"><div class="pad"><pre class="mono" style="white-space:pre-wrap">{json.dumps(mem, indent=2)}</pre></div></div>
      </div>
    </section>
    '''

    nav_marker = '<button data-page="failures">FAILURE ANALYSIS</button>'
    if nav_marker in s:
        s = s.replace(nav_marker, '<button data-page="model">MODEL LAB</button>' + nav_marker, 1)
    else:
        raise RuntimeError('V3 nav marker not found')
    page_marker = '<section class="v3page" id="v3-failures"></section>'
    if page_marker in s:
        s = s.replace(page_marker, model_page + page_marker, 1)
    else:
        raise RuntimeError('V3 failures page marker not found')

    css = r'''
<style id="assetgraph-v4-style">
.box.model{stroke:#51e5ff!important;stroke-width:2.2!important}.box.rtmdet{stroke:#b69cff!important;stroke-width:2!important;stroke-dasharray:4 3}.modelText{fill:#51e5ff!important}.rtmdetText{fill:#b69cff!important}
#v4OverlayControls{position:fixed;left:50%;transform:translateX(-50%);top:62px;z-index:90;display:flex;gap:5px;background:rgba(4,16,20,.94);border:1px solid #28505b;border-radius:999px;padding:5px;backdrop-filter:blur(12px);box-shadow:0 10px 30px #0008}#v4OverlayControls .active{border-color:#51e5ff;color:#eaffff;box-shadow:0 0 18px #51e5ff22}
@media(max-width:720px){#v4OverlayControls{top:58px;max-width:94vw;overflow:auto}.v4ov{padding:6px 8px!important;font-size:9px!important}}
</style>
'''
    s = s.replace('</head>', css + '</head>', 1)

    controls = '''
<div id="v4OverlayControls">
  <button class="btn v4ov active" data-mode="ALL" onclick="setOverlayModeV4('ALL')">ALL</button>
  <button class="btn v4ov" data-mode="GT" onclick="setOverlayModeV4('GT')">GT</button>
  <button class="btn v4ov" data-mode="MODEL" onclick="setOverlayModeV4('MODEL')">MODEL</button>
  <button class="btn v4ov" data-mode="RTMDET" onclick="setOverlayModeV4('RTMDET')">RTMDet-R</button>
</div>
'''
    if '<div class="app">' in s:
        s = s.replace('<div class="app">', controls + '<div class="app">', 1)
    else:
        s = controls + s

    # Runtime override: preserve all V3 code and replace only rendering behavior after it loads.
    runtime = r'''
<script id="assetgraph-v4-runtime">
(function(){
  let mode='ALL';
  window.setOverlayModeV4=function(m){
    mode=m;
    document.querySelectorAll('.v4ov').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
    if(typeof renderFrame==='function') renderFrame();
  };
  const svgNS='http://www.w3.org/2000/svg';
  function v4Label(b){
    const ev=String(b.evidence||'');
    if(ev.includes('MODEL_PREDICTION')) return `${b.class_name||'model'} · T${b.track_id??'—'} · ${Math.round((b.confidence||0)*100)}% · ${b.eval||''}`;
    if(ev.includes('RTMDET_R')) return `${b.class_name||'RTMDet-R'} · ${Math.round((b.confidence||0)*100)}%`;
    return `${b.class_name||'GT'} · ${b.track_id??''}`;
  }
  function v4RenderBox(ov,b){
    if(!b||!b.xyxy) return;
    const [x1,y1,x2,y2]=b.xyxy;
    const ev=String(b.evidence||'');
    const motion=ev.includes('MOTION'), model=ev.includes('MODEL_PREDICTION'), rtm=ev.includes('RTMDET_R');
    const cls=motion?'motion':model?'model':rtm?'rtmdet':'';
    const tcls=motion?'motionText':model?'modelText':rtm?'rtmdetText':'';
    const g=document.createElementNS(svgNS,'g');
    const rect=document.createElementNS(svgNS,'rect');rect.setAttribute('class','box '+cls);rect.setAttribute('x',x1);rect.setAttribute('y',y1);rect.setAttribute('width',Math.max(1,x2-x1));rect.setAttribute('height',Math.max(1,y2-y1));g.appendChild(rect);
    const bg=document.createElementNS(svgNS,'rect');bg.setAttribute('class','boxLabel');bg.setAttribute('x',x1);bg.setAttribute('y',Math.max(0,y1-18));bg.setAttribute('width',Math.max(135,x2-x1));bg.setAttribute('height',18);g.appendChild(bg);
    const tx=document.createElementNS(svgNS,'text');tx.setAttribute('class','labelText '+tcls);tx.setAttribute('x',x1+4);tx.setAttribute('y',Math.max(12,y1-5));tx.textContent=v4Label(b);g.appendChild(tx);ov.appendChild(g);
  }
  if(typeof renderFrame==='function'){
    renderFrame=function(){
      const m=DATA.missions[missionIndex],f=m.frames[frameIndex]||m.frames[0];
      const main=document.querySelector('#mainImage');if(main)main.src=f.image;
      const fid=document.querySelector('#frameId');if(fid)fid.textContent=f.frame_id;
      const fh=document.querySelector('#frameHash');if(fh)fh.textContent=(f.raw_sha256||'').slice(0,16)+'…';
      const fc=document.querySelector('#frameCount');if(fc)fc.textContent=`${frameIndex+1}/${m.frames.length}`;
      const ov=document.querySelector('#overlay');
      if(ov){
        ov.setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);ov.innerHTML='';
        let layers=[];
        if(mode==='ALL'||mode==='GT') layers.push(...(f.boxes||[]));
        if(mode==='ALL'||mode==='MODEL') layers.push(...(f.model_boxes||[]));
        if(mode==='ALL'||mode==='RTMDET') layers.push(...(f.rtmdet_boxes||[]));
        layers.forEach(b=>v4RenderBox(ov,b));
      }
      const tb=document.querySelector('#frames');
      if(tb){tb.innerHTML='';m.frames.forEach((x,i)=>{const d=document.createElement('div');d.className='thumb '+(i===frameIndex?'active':'');d.innerHTML=`<img src="${x.image}"><span>${i+1}</span>`;d.onclick=()=>selectFrame(i);tb.appendChild(d)})}
      if(typeof renderFrameEvidence==='function') renderFrameEvidence(f);
    };
  }
})();
</script>
'''
    s = s.replace('</body>', runtime + '</body>', 1)
    return s

mod.patch_html = safe_patch_html
mod.main()
