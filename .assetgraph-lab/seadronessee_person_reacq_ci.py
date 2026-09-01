from __future__ import annotations
import base64, hashlib, json, pathlib
from collections import defaultdict
import seadronessee_stress_ci as S

ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; DIST=ROOT/'dist'; OUT.mkdir(exist_ok=True); DIST.mkdir(exist_ok=True)
PERSON={1,2}

def infer_class(paths,weights):
 from ultralytics import YOLO
 model=YOLO(weights);out={}
 for fr,p in paths:
  r=model.track(str(p),persist=True,tracker='bytetrack.yaml',conf=S.CONF,iou=.5,imgsz=960,verbose=False)[0];rows=[];names=r.names
  if r.boxes is not None and len(r.boxes):
   boxes=r.boxes.xyxy.cpu().numpy().tolist();ids=r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes.id is not None else [None]*len(boxes);confs=r.boxes.conf.cpu().numpy().tolist();clss=r.boxes.cls.cpu().numpy().astype(int).tolist()
   for b,i,c,k in zip(boxes,ids,confs,clss):rows.append({'box':tuple(map(float,b)),'track_id':None if i is None else int(i),'confidence':float(c),'class_id':int(k),'class_name':str(names.get(int(k),k) if isinstance(names,dict) else names[int(k)])})
  out[fr]=rows
 return out

def pred_person(x):
 n=str(x.get('class_name','')).lower().replace('_',' ');return 'swimmer' in n or 'person' in n or 'human' in n

def episodes(window,gt,links):
 ids=[x['id'] for x in window]; pos={x:i for i,x in enumerate(ids)}; tracks=defaultdict(dict)
 for fr in ids:
  L={x['gi']:x for x in links[fr]}
  for gi,g in enumerate(gt.get(fr,[])):
   if g['category_id'] in PERSON: tracks[g['track_id']][pos[fr]]={'g':g,'m':L.get(gi)}
 out=[]
 for gid,obs in tracks.items():
  if len(obs)<4: continue
  lo,hi=min(obs),max(obs); i=lo
  while i<=hi:
   if i in obs and obs[i]['m'] is None:
    st=i
    while i<=hi and i in obs and obs[i]['m'] is None:i+=1
    en=i-1; b=st-1; a=i
    if b in obs and a in obs and obs[b]['m'] and obs[a]['m']:
     pb,pa=obs[b]['m']['pid'],obs[a]['m']['pid']; out.append({'kind':'visible_detection_loss','gt_track_id':gid,'start_idx':st,'end_idx':en,'gap_frames':en-st+1,'before_idx':b,'after_idx':a,'pred_before':pb,'pred_after':pa,'same_pred_id':pb==pa})
   else:i+=1
 out.sort(key=lambda e:(0 if e['same_pred_id'] else 1,-min(e['gap_frames'],8),e['start_idx']))
 return out

def reacq_stats(eps):
 return {'visible_loss_episodes':len(eps),'same_id_reacquired':sum(e['same_pred_id'] for e in eps),'same_id_reacquisition_rate':sum(e['same_pred_id'] for e in eps)/len(eps) if eps else None,'id_change_reacquisitions':sum(not e['same_pred_id'] for e in eps)}

def uri(p):return 'data:image/jpeg;base64,'+base64.b64encode(pathlib.Path(p).read_bytes()).decode()
def state(i,e):
 if i==e['after_idx']:return 'REACQUIRED' if e['same_pred_id'] else 'ID_SWITCH'
 if e['start_idx']<=i<=e['end_idx']:return 'LOST'
 return 'LOCKED' if i<=e['before_idx'] else 'TRACKING'

def make_html(pack,cards):
 D=json.dumps({'pack':pack,'cards':cards},separators=(',',':'))
 return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AssetGraph v0.12 SAR Reacquisition</title><style>:root{--bg:#061019;--p:#0c1822;--l:#20394b;--t:#e8f5ff;--m:#8aa8ba;--g:#47e6a5;--c:#58d8ff;--r:#ff6b6b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 70% 0,#16344b,#061019 42%);color:var(--t);font:14px system-ui}main{max-width:1280px;margin:auto;padding:18px}h1{font-size:clamp(28px,5vw,54px);margin:.2em 0}.muted{color:var(--m)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.card,.panel{background:#0c1822e8;border:1px solid var(--l);border-radius:16px;padding:14px}.metric b{display:block;font-size:28px}.timeline{display:flex;gap:6px;overflow:auto;padding:10px 0}button{background:#102638;color:var(--t);border:1px solid var(--l);border-radius:10px;padding:8px;white-space:nowrap}button.on{outline:2px solid var(--c)}#v{position:relative;background:#000;border-radius:14px;overflow:hidden}#im{display:block;width:100%}.box{position:absolute;border:3px solid}.box span{position:absolute;top:-24px;left:-2px;background:#000c;padding:2px 5px}.gt{border-color:var(--g)}.pr{border-color:var(--c)}#st{position:absolute;right:10px;top:10px;background:#000b;padding:7px 10px;border-radius:99px;font-weight:800}.bad{color:var(--r)}.good{color:var(--g)}pre{white-space:pre-wrap;word-break:break-word;background:#061019;border:1px solid var(--l);padding:10px;border-radius:10px}.chain{display:flex;gap:7px;flex-wrap:wrap}.n{border:1px solid var(--l);padding:8px;border-radius:10px}</style></head><body><main><div class="muted">ASSETGRAPH FRONTIER · v0.12 · OFFLINE EVIDENCE CUBE</div><h1>Loss → reacquisition → alert → evidence</h1><p class="muted">Real SeaDronesSee MOT validation JPEG bytes. Stress window chosen before inference from GT structure only; the episode below is selected after inference for explanation, not scoring.</p><div class="grid"><div class="card metric">Person recall<b id="r"></b><small>REQ-003 ≥ 0.90</small></div><div class="card metric">Person precision<b id="p"></b></div><div class="card metric">Loss episodes<b id="e"></b></div><div class="card metric">Same-ID reacq<b id="q"></b></div><div class="card metric">Stress IDF1<b id="i"></b></div><div class="card metric">Stress MOTA<b id="o"></b></div></div><section class="panel" style="margin-top:12px"><h2>Real person continuity episode</h2><div id="tl" class="timeline"></div><div id="v"><img id="im"><div id="gt" class="box gt"><span></span></div><div id="pr" class="box pr"><span></span></div><div id="st"></div></div><p id="meta" class="muted"></p></section><div class="grid" style="margin-top:12px"><section class="panel"><h2>Traceable alert</h2><pre id="a"></pre></section><section class="panel"><h2>Evidence chain</h2><div class="chain"><span class="n">ALERT</span>→<span class="n">EVENT</span>→<span class="n">TRACK</span>→<span class="n">OBSERVATION</span>→<span class="n">FRAME SHA</span>→<span class="n">SOURCE VIDEO</span>→<span class="n">DATASET</span></div><pre id="z"></pre></section></div><section class="panel" style="margin-top:12px"><h2>Boundary</h2><p>Research reproduction on SeaDronesSee validation. Detector checkpoint is AGPL-3.0 and is not a commercial product candidate. Green = native GT identity; cyan = model + ByteTrack. Missing cyan while green remains visible is a real model loss.</p></section><script>const D='''+D+''',P=D.pack,C=D.cards;function f(x){return x==null?'—':x.toFixed(3)}r.textContent=f(P.person.person_recall);p.textContent=f(P.person.person_precision);e.textContent=P.reacq.visible_loss_episodes;q.textContent=f(P.reacq.same_id_reacquisition_rate);i.textContent=f(P.stress.idf1);o.textContent=f(P.stress.mota);a.textContent=JSON.stringify(P.alert,null,2);z.textContent=JSON.stringify(P.provenance,null,2);function box(el,b,w,h,l){if(!b){el.style.display='none';return}el.style.display='block';el.style.left=b[0]/w*100+'%';el.style.top=b[1]/h*100+'%';el.style.width=(b[2]-b[0])/w*100+'%';el.style.height=(b[3]-b[1])/h*100+'%';el.firstChild.textContent=l}function show(n){[...tl.children].forEach((x,j)=>x.classList.toggle('on',j===n));let c=C[n];im.src=c.image;im.onload=()=>{box(gt,c.gt,c.w,c.h,'GT '+P.episode.gt_track_id);box(pr,c.pred,c.w,c.h,'P '+(c.pid??'—'))};st.textContent=c.state;st.className=c.state==='REACQUIRED'?'good':c.state==='LOST'||c.state==='ID_SWITCH'?'bad':'';meta.textContent=`image_id ${c.id} · ${c.video} frame ${c.source_frame} · GPS ${c.lat?.toFixed(6)}, ${c.lon?.toFixed(6)} · alt ${c.alt?.toFixed(1)} m · SHA ${c.sha.slice(0,16)}…`}C.forEach((c,n)=>{let b=document.createElement('button');b.textContent=c.state+' · '+c.id;b.onclick=()=>show(n);tl.appendChild(b)});show(0)</script></main></body></html>'''

def main():
 annp=S.ASSET/'instances_val.json';S.dl(S.ANN_URL,annp);data=json.load(open(annp));cats={int(c['id']):c['name'] for c in data['categories']};ignored={k for k,v in cats.items() if 'ignore' in v.lower()};gt=defaultdict(list)
 for a in data['annotations']:
  if a['category_id'] not in ignored:gt[a['image_id']].append({'box':S.xywh(a['bbox']),'track_id':a['track_id'],'category_id':a['category_id']})
 chosen=S.choose_stress_window(data['images'],gt);window=chosen['images'];acq=S.acquire(window);am={x['image_id']:x for x in acq};paths=[(x['id'],pathlib.Path(am[x['id']]['path'])) for x in window]
 from huggingface_hub import hf_hub_download
 weights=hf_hub_download(repo_id=S.MODEL_REPO,filename=S.MODEL_FILE);pred=infer_class(paths,weights);frames=[x[0] for x in paths];stress=S.mot_metrics(pred,gt,frames)
 links={};pt=pg=pfp=pp=0
 for fr in frames:
  P,G=pred[fr],gt[fr];m,up,ug=S.match_frame(P,G);links[fr]=[];pg+=sum(g['category_id'] in PERSON for g in G);pp+=sum(pred_person(x) for x in P)
  for a,b,ov in m:
   if G[b]['category_id'] in PERSON and pred_person(P[a]):pt+=1
   links[fr].append({'pi':a,'gi':b,'iou':ov,'pid':P[a].get('track_id'),'gid':G[b]['track_id'],'gcat':G[b]['category_id'],'conf':P[a].get('confidence')})
  pfp+=sum(pred_person(P[a]) for a in up)
 person={'person_recall':pt/max(pg,1),'person_precision':pt/max(pt+pfp,1),'person_tp':pt,'person_gt':pg,'person_pred':pp,'person_fp_unmatched':pfp}
 eps=episodes(window,gt,links);rs=reacq_stats(eps)
 if not eps:raise RuntimeError('No visible person loss/reacquisition episode')
 ep=eps[0];ids=[x['id'] for x in window];lo=max(0,ep['before_idx']-3);hi=min(len(window)-1,ep['after_idx']+4);sel=list(range(lo,hi+1))[:12];cards=[]
 for i in sel:
  fr=ids[i];asset=am[fr]
  from PIL import Image
  with Image.open(asset['path']) as im:w,h=im.size
  G=gt[fr];gi=next((j for j,g in enumerate(G) if g['track_id']==ep['gt_track_id']),None);g=G[gi] if gi is not None else None;mm=next((x for x in links[fr] if x['gid']==ep['gt_track_id']),None);p=pred[fr][mm['pi']] if mm else None
  cards.append({'id':fr,'state':state(i,ep),'image':uri(asset['path']),'w':w,'h':h,'gt':g['box'] if g else None,'pred':p['box'] if p else None,'pid':p.get('track_id') if p else None,'conf':p.get('confidence') if p else None,'video':asset.get('source',{}).get('video'),'source_frame':asset.get('source',{}).get('frame_no'),'lat':asset.get('meta',{}).get('gps_latitude'),'lon':asset.get('meta',{}).get('gps_longitude'),'alt':asset.get('meta',{}).get('altitude'),'sha':asset['sha256']})
 rcard=next(c for c in cards if c['id']==ids[ep['after_idx']]);alert={'type':'PERSON_REACQUIRED' if ep['same_pred_id'] else 'PERSON_REAPPEARED_ID_CHANGED','gt_track_id':ep['gt_track_id'],'pred_before':ep['pred_before'],'pred_after':ep['pred_after'],'gap_frames':ep['gap_frames'],'image_id':rcard['id'],'source_video':rcard['video'],'source_frame':rcard['source_frame'],'gps':[rcard['lat'],rcard['lon']],'altitude':rcard['alt'],'frame_sha256':rcard['sha']}
 prov={'dataset':'SeaDronesSee MOT','license':'CC0-1.0','split':'validation','annotation_sha256':S.sha_file(annp),'model_repo':S.MODEL_REPO,'model_license':'AGPL-3.0 benchmark-only','weights_sha256':S.sha_file(weights),'confidence_frozen':S.CONF,'stress_window_selection':'GT structural only, pre-inference','episode_selection':'post-inference explanatory example'}
 pack={'schema':'assetgraph-evidence/seadronessee-person-reacquisition-v1','request_id':'REQ-003','person':person,'person_recall_gate_pass':person['person_recall']>=.90,'reacq':rs,'episode':ep,'stress':stress,'alert':alert,'provenance':prov,'embedded_frame_ids':[c['id'] for c in cards]}
 (OUT/'seadronessee_person_reacquisition.json').write_text(json.dumps(pack,indent=2));(DIST/'assetgraph_frontier_v12_sar_reacquisition.html').write_text(make_html(pack,cards));print(json.dumps({'person':person,'reacq':rs,'episode':ep,'stress':stress,'gate':pack['person_recall_gate_pass'],'html':str(DIST/'assetgraph_frontier_v12_sar_reacquisition.html')},indent=2))
if __name__=='__main__':main()
