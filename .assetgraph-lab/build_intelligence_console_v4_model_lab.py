from __future__ import annotations
import base64, copy, importlib.util, io, json, math, os, re, pathlib
from collections import Counter
import numpy as np
from PIL import Image

ROOT=pathlib.Path(__file__).resolve().parent
V3=ROOT/'build_intelligence_console_v3_showcase.py'
spec=importlib.util.spec_from_file_location('assetgraph_v3_for_v4',V3)
mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
HTML=ROOT/'dist'/'ASSETGRAPH_INTELLIGENCE_CONSOLE_v3_SHOWCASE.html'
MAN=ROOT/'dist'/'assetgraph_console_v3_manifest.json'
OUT=ROOT/'dist'/'ASSETGRAPH_INTELLIGENCE_CONSOLE_v4_MODEL_LAB.html'
OUTMAN=ROOT/'dist'/'assetgraph_console_v4_manifest.json'
WORK=ROOT/'.assetgraph-lab'/'assets'/'console_v4_model_lab'; WORK.mkdir(parents=True,exist_ok=True)


def data_from_html(s):
    m=re.search(r'const DATA=(.*?);\nlet missionIndex=',s,re.S)
    if not m: raise RuntimeError('DATA payload not found')
    return json.loads(m.group(1)),m

def decode_frame(frame, path):
    b=base64.b64decode(frame['image'].split(',',1)[1]); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b); return path

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-9)

def match(pred,gt,thr=.30):
    if not pred or not gt:return [],list(range(len(pred))),list(range(len(gt)))
    from scipy.optimize import linear_sum_assignment
    C=np.array([[1-iou(p['xyxy'],g['xyxy']) for g in gt] for p in pred],float); rr,cc=linear_sum_assignment(C); pairs=[];mp=set();mg=set()
    for a,b in zip(rr,cc):
        ov=1-C[a,b]
        if ov>=thr:pairs.append((a,b,float(ov)));mp.add(a);mg.add(b)
    return pairs,[i for i in range(len(pred)) if i not in mp],[i for i in range(len(gt)) if i not in mg]

def crop_desc(path,box):
    im=Image.open(path).convert('RGB');w,h=im.size;x1,y1,x2,y2=box;x1=max(0,int(x1));y1=max(0,int(y1));x2=min(w,max(x1+1,int(x2)));y2=min(h,max(y1+1,int(y2)))
    a=np.asarray(im.crop((x1,y1,x2,y2)).resize((24,24)),dtype=np.float32)/255.0; feats=[]
    for ch in range(3):
        hist,_=np.histogram(a[:,:,ch],bins=8,range=(0,1)); hist=hist.astype(float);hist/=max(hist.sum(),1);feats.extend(hist.tolist())
    feats.extend(a.mean(axis=(0,1)).tolist());feats.extend(a.std(axis=(0,1)).tolist());g=a.mean(axis=2);gy,gx=np.gradient(g);mag=np.hypot(gx,gy);ang=(np.arctan2(gy,gx)+np.pi)%(2*np.pi);hist=np.zeros(8,float)
    for k in range(8):hist[k]=mag[(ang>=k*np.pi/4)&(ang<(k+1)*np.pi/4)].sum()
    hist/=max(hist.sum(),1e-9);feats.extend(hist.tolist());v=np.asarray(feats,float);v/=max(np.linalg.norm(v),1e-9);return v

def cosine(a,b): return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))

def sea_model_enrich(data):
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    weights=hf_hub_download(repo_id='dronefreak/seadronessee-yolov8n',filename='best.pt')
    summary={'model_repo':'dronefreak/seadronessee-yolov8n','weights_sha256':mod.sha256_file(pathlib.Path(weights)),'license':'AGPL-3.0 benchmark-only','product_candidate':False,'missions':{}}
    paths={}
    for mission in data['missions']:
        if 'maritime' not in mission.get('sensor','').lower():continue
        y=YOLO(weights); rows=[]; paths[mission['id']]=[]
        for fi,f in enumerate(mission['frames']):
            p=decode_frame(f,WORK/'sea'/mission['id']/f'{fi:02d}.jpg');paths[mission['id']].append(p)
            r=y.track(str(p),persist=True,tracker='bytetrack.yaml',conf=.10,iou=.5,imgsz=960,verbose=False)[0];pred=[];names=r.names
            if r.boxes is not None and len(r.boxes):
                bs=r.boxes.xyxy.cpu().numpy().tolist(); ids=r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes.id is not None else [None]*len(bs); cs=r.boxes.conf.cpu().numpy().tolist(); ks=r.boxes.cls.cpu().numpy().astype(int).tolist()
                for b,tid,c,k in zip(bs,ids,cs,ks):
                    nm=str(names.get(int(k),k) if isinstance(names,dict) else names[int(k)])
                    pred.append({'xyxy':list(map(float,b)),'track_id':None if tid is None else int(tid),'class_name':nm,'class_id':int(k),'confidence':float(c),'evidence':'MODEL_PREDICTION_BENCHMARK','runtime':'YOLOv8n+ByteTrack'})
            gt=[g for g in f.get('boxes',[]) if str(g.get('evidence','')).startswith('GROUND_TRUTH')]
            pairs,fp,fn=match(pred,gt,.30)
            for pi,gi,ov in pairs:pred[pi]['eval']='TP';pred[pi]['matched_gt_track']=gt[gi].get('track_id');pred[pi]['iou_to_gt']=ov
            for pi in fp:pred[pi]['eval']='FP'
            f['model_boxes']=pred; f['model_eval']={'tp':len(pairs),'fp':len(fp),'fn':len(fn),'precision':len(pairs)/max(len(pred),1),'recall':len(pairs)/max(len(gt),1),'mean_iou':float(np.mean([x[2] for x in pairs])) if pairs else None}
            rows.append(f['model_eval'])
        mission['model_runtime']={'name':'SeaDronesSee YOLOv8n + ByteTrack','status':'BENCHMARK_ONLY_AGPL','confidence':.10,'match_iou':.30}
        summary['missions'][mission['id']]={'frames':len(rows),'tp':sum(x['tp'] for x in rows),'fp':sum(x['fp'] for x in rows),'fn':sum(x['fn'] for x in rows)}
    m04=next(m for m in data['missions'] if m['id']=='M04');m07=next(m for m in data['missions'] if m['id']=='M07'); target=m04.get('primary_track')
    proto=[];enrolled=[]
    for fi,f in enumerate(m04['frames']):
        for b in f.get('model_boxes',[]):
            if str(b.get('matched_gt_track'))==str(target):proto.append(crop_desc(paths['M04'][fi],b['xyxy']));enrolled.append({'frame':fi,'track_id':b.get('track_id'),'confidence':b['confidence'],'iou':b.get('iou_to_gt')})
    retrieval={'target_asset':f'GT-{target}','protocol':'MODEL_ONLY_RANKING_AFTER_GT_BACKED_BENCHMARK_ENROLLMENT','gt_used_for_ranking':False,'enrollment_matches':enrolled,'pass':False}
    if proto:
        P=np.mean(proto,axis=0);P/=max(np.linalg.norm(P),1e-9);cand=[]
        for fi,f in enumerate(m07['frames']):
            for bi,b in enumerate(f.get('model_boxes',[])):
                d=crop_desc(paths['M07'][fi],b['xyxy']); score=cosine(P,d); cand.append((score,fi,bi,b))
        if cand:
            cand.sort(key=lambda x:x[0],reverse=True);score,fi,bi,b=cand[0]; ok=str(b.get('matched_gt_track'))==str(target)
            retrieval.update({'top1_similarity':score,'candidate_frame':fi,'candidate_model_track':b.get('track_id'),'candidate_class':b.get('class_name'),'candidate_confidence':b.get('confidence'),'evaluation_matched_gt_track':b.get('matched_gt_track'),'pass':ok,'candidate_count':len(cand)})
    data['model_memory_retrieval']=retrieval; summary['memory_retrieval']=retrieval
    return summary

def rtmdet_enrich(data):
    mmr=pathlib.Path(os.environ['MMROTATE_DIR']); ckpt=pathlib.Path(os.environ['RTMDET_CKPT']); cfg=mmr/'configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py'
    from mmrotate.utils import register_all_modules
    from mmdet.apis import init_detector, inference_detector
    register_all_modules();model=init_detector(str(cfg),str(ckpt),palette='dota',device='cpu'); classes=tuple(model.dataset_meta.get('classes') or ())
    picks={'M01':[0,6,11],'M02':[0,6,11],'M03':[0,6,11]}; out={'model':'rotated_rtmdet_tiny-3x-dota','checkpoint_sha256':mod.sha256_file(ckpt),'license':'Apache-2.0','role':'OUT_OF_DOMAIN_STRESS_CONTROL','frames':[]}
    for mid,idxs in picks.items():
        mission=next(m for m in data['missions'] if m['id']==mid)
        for fi in idxs:
            f=mission['frames'][fi];p=decode_frame(f,WORK/'rtmdet'/mid/f'{fi:02d}.jpg');res=inference_detector(model,str(p)); pred=[]
            inst=res.pred_instances.cpu(); bbs=inst.bboxes.tensor.numpy() if hasattr(inst.bboxes,'tensor') else inst.bboxes.numpy(); scores=inst.scores.numpy(); labels=inst.labels.numpy()
            order=np.argsort(-scores)[:40]
            for j in order:
                score=float(scores[j]);
                if score<.20:continue
                cx,cy,w,h,a=map(float,bbs[j]);c,s=math.cos(a),math.sin(a);corn=[]
                for dx,dy in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]:corn.append([cx+dx*c-dy*s,cy+dx*s+dy*c])
                xs=[q[0] for q in corn];ys=[q[1] for q in corn]; pred.append({'xyxy':[min(xs),min(ys),max(xs),max(ys)],'polygon':corn,'class_name':classes[int(labels[j])] if int(labels[j])<len(classes) else str(labels[j]),'confidence':score,'evidence':'RTMDET_R_OUT_OF_DOMAIN','runtime':'RTMDet-R tiny DOTA'})
            f['rtmdet_boxes']=pred; out['frames'].append({'mission':mid,'frame':fi,'detections':len(pred),'classes':dict(Counter(x['class_name'] for x in pred))})
    return out

def patch_html(s,data,manifest):
    m=re.search(r'const DATA=(.*?);\nlet missionIndex=',s,re.S); new=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    s=s[:m.start(1)]+new+s[m.end(1):]
    s=s.replace('ASSETGRAPH INTELLIGENCE CONSOLE v3','ASSETGRAPH INTELLIGENCE CONSOLE v4',2)
    s=s.replace('.box.motion{stroke:var(--amber);stroke-dasharray:7 5}', '.box.motion{stroke:var(--amber);stroke-dasharray:7 5}.box.model{stroke:var(--cyan);stroke-width:2.2}.box.rtmdet{stroke:var(--violet);stroke-dasharray:3 4}.modelText{fill:var(--cyan)}.rtmdetText{fill:var(--violet)}')
    s=s.replace("let missionIndex=0, frameIndex=0, activeTab='intel', autoplay=null;", "let missionIndex=0, frameIndex=0, activeTab='intel', autoplay=null, overlayMode='ALL';")
    old=re.search(r'function renderFrame\(\)\{.*?\}function renderFrameEvidence',s,re.S)
    if not old: raise RuntimeError('renderFrame not found')
    fn=r'''function setOverlayMode(m){overlayMode=m;document.querySelectorAll('.v4ov').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));renderFrame()}function renderFrame(){const m=DATA.missions[missionIndex],f=m.frames[frameIndex]||m.frames[0];$('#mainImage').src=f.image;$('#frameId').textContent=f.frame_id;$('#frameHash').textContent=f.raw_sha256.slice(0,16)+'…';$('#frameCount').textContent=`${frameIndex+1}/${m.frames.length}`;const ov=$('#overlay');ov.setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);ov.innerHTML='';let layers=[];if(overlayMode==='ALL'||overlayMode==='GT')layers.push(...(f.boxes||[]));if(overlayMode==='ALL'||overlayMode==='MODEL')layers.push(...(f.model_boxes||[]));if(overlayMode==='ALL'||overlayMode==='RTMDET')layers.push(...(f.rtmdet_boxes||[]));layers.forEach((b,k)=>{const [x1,y1,x2,y2]=b.xyxy;const ev=String(b.evidence||'');const motion=ev.includes('MOTION'),model=ev.includes('MODEL_PREDICTION'),rtm=ev.includes('RTMDET_R');const cls=motion?'motion':model?'model':rtm?'rtmdet':'';const tcls=motion?'motionText':model?'modelText':rtm?'rtmdetText':'';const suffix=model?` · ${Math.round((b.confidence||0)*100)}% ${b.eval||''}`:rtm?` · ${Math.round((b.confidence||0)*100)}%`:'';const g=document.createElementNS('http://www.w3.org/2000/svg','g');g.innerHTML=`<rect class="box ${cls}" x="${x1}" y="${y1}" width="${Math.max(1,x2-x1)}" height="${Math.max(1,y2-y1)}"/><rect class="boxLabel" x="${x1}" y="${Math.max(0,y1-18)}" width="${Math.max(130,(x2-x1))}" height="18"/><text class="labelText ${tcls}" x="${x1+4}" y="${Math.max(12,y1-5)}">${esc(b.class_name)} · ${esc(b.track_id??b.runtime??'')}${suffix}</text>`;ov.appendChild(g)});const tb=$('#frames');tb.innerHTML='';m.frames.forEach((x,i)=>{const d=document.createElement('div');d.className='thumb '+(i===frameIndex?'active':'');d.innerHTML=`<img src="${x.image}"><span>${i+1}</span>`;d.onclick=()=>selectFrame(i);tb.appendChild(d)});renderFrameEvidence(f)}function renderFrameEvidence'''
    s=s[:old.start()]+fn+s[old.end():]
    controls='''<div id="v4OverlayControls" style="position:fixed;left:50%;transform:translateX(-50%);top:63px;z-index:85;display:flex;gap:5px;background:#041014e8;border:1px solid #28505b;border-radius:999px;padding:5px;backdrop-filter:blur(12px)"><button class="btn v4ov active" data-mode="ALL" onclick="setOverlayMode('ALL')">ALL</button><button class="btn v4ov" data-mode="GT" onclick="setOverlayMode('GT')">GT</button><button class="btn v4ov" data-mode="MODEL" onclick="setOverlayMode('MODEL')">MODEL</button><button class="btn v4ov" data-mode="RTMDET" onclick="setOverlayMode('RTMDET')">RTMDet-R</button></div>'''
    s=s.replace('<div class="app">',controls+'<div class="app">',1)
    mem=data.get('model_memory_retrieval',{}); sea=manifest['v4']['sea_model'];rtm=manifest['v4']['rtmdet_domain_stress']
    nav='<button data-v3page="model">MODEL LAB</button>'
    s=s.replace('<button data-v3page="failure">',nav+'<button data-v3page="failure">',1)
    page=f'''<section class="v3page" id="v3-model"><div class="v3eyebrow">MODEL LAB · REAL INFERENCE</div><div class="v3section"><h2>RAW → GT → MODEL → ERROR → TRACK → IDENTITY → MEMORY</h2><div class="v3sectionLead">GT evaluates the model. It does not rank the M07 retrieval candidate.</div></div><div class="v3metrics"><div class="v3metric"><strong>{sum(x['tp'] for x in sea['missions'].values())}</strong><span>TP</span></div><div class="v3metric"><strong>{sum(x['fp'] for x in sea['missions'].values())}</strong><span>FP</span></div><div class="v3metric"><strong>{sum(x['fn'] for x in sea['missions'].values())}</strong><span>FN</span></div><div class="v3metric"><strong>{mem.get('top1_similarity',0):.3f}</strong><span>Memory similarity</span></div><div class="v3metric"><strong>{'PASS' if mem.get('pass') else 'FAIL'}</strong><span>GT-386 retrieval</span></div></div><div class="v3split"><div class="v3card"><div class="pad"><span class="v3badge gt">DOMAIN-MATCHED BENCHMARK</span><h3>SeaDronesSee YOLOv8n + ByteTrack</h3><p>Real detector/tracker output on the exact embedded maritime frames. AGPL-3.0 benchmark-only; not a product candidate.</p><pre class="mono" style="white-space:pre-wrap">{json.dumps(sea['missions'],indent=2)}</pre></div></div><div class="v3card"><div class="pad"><span class="v3badge sealed">OUT-OF-DOMAIN CONTROL</span><h3>RTMDet-R tiny DOTA</h3><p>Real RTMDet-R output on thermal/maritime scenes. Its DOTA taxonomy is intentionally mismatched: this demonstrates domain boundaries rather than pretending every model fits every sensor.</p><pre class="mono" style="white-space:pre-wrap">{json.dumps(rtm['frames'],indent=2)}</pre></div></div></div><div class="v3section"><h2>Model-only memory retrieval</h2><div class="v3card"><div class="pad"><pre class="mono" style="white-space:pre-wrap">{json.dumps(mem,indent=2)}</pre></div></div></div></section>'''
    s=s.replace('<section class="v3page" id="v3-failure">',page+'<section class="v3page" id="v3-failure">',1)
    return s

def main():
    s=HTML.read_text(encoding='utf-8'); data,_=data_from_html(s); sea=sea_model_enrich(data); rtm=rtmdet_enrich(data); manifest=json.loads(MAN.read_text());manifest['schema']='assetgraph-console/model-lab-v4';manifest['v4']={'model_vs_gt':True,'overlay_modes':['ALL','GT','MODEL','RTMDET'],'gt_policy':'evaluation-only for model comparison; M07 retrieval ranking does not consult GT','sea_model':sea,'rtmdet_domain_stress':rtm,'memory_retrieval':data['model_memory_retrieval']};OUTMAN.write_text(json.dumps(manifest,indent=2),encoding='utf-8');OUT.write_text(patch_html(s,data,manifest),encoding='utf-8');print(json.dumps({'html':str(OUT),'bytes':OUT.stat().st_size,'memory_retrieval':data['model_memory_retrieval'],'sea_totals':{k:sum(x[k] for x in sea['missions'].values()) for k in ['tp','fp','fn']},'rtmdet_frames':len(rtm['frames'])},indent=2))
if __name__=='__main__':main()
