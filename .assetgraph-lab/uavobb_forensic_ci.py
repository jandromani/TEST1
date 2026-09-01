from __future__ import annotations
import base64, collections, hashlib, json, math, pathlib, requests, shutil, time, zipfile

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
DIST=ROOT/'dist'; DIST.mkdir(exist_ok=True)
WORK=ROOT/'assets'/'uavobb_forensic'; WORK.mkdir(parents=True,exist_ok=True)
PT=ROOT/'evidence'/'c11'/'uavobb_calibrated_best.pt'
FILE_ID='1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7'
URL=f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
EXPECTED=601733395
CONF=0.30
DIAG_CONF=0.05
MATCH_IOU=0.30
RESOLUTIONS=(512,768,1024)


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def get_corpus():
    z=WORK/'UAV-OBB.zip'
    if not z.exists() or z.stat().st_size!=EXPECTED:
        with requests.get(URL,stream=True,timeout=120) as r:
            r.raise_for_status()
            with open(z,'wb') as f:
                for c in r.iter_content(1<<20):
                    if c:f.write(c)
    if z.stat().st_size!=EXPECTED:
        raise RuntimeError(f'archive-size-mismatch:{z.stat().st_size}')
    ex=WORK/'extracted'
    if not (ex/'UAV-OBB').exists():
        with zipfile.ZipFile(z) as q:q.extractall(ex)
    return z,ex/'UAV-OBB'


def class_names(root):
    import yaml
    n=yaml.safe_load((root/'data.yaml').read_text())['names']
    return {i:str(x) for i,x in enumerate(n)} if isinstance(n,list) else {int(k):str(v) for k,v in n.items()}


def poly_area(poly):
    return abs(sum(poly[i][0]*poly[(i+1)%4][1]-poly[(i+1)%4][0]*poly[i][1] for i in range(4)))*0.5


def read_gt(path,w,h):
    out=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        a=line.split(); cls=int(float(a[0])); v=list(map(float,a[1:9]))
        p=[(v[i]*w,v[i+1]*h) for i in range(0,8,2)]
        out.append({'cls':cls,'poly':p,'area_frac':poly_area(p)/(w*h)})
    return out


def read_pred(result):
    o=getattr(result,'obb',None)
    if o is None:return []
    ps=o.xyxyxyxy.cpu().numpy(); cs=o.cls.cpu().numpy(); ss=o.conf.cpu().numpy()
    return [{'cls':int(c),'poly':[(float(x),float(y)) for x,y in p],'conf':float(s)} for p,c,s in zip(ps,cs,ss)]


def poly_iou(a,b):
    import cv2,numpy as np
    A=np.array(a,dtype='float32'); B=np.array(b,dtype='float32')
    aa=abs(cv2.contourArea(A)); bb=abs(cv2.contourArea(B))
    if aa<=0 or bb<=0:return 0.0
    inter,_=cv2.intersectConvexConvex(A,B)
    return float(inter)/(aa+bb-float(inter)+1e-9)


def greedy_match(G,P,thr=MATCH_IOU,class_aware=True):
    cand=[]
    for gi,g in enumerate(G):
        for pi,p in enumerate(P):
            if class_aware and g['cls']!=p['cls']:continue
            i=poly_iou(g['poly'],p['poly'])
            if i>=thr:cand.append((i,gi,pi))
    cand.sort(reverse=True); ug=set();up=set();pairs=[]
    for i,gi,pi in cand:
        if gi in ug or pi in up:continue
        ug.add(gi);up.add(pi);pairs.append((gi,pi,i))
    return pairs,ug,up


def size_bin(frac):
    return 'small' if frac<0.00012 else ('medium' if frac<0.00055 else 'large')


def inference(model,images,imgsz,conf):
    return model.predict([str(x) for x in images],conf=conf,imgsz=imgsz,batch=8,device='cpu',verbose=False)


def class_agnostic_confusion(model,root,N,conf):
    from PIL import Image
    images=sorted((root/'valid'/'images').glob('*.jpg'))
    results=inference(model,images,512,conf)
    matrix=collections.defaultdict(collections.Counter)
    by_true=collections.Counter(); matched=collections.Counter(); examples=[]
    for im,r in zip(images,results):
        with Image.open(im) as q:w,h=q.size
        G=read_gt(root/'valid'/'labels'/(im.stem+'.txt'),w,h); P=read_pred(r)
        pairs,UG,UP=greedy_match(G,P,MATCH_IOU,class_aware=False)
        for g in G:by_true[N[g['cls']]]+=1
        for gi,pi,iou in pairs:
            t=N[G[gi]['cls']]; p=N[P[pi]['cls']]; matrix[t][p]+=1; matched[t]+=1
            if t in ('other_vehicle','truck') and len(examples)<12:
                examples.append({'image':im.name,'sha256':sha256(im),'true':t,'predicted':p,'iou':iou,'gt_poly':G[gi]['poly'],'pred_poly':P[pi]['poly'],'pred_conf':P[pi]['conf']})
        for gi,g in enumerate(G):
            if gi not in UG:matrix[N[g['cls']]]['__unmatched__']+=1
    return {'confidence':conf,'by_true':dict(by_true),'matrix':{k:dict(v) for k,v in matrix.items()},'spatial_match_rate':{k:matched[k]/v if v else None for k,v in by_true.items()},'examples':examples}


def evaluate_resolution(model,root,N,split,imgsz):
    from PIL import Image
    images=sorted((root/split/'images').glob('*.jpg'))
    results=inference(model,images,imgsz,CONF)
    total={'gt':0,'tp':0,'pred':0}; size=collections.defaultdict(lambda:{'gt':0,'tp':0}); cls=collections.defaultdict(lambda:{'gt':0,'tp':0,'pred':0})
    for im,r in zip(images,results):
        with Image.open(im) as q:w,h=q.size
        G=read_gt(root/split/'labels'/(im.stem+'.txt'),w,h); P=read_pred(r); pairs,UG,UP=greedy_match(G,P,MATCH_IOU,True)
        total['gt']+=len(G);total['tp']+=len(pairs);total['pred']+=len(P)
        for g in G:
            size[size_bin(g['area_frac'])]['gt']+=1;cls[N[g['cls']]]['gt']+=1
        for p in P:cls[N[p['cls']]]['pred']+=1
        for gi,pi,_ in pairs:
            g=G[gi];size[size_bin(g['area_frac'])]['tp']+=1;cls[N[g['cls']]]['tp']+=1
    precision=total['tp']/total['pred'] if total['pred'] else 0; recall=total['tp']/total['gt'] if total['gt'] else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
    sizes={k:{**v,'recall':v['tp']/v['gt'] if v['gt'] else None} for k,v in size.items()}
    classes={k:{**v,'recall':v['tp']/v['gt'] if v['gt'] else None,'precision':v['tp']/v['pred'] if v['pred'] else None} for k,v in cls.items()}
    return {'imgsz':imgsz,'overall':{**total,'precision':precision,'recall':recall,'f1':f1},'size':sizes,'class':classes}


def profile_score(x):
    s=x['size']; o=x['overall']
    small=(s.get('small') or {}).get('recall') or 0
    med=(s.get('medium') or {}).get('recall') or 0
    return 0.50*small+0.25*med+0.25*o['f1']


def choose_profile(sweep):
    eligible=[x for x in sweep if x['overall']['precision']>=0.60 and x['overall']['recall']>=0.60]
    if not eligible:eligible=[x for x in sweep if x['imgsz']==512]
    return max(eligible,key=profile_score)


def preprocess_image(path,imgsz=512):
    import cv2,numpy as np
    from ultralytics.data.augment import LetterBox
    im=cv2.imread(str(path))
    lb=LetterBox(new_shape=(imgsz,imgsz),auto=False,stride=32)
    im=lb(image=im)
    im=im[...,::-1].transpose(2,0,1)
    im=np.ascontiguousarray(im,dtype=np.float32)/255.0
    return im[None]


def raw_parity(pt_model,onnx_path,root):
    import numpy as np,torch,onnxruntime as ort
    net=pt_model.model.eval()
    sess=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()[0].name
    rows=[]; all_abs=[]; dots=[]
    ims=sorted((root/'test'/'images').glob('*.jpg'))[:8]
    for im in ims:
        arr=preprocess_image(im,512)
        with torch.no_grad():
            po=net(torch.from_numpy(arr))
        if isinstance(po,(tuple,list)):po=po[0]
        if isinstance(po,(tuple,list)):po=po[0]
        pa=po.detach().cpu().numpy().astype('float32')
        oa=sess.run(None,{inp:arr.astype('float32')})[0].astype('float32')
        if pa.shape!=oa.shape:
            rows.append({'image':im.name,'pt_shape':list(pa.shape),'onnx_shape':list(oa.shape),'shape_match':False});continue
        d=np.abs(pa-oa).ravel(); all_abs.append(d)
        a=pa.ravel().astype('float64'); b=oa.ravel().astype('float64'); denom=(np.linalg.norm(a)*np.linalg.norm(b)+1e-12); cos=float(np.dot(a,b)/denom);dots.append(cos)
        rows.append({'image':im.name,'shape':list(pa.shape),'mean_abs':float(d.mean()),'p99_abs':float(np.quantile(d,.99)),'max_abs':float(d.max()),'cosine':cos})
    if all_abs:
        z=np.concatenate(all_abs)
        agg={'mean_abs':float(z.mean()),'p99_abs':float(np.quantile(z,.99)),'max_abs':float(z.max()),'mean_cosine':float(np.mean(dots))}
    else:agg={}
    return {'aggregate':agg,'rows':rows}


def build_html(root,N,report):
    confusion=report['confusion']['frozen']['matrix']
    ov=confusion.get('other_vehicle',{})
    rows=''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in sorted(ov.items(),key=lambda kv:-kv[1]))
    sweep=''.join(f"<tr><td>{x['imgsz']}</td><td>{x['overall']['precision']:.3f}</td><td>{x['overall']['recall']:.3f}</td><td>{x['size'].get('small',{}).get('recall',0):.3f}</td><td>{x['size'].get('medium',{}).get('recall',0):.3f}</td><td>{profile_score(x):.3f}</td></tr>" for x in report['resolution_sweep']['validation'])
    cards=[]
    ex=report['confusion']['diagnostic_low_conf']['examples'][:4]
    for e in ex:
        p=root/'valid'/'images'/e['image']; b64=base64.b64encode(p.read_bytes()).decode()
        cards.append(f'''<section class="shot"><canvas data-img="data:image/jpeg;base64,{b64}" data-g='{json.dumps(e['gt_poly'])}' data-p='{json.dumps(e['pred_poly'])}'></canvas><p><b>{e['true']} → {e['predicted']}</b> · IoU {e['iou']:.2f} · conf {e['pred_conf']:.2f}<br><code>{e['image']}</code></p></section>''')
    payload=json.dumps(report,separators=(',',':'))
    return f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AssetGraph v15 Forensic Lab</title><style>body{{margin:0;background:#071018;color:#eef7fb;font:14px system-ui}}main{{max-width:1180px;margin:auto;padding:22px}}h1{{font-size:46px;margin-bottom:8px}}h2{{margin-top:32px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.card,.shot{{background:#0d1b25;border:1px solid #29404e;border-radius:15px;padding:15px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #29404e;text-align:left}}canvas{{width:100%;height:auto}}code{{color:#83cfe2;word-break:break-all}}.bad{{color:#ff8090}}.good{{color:#63e3a4}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style><main><h1>Cycle 13 · Failure Forensics</h1><p>Frozen Cycle 11 checkpoint. No retraining. Causes separated into class collapse, scale sensitivity and raw PT↔ONNX drift.</p><div class="grid"><div class="card"><h2>other_vehicle at conf .30</h2><table><tr><th>Spatially aligned predicted class</th><th>GT count</th></tr>{rows}</table></div><div class="card"><h2>Raw ONNX parity</h2><pre>{json.dumps(report['onnx_raw_parity']['aggregate'],indent=2)}</pre></div></div><h2>Validation resolution sweep</h2><table><tr><th>imgsz</th><th>P</th><th>R</th><th>small R</th><th>medium R</th><th>selection score</th></tr>{sweep}</table><p>Selected on validation only: <b>{report['resolution_sweep']['selected_imgsz']}</b>. Frozen test evaluated afterward.</p><h2>Real class-collapse examples</h2><div class="grid">{''.join(cards) if cards else '<div class="card">No spatially matched other_vehicle/truck examples found even at diagnostic confidence.</div>'}</div><script>document.querySelectorAll('canvas').forEach(c=>{{let im=new Image();im.onload=()=>{{c.width=im.width;c.height=im.height;let x=c.getContext('2d');x.drawImage(im,0,0);for(const [poly,col] of [[JSON.parse(c.dataset.g),'#59e696'],[JSON.parse(c.dataset.p),'#5de1ff']]){{x.strokeStyle=col;x.lineWidth=4;x.beginPath();poly.forEach((q,i)=>i?x.lineTo(...q):x.moveTo(...q));x.closePath();x.stroke()}}}};im.src=c.dataset.img}});window.EVIDENCE={payload}</script></main>'''


def main():
    if not PT.exists():raise RuntimeError('Cycle11 checkpoint missing')
    from ultralytics import YOLO
    t=time.time(); z,root=get_corpus(); N=class_names(root); model=YOLO(str(PT))
    conf_frozen=class_agnostic_confusion(model,root,N,CONF)
    conf_diag=class_agnostic_confusion(model,root,N,DIAG_CONF)
    sweep=[evaluate_resolution(model,root,N,'valid',r) for r in RESOLUTIONS]
    chosen=choose_profile(sweep)
    frozen_test=evaluate_resolution(model,root,N,'test',chosen['imgsz'])
    onnx=pathlib.Path(model.export(format='onnx',imgsz=512,opset=17,dynamic=False,simplify=False,device='cpu',verbose=False))
    raw=raw_parity(model,onnx,root)
    ov=conf_frozen['matrix'].get('other_vehicle',{}); ov_total=conf_frozen['by_true'].get('other_vehicle',0); ov_unmatched=ov.get('__unmatched__',0)
    nonmatch=ov_unmatched/ov_total if ov_total else None
    top_wrong=sorted([(k,v) for k,v in ov.items() if k!='__unmatched__'],key=lambda kv:-kv[1])[:3]
    rawagg=raw['aggregate']; raw_close=bool(rawagg) and rawagg.get('mean_cosine',0)>=0.999 and rawagg.get('p99_abs',999)<=0.01
    report={'schema':'assetgraph-evidence/uavobb-forensic-v1','request_id':'REQ-UAVOBB-FAILURE-FORENSIC','training_in_cycle13':False,'dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':sha256(z)},'model':{'pt_sha256':sha256(PT),'confidence_frozen':CONF},'classes':N,'confusion':{'frozen':conf_frozen,'diagnostic_low_conf':conf_diag},'resolution_sweep':{'selection_rule':'validation-only: eligible P>=0.60 and R>=0.60; maximize 0.50*small_recall + 0.25*medium_recall + 0.25*overall_f1','validation':sweep,'selected_imgsz':chosen['imgsz'],'frozen_test':frozen_test},'onnx_raw_parity':raw,'diagnosis':{'other_vehicle_gt':ov_total,'other_vehicle_unmatched_fraction':nonmatch,'other_vehicle_top_spatial_predictions':top_wrong,'raw_network_outputs_close':raw_close,'small_object_problem_confirmed':(chosen['size'].get('small',{}).get('recall') or 0)<0.25},'elapsed_seconds':time.time()-t}
    (OUT/'uavobb_forensic.json').write_text(json.dumps(report,indent=2));shutil.copy2(onnx,OUT/'uavobb_forensic.onnx');(DIST/'assetgraph_frontier_v15_failure_forensics.html').write_text(build_html(root,N,report))
    print(json.dumps({'diagnosis':report['diagnosis'],'selected_imgsz':chosen['imgsz'],'validation_sweep':[{'imgsz':x['imgsz'],'overall':x['overall'],'small':x['size'].get('small'),'medium':x['size'].get('medium')} for x in sweep],'frozen_test':frozen_test['overall'],'frozen_test_sizes':frozen_test['size'],'raw_parity':rawagg,'other_vehicle_confusion':ov,'other_vehicle_diag_confusion':conf_diag['matrix'].get('other_vehicle',{})},indent=2))

if __name__=='__main__':main()
