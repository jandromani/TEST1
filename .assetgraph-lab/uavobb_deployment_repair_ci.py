from __future__ import annotations
import collections, json, math, pathlib, shutil, time
import uavobb_forensic_ci as F

ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; DIST=ROOT/'dist'
CONF=.30; IOU=.30
PROFILES=[
 {'name':'single1024','mode':'single','imgsz':1024,'tile':0,'overlap':0},
 {'name':'tile1024_o20','mode':'tile','imgsz':1024,'tile':1024,'overlap':.20},
 {'name':'tile768_o20','mode':'tile','imgsz':768,'tile':768,'overlap':.20},
]
MODEL=None

def starts(n,t,o):
    if n<=t:return [0]
    step=max(1,int(t*(1-o))); a=list(range(0,max(1,n-t+1),step)); last=n-t
    if a[-1]!=last:a.append(last)
    return sorted(set(a))

def pred_result(r,dx=0,dy=0):
    out=[]; ob=getattr(r,'obb',None)
    if ob is None:return out
    for p,c,s in zip(ob.xyxyxyxy.cpu().numpy(),ob.cls.cpu().numpy(),ob.conf.cpu().numpy()):
        out.append({'cls':int(c),'poly':[(float(x)+dx,float(y)+dy) for x,y in p],'conf':float(s)})
    return out

def merge(P,thr=.50):
    keep=[]
    for p in sorted(P,key=lambda x:x['conf'],reverse=True):
        if any(p['cls']==q['cls'] and F.poly_iou(p['poly'],q['poly'])>=thr for q in keep):continue
        keep.append(p)
    return keep

def predict(path,p):
    from PIL import Image
    if p['mode']=='single':return pred_result(MODEL.predict(str(path),conf=CONF,imgsz=p['imgsz'],device='cpu',verbose=False)[0])
    im=Image.open(path).convert('RGB');w,h=im.size;crops=[];offs=[]
    for y in starts(h,p['tile'],p['overlap']):
      for x in starts(w,p['tile'],p['overlap']):crops.append(im.crop((x,y,min(w,x+p['tile']),min(h,y+p['tile']))));offs.append((x,y))
    rs=MODEL.predict(crops,conf=CONF,imgsz=p['imgsz'],batch=min(8,len(crops)),device='cpu',verbose=False)
    P=[]
    for r,(x,y) in zip(rs,offs):P.extend(pred_result(r,x,y))
    return merge(P)

def evaluate(root,N,split,p):
    from PIL import Image
    tot={'gt':0,'tp':0,'pred':0}; coarse={'gt':0,'tp':0,'pred':0}; sz=collections.defaultdict(lambda:{'gt':0,'tp':0}); cls=collections.defaultdict(lambda:{'gt':0,'tp':0,'pred':0}); ov={'gt':0,'tp':0}
    for im in sorted((root/split/'images').glob('*.jpg')):
        with Image.open(im) as q:w,h=q.size
        G=F.read_gt(root/split/'labels'/(im.stem+'.txt'),w,h);P=predict(im,p)
        pairs,_,_=F.greedy_match(G,P,IOU,True); cpairs,_,_=F.greedy_match(G,P,IOU,False)
        tot['gt']+=len(G);tot['tp']+=len(pairs);tot['pred']+=len(P);coarse['gt']+=len(G);coarse['tp']+=len(cpairs);coarse['pred']+=len(P)
        for g in G:
            b=F.size_bin(g['area_frac']);sz[b]['gt']+=1;cls[N[g['cls']]]['gt']+=1
            if N[g['cls']]=='other_vehicle':ov['gt']+=1
        for q in P:cls[N[q['cls']]]['pred']+=1
        for gi,pi,_ in pairs:sz[F.size_bin(G[gi]['area_frac'])]['tp']+=1;cls[N[G[gi]['cls']]]['tp']+=1
        for gi,pi,_ in cpairs:
            if N[G[gi]['cls']]=='other_vehicle':ov['tp']+=1
    def prf(x):
        p=x['tp']/x['pred'] if x['pred'] else 0;r=x['tp']/x['gt'] if x['gt'] else 0;return {**x,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0}
    sizes={k:{**v,'recall':v['tp']/v['gt'] if v['gt'] else None} for k,v in sz.items()}; classes={k:{**v,'recall':v['tp']/v['gt'] if v['gt'] else None,'precision':v['tp']/v['pred'] if v['pred'] else None} for k,v in cls.items()};ov['recall']=ov['tp']/ov['gt'] if ov['gt'] else 0
    return {'profile':p,'fine':prf(tot),'coarse_vehicle':prf(coarse),'size':sizes,'class':classes,'other_vehicle_coarse':ov}

def score(x):return .55*((x['size'].get('small')or{}).get('recall')or 0)+.25*x['fine']['f1']+.20*x['coarse_vehicle']['f1']
def choose(V):
    b=next(x for x in V if x['profile']['name']=='single1024'); E=[x for x in V if x['fine']['f1']>=b['fine']['f1']-.05 and x['fine']['precision']>=.50];return max(E or [b],key=score)

def parity(onnx,root,nc):
    import numpy as np,torch,onnxruntime as ort
    from ultralytics.utils import ops
    net=MODEL.model.eval();sess=ort.InferenceSession(str(onnx),providers=['CPUExecutionProvider']);inp=sess.get_inputs()[0].name;tp=ptn=onn=0;rows=[]
    for im in sorted((root/'test'/'images').glob('*.jpg'))[:12]:
        a=F.preprocess_image(im,512)
        with torch.no_grad():po=net(torch.from_numpy(a))
        if isinstance(po,(tuple,list)):po=po[0]
        if isinstance(po,(tuple,list)):po=po[0]
        oo=sess.run(None,{inp:a.astype('float32')})[0]
        A=ops.non_max_suppression(po,CONF,.70,nc=nc,rotated=True,max_det=300)[0].detach().cpu().numpy();B=ops.non_max_suppression(torch.from_numpy(oo),CONF,.70,nc=nc,rotated=True,max_det=300)[0].detach().cpu().numpy();ptn+=len(A);onn+=len(B)
        C=[]
        for i,x in enumerate(A):
          for j,y in enumerate(B):
            if int(x[5])!=int(y[5]):continue
            dc=math.hypot(float(x[0]-y[0]),float(x[1]-y[1]));ds=abs(float(x[2]-y[2]))+abs(float(x[3]-y[3]));da=abs(float(x[-1]-y[-1]))
            if dc<=2 and ds<=4 and da<=.03:C.append((dc+.2*ds+5*da,i,j))
        ua=set();ub=set();m=0
        for _,i,j in sorted(C):
            if i in ua or j in ub:continue
            ua.add(i);ub.add(j);m+=1
        tp+=m;rows.append({'image':im.name,'pt':len(A),'onnx':len(B),'matched':m})
    p=tp/onn if onn else 0;r=tp/ptn if ptn else 0;return {'pt_total':ptn,'onnx_total':onn,'matched':tp,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'same_postprocess':True,'rows':rows}

def decision(root,N,p,ptsha,oxsha):
    from PIL import Image
    im=sorted((root/'test'/'images').glob('*.jpg'))[0]
    with Image.open(im) as q:w,h=q.size
    P=predict(im,p)
    return {'schema':'assetgraph-decision-object/0.2','decision_object_id':'DO-UAVOBB-C14-001','mission':'overhead-vehicle-enumeration','asset_semantics':{'stable_class':'vehicle','subtype_is_hypothesis':True},'source':{'image':im.name,'sha256':F.sha256(im),'width':w,'height':h},'observations':[{'observation_id':f'OBS-{i+1:04d}','asset_class':'vehicle','subtype_hypothesis':N[x['cls']],'subtype_confidence':x['conf'],'polygon_px':x['poly']} for i,x in enumerate(P)],'model':{'pt_sha256':ptsha,'onnx_sha256':oxsha,'confidence':CONF,'profile':p},'review':{'status':'machine_hypothesis'}}

def html(report,d):
    V=''.join(f"<tr><td>{x['profile']['name']}</td><td>{x['fine']['f1']:.3f}</td><td>{(x['size'].get('small')or{}).get('recall',0):.3f}</td><td>{x['other_vehicle_coarse']['recall']:.3f}</td><td>{score(x):.3f}</td></tr>" for x in report['validation_profiles']);t=report['frozen_test'];p=report['shared_postprocess_parity']
    return f'''<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>AssetGraph v16 Deployment Repair</title><style>body{{margin:0;background:#071018;color:#eef7fb;font:14px system-ui}}main{{max-width:1100px;margin:auto;padding:22px}}h1{{font-size:48px}}.g{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.c{{background:#0d1b25;border:1px solid #29404e;border-radius:15px;padding:16px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #29404e;text-align:left}}@media(max-width:700px){{.g{{grid-template-columns:1fr}}}}</style><main><p>ASSETGRAPH · CYCLE 14</p><h1>Deployment Repair</h1><p>Frozen weights. Architecture only: tiling + stable vehicle semantics + shared PT/ONNX postprocess.</p><div class=g><div class=c><h2>Gate</h2><pre>{json.dumps(report['gates'],indent=2)}</pre></div><div class=c><h2>Frozen test</h2><p><b>{t['profile']['name']}</b><br>Fine F1 {t['fine']['f1']:.3f}<br>Small recall {(t['size'].get('small')or{}).get('recall',0):.3f}<br>Coarse vehicle F1 {t['coarse_vehicle']['f1']:.3f}<br>other_vehicle coarse recall {t['other_vehicle_coarse']['recall']:.3f}</p></div><div class=c><h2>PT ↔ ONNX</h2><p>Shared-postprocess F1 <b>{p['f1']:.4f}</b></p></div><div class=c><h2>DecisionObject</h2><p>{len(d['observations'])} vehicle observations; subtype is explicitly a hypothesis.</p></div></div><h2>Validation-only selection</h2><table><tr><th>Profile</th><th>Fine F1</th><th>Small R</th><th>other_vehicle coarse R</th><th>Score</th></tr>{V}</table><script>window.EVIDENCE={json.dumps(report,separators=(',',':'))};window.DECISION_OBJECT={json.dumps(d,separators=(',',':'))}</script></main>'''

def main():
    global MODEL
    from ultralytics import YOLO
    if not F.PT.exists():raise RuntimeError('Cycle11 checkpoint missing')
    t=time.time();z,root=F.get_corpus();N=F.class_names(root);MODEL=YOLO(str(F.PT));V=[evaluate(root,N,'valid',p) for p in PROFILES];C=choose(V);T=evaluate(root,N,'test',C['profile']);onnx=pathlib.Path(MODEL.export(format='onnx',imgsz=512,opset=17,dynamic=False,simplify=False,device='cpu',verbose=False));P=parity(onnx,root,len(N));small=(T['size'].get('small')or{}).get('recall')or 0;ov=T['other_vehicle_coarse']['recall'];base=next(x for x in V if x['profile']['name']=='single1024')
    G={'shared_postprocess_parity_pass':P['f1']>=.98,'small_recall_pass':small>=.30,'fine_f1_nonregression_pass':T['fine']['f1']>=base['fine']['f1']-.05,'other_vehicle_coarse_pass':ov>=.70};G['deployment_repair_pass']=all(G.values())
    R={'schema':'assetgraph-evidence/uavobb-deployment-repair-v1','training_in_cycle14':False,'dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':F.sha256(z)},'model':{'pt_sha256':F.sha256(F.PT),'onnx_sha256':F.sha256(onnx),'confidence':CONF},'selection_rule':'validation only: F1 >= single1024 F1-.05, P>=.50; maximize .55*smallR+.25*fineF1+.20*coarseF1','validation_profiles':V,'selected_profile':C['profile'],'frozen_test':T,'shared_postprocess_parity':P,'hierarchical_semantics':{'stable_asset_class':'vehicle','fine_subtype_is_hypothesis':True},'gates':G,'elapsed_seconds':time.time()-t};D=decision(root,N,C['profile'],F.sha256(F.PT),F.sha256(onnx));(OUT/'uavobb_deployment_repair.json').write_text(json.dumps(R,indent=2));(OUT/'decision_object_cycle14.json').write_text(json.dumps(D,indent=2));shutil.copy2(onnx,OUT/'uavobb_deployment_repair.onnx');(DIST/'assetgraph_frontier_v16_deployment_repair.html').write_text(html(R,D));print(json.dumps({'selected':C['profile'],'test':T,'parity':P,'gates':G},indent=2))
if __name__=='__main__':main()
