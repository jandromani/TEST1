from __future__ import annotations
import base64,hashlib,json,math,pathlib,requests,shutil,sys,time,zipfile
from collections import defaultdict
ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True); DIST=ROOT/'dist'; DIST.mkdir(exist_ok=True)
WORK=ROOT/'assets'/'uavobb_promotion_audit_fixed'; WORK.mkdir(parents=True,exist_ok=True)
FILE_ID='1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7'; URL=f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'; EXPECTED=601733395
IMGSZ=512; CONF=.30; BATCH=8; IOU_MATCH=.30; PT=ROOT/'evidence'/'c11'/'uavobb_calibrated_best.pt'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def corpus():
 z=WORK/'UAV-OBB.zip'
 if not z.exists() or z.stat().st_size!=EXPECTED:
  with requests.get(URL,stream=True,timeout=120) as r:
   r.raise_for_status()
   with open(z,'wb') as f:
    for c in r.iter_content(1<<20):
     if c:f.write(c)
 if z.stat().st_size!=EXPECTED: raise RuntimeError(f'archive-size-mismatch:{z.stat().st_size}')
 ex=WORK/'extracted'
 if not (ex/'UAV-OBB').exists():
  with zipfile.ZipFile(z) as q:q.extractall(ex)
 return z,ex/'UAV-OBB'

def names(root):
 import yaml
 n=yaml.safe_load((root/'data.yaml').read_text())['names']
 return {i:str(x) for i,x in enumerate(n)} if isinstance(n,list) else {int(k):str(v) for k,v in n.items()}

def area(poly):
 return abs(sum(poly[i][0]*poly[(i+1)%4][1]-poly[(i+1)%4][0]*poly[i][1] for i in range(4)))*.5

def gt(path,w,h):
 out=[]
 for line in path.read_text().splitlines():
  if not line.strip():continue
  a=line.split(); c=int(float(a[0])); v=list(map(float,a[1:9])); p=[(v[i]*w,v[i+1]*h) for i in range(0,8,2)]
  ang=math.degrees(math.atan2(p[1][1]-p[0][1],p[1][0]-p[0][0]))%180
  out.append({'cls':c,'poly':p,'area_frac':area(p)/(w*h),'angle':ang})
 return out

def pred(r):
 o=getattr(r,'obb',None)
 if o is None:return []
 ps=o.xyxyxyxy.cpu().numpy(); cs=o.cls.cpu().numpy(); ss=o.conf.cpu().numpy()
 return [{'cls':int(c),'poly':[(float(x),float(y)) for x,y in p],'conf':float(s)} for p,c,s in zip(ps,cs,ss)]

def piou(a,b):
 import cv2,numpy as np
 A=np.array(a,dtype='float32'); B=np.array(b,dtype='float32'); aa=abs(cv2.contourArea(A)); bb=abs(cv2.contourArea(B))
 if aa<=0 or bb<=0:return 0.0
 inter,_=cv2.intersectConvexConvex(A,B); return float(inter)/(aa+bb-float(inter)+1e-9)

def match(G,P,thr):
 cand=[]
 for gi,g in enumerate(G):
  for pi,p in enumerate(P):
   if g['cls']==p['cls']:
    i=piou(g['poly'],p['poly'])
    if i>=thr:cand.append((i,gi,pi))
 cand.sort(reverse=True); ug=set();up=set();pairs=[]
 for i,gi,pi in cand:
  if gi in ug or pi in up:continue
  ug.add(gi);up.add(pi);pairs.append((gi,pi,i))
 return pairs,ug,up

def sb(x):return 'small' if x<.00012 else ('medium' if x<.00055 else 'large')
def ab(x):return 'axis_0' if x<22.5 or x>=157.5 else ('diag_45' if x<67.5 else ('axis_90' if x<112.5 else 'diag_135'))

def audit(model,root,split,N):
 from PIL import Image
 ims=sorted((root/split/'images').glob('*.jpg')); rr=model.predict([str(x) for x in ims],conf=CONF,imgsz=IMGSZ,batch=BATCH,device='cpu',verbose=False)
 sl={k:defaultdict(lambda:{'gt':0,'tp':0,'pred':0}) for k in ('class','size','angle')}; total={'gt':0,'tp':0,'pred':0}; examples=[]
 for im,r in zip(ims,rr):
  with Image.open(im) as q:w,h=q.size
  G=gt(root/split/'labels'/(im.stem+'.txt'),w,h); P=pred(r); pairs,UG,UP=match(G,P,IOU_MATCH); total['gt']+=len(G);total['tp']+=len(pairs);total['pred']+=len(P)
  for g in G:
   sl['class'][N.get(g['cls'],str(g['cls']))]['gt']+=1;sl['size'][sb(g['area_frac'])]['gt']+=1;sl['angle'][ab(g['angle'])]['gt']+=1
  for p in P:sl['class'][N.get(p['cls'],str(p['cls']))]['pred']+=1
  for gi,pi,_ in pairs:
   g=G[gi];sl['class'][N.get(g['cls'],str(g['cls']))]['tp']+=1;sl['size'][sb(g['area_frac'])]['tp']+=1;sl['size'][sb(g['area_frac'])]['pred']+=1;sl['angle'][ab(g['angle'])]['tp']+=1;sl['angle'][ab(g['angle'])]['pred']+=1
  if len(examples)<6:examples.append({'image':im.name,'sha256':sha(im),'w':w,'h':h,'gt':G,'pred':P})
 def done(d):
  z={}
  for k,v in d.items():
   z[k]={**v,'recall':v['tp']/v['gt'] if v['gt'] else None,'precision':v['tp']/v['pred'] if v['pred'] else None}
  return z
 return {'overall':{**total,'precision':total['tp']/total['pred'] if total['pred'] else 0,'recall':total['tp']/total['gt'] if total['gt'] else 0},'slices':{k:done(v) for k,v in sl.items()},'examples':examples}

def parity(pt,ox,root):
 # Same 8 frozen images as Cycle 12, but one image per call because exported ONNX has fixed batch=1.
 ims=sorted((root/'test'/'images').glob('*.jpg'))[:8]; TP=FP=FN=0;rows=[]
 for im in ims:
  A=pred(pt.predict(str(im),conf=CONF,imgsz=IMGSZ,batch=1,device='cpu',verbose=False)[0]); B=pred(ox.predict(str(im),conf=CONF,imgsz=IMGSZ,batch=1,device='cpu',verbose=False)[0]); pairs,_,_=match(A,B,.85)
  t=len(pairs); f=len(B)-t; n=len(A)-t; TP+=t;FP+=f;FN+=n;rows.append({'image':im.name,'pt':len(A),'onnx':len(B),'matched_iou85':t})
 pr=TP/(TP+FP) if TP+FP else 1.;re=TP/(TP+FN) if TP+FN else 1.;f1=2*pr*re/(pr+re) if pr+re else 0
 return {'precision':pr,'recall':re,'f1':f1,'tp':TP,'fp':FP,'fn':FN,'rows':rows,'batch_adapter':'fixed ONNX batch=1; 8 images evaluated sequentially'}

def decision(ex,N,msha):
 return {'schema':'assetgraph/decision-object-v1','decision_object_id':'AG-UAVOBB-'+ex['sha256'][:12],'mission':{'type':'civil_overhead_inventory','source':'UAV-OBB/test'},'observation':{'image':ex['image'],'sha256':ex['sha256'],'width':ex['w'],'height':ex['h']},'assets':[{'asset_hypothesis_id':f'obs-{i:03d}','class':N.get(p['cls'],str(p['cls'])),'confidence':p['conf'],'polygon_px':p['poly'],'identity_status':'observation_only'} for i,p in enumerate(ex['pred'])],'provenance':{'model_sha256':msha,'confidence_threshold':CONF,'imgsz':IMGSZ,'framework':'Ultralytics benchmark adapter'},'review':{'status':'machine_hypothesis','analyst':None}}

def page(examples,N,pack,root):
 cards=[]
 for e in examples[:4]:
  b=base64.b64encode((root/'test'/'images'/e['image']).read_bytes()).decode(); cards.append(f'''<section><canvas data-img="data:image/jpeg;base64,{b}" data-p='{json.dumps(e['pred'])}'></canvas><p><b>{e['image']}</b><br><code>{e['sha256']}</code></p></section>''')
 payload=json.dumps(pack,separators=(',',':'))
 return f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AssetGraph v14 Promotion Audit</title><style>body{{margin:0;background:#071018;color:#eaf4fa;font:14px system-ui}}main{{max-width:1100px;margin:auto;padding:22px}}h1{{font-size:44px}}.k{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}}.k div,section{{background:#0d1b25;border:1px solid #29404e;border-radius:14px;padding:13px;margin:12px 0}}canvas{{width:100%;height:auto}}code{{color:#83cfe2;word-break:break-all}}@media(max-width:700px){{.k{{grid-template-columns:1fr}}}}</style><main><h1>Promotion Audit · {str(pack['gate']['pass']).upper()}</h1><p>Frozen Cycle 11 checkpoint → robustness → ONNX parity → replay → DecisionObject. Real UAV-OBB pixels only.</p><div class="k"><div>Recall<br><b>{pack['test']['overall']['recall']:.3f}</b></div><div>Precision<br><b>{pack['test']['overall']['precision']:.3f}</b></div><div>ONNX parity F1<br><b>{pack['onnx']['parity']['f1']:.3f}</b></div><div>ONNX<br><b>{pack['onnx']['bytes']/1048576:.1f} MB</b></div></div>{''.join(cards)}<script>const N={json.dumps(N)};document.querySelectorAll('canvas').forEach(c=>{{let im=new Image();im.onload=()=>{{c.width=im.width;c.height=im.height;let x=c.getContext('2d');x.drawImage(im,0,0);JSON.parse(c.dataset.p).forEach(p=>{{x.strokeStyle='#5de1ff';x.fillStyle='#5de1ff';x.lineWidth=3;x.beginPath();p.poly.forEach((q,i)=>i?x.lineTo(...q):x.moveTo(...q));x.closePath();x.stroke();x.font='22px monospace';x.fillText((N[p.cls]||p.cls)+' '+p.conf.toFixed(2),p.poly[0][0],p.poly[0][1])}})}};im.src=c.dataset.img}});window.EVIDENCE={payload}</script></main>'''

def main():
 if not PT.exists():raise RuntimeError('Cycle11 checkpoint missing')
 from ultralytics import YOLO
 t=time.time();z,root=corpus();N=names(root);msha=sha(PT);m=YOLO(str(PT));val=audit(m,root,'valid',N);test=audit(m,root,'test',N)
 heavy={k:v for k,v in val['slices']['class'].items() if v['gt']>=100}; robust_classes=all((v['recall'] or 0)>=.20 for v in heavy.values()); robust_slices=all((v['recall'] or 0)>=.25 for typ in ('size','angle') for v in val['slices'][typ].values() if v['gt']>=100); robust=test['overall']['recall']>=.50 and robust_classes and robust_slices
 export=pathlib.Path(m.export(format='onnx',imgsz=IMGSZ,opset=17,dynamic=False,simplify=False,device='cpu',verbose=False));ox=YOLO(str(export),task='obb');par=parity(m,ox,root);opass=par['f1']>=.98
 ims=sorted((root/'test'/'images').glob('*.jpg'))[:8]
 def sig():
  r=m.predict([str(x) for x in ims],conf=CONF,imgsz=IMGSZ,batch=8,device='cpu',verbose=False);return [[(p['cls'],round(p['conf'],5)) for p in pred(q)] for q in r]
 replay=sig()==sig();dec=decision(test['examples'][0],N,msha);gate=robust and opass and replay
 pack={'schema':'assetgraph-evidence/uavobb-promotion-audit-v1.1','request_id':'REQ-OVERHEAD-PROMOTION-AUDIT','repair_of':'Cycle12 harness batch mismatch; gates unchanged','dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':sha(z),'validation_images':218,'test_images':16},'protocol':{'checkpoint_source':'Cycle11 artifact run 33485780370','training_in_cycle12b':False,'confidence_frozen':CONF,'imgsz':IMGSZ,'iou_match':IOU_MATCH,'robustness_gate':'test recall >=0.50 AND validation classes >=100 GT recall >=0.20 AND validation size/orientation slices >=100 GT recall >=0.25','onnx_gate':'PT↔ONNX class-aware IoU>=0.85 parity F1 >=0.98 on same first 8 frozen test images','replay_gate':'two PT runs produce identical class/conf signatures'},'model':{'pt_sha256':msha,'pt_bytes':PT.stat().st_size},'validation':val,'test':test,'robustness':{'heavy_classes':heavy,'pass':robust},'onnx':{'sha256':sha(export),'bytes':export.stat().st_size,'parity':par,'pass':opass},'replay':{'pass':replay},'decision_object':dec,'gate':{'pass':gate,'robustness':robust,'onnx_parity':opass,'replay':replay},'elapsed_seconds':time.time()-t}
 (OUT/'uavobb_promotion_audit.json').write_text(json.dumps(pack,indent=2));(OUT/'decision_object_uavobb.json').write_text(json.dumps(dec,indent=2));shutil.copy2(export,OUT/'uavobb_calibrated_best.onnx');(DIST/'assetgraph_frontier_v14_overhead_decision_object.html').write_text(page(test['examples'],N,pack,root))
 print(json.dumps({'gate':pack['gate'],'test':test['overall'],'heavy_classes':heavy,'size':val['slices']['size'],'angle':val['slices']['angle'],'onnx':pack['onnx'],'decision_assets':len(dec['assets']),'html':'dist/assetgraph_frontier_v14_overhead_decision_object.html'},indent=2))
 if not gate:sys.exit(2)
if __name__=='__main__':main()
