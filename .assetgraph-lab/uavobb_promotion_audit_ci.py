from __future__ import annotations
import base64, hashlib, json, math, pathlib, requests, shutil, sys, time, zipfile
from collections import defaultdict

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
WORK=ROOT/'assets'/'uavobb_promotion_audit'; WORK.mkdir(parents=True,exist_ok=True)
FILE_ID='1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7'
URL=f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
EXPECTED=601733395
IMGSZ=512; CONF=.30; BATCH=8; IOU_MATCH=.30
C11=ROOT/'evidence'/'c11'
PT=C11/'uavobb_calibrated_best.pt'

def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def download():
 p=WORK/'UAV-OBB.zip'
 if not p.exists() or p.stat().st_size!=EXPECTED:
  with requests.get(URL,stream=True,timeout=120) as r:
   r.raise_for_status()
   with open(p,'wb') as f:
    for c in r.iter_content(1<<20):
     if c:f.write(c)
 if p.stat().st_size!=EXPECTED: raise RuntimeError(f'archive size {p.stat().st_size} != {EXPECTED}')
 return p

def extract(z):
 ex=WORK/'extracted'
 if not (ex/'UAV-OBB').exists():
  with zipfile.ZipFile(z) as zz: zz.extractall(ex)
 return ex/'UAV-OBB'

def names_from_yaml(root):
 import yaml
 y=yaml.safe_load((root/'data.yaml').read_text())
 n=y.get('names',{})
 if isinstance(n,list): return {i:str(v) for i,v in enumerate(n)}
 return {int(k):str(v) for k,v in n.items()}

def poly_area(poly):
 s=0.0
 for i,(x1,y1) in enumerate(poly):
  x2,y2=poly[(i+1)%len(poly)]; s+=x1*y2-x2*y1
 return abs(s)*.5

def gt_objects(label_path,w=1920,h=1080):
 out=[]
 for line in label_path.read_text().splitlines():
  if not line.strip(): continue
  a=line.split(); cls=int(float(a[0])); vals=list(map(float,a[1:9]))
  pts=[(vals[i]*w,vals[i+1]*h) for i in range(0,8,2)]
  area=poly_area(pts); frac=area/(w*h)
  dx=pts[1][0]-pts[0][0]; dy=pts[1][1]-pts[0][1]
  ang=(math.degrees(math.atan2(dy,dx))%180.0)
  out.append({'cls':cls,'poly':pts,'area_frac':frac,'angle':ang})
 return out

def pred_objects(result):
 obb=getattr(result,'obb',None)
 if obb is None: return []
 polys=obb.xyxyxyxy.cpu().numpy(); cls=obb.cls.cpu().numpy(); conf=obb.conf.cpu().numpy()
 return [{'cls':int(c),'poly':[(float(x),float(y)) for x,y in p],'conf':float(s)} for p,c,s in zip(polys,cls,conf)]

def iou_poly(a,b):
 import cv2, numpy as np
 A=np.array(a,dtype='float32'); B=np.array(b,dtype='float32')
 aa=abs(cv2.contourArea(A)); bb=abs(cv2.contourArea(B))
 if aa<=0 or bb<=0:return 0.0
 inter,_=cv2.intersectConvexConvex(A,B)
 return float(inter)/(aa+bb-float(inter)+1e-9)

def match(gt,pred,thr=IOU_MATCH):
 cand=[]
 for gi,g in enumerate(gt):
  for pi,p in enumerate(pred):
   if g['cls']!=p['cls']: continue
   i=iou_poly(g['poly'],p['poly'])
   if i>=thr:cand.append((i,gi,pi))
 cand.sort(reverse=True); G=set();P=set();pairs=[]
 for i,gi,pi in cand:
  if gi in G or pi in P: continue
  G.add(gi);P.add(pi);pairs.append((gi,pi,i))
 return pairs,G,P

def size_bin(frac):
 return 'small' if frac<0.00012 else ('medium' if frac<0.00055 else 'large')

def angle_bin(a):
 if a<22.5 or a>=157.5:return 'axis_0'
 if a<67.5:return 'diag_45'
 if a<112.5:return 'axis_90'
 return 'diag_135'

def audit_split(model,root,split,names):
 images=sorted((root/split/'images').glob('*.jpg'))
 results=model.predict([str(x) for x in images],conf=CONF,imgsz=IMGSZ,batch=BATCH,device='cpu',verbose=False)
 slices={k:defaultdict(lambda:{'gt':0,'tp':0,'pred':0}) for k in ('class','size','angle')}
 totals={'gt':0,'tp':0,'pred':0}; examples=[]
 for im,r in zip(images,results):
  from PIL import Image
  with Image.open(im) as img:w,h=img.size
  gt=gt_objects(root/split/'labels'/(im.stem+'.txt'),w,h); pred=pred_objects(r)
  pairs,G,P=match(gt,pred)
  totals['gt']+=len(gt);totals['tp']+=len(pairs);totals['pred']+=len(pred)
  for g in gt:
   slices['class'][names.get(g['cls'],str(g['cls']))]['gt']+=1
   slices['size'][size_bin(g['area_frac'])]['gt']+=1
   slices['angle'][angle_bin(g['angle'])]['gt']+=1
  for p in pred:slices['class'][names.get(p['cls'],str(p['cls']))]['pred']+=1
  for gi,pi,_ in pairs:
   g=gt[gi]
   slices['class'][names.get(g['cls'],str(g['cls']))]['tp']+=1
   slices['size'][size_bin(g['area_frac'])]['tp']+=1
   slices['angle'][angle_bin(g['angle'])]['tp']+=1
  for pi,p in enumerate(pred):
   if pi in P:
    gi=next(gi for gi,pj,_ in pairs if pj==pi);g=gt[gi]
    slices['size'][size_bin(g['area_frac'])]['pred']+=1
    slices['angle'][angle_bin(g['angle'])]['pred']+=1
  if len(examples)<6:
   examples.append({'image':im.name,'sha256':sha_file(im),'w':w,'h':h,'gt':gt,'pred':pred})
 def finish(d):
  out={}
  for k,v in d.items():
   gt=v['gt'];tp=v['tp'];pr=v['pred'];out[k]={**v,'recall':tp/gt if gt else None,'precision':tp/pr if pr else None}
  return out
 overall={'precision':totals['tp']/totals['pred'] if totals['pred'] else 0,'recall':totals['tp']/totals['gt'] if totals['gt'] else 0,**totals}
 return {'overall':overall,'slices':{k:finish(v) for k,v in slices.items()},'examples':examples}

def parity(pt_model,onnx_model,root):
 ims=sorted((root/'test'/'images').glob('*.jpg'))[:8]
 a=pt_model.predict([str(x) for x in ims],conf=CONF,imgsz=IMGSZ,batch=8,device='cpu',verbose=False)
 b=onnx_model.predict([str(x) for x in ims],conf=CONF,imgsz=IMGSZ,batch=8,device='cpu',verbose=False)
 TP=FP=FN=0;rows=[]
 for im,ra,rb in zip(ims,a,b):
  ga=pred_objects(ra); pb=pred_objects(rb); pairs,G,P=match(ga,pb,thr=.85)
  tp=len(pairs);fp=len(pb)-tp;fn=len(ga)-tp;TP+=tp;FP+=fp;FN+=fn
  rows.append({'image':im.name,'pt':len(ga),'onnx':len(pb),'matched_iou85':tp})
 precision=TP/(TP+FP) if TP+FP else 1.0;recall=TP/(TP+FN) if TP+FN else 1.0
 f1=2*precision*recall/(precision+recall) if precision+recall else 0
 return {'precision':precision,'recall':recall,'f1':f1,'tp':TP,'fp':FP,'fn':FN,'rows':rows}

def decision_object(example,names,model_sha,source='UAV-OBB/test'):
 return {'schema':'assetgraph/decision-object-v1','decision_object_id':'AG-UAVOBB-'+example['sha256'][:12],
  'mission':{'type':'civil_overhead_inventory','source':source},
  'observation':{'image':example['image'],'sha256':example['sha256'],'width':example['w'],'height':example['h']},
  'assets':[{'asset_hypothesis_id':f"obs-{i:03d}",'class':names.get(p['cls'],str(p['cls'])),'confidence':p['conf'],'polygon_px':p['poly'],'identity_status':'observation_only'} for i,p in enumerate(example['pred'])],
  'provenance':{'model_sha256':model_sha,'confidence_threshold':CONF,'imgsz':IMGSZ,'framework':'Ultralytics benchmark adapter'},
  'review':{'status':'machine_hypothesis','analyst':None}}

def offline_html(examples,names,pack):
 cards=[]
 for e in examples[:4]:
  p=pathlib.Path(pack['_root'])/'test'/'images'/e['image']; b64=base64.b64encode(p.read_bytes()).decode()
  cards.append(f'''<section class="shot"><canvas data-img="data:image/jpeg;base64,{b64}" data-preds='{json.dumps(e['pred'])}' width="960" height="540"></canvas><div><b>{e['image']}</b><br><code>{e['sha256']}</code></div></section>''')
 payload=json.dumps({k:v for k,v in pack.items() if not k.startswith('_')},separators=(',',':'))
 return f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AssetGraph Cycle 12 Promotion Audit</title><style>body{{margin:0;background:#071018;color:#eaf4fa;font:14px system-ui}}main{{max-width:1100px;margin:auto;padding:24px}}h1{{font-size:44px}}.k{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.k div,.shot{{background:#0d1b25;border:1px solid #29404e;border-radius:14px;padding:14px;margin:12px 0}}canvas{{width:100%;height:auto;border-radius:10px}}code{{color:#83cfe2;word-break:break-all}}.pass{{color:#55e4a0}}@media(max-width:700px){{.k{{grid-template-columns:1fr}}}}</style><main><h1>Promotion Audit <span class="pass">{str(pack['gate']['pass']).upper()}</span></h1><p>Frozen Cycle 11 checkpoint → robustness → ONNX parity → DecisionObject. Real UAV-OBB pixels only.</p><div class="k"><div><b>Test recall</b><br>{pack['test']['overall']['recall']:.3f}</div><div><b>Test precision</b><br>{pack['test']['overall']['precision']:.3f}</div><div><b>ONNX parity F1</b><br>{pack['onnx']['parity']['f1']:.3f}</div><div><b>ONNX MB</b><br>{pack['onnx']['bytes']/1048576:.1f}</div></div>{''.join(cards)}<script>const N={json.dumps(names)};document.querySelectorAll('canvas').forEach(c=>{{let im=new Image();im.onload=()=>{{c.width=im.width;c.height=im.height;let x=c.getContext('2d');x.drawImage(im,0,0);let P=JSON.parse(c.dataset.preds);x.lineWidth=Math.max(2,im.width/700);x.font=Math.max(14,im.width/80)+'px monospace';P.forEach(p=>{{x.strokeStyle='#5de1ff';x.fillStyle='#5de1ff';x.beginPath();p.poly.forEach((q,i)=>i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]));x.closePath();x.stroke();x.fillText((N[p.cls]||p.cls)+' '+p.conf.toFixed(2),p.poly[0][0],p.poly[0][1]);}})}};im.src=c.dataset.img;}});window.EVIDENCE={payload}</script></main>'''

def main():
 if not PT.exists(): raise RuntimeError(f'Cycle11 checkpoint missing: {PT}')
 from ultralytics import YOLO
 t0=time.time(); z=download(); root=extract(z); names=names_from_yaml(root); model_sha=sha_file(PT); model=YOLO(str(PT))
 val=audit_split(model,root,'valid',names); test=audit_split(model,root,'test',names)
 heavy_classes={k:v for k,v in val['slices']['class'].items() if v['gt']>=100}
 robust_classes=all((v['recall'] or 0)>=.20 for v in heavy_classes.values())
 robust_slices=True
 for typ in ('size','angle'):
  for v in val['slices'][typ].values():
   if v['gt']>=100 and (v['recall'] or 0)<.25: robust_slices=False
 robustness_pass=test['overall']['recall']>=.50 and robust_classes and robust_slices
 export=model.export(format='onnx',imgsz=IMGSZ,opset=17,dynamic=False,simplify=False,device='cpu',verbose=False)
 onnx=pathlib.Path(export); onnx_model=YOLO(str(onnx),task='obb'); parity_res=parity(model,onnx_model,root); parity_pass=parity_res['f1']>=.98
 ims=sorted((root/'test'/'images').glob('*.jpg'))[:8]
 def signature():
  rr=model.predict([str(x) for x in ims],conf=CONF,imgsz=IMGSZ,batch=8,device='cpu',verbose=False)
  return [[(p['cls'],round(p['conf'],5)) for p in pred_objects(r)] for r in rr]
 s1=signature();s2=signature(); replay_pass=s1==s2
 decision=decision_object(test['examples'][0],names,model_sha)
 gate=robustness_pass and parity_pass and replay_pass
 pack={'schema':'assetgraph-evidence/uavobb-promotion-audit-v1','request_id':'REQ-OVERHEAD-PROMOTION-AUDIT','dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':sha_file(z),'test_images':16,'validation_images':218},'protocol':{'checkpoint_source':'Cycle11 artifact run 33485780370','training_in_cycle12':False,'confidence_frozen':CONF,'imgsz':IMGSZ,'iou_match':IOU_MATCH,'robustness_gate':'test recall >=0.50 AND validation classes >=100 GT recall >=0.20 AND validation size/orientation slices >=100 GT recall >=0.25','onnx_gate':'PT↔ONNX class-aware IoU>=0.85 parity F1 >=0.98 on first 8 frozen test images','replay_gate':'two PT runs produce identical class/conf signatures'},'model':{'pt_sha256':model_sha,'pt_bytes':PT.stat().st_size},'validation':val,'test':test,'robustness':{'heavy_classes':heavy_classes,'pass':robustness_pass},'onnx':{'path':onnx.name,'sha256':sha_file(onnx),'bytes':onnx.stat().st_size,'parity':parity_res,'pass':parity_pass},'replay':{'pass':replay_pass},'decision_object':decision,'gate':{'pass':gate,'robustness':robustness_pass,'onnx_parity':parity_pass,'replay':replay_pass},'elapsed_seconds':time.time()-t0,'_root':str(root)}
 clean={k:v for k,v in pack.items() if k!='_root'}
 (OUT/'uavobb_promotion_audit.json').write_text(json.dumps(clean,indent=2));(OUT/'decision_object_uavobb.json').write_text(json.dumps(decision,indent=2));shutil.copy2(onnx,OUT/'uavobb_calibrated_best.onnx')
 (ROOT/'dist').mkdir(exist_ok=True);(ROOT/'dist'/'assetgraph_frontier_v14_overhead_decision_object.html').write_text(offline_html(test['examples'],names,pack))
 print(json.dumps({'gate':clean['gate'],'test':test['overall'],'onnx':clean['onnx'],'replay':clean['replay'],'decision_assets':len(decision['assets']),'html':'dist/assetgraph_frontier_v14_overhead_decision_object.html'},indent=2))
 if not gate: sys.exit(2)
if __name__=='__main__':main()
