from __future__ import annotations
import json,hashlib,pathlib,urllib.request
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
ROOT=pathlib.Path(__file__).resolve().parent;ASSETS=ROOT/'assets'/'auair_overhead';OUT=ROOT/'evidence';ASSETS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
BASE='https://raw.githubusercontent.com/freeridering/auair-dataset/master/examples/auair_subset';ANN_URL=f'{BASE}/annotations.json';VIS_URL='https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/best.pt?download=true';VIS_SHA='e5f2685462a10a24e05bd6e5db33e11eb46e0113c96b3440db61483d70831a60';IOU=.35;THRESHOLDS=[.05,.10,.15,.20,.25];AU_VEH={1,2,3,6,7}
def dl(u,p):
 if not p.exists():urllib.request.urlretrieve(u,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def iou(a,b):
 ax,ay,aw,ah=a;bx,by,bw,bh=b;x1=max(ax,bx);y1=max(ay,by);x2=min(ax+aw,bx+bw);y2=min(ay+ah,by+bh);inter=max(0.,x2-x1)*max(0.,y2-y1);return inter/max(aw*ah+bw*bh-inter,1e-9)
def score(pred,gt):
 if not pred and not gt:return 0,0,0,[]
 if not pred:return 0,0,len(gt),[]
 if not gt:return 0,len(pred),0,[]
 C=np.ones((len(pred),len(gt)))
 for i,p in enumerate(pred):
  for j,g in enumerate(gt):C[i,j]=1-iou(p,g)
 rr,cc=linear_sum_assignment(C);m=[];mp=set();mg=set()
 for i,j in zip(rr,cc):
  ov=1-C[i,j]
  if ov>=IOU:m.append(float(ov));mp.add(i);mg.add(j)
 return len(m),len(pred)-len(mp),len(gt)-len(mg),m
def metric(rows,t):
 tp=fp=fn=0;ovs=[]
 for r in rows:
  a,b,c,m=score([x['bbox'] for x in r['pred'] if x['confidence']>=t],r['gt']);tp+=a;fp+=b;fn+=c;ovs+=m
 p=tp/max(tp+fp,1);rec=tp/max(tp+fn,1);f1=2*p*rec/max(p+rec,1e-9);return {'precision':p,'recall':rec,'f1':f1,'tp':tp,'fp':fp,'fn':fn,'mean_iou':float(np.mean(ovs)) if ovs else None}
def split_names(names):
 o=sorted(names,key=lambda n:hashlib.sha256(n.encode()).hexdigest());return set(o[:40]),set(o[40:])
dl(ANN_URL,ASSETS/'annotations.json');data=json.loads((ASSETS/'annotations.json').read_text())
for a in data['annotations']:dl(f'{BASE}/images/{a["image_name"]}',ASSETS/a['image_name'])
vis=ASSETS/'visdrone.pt';dl(VIS_URL,vis)
if sha(vis)!=VIS_SHA:raise RuntimeError('VisDrone checkpoint hash mismatch')
models={'coco_yolo11n':YOLO('yolo11n.pt'),'dota_yolo11n_obb':YOLO('yolo11n-obb.pt'),'visdrone_yolov8n':YOLO(str(vis))}
def extract(name,r):
 out=[]
 if name=='dota_yolo11n_obb':
  if r.obb is None:return out
  for b,c,cf in zip(r.obb.xyxy.cpu().numpy(),r.obb.cls.cpu().numpy().astype(int),r.obb.conf.cpu().numpy()):
   if c in (9,10):x1,y1,x2,y2=map(float,b);out.append({'bbox':[x1,y1,x2-x1,y2-y1],'confidence':float(cf),'source_class':int(c)})
 else:
  if r.boxes is None:return out
  for b,c,cf in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy().astype(int),r.boxes.conf.cpu().numpy()):
   label=str(r.names[int(c)]).lower().replace('-','_');ok=label in ({'car','bus','truck'} if name=='coco_yolo11n' else {'car','van','truck','bus'})
   if ok:x1,y1,x2,y2=map(float,b);out.append({'bbox':[x1,y1,x2-x1,y2-y1],'confidence':float(cf),'source_class':label})
 return out
val_names,test_names=split_names([a['image_name'] for a in data['annotations']]);frames={k:[] for k in models}
for mn,model in models.items():
 for a in data['annotations']:
  gt=[[float(b['left']),float(b['top']),float(b['width']),float(b['height'])] for b in a['bbox'] if int(b['class']) in AU_VEH];r=model.predict(str(ASSETS/a['image_name']),conf=.04,iou=.55,imgsz=1280,device='cpu',verbose=False)[0];frames[mn].append({'image_name':a['image_name'],'split':'validation' if a['image_name'] in val_names else 'test','gt':gt,'pred':extract(mn,r)})
results={};ranking=[]
for name,rows in frames.items():
 val=[r for r in rows if r['split']=='validation'];test=[r for r in rows if r['split']=='test'];grid=[]
 for t in THRESHOLDS:
  m=metric(val,t);grid.append((m['f1'],t,m))
 grid.sort(key=lambda x:(-x[0],x[1]));_,bt,bv=grid[0];tm=metric(test,bt);results[name]={'selected_threshold':bt,'validation':bv,'test':tm,'validation_grid':[{'threshold':t,**m} for _,t,m in grid]};ranking.append((tm['f1'],name))
ranking.sort(reverse=True);winner=ranking[0][1];general=results['coco_yolo11n']['test']['f1'];winf=results[winner]['test']['f1'];rel=(winf-general)/max(general,1e-9);passed=winf>=.45 and rel>=.15
pack={'schema':'assetgraph-evidence/overhead-detector-tournament-v1','request_id':'REQ-006','dataset':'AU-AIR public 100-image subset','task':'binary motor-vehicle overhead detection','split_protocol':{'method':'sha256(image_name)','validation_images':40,'blind_test_images':60,'thresholds':THRESHOLDS,'test_never_used_for_model_or_threshold_selection':True},'models':results,'winner':winner,'winner_test_f1':winf,'coco_baseline_test_f1':general,'relative_improvement_vs_coco':rel,'pass_rule':passed,'gate':{'minimum_test_f1':.45,'minimum_relative_improvement_vs_generalist':.15},'provenance':{'annotation_url':ANN_URL,'annotation_sha256':sha(ASSETS/'annotations.json'),'visdrone_model_url':VIS_URL,'visdrone_model_sha256':sha(vis),'visdrone_model_license':'OpenRAIL - legal review before commercial embedding','dota_model':'Ultralytics yolo11n-obb.pt / DOTA','dota_license_note':'AGPL/commercial licensing review required'}}
(OUT/'overhead_detector_tournament.json').write_text(json.dumps(pack,indent=2));print(json.dumps({k:{'threshold':v['selected_threshold'],'test':v['test']} for k,v in results.items()},indent=2));print('WINNER',winner,'REL',rel,'PASS',passed)