from __future__ import annotations
import json, math, pathlib, urllib.request, hashlib
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from ultralytics import YOLO
except Exception:
    YOLO=None

ROOT=pathlib.Path(__file__).resolve().parent
ASSETS=ROOT/'assets'/'auair_subset'; OUT=ROOT/'evidence'
ASSETS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
BASE='https://raw.githubusercontent.com/freeridering/auair-dataset/master/examples/auair_subset'
ANN_URL=f'{BASE}/annotations.json'
MAP={0:0,1:2,2:7,3:2,4:3,5:1,6:5,7:7}
COARSE={0:'human',2:'light_vehicle',7:'heavy_vehicle',3:'motorbike',1:'bicycle',5:'bus'}
IOU=.35

def dl(u,p):
    if not p.exists(): urllib.request.urlretrieve(u,p)
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    x1=max(ax,bx);y1=max(ay,by);x2=min(ax+aw,bx+bw);y2=min(ay+ah,by+bh)
    inter=max(0,x2-x1)*max(0,y2-y1);return inter/max(aw*ah+bw*bh-inter,1e-9)
def match(pred,gt):
    if not pred or not gt:return [], list(range(len(pred))), list(range(len(gt)))
    C=np.full((len(pred),len(gt)),999.0)
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):
            if p['cls']==g['cls']: C[i,j]=1-iou(p['bbox'],g['bbox'])
    rr,cc=linear_sum_assignment(C);m=[];mp=set();mg=set()
    for i,j in zip(rr,cc):
        ov=1-C[i,j]
        if ov>=IOU:m.append((i,j,ov));mp.add(i);mg.add(j)
    return m,[i for i in range(len(pred)) if i not in mp],[j for j in range(len(gt)) if j not in mg]

dl(ANN_URL,ASSETS/'annotations.json'); data=json.loads((ASSETS/'annotations.json').read_text())
for a in data['annotations']: dl(f'{BASE}/images/{a["image_name"]}',ASSETS/a['image_name'])
model=YOLO('yolo11n.pt') if YOLO else None
TP=FP=FN=0; byclass=defaultdict(lambda:{'tp':0,'fp':0,'fn':0}); frames=[]; completeness=[]
fields=['latitude','longtitude','altitude','linear_x','linear_y','linear_z','angle_phi','angle_theta','angle_psi','time']
for a in data['annotations']:
    comp=sum(a.get(k) is not None for k in fields)/len(fields);completeness.append(comp)
    gt=[]
    for b in a['bbox']:
        gt.append({'cls':MAP[int(b['class'])],'bbox':[float(b['left']),float(b['top']),float(b['width']),float(b['height'])]})
    pred=[]
    if model:
        r=model.predict(str(ASSETS/a['image_name']),classes=[0,1,2,3,5,7],conf=.18,iou=.55,verbose=False,device='cpu')[0]
        if r.boxes is not None:
            for xyxy,c,cf in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy().astype(int),r.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2=map(float,xyxy);pred.append({'cls':int(c),'bbox':[x1,y1,x2-x1,y2-y1],'confidence':float(cf)})
    m,up,ug=match(pred,gt);TP+=len(m);FP+=len(up);FN+=len(ug)
    for i,j,_ in m:byclass[COARSE.get(pred[i]['cls'],str(pred[i]['cls']))]['tp']+=1
    for i in up:byclass[COARSE.get(pred[i]['cls'],str(pred[i]['cls']))]['fp']+=1
    for j in ug:byclass[COARSE.get(gt[j]['cls'],str(gt[j]['cls']))]['fn']+=1
    altitude_m=float(a['altitude'])/1000.0
    frames.append({'image_name':a['image_name'],'telemetry':{'lat':a['latitude'],'lon':a['longtitude'],'altitude_m':altitude_m,'velocity_mps':[a['linear_x'],a['linear_y'],a['linear_z']],'attitude_rad':[a['angle_phi'],a['angle_theta'],a['angle_psi']]},'telemetry_completeness':comp,'gt_objects':len(gt),'pred_objects':len(pred),'matches':len(m),'projection_status':'UNCERTAIN_NO_DATASET_CAMERA_INTRINSICS_EXTRINSICS'})
precision=TP/max(TP+FP,1);recall=TP/max(TP+FN,1);f1=2*precision*recall/max(precision+recall,1e-9)
for k,v in byclass.items():
    p=v['tp']/max(v['tp']+v['fp'],1);r=v['tp']/max(v['tp']+v['fn'],1);v['precision']=p;v['recall']=r;v['f1']=2*p*r/max(p+r,1e-9)
pack={'schema':'assetgraph-evidence/auair-sensor-aware-v1','request_id':'REQ-007','dataset':'AU-AIR public GitHub 100-sample subset','metrics':{'precision':precision,'recall':recall,'f1':f1,'tp':TP,'fp':FP,'fn':FN,'telemetry_completeness_mean':float(np.mean(completeness)),'sample_count':len(frames)},'per_class':dict(byclass),'component_pass':f1>=.45 and float(np.mean(completeness))==1.0,'world_projection_gate':'BLOCKED_BY_CAMERA_CALIBRATION','provenance':{'annotation_url':ANN_URL,'annotation_sha256':sha(ASSETS/'annotations.json'),'source_repo':'freeridering/auair-dataset','license_note':'Dataset website states CC-BY, but example annotations embed NC license entries; commercial redistribution requires reconciliation/legal review'},'frames':frames}
(OUT/'auair_sensor_benchmark.json').write_text(json.dumps(pack,indent=2));print(json.dumps(pack['metrics'],indent=2));print('COMPONENT_PASS',pack['component_pass'])
