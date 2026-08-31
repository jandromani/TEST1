from __future__ import annotations
import csv,json,math,hashlib,pathlib,urllib.request
import numpy as np,cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
ROOT=pathlib.Path(__file__).resolve().parent;ASSETS=ROOT/'assets'/'pets';OUT=ROOT/'evidence';ASSETS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
IMG='https://raw.githubusercontent.com/nathanrooy/rpi-urban-mobility-tracker/master/data/images/PETS09-S2L1';GT='https://raw.githubusercontent.com/gonzalocarreteroh/MASA-OA/master/benchmarks/MOT15/PETS09-S2L1/gt/gt.txt';IOU=.35

def dl(u,p):
    if not p.exists():urllib.request.urlretrieve(u,p)
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b;x1=max(ax,bx);y1=max(ay,by);x2=min(ax+aw,bx+bw);y2=min(ay+ah,by+bh);inter=max(0,x2-x1)*max(0,y2-y1);return inter/max(aw*ah+bw*bh-inter,1e-9)
def foot(b):x,y,w,h=b;return x+w/2,y+h
def proj(H,p):q=H@np.array([p[0],p[1],1.]);return float(q[0]/q[2]),float(q[1]/q[2])
def fm(pred,gt):
    if not pred or not gt:return [],list(range(len(pred))),list(range(len(gt)))
    C=np.ones((len(pred),len(gt)))
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):C[i,j]=1-iou(p['bbox'],g['bbox'])
    rr,cc=linear_sum_assignment(C);m=[];mp=set();mg=set()
    for i,j in zip(rr,cc):
        ov=1-C[i,j]
        if ov>=IOU:m.append((i,j,ov));mp.add(i);mg.add(j)
    return m,[i for i in range(len(pred)) if i not in mp],[j for j in range(len(gt)) if j not in mg]
def acquire():
    for f in range(1,121):dl(f'{IMG}/{f:06d}.jpg',ASSETS/f'{f:06d}.jpg')
    dl(GT,ASSETS/'gt.txt')
def loadgt():
    out={}
    with open(ASSETS/'gt.txt',newline='') as h:
        for r in csv.reader(h):
            if len(r)<10:continue
            f=int(float(r[0]));
            if f<=120:out.setdefault(f,[]).append({'id':int(float(r[1])),'bbox':[float(r[2]),float(r[3]),float(r[4]),float(r[5])],'world':[float(r[7]),float(r[8])]})
    return out
def run_range(frames,tracker,conf,H,gt,store=False):
    model=YOLO('yolo11n.pt');tp=fp=fn=idsw=0;gtot=ptot=0;last={};pairs={};pids=set();gids=set();errs=[];records=[]
    for f in frames:
        r=model.track(str(ASSETS/f'{f:06d}.jpg'),persist=True,tracker=tracker,classes=[0],conf=conf,iou=.55,verbose=False,device='cpu')[0];pred=[]
        if r.boxes is not None and len(r.boxes):
            xy=r.boxes.xywh.cpu().numpy();ids=r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.arange(len(xy))+100000+f*100;cf=r.boxes.conf.cpu().numpy()
            for b,tid,c in zip(xy,ids,cf):cx,cy,w,h=map(float,b);pred.append({'track_id':int(tid),'bbox':[cx-w/2,cy-h/2,w,h],'confidence':float(c)})
        truth=gt.get(f,[]);m,up,ug=fm(pred,truth);tp+=len(m);fp+=len(up);fn+=len(ug);gtot+=len(truth);ptot+=len(pred)
        for p in pred:pids.add(p['track_id'])
        for g in truth:gids.add(g['id'])
        mr=[]
        for pi,gi,ov in m:
            p=pred[pi];g=truth[gi];pairs[(p['track_id'],g['id'])]=pairs.get((p['track_id'],g['id']),0)+1;prev=last.get(g['id']);idsw+=1 if prev is not None and prev!=p['track_id'] else 0;last[g['id']]=p['track_id'];wx,wy=proj(H,foot(p['bbox']));e=math.hypot(wx-g['world'][0],wy-g['world'][1]);errs.append(e)
            if store:mr.append({'track_id':p['track_id'],'gt_id':g['id'],'iou':ov,'world_error':e})
        if store:records.append({'frame':f,'predictions':len(pred),'ground_truth':len(truth),'matches':mr,'fp':len(up),'fn':len(ug)})
    P=tp/max(tp+fp,1);R=tp/max(tp+fn,1);mota=1-(fn+fp+idsw)/max(gtot,1);pl=sorted(pids);gl=sorted(gids);M=np.zeros((len(pl),len(gl)),int);pm={x:i for i,x in enumerate(pl)};gm={x:i for i,x in enumerate(gl)}
    for (p,g),n in pairs.items():M[pm[p],gm[g]]=n
    if M.size:rr,cc=linear_sum_assignment(-M);idtp=int(M[rr,cc].sum())
    else:idtp=0
    idfp=ptot-idtp;idfn=gtot-idtp;idf1=2*idtp/max(2*idtp+idfp+idfn,1);rmse=float(np.sqrt(np.mean(np.square(errs)))) if errs else None
    return {'precision':P,'recall':R,'idf1':idf1,'mota':mota,'id_switches':idsw,'tp':tp,'fp':fp,'fn':fn,'gt_detections':gtot,'pred_detections':ptot,'world_rmse_pets_units':rmse,'world_p95_pets_units':float(np.percentile(errs,95)) if errs else None,'frames':records}
acquire();gt=loadgt();src=[];dst=[]
for f in range(1,21):
    for g in gt.get(f,[]):src.append(foot(g['bbox']));dst.append(g['world'])
H,_=cv2.findHomography(np.asarray(src,np.float32),np.asarray(dst,np.float32),cv2.RANSAC,1.);cal=np.asarray([proj(H,p) for p in src]);calrmse=float(np.sqrt(np.mean(np.sum((cal-np.asarray(dst))**2,axis=1))))
candidates=[]
for tracker in ['bytetrack.yaml','botsort.yaml']:
    for conf in [.12,.18,.25]:
        met=run_range(range(21,61),tracker,conf,H,gt,False);candidates.append({'tracker':tracker,'conf':conf,'metrics':{k:v for k,v in met.items() if k!='frames'}});print('VAL',tracker,conf,met['idf1'],met['precision'],met['recall'])
sel=sorted(candidates,key=lambda c:(-c['metrics']['idf1'],-c['metrics']['precision'],-c['metrics']['recall'],c['tracker'],c['conf']))[0];test=run_range(range(61,121),sel['tracker'],sel['conf'],H,gt,True);strict=test['precision']>=.70 and test['recall']>=.70 and test['idf1']>=.80 and test['world_rmse_pets_units'] is not None and test['world_rmse_pets_units']<=1.0
pack={'schema':'assetgraph-evidence/pets-rgb-world-e2e-v2','request_id':'REQ-001','dataset':'PETS09-S2L1/MOT15','split':{'calibration':[1,20],'validation':[21,60],'blind_test':[61,120]},'selection':sel,'validation_candidates':candidates,'test_metrics':{k:v for k,v in test.items() if k!='frames'},'strict_pass_rule':strict,'strict_thresholds':{'precision':.70,'recall':.70,'idf1':.80,'world_rmse_pets_units':1.0},'calibration_rmse_pets_units':calrmse,'ground_truth_policy':'GT allowed for calibration and validation selection only; blind test GT unlocked after predictions','provenance':{'images':IMG,'ground_truth':GT,'gt_sha256':sha(ASSETS/'gt.txt'),'detector':'YOLO11n COCO person','license_note':'Ultralytics/AGPL benchmark component only'},'frames':test['frames']}
(OUT/'pets_rgb_world_e2e_v2.json').write_text(json.dumps(pack,indent=2));print('SELECTED',sel['tracker'],sel['conf']);print(json.dumps(pack['test_metrics'],indent=2));print('STRICT_PASS',strict)
