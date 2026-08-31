from __future__ import annotations
import csv,json,math,hashlib,pathlib,urllib.request
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT=pathlib.Path(__file__).resolve().parent; ASSETS=ROOT/'assets'/'pets'; OUT=ROOT/'evidence'; ASSETS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
IMG_BASE='https://raw.githubusercontent.com/nathanrooy/rpi-urban-mobility-tracker/master/data/images/PETS09-S2L1'
GT_URL='https://raw.githubusercontent.com/gonzalocarreteroh/MASA-OA/master/benchmarks/MOT15/PETS09-S2L1/gt/gt.txt'
CAL_FRAMES=range(1,21); TEST_FRAMES=range(21,121); IOU_THR=.35

def dl(url,p):
    if not p.exists(): urllib.request.urlretrieve(url,p)
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def iou(a,b):
    ax1,ay1,aw,ah=a;bx1,by1,bw,bh=b;ax2=ax1+aw;ay2=ay1+ah;bx2=bx1+bw;by2=by1+bh
    x1=max(ax1,bx1);y1=max(ay1,by1);x2=min(ax2,bx2);y2=min(ay2,by2);inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(aw*ah+bw*bh-inter,1e-9)
def foot(b):x,y,w,h=b;return (x+w/2,y+h)
def project(H,pt):
    p=np.array([pt[0],pt[1],1.],dtype=float);q=H@p;return (float(q[0]/q[2]),float(q[1]/q[2]))
def match(pred,gt):
    if not pred or not gt:return [],list(range(len(pred))),list(range(len(gt)))
    C=np.ones((len(pred),len(gt)),dtype=float)
    for i,p in enumerate(pred):
        for j,g in enumerate(gt):C[i,j]=1-iou(p['bbox'],g['bbox'])
    ri,ci=linear_sum_assignment(C);matches=[];mp=set();mg=set()
    for i,j in zip(ri,ci):
        ov=1-C[i,j]
        if ov>=IOU_THR:matches.append((i,j,ov));mp.add(i);mg.add(j)
    return matches,[i for i in range(len(pred)) if i not in mp],[j for j in range(len(gt)) if j not in mg]

# acquire exactly matched RGB + MOT ground truth
for f in range(1,121):dl(f'{IMG_BASE}/{f:06d}.jpg',ASSETS/f'{f:06d}.jpg')
gtp=ASSETS/'gt.txt';dl(GT_URL,gtp)
gt={}
with open(gtp,newline='') as fh:
    for r in csv.reader(fh):
        if len(r)<10:continue
        fr=int(float(r[0]));
        if fr>120:continue
        gt.setdefault(fr,[]).append({'id':int(float(r[1])),'bbox':[float(r[2]),float(r[3]),float(r[4]),float(r[5])],'world':[float(r[7]),float(r[8])]})
# calibration uses GT only in an explicitly separated calibration window, then freezes H
src=[];dst=[]
for f in CAL_FRAMES:
    for g in gt.get(f,[]):src.append(foot(g['bbox']));dst.append(g['world'])
H,mask=cv2.findHomography(np.asarray(src,np.float32),np.asarray(dst,np.float32),cv2.RANSAC,1.0)
if H is None: raise RuntimeError('homography fit failed')
cal_pred=np.asarray([project(H,p) for p in src]);cal_true=np.asarray(dst);cal_rmse=float(np.sqrt(np.mean(np.sum((cal_pred-cal_true)**2,axis=1))))

model=YOLO('yolo11n.pt')
frame_records=[];tp=fp=fn=idsw=gt_total=pred_total=0;world_err=[];last_for_gt={};pair_counts={};all_pred_ids=set();all_gt_ids=set()
for f in TEST_FRAMES:
    img=str(ASSETS/f'{f:06d}.jpg')
    res=model.track(img,persist=True,tracker='bytetrack.yaml',classes=[0],conf=.18,iou=.55,verbose=False,device='cpu')[0]
    pred=[]
    if res.boxes is not None and len(res.boxes):
        xywh=res.boxes.xywh.cpu().numpy();ids=res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.arange(len(xywh))+100000+f*100
        conf=res.boxes.conf.cpu().numpy()
        for b,tid,c in zip(xywh,ids,conf):
            cx,cy,w,h=map(float,b);pred.append({'track_id':int(tid),'bbox':[cx-w/2,cy-h/2,w,h],'confidence':float(c)})
    truth=gt.get(f,[]);matches,up,ug=match(pred,truth);tp+=len(matches);fp+=len(up);fn+=len(ug);gt_total+=len(truth);pred_total+=len(pred)
    for p in pred:all_pred_ids.add(p['track_id'])
    for g in truth:all_gt_ids.add(g['id'])
    mrecs=[]
    for pi,gi,ov in matches:
        p=pred[pi];g=truth[gi];key=(p['track_id'],g['id']);pair_counts[key]=pair_counts.get(key,0)+1
        prev=last_for_gt.get(g['id'])
        if prev is not None and prev!=p['track_id']:idsw+=1
        last_for_gt[g['id']]=p['track_id']
        wx,wy=project(H,foot(p['bbox']));err=math.hypot(wx-g['world'][0],wy-g['world'][1]);world_err.append(err)
        mrecs.append({'track_id':p['track_id'],'gt_id':g['id'],'iou':ov,'world_error':err,'pred_world':[wx,wy],'gt_world':g['world']})
    frame_records.append({'frame':f,'predictions':len(pred),'ground_truth':len(truth),'matches':mrecs,'fp':len(up),'fn':len(ug)})

precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);mota=1-(fn+fp+idsw)/max(gt_total,1)
# global identity assignment for IDF1
pids=sorted(all_pred_ids);gids=sorted(all_gt_ids);M=np.zeros((len(pids),len(gids)),dtype=int);pi={x:i for i,x in enumerate(pids)};gi={x:i for i,x in enumerate(gids)}
for (p,g),n in pair_counts.items():M[pi[p],gi[g]]=n
if M.size:
    rr,cc=linear_sum_assignment(-M);idtp=int(M[rr,cc].sum())
else:idtp=0
idfp=pred_total-idtp;idfn=gt_total-idtp;idf1=2*idtp/max(2*idtp+idfp+idfn,1)
world_rmse=float(np.sqrt(np.mean(np.square(world_err)))) if world_err else None
world_p95=float(np.percentile(world_err,95)) if world_err else None
pass_rule=precision>=.50 and recall>=.50 and idf1>=.45 and world_rmse is not None and world_rmse<=2.5
weights=[]
for p in [pathlib.Path('yolo11n.pt'),ROOT/'yolo11n.pt']:
    if p.exists():weights.append({'path':str(p),'sha256':sha(p)})
pack={'schema':'assetgraph-evidence/pets-rgb-world-e2e-v1','request_id':'REQ-001','dataset':'PETS09-S2L1/MOT15','protocol':{'calibration_frames':[1,20],'test_frames':[21,120],'ground_truth_locked_during_inference':True,'iou_threshold':IOU_THR,'detector':'Ultralytics YOLO11n COCO person','tracker':'ByteTrack','license_note':'Ultralytics benchmark component; review AGPL/commercial licensing before product use'},'metrics':{'precision':precision,'recall':recall,'idf1':idf1,'mota':mota,'id_switches':idsw,'tp':tp,'fp':fp,'fn':fn,'gt_detections':gt_total,'pred_detections':pred_total,'world_rmse_pets_units':world_rmse,'world_p95_pets_units':world_p95,'calibration_rmse_pets_units':cal_rmse},'pass_rule':pass_rule,'thresholds':{'precision':.50,'recall':.50,'idf1':.45,'world_rmse_pets_units':2.5},'input_provenance':{'images_source':IMG_BASE,'gt_source':GT_URL,'gt_sha256':sha(gtp),'first_image_sha256':sha(ASSETS/'000001.jpg'),'last_image_sha256':sha(ASSETS/'000120.jpg'),'weights':weights},'frames':frame_records}
(OUT/'pets_rgb_world_e2e.json').write_text(json.dumps(pack,indent=2));print(json.dumps(pack['metrics'],indent=2));print('PASS',pass_rule)
