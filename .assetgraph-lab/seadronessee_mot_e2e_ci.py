from __future__ import annotations
import json,pathlib,urllib.request,hashlib,statistics
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=pathlib.Path(__file__).resolve().parent;ASSET=ROOT/'assets'/'seadronessee_mot_e2e';OUT=ROOT/'evidence';ASSET.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
ANN_URL='https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/annotations/instances_val_objects_in_water.json?download=true'
ZIP_URL='https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/SeaDronesSee_MOT_jpg_compressed.zip?download=true'
MODEL_REPO='dronefreak/seadronessee-yolov8n';MODEL_FILE='best.pt';VIDEO_ID=19;N_FRAMES=80;VAL_FRAMES=20

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dl(url,p):
 if not p.exists():urllib.request.urlretrieve(url,p)
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(aa+bb-inter,1e-9)
def xywh(v):x,y,w,h=v;return (x,y,x+w,y+h)
def match_frame(pred,gt,thr=.30):
 if not pred or not gt:return [],list(range(len(pred))),list(range(len(gt)))
 cost=np.array([[1-iou(p['box'],g['box']) for g in gt] for p in pred],float);rr,cc=linear_sum_assignment(cost);matches=[];mp=set();mg=set()
 for a,b in zip(rr,cc):
  ov=1-cost[a,b]
  if ov>=thr:matches.append((a,b,ov));mp.add(a);mg.add(b)
 return matches,[i for i in range(len(pred)) if i not in mp],[j for j in range(len(gt)) if j not in mg]
def metrics(pred_by_frame,gt_by_frame,frames):
 tp=fp=fn=idsw=0;ious=[];pairs=defaultdict(int);last={};pred_total=gt_total=0
 for fr in frames:
  p=pred_by_frame.get(fr,[]);g=gt_by_frame.get(fr,[]);pred_total+=len(p);gt_total+=len(g);m,up,ug=match_frame(p,g);tp+=len(m);fp+=len(up);fn+=len(ug)
  for a,b,ov in m:
   ious.append(ov);pid=p[a].get('track_id');gid=g[b]['track_id']
   if pid is not None:
    pairs[(pid,gid)]+=1
    if gid in last and last[gid]!=pid:idsw+=1
    last[gid]=pid
 pids=sorted({a for a,_ in pairs});gids=sorted({b for _,b in pairs});idtp=0
 if pids and gids:
  M=np.zeros((len(pids),len(gids)));pi={x:i for i,x in enumerate(pids)};gi={x:i for i,x in enumerate(gids)}
  for (p,g),c in pairs.items():M[pi[p],gi[g]]=c
  r,c=linear_sum_assignment(-M);idtp=int(M[r,c].sum())
 idfp=pred_total-idtp;idfn=gt_total-idtp;idf1=2*idtp/max(2*idtp+idfp+idfn,1);precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*precision*recall/max(precision+recall,1e-9);mota=1-(fp+fn+idsw)/max(gt_total,1)
 return {'precision':precision,'recall':recall,'f1':f1,'idf1':idf1,'mota':mota,'id_switches':idsw,'mean_match_iou':statistics.fmean(ious) if ious else None,'tp':tp,'fp':fp,'fn':fn,'idtp':idtp,'idfp':idfp,'idfn':idfn,'pred_detections':pred_total,'gt_detections':gt_total}
def acquire_frames(images):
 from remotezip import RemoteZip
 from PIL import Image
 rgb=ASSET/'rgb';rgb.mkdir(exist_ok=True);rz=RemoteZip(ZIP_URL);members=set(rz.namelist());acquired=[]
 for im in images:
  # Discovery gate proved exact transport mapping: Compressed/val/<COCO image_id>.jpg.
  member=f'Compressed/val/{im["id"]}.jpg'
  if member not in members:raise RuntimeError(f'Expected mapped ZIP member missing: {member}')
  b=rz.read(member);p=rgb/f'{im["id"]}.jpg';p.write_bytes(b)
  with Image.open(p) as chk:chk.verify()
  acquired.append({'image_id':im['id'],'member':member,'path':str(p),'sha256':sha_bytes(b),'bytes':len(b),'source':im.get('source'),'meta':im.get('meta')})
 rz.close();return acquired
def get_weights():
 from huggingface_hub import hf_hub_download
 return hf_hub_download(repo_id=MODEL_REPO,filename=MODEL_FILE)
def run_tracking(paths,conf,weights):
 from ultralytics import YOLO
 model=YOLO(weights);out={}
 for fr,p in paths:
  r=model.track(source=str(p),persist=True,tracker='bytetrack.yaml',conf=conf,iou=.5,imgsz=960,verbose=False)[0];rows=[]
  if r.boxes is not None and len(r.boxes):
   boxes=r.boxes.xyxy.cpu().numpy().tolist();ids=r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes.id is not None else [None]*len(boxes);confs=r.boxes.conf.cpu().numpy().tolist();clss=r.boxes.cls.cpu().numpy().astype(int).tolist()
   for b,i,c,k in zip(boxes,ids,confs,clss):rows.append({'box':tuple(b),'track_id':i,'confidence':float(c),'class_id':int(k)})
  out[fr]=rows
 return out
def detection_f1_for_conf(paths,gt_by_frame,conf,weights):
 from ultralytics import YOLO
 model=YOLO(weights);tp=fp=fn=0
 for fr,p in paths:
  r=model.predict(str(p),conf=conf,imgsz=960,verbose=False)[0];pred=[]
  if r.boxes is not None and len(r.boxes):pred=[{'box':tuple(x)} for x in r.boxes.xyxy.cpu().numpy().tolist()]
  m,up,ug=match_frame(pred,gt_by_frame.get(fr,[]));tp+=len(m);fp+=len(up);fn+=len(ug)
 pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1);return {'precision':pr,'recall':rc,'f1':2*pr*rc/max(pr+rc,1e-9),'tp':tp,'fp':fp,'fn':fn}
def contact_sheet(paths,gt,pred,outpath):
 from PIL import Image,ImageDraw
 chosen=paths[:8];thumbs=[]
 for fr,p in chosen:
  orig=Image.open(p).convert('RGB');ow,oh=orig.size;orig.thumbnail((480,270));im=orig;sx=im.width/ow;sy=im.height/oh;d=ImageDraw.Draw(im)
  for g in gt.get(fr,[]):
   x1,y1,x2,y2=g['box'];d.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline='lime',width=2);d.text((x1*sx,y1*sy),f"GT{g['track_id']}",fill='lime')
  for q in pred.get(fr,[]):
   x1,y1,x2,y2=q['box'];d.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline='cyan',width=2);d.text((x1*sx,max(0,y1*sy+12)),f"P{q.get('track_id')}",fill='cyan')
  d.text((8,8),f'image_id {fr}',fill='white');thumbs.append(im)
 w=max(i.width for i in thumbs);h=max(i.height for i in thumbs);sheet=Image.new('RGB',(w*4,h*2),(15,15,15))
 for idx,im in enumerate(thumbs):sheet.paste(im,((idx%4)*w,(idx//4)*h))
 sheet.save(outpath,quality=88)
def main():
 annp=ASSET/'instances_val_objects_in_water.json';dl(ANN_URL,annp);data=json.load(open(annp));cats={c['id']:c['name'] for c in data.get('categories',[])};ignored={k for k,v in cats.items() if 'ignore' in v.lower()}
 video=[im for im in data['images'] if im.get('video_id')==VIDEO_ID];video=sorted(video,key=lambda x:x.get('frame_index',x.get('id',0)))[:N_FRAMES]
 if len(video)<VAL_FRAMES+20:raise RuntimeError(f'Not enough frames video_id={VIDEO_ID}: {len(video)}')
 acquired=acquire_frames(video);pmap={a['image_id']:pathlib.Path(a['path']) for a in acquired};anns=defaultdict(list);image_ids={im['id'] for im in video}
 for a in data['annotations']:
  if a['image_id'] in image_ids and a.get('category_id') not in ignored:anns[a['image_id']].append({'box':xywh(a['bbox']),'track_id':a['track_id'],'category_id':a.get('category_id')})
 val=[(im['id'],pmap[im['id']]) for im in video[:VAL_FRAMES]];test=[(im['id'],pmap[im['id']]) for im in video[VAL_FRAMES:]];weights=get_weights();cand=[]
 for c in (.05,.10,.15,.20,.30):cand.append((detection_f1_for_conf(val,anns,c,weights),c))
 cand.sort(key=lambda x:(-x[0]['f1'],x[1]));conf=cand[0][1];pred=run_tracking(test,conf,weights);m=metrics(pred,anns,[x[0] for x in test]);technical=m['f1']>=.45 and m['idf1']>=.35
 sheet=OUT/'seadronessee_mot_contact_sheet.jpg';contact_sheet(test,anns,pred,sheet)
 pack={'schema':'assetgraph-evidence/seadronessee-mot-e2e-v2','request_id':'REQ-003','dataset':{'official_dataset':'SeaDronesSee MOT','official_license':'CC0-1.0','transport_mirror':'ObjEarth Hugging Face','annotation_url':ANN_URL,'remote_zip_url':ZIP_URL,'annotation_sha256':sha_file(annp),'video_id':VIDEO_ID,'source_video':video[0].get('source',{}).get('video'),'frames_acquired':len(acquired),'transport_mapping':'Compressed/val/<image_id>.jpg','frame_assets':acquired},'ground_truth':{'persistent_id_field':'track_id','categories':cats,'ignored_category_ids':sorted(ignored)},'model':{'repo':MODEL_REPO,'file':MODEL_FILE,'license':'AGPL-3.0','weights_sha256':sha_file(weights),'product_candidate':False},'selection':{'validation_frames':VAL_FRAMES,'conf_candidates':[{'conf':c,**x} for x,c in cand],'selected_conf':conf},'blind_tracking_test':{'frames':len(test),'metrics':m,'technical_component_pass':technical,'gate':{'detection_f1_gte':.45,'idf1_gte':.35}},'contact_sheet':str(sheet),'provenance_note':'RGB bytes, native track_id GT, source video mapping and telemetry originate from the same SeaDronesSee MOT release; only transport mirror differs. Model is benchmark-only because of AGPL.'}
 p=OUT/'seadronessee_mot_e2e.json';p.write_text(json.dumps(pack,indent=2));print(json.dumps({'pass':technical,'selected_conf':conf,'validation_detection':cand[0][0],'metrics':m,'frames':len(test),'contact_sheet':str(sheet)},indent=2))
if __name__=='__main__':main()
