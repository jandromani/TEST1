from __future__ import annotations
import json, pathlib, urllib.request, hashlib, statistics
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=pathlib.Path(__file__).resolve().parent
ASSET=ROOT/'assets'/'seadronessee_stress'; OUT=ROOT/'evidence'
ASSET.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
ANN_URL='https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/annotations/instances_val_objects_in_water.json?download=true'
ZIP_URL='https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/SeaDronesSee_MOT_jpg_compressed.zip?download=true'
MODEL_REPO='dronefreak/seadronessee-yolov8n'; MODEL_FILE='best.pt'
CONF=0.10  # frozen from previous micro-sequence, never tuned here
WINDOW=120; STRIDE=30; MATCH_IOU=.30

def dl(u,p):
    if not p.exists(): urllib.request.urlretrieve(u,p)
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def xywh(v):
    x,y,w,h=v; return (x,y,x+w,y+h)
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-9)
def match_frame(pred,gt,thr=MATCH_IOU):
    if not pred or not gt:return [],list(range(len(pred))),list(range(len(gt)))
    cost=np.array([[1-iou(p['box'],g['box']) for g in gt] for p in pred],float)
    rr,cc=linear_sum_assignment(cost); m=[]; mp=set(); mg=set()
    for a,b in zip(rr,cc):
        ov=1-cost[a,b]
        if ov>=thr:m.append((a,b,ov));mp.add(a);mg.add(b)
    return m,[i for i in range(len(pred)) if i not in mp],[j for j in range(len(gt)) if j not in mg]
def mot_metrics(pred_by,gt_by,frames):
    tp=fp=fn=idsw=0; ious=[]; pairs=defaultdict(int); last={}; pred_total=gt_total=0
    for fr in frames:
        p=pred_by.get(fr,[]); g=gt_by.get(fr,[]); pred_total+=len(p); gt_total+=len(g)
        m,up,ug=match_frame(p,g); tp+=len(m); fp+=len(up); fn+=len(ug)
        for a,b,ov in m:
            ious.append(ov); pid=p[a].get('track_id'); gid=g[b]['track_id']
            if pid is not None:
                pairs[(pid,gid)]+=1
                if gid in last and last[gid]!=pid: idsw+=1
                last[gid]=pid
    pids=sorted({p for p,_ in pairs}); gids=sorted({g for _,g in pairs}); idtp=0
    if pids and gids:
        M=np.zeros((len(pids),len(gids))); pi={x:i for i,x in enumerate(pids)}; gi={x:i for i,x in enumerate(gids)}
        for (p,g),n in pairs.items(): M[pi[p],gi[g]]=n
        r,c=linear_sum_assignment(-M); idtp=int(M[r,c].sum())
    idfp=pred_total-idtp; idfn=gt_total-idtp
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-9)
    idf1=2*idtp/max(2*idtp+idfp+idfn,1); mota=1-(fp+fn+idsw)/max(gt_total,1)
    return {'precision':precision,'recall':recall,'f1':f1,'idf1':idf1,'mota':mota,'id_switches':idsw,'mean_iou':statistics.fmean(ious) if ious else None,'tp':tp,'fp':fp,'fn':fn,'gt_detections':gt_total,'pred_detections':pred_total,'idtp':idtp,'idfp':idfp,'idfn':idfn}
def choose_stress_window(images,anns):
    byvid=defaultdict(list)
    for im in images: byvid[im['video_id']].append(im)
    candidates=[]
    for vid,ims in byvid.items():
        ims=sorted(ims,key=lambda x:x.get('frame_index',x['id']))
        if len(ims)<WINDOW: continue
        for st in range(0,len(ims)-WINDOW+1,STRIDE):
            w=ims[st:st+WINDOW]; ids=[x['id'] for x in w]
            counts=[len(anns.get(i,[])) for i in ids]; tracks={a['track_id'] for i in ids for a in anns.get(i,[])}
            # Structural challenge selection only; no model output involved.
            maxc=max(counts) if counts else 0; meanc=statistics.fmean(counts) if counts else 0
            occupancy=sum(1 for c in counts if c>0)/len(counts)
            score=10*maxc + 3*len(tracks) + 2*meanc + occupancy
            candidates.append({'score':score,'video_id':vid,'start':st,'unique_tracks':len(tracks),'max_concurrent':maxc,'mean_concurrent':meanc,'occupancy':occupancy,'images':w})
    if not candidates: raise RuntimeError('No eligible 120-frame stress window')
    candidates.sort(key=lambda x:(-x['score'],-x['unique_tracks'],-x['max_concurrent'],x['video_id'],x['start']))
    return candidates[0], [{k:v for k,v in c.items() if k!='images'} for c in candidates[:20]]
def acquire(window):
    from remotezip import RemoteZip
    from PIL import Image
    rgb=ASSET/'rgb'; rgb.mkdir(exist_ok=True); rz=RemoteZip(ZIP_URL); names=set(rz.namelist()); out=[]
    for im in window:
        member=f'Compressed/val/{im["id"]}.jpg'
        if member not in names: raise RuntimeError(f'missing {member}')
        b=rz.read(member); p=rgb/f'{im["id"]}.jpg'; p.write_bytes(b)
        with Image.open(p) as x: x.verify()
        out.append({'image_id':im['id'],'path':str(p),'member':member,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'source':im.get('source'),'meta':im.get('meta')})
    rz.close(); return out
def infer(paths,weights):
    from ultralytics import YOLO
    model=YOLO(weights); pred={}
    for fr,p in paths:
        r=model.track(str(p),persist=True,tracker='bytetrack.yaml',conf=CONF,iou=.5,imgsz=960,verbose=False)[0]; rows=[]
        if r.boxes is not None and len(r.boxes):
            boxes=r.boxes.xyxy.cpu().numpy().tolist(); ids=r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes.id is not None else [None]*len(boxes); confs=r.boxes.conf.cpu().numpy().tolist()
            for b,i,c in zip(boxes,ids,confs): rows.append({'box':tuple(b),'track_id':i,'confidence':float(c)})
        pred[fr]=rows
    return pred
def sheet(paths,gt,pred,p):
    from PIL import Image,ImageDraw
    sample=[paths[round(i*(len(paths)-1)/7)] for i in range(8)]; thumbs=[]
    for fr,path in sample:
        im=Image.open(path).convert('RGB'); ow,oh=im.size; im.thumbnail((480,270)); sx=im.width/ow; sy=im.height/oh; d=ImageDraw.Draw(im)
        for g in gt.get(fr,[]):
            x1,y1,x2,y2=g['box']; d.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline='lime',width=2); d.text((x1*sx,y1*sy),f'G{g["track_id"]}',fill='lime')
        for q in pred.get(fr,[]):
            x1,y1,x2,y2=q['box']; d.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline='cyan',width=2); d.text((x1*sx,y1*sy+12),f'P{q.get("track_id")}',fill='cyan')
        d.text((6,6),f'id {fr}',fill='white'); thumbs.append(im)
    w=max(x.width for x in thumbs); h=max(x.height for x in thumbs); out=Image.new('RGB',(w*4,h*2),(10,10,10))
    for i,x in enumerate(thumbs):out.paste(x,((i%4)*w,(i//4)*h))
    out.save(p,quality=88)
def main():
    annp=ASSET/'instances_val.json'; dl(ANN_URL,annp); data=json.load(open(annp)); cats={c['id']:c['name'] for c in data.get('categories',[])}; ignored={k for k,v in cats.items() if 'ignore' in v.lower()}
    anns=defaultdict(list)
    for a in data['annotations']:
        if a.get('category_id') not in ignored: anns[a['image_id']].append({'box':xywh(a['bbox']),'track_id':a['track_id'],'category_id':a.get('category_id')})
    chosen,top=choose_stress_window(data['images'],anns); acquired=acquire(chosen['images']); pmap={x['image_id']:pathlib.Path(x['path']) for x in acquired}; paths=[(im['id'],pmap[im['id']]) for im in chosen['images']]
    from huggingface_hub import hf_hub_download
    weights=hf_hub_download(repo_id=MODEL_REPO,filename=MODEL_FILE); pred=infer(paths,weights); frames=[x[0] for x in paths]; m=mot_metrics(pred,anns,frames)
    gate=m['f1']>=.45 and m['idf1']>=.35
    sp=OUT/'seadronessee_stress_contact_sheet.jpg'; sheet(paths,anns,pred,sp)
    clean={k:v for k,v in chosen.items() if k!='images'}
    pack={'schema':'assetgraph-evidence/seadronessee-mot-stress-v1','request_id':'REQ-003','protocol':{'selection':'highest structural GT challenge score among all 120-frame validation windows; no model output used','window':WINDOW,'stride':STRIDE,'model_and_conf_frozen_before_selection':True,'confidence':CONF,'match_iou':MATCH_IOU},'dataset':{'name':'SeaDronesSee MOT','official_license':'CC0-1.0','annotation_sha256':sha_file(annp),'archive_transport':ZIP_URL,'selected_window':clean,'top20_structural_candidates':top,'frame_assets':acquired,'categories':cats},'model':{'repo':MODEL_REPO,'file':MODEL_FILE,'license':'AGPL-3.0','product_candidate':False,'weights_sha256':sha_file(weights)},'stress_metrics':m,'technical_component_pass':gate,'gate':{'f1_gte':.45,'idf1_gte':.35},'contact_sheet':str(sp)}
    out=OUT/'seadronessee_mot_stress.json'; out.write_text(json.dumps(pack,indent=2)); print(json.dumps({'pass':gate,'selected_window':clean,'metrics':m,'contact_sheet':str(sp)},indent=2))
if __name__=='__main__': main()
