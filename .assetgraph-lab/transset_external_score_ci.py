from __future__ import annotations
import json, pathlib, hashlib, io, time
import requests
from remotezip import RemoteZip
from PIL import Image

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
LOCK=OUT/'c15c'/'transset_lockbox_128.json'
C14=OUT/'c14'/'uavobb_deployment_repair.json'
PT=OUT/'c11'/'uavobb_calibrated_best.pt'
GATE=ROOT/'transset_external_gate_v1.json'
ARTICLE=26082217
MODEL=None

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def starts(n,t,o):
    if n<=t:return [0]
    step=max(1,int(t*(1-o))); a=list(range(0,max(1,n-t+1),step)); last=n-t
    if a[-1]!=last:a.append(last)
    return sorted(set(a))

def pred_result(r,dx=0,dy=0):
    out=[]; ob=getattr(r,'obb',None)
    if ob is None:return out
    for poly,cls,conf in zip(ob.xyxyxyxy.cpu().numpy(),ob.cls.cpu().numpy(),ob.conf.cpu().numpy()):
        pts=[(float(x)+dx,float(y)+dy) for x,y in poly]
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        out.append({'cls':int(cls),'conf':float(conf),'poly':pts,'aabb':[min(xs),min(ys),max(xs),max(ys)]})
    return out

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb-inter>0 else 0

def merge_exact_cycle14(P,thr=.50):
    """Replicate Cycle 14 tile merge exactly: class-aware suppression before coarse external scoring."""
    keep=[]
    for p in sorted(P,key=lambda x:x['conf'],reverse=True):
        if any(p['cls']==q['cls'] and iou(p['aabb'],q['aabb'])>=thr for q in keep):continue
        keep.append(p)
    return keep

def predict(im,profile,conf=.30):
    if profile['mode']=='single': return pred_result(MODEL.predict(im,conf=conf,imgsz=profile['imgsz'],device='cpu',verbose=False)[0])
    w,h=im.size;crops=[];offs=[];tile=profile['tile'];overlap=profile['overlap']
    for y in starts(h,tile,overlap):
        for x in starts(w,tile,overlap):
            crops.append(im.crop((x,y,min(w,x+tile),min(h,y+tile))));offs.append((x,y))
    rs=MODEL.predict(crops,conf=conf,imgsz=profile['imgsz'],batch=min(8,len(crops)),device='cpu',verbose=False)
    P=[]
    for r,(x,y) in zip(rs,offs):P.extend(pred_result(r,x,y))
    return merge_exact_cycle14(P)

def greedy(gt,pred,thr=.30):
    cand=[]
    for i,g in enumerate(gt):
        for j,p in enumerate(pred):
            v=iou(g,p['aabb'])
            if v>=thr:cand.append((v,i,j))
    ug=set();up=set();pairs=[]
    for v,i,j in sorted(cand,reverse=True):
        if i in ug or j in up:continue
        ug.add(i);up.add(j);pairs.append((i,j,v))
    return pairs

def main():
    global MODEL
    from ultralytics import YOLO
    t=time.time(); lock=json.loads(LOCK.read_text()); c14=json.loads(C14.read_text()); gate=json.loads(GATE.read_text())
    assert lock['lockbox_frozen'] and lock['selected_count']==128
    assert gate['anti_leakage']['gate_defined_before_first_inference'] is True
    profile=c14['selected_profile']; conf=float(c14['model']['confidence']); MODEL=YOLO(str(PT))
    meta=requests.get(f'https://api.figshare.com/v2/articles/{ARTICLE}',timeout=30).json(); zf=next(x for x in meta['files'] if x['name']=='TRANSSET_v2.zip')
    totals={'gt':0,'pred':0,'tp':0}; per=[]; exact_hash=True
    with RemoteZip(zf['download_url']) as rz:
        for item in lock['items']:
            b=rz.read(item['image_member']); exact_hash &= sha256_bytes(b)==item['image_sha256']; im=Image.open(io.BytesIO(b)).convert('RGB')
            gt=[o['bbox'] for o in item['objects'] if o.get('bbox')]; P=predict(im,profile,conf); pairs=greedy(gt,P,gate['evaluation_semantics']['match_iou'])
            totals['gt']+=len(gt);totals['pred']+=len(P);totals['tp']+=len(pairs)
            per.append({'image_member':item['image_member'],'gt':len(gt),'pred':len(P),'tp':len(pairs),'fn':len(gt)-len(pairs),'fp':len(P)-len(pairs),'mean_iou':sum(x[2] for x in pairs)/len(pairs) if pairs else None})
    p=totals['tp']/totals['pred'] if totals['pred'] else 0;r=totals['tp']/totals['gt'] if totals['gt'] else 0;f1=2*p*r/(p+r) if p+r else 0;cre=abs(totals['pred']-totals['gt'])/totals['gt'] if totals['gt'] else 0
    rule=gate['pass_rule']; gates={'precision_pass':p>=rule['precision_min'],'recall_pass':r>=rule['recall_min'],'f1_pass':f1>=rule['f1_min'],'aggregate_count_relative_error_pass':cre<=rule['aggregate_count_relative_error_max'],'input_hashes_pass':bool(exact_hash)};gates['external_generalization_pass']=all(gates.values())
    report={'schema':'assetgraph-evidence/transset-external-score-v1','dataset':{'name':'TRANSSET','article_id':ARTICLE,'license':'CC BY 4.0','zip_size':zf['size']},'lockbox':{'selected_count':lock['selected_count'],'selection':lock['selection'],'gt_boxes':totals['gt'],'lockbox_sha256':sha256_file(LOCK)},'model':{'checkpoint_sha256':sha256_file(PT),'source':'Cycle 11','cycle14_profile':profile,'confidence':conf,'cycle14_evidence_sha256':sha256_file(C14),'tile_merge':'exact Cycle14 class-aware NMS, then coarse class-agnostic scoring'},'anti_leakage':gate['anti_leakage'],'metrics':{'precision':p,'recall':r,'f1':f1,'tp':totals['tp'],'fp':totals['pred']-totals['tp'],'fn':totals['gt']-totals['tp'],'gt':totals['gt'],'pred':totals['pred'],'aggregate_count_relative_error':cre},'gates':gates,'per_image':per,'elapsed_seconds':time.time()-t}
    (OUT/'transset_external_score.json').write_text(json.dumps(report,indent=2));print(json.dumps({'profile':profile,'metrics':report['metrics'],'gates':gates},indent=2))
if __name__=='__main__':main()
