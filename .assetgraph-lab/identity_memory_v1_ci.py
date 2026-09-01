from __future__ import annotations
import json, math, pathlib, statistics
from collections import defaultdict
import numpy as np
import seadronessee_stress_ci as S

ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
WINDOW=120; STRIDE=30; EXCLUDE_VIDEO=21
MAX_GAP=4; MAX_NORM_DIST=3.0; MIN_APPEARANCE=0.35; MIN_SCORE=0.62
PERSON={1,2}

def choose_holdout(images,anns):
    byvid=defaultdict(list)
    for im in images: byvid[im['video_id']].append(im)
    cs=[]
    for vid,ims in byvid.items():
        if vid==EXCLUDE_VIDEO: continue
        ims=sorted(ims,key=lambda x:x.get('frame_index',x['id']))
        for st in range(0,max(0,len(ims)-WINDOW+1),STRIDE):
            w=ims[st:st+WINDOW]
            if len(w)<WINDOW: continue
            ids=[x['id'] for x in w]; counts=[len(anns.get(i,[])) for i in ids]; tracks={a['track_id'] for i in ids for a in anns.get(i,[])}
            maxc=max(counts) if counts else 0; meanc=statistics.fmean(counts) if counts else 0; occ=sum(c>0 for c in counts)/len(counts)
            score=10*maxc+3*len(tracks)+2*meanc+occ
            cs.append({'score':score,'video_id':vid,'start':st,'unique_tracks':len(tracks),'max_concurrent':maxc,'mean_concurrent':meanc,'occupancy':occ,'images':w})
    if not cs: raise RuntimeError('No independent video window')
    cs.sort(key=lambda x:(-x['score'],-x['unique_tracks'],-x['max_concurrent'],x['video_id'],x['start']))
    return cs[0],[{k:v for k,v in x.items() if k!='images'} for x in cs[:20]]

def crop_desc(path,box):
    from PIL import Image
    with Image.open(path) as im:
        im=im.convert('RGB'); w,h=im.size
        x1,y1,x2,y2=box; x1=max(0,int(x1)); y1=max(0,int(y1)); x2=min(w,max(x1+1,int(x2))); y2=min(h,max(y1+1,int(y2)))
        c=im.crop((x1,y1,x2,y2)).resize((24,24)); a=np.asarray(c,dtype=np.float32)/255.0
    feats=[]
    for ch in range(3):
        hist,_=np.histogram(a[:,:,ch],bins=8,range=(0,1),density=False); hist=hist.astype(float); hist/=max(hist.sum(),1); feats.extend(hist.tolist())
    feats.extend(a.mean(axis=(0,1)).tolist()); feats.extend(a.std(axis=(0,1)).tolist())
    g=a.mean(axis=2); gy,gx=np.gradient(g); mag=np.hypot(gx,gy); ang=(np.arctan2(gy,gx)+np.pi)%(2*np.pi); hist=np.zeros(8,float)
    for k in range(8): hist[k]=mag[(ang>=k*np.pi/4)&(ang<(k+1)*np.pi/4)].sum()
    hist/=max(hist.sum(),1e-9); feats.extend(hist.tolist())
    v=np.asarray(feats,float); v/=max(np.linalg.norm(v),1e-9); return v.tolist()

def infer(paths,weights):
    from ultralytics import YOLO
    model=YOLO(weights); pred={}
    for idx,(fr,p) in enumerate(paths):
        r=model.track(str(p),persist=True,tracker='bytetrack.yaml',conf=S.CONF,iou=.5,imgsz=960,verbose=False)[0]; rows=[]; names=r.names
        if r.boxes is not None and len(r.boxes):
            boxes=r.boxes.xyxy.cpu().numpy().tolist(); ids=r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes.id is not None else [None]*len(boxes); confs=r.boxes.conf.cpu().numpy().tolist(); clss=r.boxes.cls.cpu().numpy().astype(int).tolist()
            for b,i,c,k in zip(boxes,ids,confs,clss):
                if i is None: continue
                name=str(names.get(int(k),k) if isinstance(names,dict) else names[int(k)])
                rows.append({'box':tuple(map(float,b)),'track_id':int(i),'confidence':float(c),'class_id':int(k),'class_name':name,'desc':crop_desc(p,b),'idx':idx})
        pred[fr]=rows
    return pred

def center(b): return np.array([(b[0]+b[2])/2,(b[1]+b[3])/2],float)
def area(b): return max((b[2]-b[0])*(b[3]-b[1]),1.0)
def cosine(a,b): return float(np.dot(a,b)/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-9)))
def compatible(a,b): return a['class_id']==b['class_id'] or a['class_name'].lower()==b['class_name'].lower()

def fragment_memory(pred):
    tracks=defaultdict(list)
    for fr,rows in pred.items():
        for r in rows: tracks[r['track_id']].append((fr,r))
    for t in tracks.values(): t.sort(key=lambda z:z[1]['idx'])
    cand=[]
    for aid,A in tracks.items():
        ar=A[-1][1]; ai=ar['idx']; prev=A[-2][1] if len(A)>1 else None
        vel=center(ar['box'])-center(prev['box']) if prev is not None and prev['idx']==ai-1 else np.zeros(2)
        da=np.mean([np.asarray(x[1]['desc']) for x in A[-3:]],axis=0)
        for bid,B in tracks.items():
            if aid==bid: continue
            br=B[0][1]; gap=br['idx']-ai-1
            if gap<1 or gap>MAX_GAP or not compatible(ar,br): continue
            expected=center(ar['box'])+vel*(gap+1); scale=max(math.sqrt(area(ar['box'])),math.sqrt(area(br['box'])),5.0); nd=float(np.linalg.norm(expected-center(br['box']))/scale)
            if nd>MAX_NORM_DIST: continue
            db=np.mean([np.asarray(x[1]['desc']) for x in B[:3]],axis=0); app=max(0.0,min(1.0,cosine(da,db)))
            if app<MIN_APPEARANCE: continue
            ms=math.exp(-nd/2); ss=math.exp(-abs(math.log(area(br['box'])/area(ar['box'])))); score=.50*app+.35*ms+.15*ss
            if score>=MIN_SCORE: cand.append({'from':aid,'to':bid,'gap':gap,'score':score,'appearance':app,'motion_similarity':ms,'size_similarity':ss,'norm_distance':nd})
    cand.sort(key=lambda x:(-x['score'],x['gap'],x['from'],x['to']))
    used_from=set(); used_to=set(); selected=[]
    for c in cand:
        if c['from'] in used_from or c['to'] in used_to: continue
        used_from.add(c['from']);used_to.add(c['to']);selected.append(c)
    parent={x:x for x in tracks}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for c in selected: union(c['from'],c['to'])
    out={}
    for fr,rows in pred.items():
        out[fr]=[{**r,'track_id':find(r['track_id'])} for r in rows]
    return out,selected,cand

def person_like(r):
    n=r.get('class_name','').lower(); return 'swimmer' in n or 'person' in n or 'human' in n

def strip_desc(pred):
    return {fr:[{k:v for k,v in r.items() if k not in ('desc','idx')} for r in rows] for fr,rows in pred.items()}

def main():
    annp=S.ASSET/'instances_val.json'; S.dl(S.ANN_URL,annp); data=json.load(open(annp)); cats={c['id']:c['name'] for c in data.get('categories',[])}; ignored={k for k,v in cats.items() if 'ignore' in v.lower()}; gt=defaultdict(list)
    for a in data['annotations']:
        if a.get('category_id') not in ignored: gt[a['image_id']].append({'box':S.xywh(a['bbox']),'track_id':a['track_id'],'category_id':a.get('category_id')})
    chosen,top=choose_holdout(data['images'],gt); window=chosen['images']; acquired=S.acquire(window); pmap={x['image_id']:pathlib.Path(x['path']) for x in acquired}; paths=[(im['id'],pmap[im['id']]) for im in window]; frames=[x[0] for x in paths]
    from huggingface_hub import hf_hub_download
    weights=hf_hub_download(repo_id=S.MODEL_REPO,filename=S.MODEL_FILE); base=infer(paths,weights); mem,selected,allc=fragment_memory(base)
    base_clean=strip_desc(base); mem_clean=strip_desc(mem); bm=S.mot_metrics(base_clean,gt,frames); mm=S.mot_metrics(mem_clean,gt,frames)
    pgt={fr:[g for g in gt.get(fr,[]) if g['category_id'] in PERSON] for fr in frames}; bp={fr:[r for r in base_clean.get(fr,[]) if person_like(r)] for fr in frames}; mp={fr:[r for r in mem_clean.get(fr,[]) if person_like(r)] for fr in frames}; bpm=S.mot_metrics(bp,pgt,frames); mpm=S.mot_metrics(mp,pgt,frames)
    delta={'idf1':mm['idf1']-bm['idf1'],'mota':mm['mota']-bm['mota'],'id_switches':mm['id_switches']-bm['id_switches'],'person_idf1':mpm['idf1']-bpm['idf1'],'person_id_switches':mpm['id_switches']-bpm['id_switches']}
    passed=(delta['idf1']>=.01 or delta['id_switches']<=-1) and delta['mota']>=-1e-12 and delta['person_idf1']>=-1e-12
    pack={'schema':'assetgraph-evidence/identity-memory-v1','request_id':'REQ-008-ID','protocol':{'lockbox_selection':'highest structural GT challenge on a video different from Cycle07 video_id=21; no model output used','excluded_video_id':EXCLUDE_VIDEO,'window':WINDOW,'stride':STRIDE,'detector_tracker_frozen':True,'confidence':S.CONF,'memory_uses_gt':False,'memory_features':['pixel crop descriptor','motion extrapolation','scale continuity','class compatibility'],'memory_thresholds':{'max_gap':MAX_GAP,'max_norm_distance':MAX_NORM_DIST,'min_appearance':MIN_APPEARANCE,'min_score':MIN_SCORE}},'dataset':{'name':'SeaDronesSee MOT','split':'validation reproduction','selected_window':{k:v for k,v in chosen.items() if k!='images'},'top20_structural_candidates':top,'annotation_sha256':S.sha_file(annp)},'model':{'repo':S.MODEL_REPO,'license':'AGPL-3.0 benchmark-only','weights_sha256':S.sha_file(weights)},'baseline':bm,'identity_memory':mm,'person_baseline':bpm,'person_identity_memory':mpm,'delta':delta,'selected_fragment_merges':selected,'candidate_merge_count':len(allc),'pass_rule':passed,'promotion_rule':'IDF1 +1pp OR >=1 fewer ID switch; no MOTA degradation; no person-IDF1 degradation'}
    p=OUT/'identity_memory_v1_lockbox.json';p.write_text(json.dumps(pack,indent=2));print(json.dumps({'pass':passed,'window':pack['dataset']['selected_window'],'baseline':bm,'memory':mm,'person_base':bpm,'person_memory':mpm,'delta':delta,'merges':selected},indent=2))
if __name__=='__main__':main()
