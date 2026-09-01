from __future__ import annotations
import collections, hashlib, json, math, pathlib, shutil, time

import uavobb_forensic_ci as F
import uavobb_deployment_repair_ci as C

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
WORK=ROOT/'assets'/'cycle16_recovery'; WORK.mkdir(parents=True,exist_ok=True)
PT=ROOT/'evidence'/'c11'/'uavobb_calibrated_best.pt'
PROTOCOL=ROOT/'cycle16_recovery_protocol.json'
CONF=.30
IOU=.30
TRAIN_IMGSZ=768
EVAL_IMGSZ=1024
SEED=31
MAX_FOCUS=768
RARE={'other_vehicle','truck','bike'}
PROFILE={'name':'single1024','mode':'single','imgsz':1024,'tile':0,'overlap':0}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def poly_area(poly):
    return abs(sum(poly[i][0]*poly[(i+1)%4][1]-poly[(i+1)%4][0]*poly[i][1] for i in range(4)))*.5


def load_norm_labels(path):
    out=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        a=line.split(); cls=int(float(a[0])); v=list(map(float,a[1:9])); pts=[(v[i],v[i+1]) for i in range(0,8,2)]
        out.append({'cls':cls,'pts':pts,'area_frac':poly_area(pts)})
    return out


def clamp_crop(cx,cy,size,w,h):
    size=min(size,w,h); x0=int(round(cx-size/2)); y0=int(round(cy-size/2));
    x0=max(0,min(x0,w-size)); y0=max(0,min(y0,h-size)); return x0,y0,x0+size,y0+size


def inside(pts,x0,y0,x1,y1,margin=1.0):
    return all(x0+margin<=x<=x1-margin and y0+margin<=y<=y1-margin for x,y in pts)


def focus_candidates(root,N):
    from PIL import Image
    cand=[]
    for im in sorted((root/'train'/'images').glob('*.jpg')):
        lab=root/'train'/'labels'/(im.stem+'.txt')
        objs=load_norm_labels(lab)
        if not objs:continue
        with Image.open(im) as q:w,h=q.size
        for oi,o in enumerate(objs):
            name=N[o['cls']]; is_small=o['area_frac']<0.00012; is_rare=name in RARE
            if not (is_small or is_rare):continue
            key=hashlib.sha256(f'{im.name}:{oi}:{name}:{is_small}'.encode()).hexdigest()
            priority=(0 if name in ('other_vehicle','truck') else 1 if is_small else 2, key)
            cand.append({'priority':priority,'image':im,'label':lab,'object_index':oi,'class':name,'small':is_small,'rare':is_rare,'w':w,'h':h,'objects':objs})
    cand.sort(key=lambda x:x['priority'])
    # Deterministic cap, with rare classes naturally first by the predeclared priority.
    return cand[:MAX_FOCUS]


def build_focus(root,N):
    from PIL import Image
    import yaml
    dst=WORK/'focus'; imgs=dst/'images'; labs=dst/'labels'
    if dst.exists():shutil.rmtree(dst)
    imgs.mkdir(parents=True); labs.mkdir(parents=True)
    cand=focus_candidates(root,N); class_kept=collections.Counter(); small_count=0
    for k,c in enumerate(cand):
        im=Image.open(c['image']).convert('RGB'); w,h=im.size; target=c['objects'][c['object_index']]
        pts=[(x*w,y*h) for x,y in target['pts']]; cx=sum(x for x,_ in pts)/4; cy=sum(y for _,y in pts)/4
        crop_size=min(768,w,h)
        x0,y0,x1,y1=clamp_crop(cx,cy,crop_size,w,h); cw=x1-x0; ch=y1-y0
        kept=[]
        for o in c['objects']:
            p=[(x*w,y*h) for x,y in o['pts']]
            if not inside(p,x0,y0,x1,y1):continue
            q=[((x-x0)/cw,(y-y0)/ch) for x,y in p]
            kept.append((o['cls'],q))
        if not kept:continue
        stem=f'focus_{k:04d}_{c["image"].stem}'
        im.crop((x0,y0,x1,y1)).save(imgs/(stem+'.jpg'),quality=95)
        with open(labs/(stem+'.txt'),'w') as f:
            for cls,q in kept:
                f.write(str(cls)+' '+' '.join(f'{v:.8f}' for pt in q for v in pt)+'\n')
                class_kept[N[cls]]+=1
        small_count+=int(c['small'])
    src=yaml.safe_load((root/'data.yaml').read_text())
    focus_yaml=WORK/'focus_data.yaml'
    focus_yaml.write_text(yaml.safe_dump({'path':str(WORK.resolve()),'train':str(imgs.resolve()),'val':str((root/'valid'/'images').resolve()),'test':str((root/'test'/'images').resolve()),'names':src['names']},sort_keys=False))
    full_yaml=WORK/'full_data.yaml'
    full_yaml.write_text(yaml.safe_dump({'path':str(root.resolve()),'train':'train/images','val':'valid/images','test':'test/images','names':src['names']},sort_keys=False))
    return focus_yaml,full_yaml,{'candidate_focus_objects':len(cand),'patches_written':len(list(imgs.glob('*.jpg'))),'focus_labels_by_class':dict(class_kept),'small_focus_objects':small_count}


def val_score(x):
    s=x['size']; small=(s.get('small') or {}).get('recall') or 0; med=(s.get('medium') or {}).get('recall') or 0
    ov=x['other_vehicle_coarse']['recall']; return .40*small+.20*med+.20*ov+.20*x['fine']['f1']


def parity1024(model,onnx,root,nc):
    import numpy as np, torch, onnxruntime as ort
    from ultralytics.utils.nms import non_max_suppression
    net=model.model.eval(); sess=ort.InferenceSession(str(onnx),providers=['CPUExecutionProvider']); inp=sess.get_inputs()[0].name
    matched=ptn=onn=0; rows=[]
    for im in sorted((root/'test'/'images').glob('*.jpg'))[:12]:
        a=F.preprocess_image(im,1024)
        with torch.no_grad():po=net(torch.from_numpy(a))
        if isinstance(po,(tuple,list)):po=po[0]
        if isinstance(po,(tuple,list)):po=po[0]
        oo=sess.run(None,{inp:a.astype('float32')})[0]
        A=non_max_suppression(po,CONF,.70,nc=nc,rotated=True,max_det=500)[0].detach().cpu().numpy()
        B=non_max_suppression(torch.from_numpy(oo),CONF,.70,nc=nc,rotated=True,max_det=500)[0].detach().cpu().numpy()
        ptn+=len(A);onn+=len(B);cand=[]
        for i,x in enumerate(A):
            for j,y in enumerate(B):
                if int(x[5])!=int(y[5]):continue
                dc=math.hypot(float(x[0]-y[0]),float(x[1]-y[1]));ds=abs(float(x[2]-y[2]))+abs(float(x[3]-y[3]));da=abs(float(x[-1]-y[-1]))
                if dc<=2 and ds<=4 and da<=.03:cand.append((dc+.2*ds+5*da,i,j))
        ua=set();ub=set();m=0
        for _,i,j in sorted(cand):
            if i in ua or j in ub:continue
            ua.add(i);ub.add(j);m+=1
        matched+=m;rows.append({'image':im.name,'pt':len(A),'onnx':len(B),'matched':m})
    p=matched/onn if onn else 0;r=matched/ptn if ptn else 0
    return {'pt_total':ptn,'onnx_total':onn,'matched':matched,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'imgsz':1024,'same_postprocess':True,'rows':rows}


def main():
    global_start=time.time()
    from ultralytics import YOLO
    protocol=json.loads(PROTOCOL.read_text())
    assert protocol['frozen_before_transset_h00_score'] is True
    if not PT.exists():raise RuntimeError('Cycle11 checkpoint missing')
    z,root=F.get_corpus();N=F.class_names(root);focus_yaml,full_yaml,focus_stats=build_focus(root,N)

    # Baseline validation, fixed Cycle14 production profile.
    base=YOLO(str(PT));C.MODEL=base;base_val=C.evaluate(root,N,'valid',PROFILE)

    # Stage 1: deterministic object-centered focus patches from TRAIN only.
    stage1=YOLO(str(PT))
    r1=stage1.train(data=str(focus_yaml),epochs=1,imgsz=TRAIN_IMGSZ,batch=4,device='cpu',workers=2,project=str(WORK/'runs'),name='stage1_focus',exist_ok=True,plots=False,verbose=False,cache=False,seed=SEED,deterministic=True,patience=0)
    p1=pathlib.Path(r1.save_dir)/'weights'/'best.pt'
    if not p1.exists():raise RuntimeError('stage1 best.pt missing')

    # Stage 2: consolidate on all original full TRAIN scenes; still no validation/test leakage into training.
    stage2=YOLO(str(p1))
    r2=stage2.train(data=str(full_yaml),epochs=1,imgsz=TRAIN_IMGSZ,batch=4,device='cpu',workers=2,project=str(WORK/'runs'),name='stage2_full',exist_ok=True,plots=False,verbose=False,cache=False,seed=SEED,deterministic=True,patience=0)
    p2=pathlib.Path(r2.save_dir)/'weights'/'best.pt'
    if not p2.exists():raise RuntimeError('stage2 best.pt missing')

    cand=YOLO(str(p2));C.MODEL=cand;cand_val=C.evaluate(root,N,'valid',PROFILE)
    baseline_score=val_score(base_val);candidate_score=val_score(cand_val)
    nonreg=cand_val['fine']['f1']>=base_val['fine']['f1']-.05
    candidate_selected=bool(nonreg and candidate_score>baseline_score)
    selected_path=p2 if candidate_selected else PT
    selected=YOLO(str(selected_path));C.MODEL=selected

    # Test is consulted only after validation-only selection has been frozen.
    selected_test=C.evaluate(root,N,'test',PROFILE)
    onnx=pathlib.Path(selected.export(format='onnx',imgsz=1024,opset=17,dynamic=False,simplify=False,device='cpu',verbose=False))
    parity=parity1024(selected,onnx,root,len(N))
    small=(selected_test['size'].get('small') or {}).get('recall') or 0
    ov=selected_test['other_vehicle_coarse']['recall']; fine=selected_test['fine']['f1']; coarse=selected_test['coarse_vehicle']['f1']
    gates={
      'small_recall_pass':small>=.30,
      'other_vehicle_spatial_recall_pass':ov>=.70,
      'fine_f1_pass':fine>=.56,
      'coarse_vehicle_f1_pass':coarse>=.70,
      'pt_onnx_shared_postprocess_f1_pass':parity['f1']>=.98,
    }
    gates['internal_promotion_pass']=all(gates.values()) and candidate_selected
    report={
      'schema':'assetgraph-evidence/uavobb-small-rare-recovery-v1',
      'protocol_sha256':sha256(PROTOCOL),
      'training_sources':['UAV-OBB train only'],
      'transset_used_for_training':False,
      'transset_used_for_selection':False,
      'dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':F.sha256(z)},
      'focus_dataset':focus_stats,
      'training':{'stage1':{'epochs':1,'imgsz':TRAIN_IMGSZ,'checkpoint_sha256':sha256(p1)},'stage2':{'epochs':1,'imgsz':TRAIN_IMGSZ,'checkpoint_sha256':sha256(p2)}},
      'selection':{'profile':PROFILE,'rule':protocol['candidate_selection'],'baseline_validation':base_val,'candidate_validation':cand_val,'baseline_score':baseline_score,'candidate_score':candidate_score,'candidate_nonregression':nonreg,'selected':'cycle16_candidate' if candidate_selected else 'cycle11_baseline'},
      'frozen_test':selected_test,
      'deployment':{'selected_pt_sha256':sha256(selected_path),'onnx_sha256':sha256(onnx),'onnx_bytes':onnx.stat().st_size,'parity':parity},
      'gates':gates,
      'external_next':'TRANSSET-H01 only if internal_promotion_pass=true',
      'elapsed_seconds':time.time()-global_start,
    }
    (OUT/'cycle16_recovery.json').write_text(json.dumps(report,indent=2))
    shutil.copy2(p2,OUT/'cycle16_candidate.pt');shutil.copy2(onnx,OUT/'cycle16_selected_1024.onnx')
    print(json.dumps({'focus':focus_stats,'baseline_val_score':baseline_score,'candidate_val_score':candidate_score,'candidate_selected':candidate_selected,'test':selected_test,'parity_f1':parity['f1'],'gates':gates},indent=2))

if __name__=='__main__':main()
