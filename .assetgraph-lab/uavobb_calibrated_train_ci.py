from __future__ import annotations
import hashlib,json,os,pathlib,requests,shutil,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; WORK=ROOT/'assets'/'uavobb_calibrated'; OUT.mkdir(exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
FILE_ID='1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7'; URL=f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'; EXPECTED=601733395
EPOCHS=3; IMGSZ=512; BATCH=8; SEED=23; CONF_GRID=(.01,.02,.03,.05,.08,.10,.15,.20,.30,.40,.50)

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
 if p.stat().st_size!=EXPECTED:raise RuntimeError(f'archive size {p.stat().st_size} != {EXPECTED}')
 return p
def extract(p):
 ex=WORK/'extracted'
 if not (ex/'UAV-OBB').exists():
  with zipfile.ZipFile(p) as z:z.extractall(ex)
 return ex/'UAV-OBB'
def dataset(root):
 import yaml
 src=yaml.safe_load((root/'data.yaml').read_text())
 data={'path':str(root.resolve()),'train':'train/images','val':'valid/images','test':'test/images','names':src['names']}
 p=WORK/'data.yaml';p.write_text(yaml.safe_dump(data,sort_keys=False));return p,data
def mobj(res):
 d={}
 for k,v in (getattr(res,'results_dict',{}) or {}).items():
  try:d[str(k)]=float(v)
  except:pass
 def exact(term):
  return next((v for k,v in d.items() if term.lower() in k.lower()),None)
 m50=next((v for k,v in d.items() if 'map50' in k.lower() and 'map50-95' not in k.lower()),None)
 return {'precision':exact('precision'),'recall':exact('recall'),'map50':m50,'map50_95':exact('map50-95'),'all':d}
def gt_counts(root,split):
 out=[]
 for im in sorted((root/split/'images').glob('*.jpg')):
  lab=root/split/'labels'/(im.stem+'.txt');out.append((im,sum(1 for x in open(lab,encoding='utf-8') if x.strip())))
 return out
def count_eval(model,pairs,conf):
 src=[str(p) for p,_ in pairs]; results=model.predict(src,conf=conf,imgsz=IMGSZ,batch=BATCH,device='cpu',verbose=False)
 rows=[];errs=[]
 for (p,gt),r in zip(pairs,results):
  pred=len(r.obb) if getattr(r,'obb',None) is not None else len(r.boxes);e=abs(pred-gt);errs.append(e);rows.append({'image':p.name,'gt':gt,'pred':pred,'abs_error':e})
 return {'conf':conf,'mae':sum(errs)/len(errs) if errs else None,'rows':rows}
def calibrate(model,pairs):
 vals=[count_eval(model,pairs,c) for c in CONF_GRID]; vals.sort(key=lambda x:(x['mae'],x['conf']));return vals[0],[{'conf':x['conf'],'mae':x['mae']} for x in vals]
def main():
 from ultralytics import YOLO
 t0=time.time();z=download();root=extract(z);dp,data=dataset(root);counts={s:len(list((root/s/'images').glob('*.jpg'))) for s in ('train','valid','test')}
 valid_pairs=gt_counts(root,'valid');test_pairs=gt_counts(root,'test')
 base=YOLO('yolo11n-obb.pt');base_sha=sha_file(pathlib.Path(base.ckpt_path));bval=mobj(base.val(data=str(dp),split='val',imgsz=IMGSZ,batch=BATCH,device='cpu',workers=2,plots=False,verbose=False));btest=mobj(base.val(data=str(dp),split='test',imgsz=IMGSZ,batch=BATCH,device='cpu',workers=2,plots=False,verbose=False));bcal,bcurve=calibrate(base,valid_pairs);bcount=count_eval(base,test_pairs,bcal['conf'])
 train=YOLO('yolo11n-obb.pt');tr=train.train(data=str(dp),epochs=EPOCHS,imgsz=IMGSZ,batch=BATCH,device='cpu',workers=2,project=str(WORK/'runs'),name='uavobb_full3',exist_ok=True,plots=False,verbose=False,cache=False,seed=SEED,deterministic=True,patience=0)
 best=pathlib.Path(tr.save_dir)/'weights'/'best.pt'
 if not best.exists():raise RuntimeError('best.pt missing')
 tuned=YOLO(str(best));tval=mobj(tuned.val(data=str(dp),split='val',imgsz=IMGSZ,batch=BATCH,device='cpu',workers=2,plots=False,verbose=False));ttest=mobj(tuned.val(data=str(dp),split='test',imgsz=IMGSZ,batch=BATCH,device='cpu',workers=2,plots=False,verbose=False));tcal,tcurve=calibrate(tuned,valid_pairs);tcount=count_eval(tuned,test_pairs,tcal['conf'])
 dmap=(ttest['map50'] or 0)-(btest['map50'] or 0);dcount=tcount['mae']-bcount['mae'];uplift=dmap>=.05 and dcount<=0;deployment=(ttest['map50'] or 0)>=.50 and (ttest['recall'] or 0)>=.50 and dcount<=0
 pack={'schema':'assetgraph-evidence/uavobb-calibrated-training-v1','request_id':'REQ-OVERHEAD-TRAIN-CAL','dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':sha_file(z),'archive_bytes':z.stat().st_size,'counts':counts,'label_format':'YOLOv8 OBB'},'protocol':{'training_images':'all 1383 published train images','validation':'all 218 valid; used for confidence calibration only','test':'all 16 test; frozen until final scoring','epochs':EPOCHS,'imgsz':IMGSZ,'batch':BATCH,'seed':SEED,'count_conf_grid':CONF_GRID,'calibration_rule':'choose minimum count MAE on validation separately for baseline and tuned model','capability_uplift_gate':'test mAP50 +5pp AND calibrated test count MAE non-worse','deployment_gate':'test mAP50 >=0.50 AND recall >=0.50 AND calibrated count MAE non-worse','note':'Test is not used for training or confidence selection.'},'model':{'base':'yolo11n-obb.pt','framework_license':'AGPL-3.0 benchmark-only','base_sha256':base_sha,'fine_tuned_sha256':sha_file(best),'fine_tuned_bytes':best.stat().st_size,'product_candidate':False},'baseline':{'val':bval,'test':btest,'validation_count_curve':bcurve,'selected_count_conf':bcal['conf'],'validation_count_mae':bcal['mae'],'test_count':bcount},'fine_tuned':{'val':tval,'test':ttest,'validation_count_curve':tcurve,'selected_count_conf':tcal['conf'],'validation_count_mae':tcal['mae'],'test_count':tcount},'delta':{'test_map50':dmap,'test_count_mae':dcount},'technical_pipeline_pass':True,'capability_uplift_pass':uplift,'deployment_promotion_pass':deployment,'elapsed_seconds':time.time()-t0}
 (OUT/'uavobb_calibrated_training.json').write_text(json.dumps(pack,indent=2));shutil.copy2(best,OUT/'uavobb_calibrated_best.pt');print(json.dumps({'pipeline':True,'capability_uplift_pass':uplift,'deployment_promotion_pass':deployment,'counts':counts,'baseline_test':btest,'tuned_test':ttest,'baseline_conf':bcal['conf'],'tuned_conf':tcal['conf'],'baseline_count_mae':bcount['mae'],'tuned_count_mae':tcount['mae'],'delta':pack['delta'],'elapsed_seconds':pack['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
