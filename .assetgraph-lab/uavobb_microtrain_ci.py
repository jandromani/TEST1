from __future__ import annotations
import hashlib,json,math,os,pathlib,requests,shutil,time,zipfile
ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; WORK=ROOT/'assets'/'uavobb_train'; OUT.mkdir(exist_ok=True); WORK.mkdir(parents=True,exist_ok=True)
FILE_ID='1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7'; URL=f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'; EXPECTED=601733395
TRAIN_N=256; EPOCHS=2; IMGSZ=512; CONF=.25

def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def download():
 p=WORK/'UAV-OBB.zip'
 if p.exists() and p.stat().st_size==EXPECTED:return p
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
def label_for(img,root,split):
 return root/split/'labels'/(img.stem+'.txt')
def make_micro(root):
 import yaml
 m=WORK/'micro'
 if m.exists():shutil.rmtree(m)
 for sp in ('train','valid','test'):
  (m/sp/'images').mkdir(parents=True,exist_ok=True);(m/sp/'labels').mkdir(parents=True,exist_ok=True)
  imgs=sorted((root/sp/'images').glob('*.jpg'))
  if sp=='train':imgs=imgs[:TRAIN_N]
  for im in imgs:
   lab=label_for(im,root,sp)
   if not lab.exists():raise RuntimeError(f'missing label {lab}')
   os.symlink(im.resolve(),m/sp/'images'/im.name);os.symlink(lab.resolve(),m/sp/'labels'/lab.name)
 src=yaml.safe_load((root/'data.yaml').read_text())
 data={'path':str(m.resolve()),'train':'train/images','val':'valid/images','test':'test/images','names':src.get('names',src.get('nc'))}
 (m/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
 return m,data

def metrics_obj(res):
 d={}
 rd=getattr(res,'results_dict',{}) or {}
 for k,v in rd.items():
  try:d[str(k)]=float(v)
  except:pass
 # canonical summary, robust to OBB key naming
 def pick(parts):
  for k,v in d.items():
   lk=k.lower()
   if all(x in lk for x in parts):return v
  return None
 return {'all':d,'precision':pick(['precision']),'recall':pick(['recall']),'map50':pick(['map50']) if pick(['map50-95']) is None else next((v for k,v in d.items() if 'map50' in k.lower() and 'map50-95' not in k.lower()),None),'map50_95':pick(['map50-95'])}
def count_mae(model,root):
 vals=[];rows=[]
 for im in sorted((root/'test'/'images').glob('*.jpg')):
  gt=sum(1 for x in open(root/'test'/'labels'/(im.stem+'.txt')) if x.strip())
  r=model.predict(str(im),conf=CONF,imgsz=IMGSZ,device='cpu',verbose=False)[0]
  pred=len(r.obb) if getattr(r,'obb',None) is not None else len(r.boxes)
  vals.append(abs(pred-gt));rows.append({'image':im.name,'gt':gt,'pred':pred,'abs_error':abs(pred-gt)})
 return {'mae':sum(vals)/len(vals) if vals else None,'rows':rows}
def main():
 from ultralytics import YOLO
 t0=time.time();z=download();archive_sha=sha_file(z);root=extract(z);micro,data=make_micro(root)
 counts={sp:{'images':len(list((root/sp/'images').glob('*.jpg'))),'labels':len(list((root/sp/'labels').glob('*.txt')))} for sp in ('train','valid','test')};micro_counts={sp:len(list((micro/sp/'images').glob('*.jpg'))) for sp in ('train','valid','test')}
 base=YOLO('yolo11n-obb.pt');base_weights=pathlib.Path(base.ckpt_path);base_sha=sha_file(base_weights)
 bv=metrics_obj(base.val(data=str(micro/'data.yaml'),split='val',imgsz=IMGSZ,batch=8,device='cpu',workers=2,verbose=False,plots=False));bt=metrics_obj(base.val(data=str(micro/'data.yaml'),split='test',imgsz=IMGSZ,batch=8,device='cpu',workers=2,verbose=False,plots=False));bc=count_mae(base,micro)
 train=YOLO('yolo11n-obb.pt');tr=train.train(data=str(micro/'data.yaml'),epochs=EPOCHS,imgsz=IMGSZ,batch=8,device='cpu',workers=2,project=str(WORK/'runs'),name='uavobb_micro',exist_ok=True,plots=False,verbose=False,cache=False,seed=17,deterministic=True,patience=0)
 best=pathlib.Path(tr.save_dir)/'weights'/'best.pt'
 if not best.exists():raise RuntimeError('best.pt not produced')
 tuned=YOLO(str(best));tv=metrics_obj(tuned.val(data=str(micro/'data.yaml'),split='val',imgsz=IMGSZ,batch=8,device='cpu',workers=2,verbose=False,plots=False));tt=metrics_obj(tuned.val(data=str(micro/'data.yaml'),split='test',imgsz=IMGSZ,batch=8,device='cpu',workers=2,verbose=False,plots=False));tc=count_mae(tuned,micro)
 bmap=bt.get('map50');tmap=tt.get('map50');map_delta=None if bmap is None or tmap is None else tmap-bmap;count_delta=None if bc['mae'] is None or tc['mae'] is None else tc['mae']-bc['mae']
 promotion=bool(map_delta is not None and count_delta is not None and map_delta>=-.01 and count_delta<=0 and (map_delta>=.005 or count_delta<0))
 pack={'schema':'assetgraph-evidence/uavobb-microtrain-v1','request_id':'REQ-OVERHEAD-TRAIN','dataset':{'name':'UAV-OBB','license':'CC BY 4.0','drive_file_id':FILE_ID,'archive_bytes':z.stat().st_size,'archive_sha256':archive_sha,'published_split_counts':counts,'micro_split_counts':micro_counts,'label_format':'YOLO Oriented Bounding Box','data_yaml':data},'protocol':{'train_subset_rule':f'first {TRAIN_N} lexicographically sorted training images only','validation':'complete published valid split','test':'complete published test split; never used for training','epochs':EPOCHS,'imgsz':IMGSZ,'batch':8,'seed':17,'fixed_count_conf':CONF,'note':'This is a deterministic micro-finetune capability test, not a full-dataset SOTA benchmark.'},'model':{'base':'yolo11n-obb.pt','framework_license':'AGPL-3.0 benchmark-only','base_sha256':base_sha,'fine_tuned_sha256':sha_file(best),'fine_tuned_bytes':best.stat().st_size,'product_candidate':False},'baseline':{'val':bv,'test':bt,'test_count':bc},'fine_tuned':{'val':tv,'test':tt,'test_count':tc},'delta':{'test_map50':map_delta,'test_count_mae':count_delta},'technical_pipeline_pass':True,'model_promotion_pass':promotion,'promotion_rule':'test mAP50 no worse by >1pp AND count MAE non-worse AND either mAP50 +0.5pp or count MAE improves','elapsed_seconds':time.time()-t0}
 p=OUT/'uavobb_microtrain.json';p.write_text(json.dumps(pack,indent=2));shutil.copy2(best,OUT/'uavobb_micro_best.pt');print(json.dumps({'pipeline_pass':True,'promotion_pass':promotion,'counts':counts,'micro':micro_counts,'baseline_test':bt,'tuned_test':tt,'baseline_count_mae':bc['mae'],'tuned_count_mae':tc['mae'],'delta':pack['delta'],'best':str(best),'elapsed_seconds':pack['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
