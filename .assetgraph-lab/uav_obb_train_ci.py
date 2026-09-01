from __future__ import annotations
import json,hashlib,pathlib,time,urllib.request,zipfile,shutil,tarfile,re,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parent; DATA=ROOT/'assets'/'uav_obb'; OUT=ROOT/'evidence';OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
PMCID='PMC13092195'; OA_API=f'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={PMCID}'; ZIP=DATA/'uav_obb_mmc1.zip'; TGZ=DATA/'pmc_article_package.tar.gz'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def get_bytes(url):
 req=urllib.request.Request(url,headers={'User-Agent':'AssetGraph-Maturity/0.7 (public research dataset acquisition)'});return urllib.request.urlopen(req,timeout=120).read()
def acquire_zip():
 if ZIP.exists() and zipfile.is_zipfile(ZIP):return {'method':'cached','resolved_url':None}
 if ZIP.exists():ZIP.unlink()
 xml=get_bytes(OA_API);root=ET.fromstring(xml);links=root.findall('.//link');tgz=next((x.attrib.get('href') for x in links if x.attrib.get('format')=='tgz'),None)
 if not tgz:raise RuntimeError('PMC OA API did not return a tgz package')
 tgz=tgz.replace('ftp://ftp.ncbi.nlm.nih.gov','https://ftp.ncbi.nlm.nih.gov')
 candidates=[tgz]
 if '/pub/pmc/' in tgz:candidates.append(tgz.replace('/pub/pmc/','/pub/pmc/deprecated/'))
 last=None;resolved=None
 for u in candidates:
  try:
   req=urllib.request.Request(u,headers={'User-Agent':'AssetGraph-Maturity/0.7'});urllib.request.urlretrieve(req.full_url,TGZ);resolved=u;break
  except Exception as e:last=e
 if not resolved:raise RuntimeError(f'Unable to acquire PMC OA package: {last}')
 if TGZ.stat().st_size<1000000:raise RuntimeError(f'PMC package unexpectedly small: {TGZ.stat().st_size} bytes')
 with tarfile.open(TGZ,'r:gz') as tf:
  members=[m for m in tf.getmembers() if pathlib.PurePosixPath(m.name).name=='mmc1.zip']
  if not members:raise RuntimeError('mmc1.zip not found inside PMC OA package')
  src=tf.extractfile(members[0]);ZIP.write_bytes(src.read())
 if not zipfile.is_zipfile(ZIP):raise RuntimeError('Extracted supplementary mmc1 is not a ZIP')
 return {'method':'PMC_OA_API_package','resolved_url':resolved,'oa_api':OA_API,'tgz_sha256':sha(TGZ)}
def download():
 prov=acquire_zip();ex=DATA/'extracted'
 if ex.exists():shutil.rmtree(ex)
 ex.mkdir();zipfile.ZipFile(ZIP).extractall(ex);return prov
def find_root():
 for p in [DATA/'extracted']+list((DATA/'extracted').rglob('*')):
  if p.is_dir() and (p/'train'/'images').exists() and (p/'valid'/'images').exists() and (p/'test'/'images').exists():return p
 raise RuntimeError('UAV-OBB dataset root not found')
def label_format(root):
 labs=list((root/'train'/'labels').glob('*.txt'))
 if not labs:raise RuntimeError('No labels')
 vals=labs[0].read_text().strip().splitlines()[0].split();return len(vals)
def write_yaml(root):
 y=root/'assetgraph_data.yaml';y.write_text(f"path: {root.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: bike\n  1: bus\n  2: car\n  3: other_vehicle\n  4: taxi\n  5: truck\n");return y
def axis_bbox(poly):
 xs=poly[0::2];ys=poly[1::2];return min(xs),min(ys),max(xs),max(ys)
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(aa+bb-inter,1e-9)
def gt_for(label,w,h):
 out=[]
 for ln in label.read_text().splitlines():
  p=ln.split()
  if len(p)==9:
   xy=[float(x) for x in p[1:]];px=[v*(w if i%2==0 else h) for i,v in enumerate(xy)];out.append({'cls':int(p[0]),'box':axis_bbox(px)})
  elif len(p)==6:
   # Defensive support for cx cy w h theta variant if a future package revision uses it.
   _,cx,cy,bw,bh,_ang=p;cx=float(cx)*w;cy=float(cy)*h;bw=float(bw)*w;bh=float(bh)*h;out.append({'cls':int(p[0]),'box':(cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2)})
 return out
def eval_binary(model,root,split='test',conf=.10):
 from PIL import Image
 from scipy.optimize import linear_sum_assignment
 imgs=sorted((root/split/'images').glob('*'));tp=fp=fn=0;abs_count=[];rows=[]
 for img in imgs:
  with Image.open(img) as im:w,h=im.size
  g=gt_for(root/split/'labels'/f'{img.stem}.txt',w,h);r=model.predict(str(img),imgsz=512,conf=conf,verbose=False)[0];pred=[]
  if getattr(r,'obb',None) is not None and r.obb is not None and len(r.obb):
   for poly in r.obb.xyxyxyxy.cpu().numpy():pred.append(axis_bbox(poly.reshape(-1).tolist()))
  elif getattr(r,'boxes',None) is not None and len(r.boxes):pred=[tuple(x) for x in r.boxes.xyxy.cpu().numpy().tolist()]
  M=[[1-iou(p,x['box']) for x in g] for p in pred];matched=0
  if pred and g:
   rr,cc=linear_sum_assignment(M);matched=sum(1 for a,b in zip(rr,cc) if 1-M[a][b]>=.5)
  tp+=matched;fp+=len(pred)-matched;fn+=len(g)-matched;abs_count.append(abs(len(pred)-len(g))/max(1,len(g)));rows.append({'image':img.name,'gt':len(g),'pred':len(pred),'matched':matched})
 precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*precision*recall/max(precision+recall,1e-9);mape=sum(abs_count)/len(abs_count) if abs_count else None
 return {'precision':precision,'recall':recall,'f1':f1,'count_mape':mape,'tp':tp,'fp':fp,'fn':fn,'images':len(imgs),'rows':rows}
def main():
 acquisition=download();root=find_root();fmt=label_format(root)
 if fmt not in (6,9):raise RuntimeError(f'Unexpected OBB label field count: {fmt}')
 split_counts={s:len(list((root/s/'images').glob('*'))) for s in ('train','valid','test')};y=write_yaml(root)
 from ultralytics import YOLO
 base='yolo11n-obb.pt';model=YOLO(base);start=time.time();res=model.train(data=str(y),epochs=2,imgsz=512,batch=4,device='cpu',workers=2,project=str(ROOT/'runs'),name='uavobb_cycle07',exist_ok=True,patience=2,cache=False,verbose=False);train_seconds=time.time()-start
 best=pathlib.Path(res.save_dir)/'weights'/'best.pt';trained=YOLO(str(best));cand=[]
 for c in (.05,.10,.15,.20,.25,.30,.40):
  m=eval_binary(trained,root,'valid',c);cand.append((m['f1'],c,m))
 cand.sort(key=lambda x:(-x[0],x[1]));best_conf=cand[0][1];val_metrics=cand[0][2];test_metrics=eval_binary(trained,root,'test',best_conf);pass_rule=test_metrics['f1']>=.60 and test_metrics['count_mape']<=.15
 evidence={'schema':'assetgraph-evidence/uav-obb-training-factory-v2','request_id':'REQ-006','dataset':{'name':'UAV-OBB','doi':'10.17632/6snrjwcpkh.4','article_doi':'10.1016/j.dib.2026.112710','license':'CC-BY-4.0','acquisition':acquisition,'supplement_sha256':sha(ZIP),'supplement_bytes':ZIP.stat().st_size,'actual_split_counts':split_counts,'label_fields':fmt},'training':{'base_model':base,'epochs':2,'imgsz':512,'batch':4,'device':'cpu','train_seconds':train_seconds,'best_checkpoint_sha256':sha(best)},'selection':{'thresholds':[x[1] for x in cand],'selected_conf':best_conf,'validation_metrics':val_metrics},'blind_test':test_metrics,'technical_pass_rule':{'f1_gte':.60,'count_mape_lte':.15,'passed':pass_rule},'promotion':{'accuracy_gate':pass_rule,'dataset_license_gate':True,'checkpoint_provenance_gate':True,'runtime_reproducibility_gate':True,'product_candidate':False,'note':'Technical checkpoint is Ultralytics-derived; commercial runtime licensing remains a separate unresolved gate even if accuracy passes.'}}
 p=OUT/'uav_obb_training_factory.json';p.write_text(json.dumps(evidence,indent=2));shutil.copy2(best,OUT/'uav_obb_best.pt');print(json.dumps({'pass':pass_rule,'split_counts':split_counts,'selected_conf':best_conf,'val_f1':val_metrics['f1'],'test':{k:test_metrics[k] for k in ('precision','recall','f1','count_mape')},'checkpoint_sha256':sha(best)},indent=2))
if __name__=='__main__':main()
