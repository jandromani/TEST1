from __future__ import annotations
import json,pathlib,urllib.request,hashlib,os
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(parents=True,exist_ok=True)
# This probe deliberately does not pull the 29.6GB MOT archive. It attempts to acquire native train/val annotation metadata and records exact blockers for a physically small clip.
HF_BASE='https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/annotations'
FILES=['instances_train_objects_in_water.json','instances_val_objects_in_water.json']
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dl(url,p):
 if not p.exists():urllib.request.urlretrieve(url,p)
assets=ROOT/'assets'/'seadronessee_mot';assets.mkdir(parents=True,exist_ok=True)
report={'schema':'assetgraph-evidence/seadronessee-native-mot-probe-v1','request_id':'REQ-003','license':'CC0-1.0 per official SeaDronesSee release','sources':{},'status':'PROBE'}
anns={}
for fn in FILES:
 p=assets/fn;url=f'{HF_BASE}/{fn}?download=true';dl(url,p);report['sources'][fn]={'url':url,'sha256':sha(p),'bytes':p.stat().st_size};anns[fn]=json.load(open(p))
# COCO-video style: images carry video_id/frame_index/source/meta; annotations may or may not carry track_id/instance_id depending release.
def inspect(name,data):
 images=data.get('images',[]); ann=data.get('annotations',[]); vids=data.get('videos',[])
 keys=sorted({k for a in ann[:5000] for k in a.keys()}); idkeys=[k for k in keys if any(x in k.lower() for x in ('track','instance','object_id','person_id'))]
 byvid={}
 for im in images:
  vid=im.get('video_id');byvid.setdefault(vid,[]).append(im)
 candidates=[]
 for vid,ims in byvid.items():
  ims=sorted(ims,key=lambda x:x.get('frame_index',x.get('id',0)))
  if len(ims)>=12:
   candidates.append({'video_id':vid,'frames':len(ims),'first_image_id':ims[0].get('id'),'last_image_id':ims[-1].get('id'),'source':ims[0].get('source'),'meta':ims[0].get('meta')})
 candidates=sorted(candidates,key=lambda x:-x['frames'])[:10]
 return {'images':len(images),'annotations':len(ann),'videos_declared':len(vids),'annotation_keys':keys,'persistent_id_candidate_keys':idkeys,'top_video_candidates':candidates}
report['train']=inspect('train',anns[FILES[0]]);report['val']=inspect('val',anns[FILES[1]])
# We can only score native MOT locally if annotations expose a persistent ID and RGB frames for one candidate are physically acquired.
has_pid=bool(report['train']['persistent_id_candidate_keys'] or report['val']['persistent_id_candidate_keys'])
report['native_persistent_id_exposed']=has_pid
report['rgb_archive']={'known_archive':'SeaDronesSee_MOT_jpg_compressed.zip','size_bytes':29588897110,'sha256':'53c7a6329e6bd1184b06c1f743943555ae5da7bc974b06c0f8b92c1d57b46dc3','downloaded':False,'reason':'Cycle 07 probe avoids pulling 29.6GB before confirming a small native clip can be isolated.'}
report['pass_rule']=False
report['blockers']=[]
if not has_pid:report['blockers'].append('public train/val COCO annotation JSON does not expose a persistent identity field usable as native MOT ground truth')
report['blockers'].append('RGB frames are inside a 29.6GB compressed archive; no small official per-video archive has yet been identified')
report['next_acquisition']='Identify an official/public per-video or ranged-download source for one train/val video; otherwise use the 29.6GB archive in a dedicated data job and extract one clip.'
p=OUT/'seadronessee_native_mot_probe.json';p.write_text(json.dumps(report,indent=2));print(json.dumps({'native_persistent_id_exposed':has_pid,'train':report['train'],'val':report['val'],'blockers':report['blockers']},indent=2))
