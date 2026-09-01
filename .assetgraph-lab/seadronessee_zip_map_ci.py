from __future__ import annotations
import json,pathlib,re
from collections import Counter
from remotezip import RemoteZip
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(parents=True,exist_ok=True)
URL='https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/resolve/main/SeaDronesSee/MOT/SeaDronesSee_MOT_jpg_compressed.zip?download=true'
rz=RemoteZip(URL);names=rz.namelist()
def hits(term,n=100):return [x for x in names if term.lower() in x.lower()][:n]
ext=Counter(pathlib.PurePosixPath(n).suffix.lower() for n in names)
roots=Counter(pathlib.PurePosixPath(n).parts[0] if pathlib.PurePosixPath(n).parts else '' for n in names)
report={'schema':'assetgraph-debug/seadronessee-remotezip-map-v1','archive_url':URL,'member_count':len(names),'first_120':names[:120],'extensions':ext,'top_roots':roots.most_common(30),'queries':{q:hits(q) for q in ['42875','DJI_0003','000770','770','DJI_0003.mov','video_19','val','validation']}}
# Sample image names near likely structures.
report['image_samples']=[n for n in names if pathlib.PurePosixPath(n).suffix.lower() in ('.jpg','.jpeg','.png')][:200]
(OUT/'seadronessee_zip_map.json').write_text(json.dumps(report,indent=2,default=lambda x:dict(x)))
print(json.dumps({'member_count':len(names),'extensions':dict(ext),'top_roots':roots.most_common(10),'queries':report['queries']},indent=2));rz.close()
