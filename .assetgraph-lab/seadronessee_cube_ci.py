from __future__ import annotations
import json,hashlib,pathlib,shutil
from PIL import Image,ImageDraw
from huggingface_hub import HfApi,hf_hub_download
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'evidence'/'seadronessee_cc0_cube';RAW=OUT/'raw';ANN=OUT/'annotated';RAW.mkdir(parents=True,exist_ok=True);ANN.mkdir(parents=True,exist_ok=True)
REPO='dronefreak/SeaDronesSee';TARGET=[f'{i}.jpg' for i in range(10119,10127)];NAMES=['swimmer','boat','jetski','life_saving_appliances','buoy']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
files=HfApi().list_repo_files(REPO,repo_type='dataset');bybase={}
for p in files:bybase.setdefault(pathlib.PurePosixPath(p).name,[]).append(p)
frames=[];thumbs=[]
for fn in TARGET:
 cand=[p for p in bybase.get(fn,[]) if '/images/train/' in p]
 if not cand:raise RuntimeError(f'image missing {fn}')
 ipath=sorted(cand)[0];shard=pathlib.PurePosixPath(ipath).parent.name;stem=pathlib.PurePosixPath(fn).stem;labs=[p for p in files if p.endswith(f'/labels/train/{shard}/{stem}.txt') or (p.endswith(f'/{stem}.txt') and '/labels/train/' in p)]
 if not labs:raise RuntimeError(f'label missing {fn}')
 lpath=sorted(labs)[0];src_img=pathlib.Path(hf_hub_download(REPO,ipath,repo_type='dataset'));src_lab=pathlib.Path(hf_hub_download(REPO,lpath,repo_type='dataset'));dst_img=RAW/fn;dst_lab=RAW/f'{stem}.txt';shutil.copy2(src_img,dst_img);shutil.copy2(src_lab,dst_lab);im=Image.open(dst_img).convert('RGB');w,h=im.size;draw=ImageDraw.Draw(im);objects=[]
 for line in dst_lab.read_text().splitlines():
  if not line.strip():continue
  c,xc,yc,bw,bh=map(float,line.split()[:5]);c=int(c);x=(xc-bw/2)*w;y=(yc-bh/2)*h;ww=bw*w;hh=bh*h;name=NAMES[c] if 0<=c<len(NAMES) else str(c);objects.append({'class_id':c,'class_name':name,'bbox_xywh_px':[x,y,ww,hh]});draw.rectangle((x,y,x+ww,y+hh),outline='white',width=max(2,w//1600));draw.text((x,max(0,y-18)),name,fill='white')
 im.save(ANN/fn,quality=90);t=im.copy();t.thumbnail((640,360));thumbs.append((fn,t.copy()));frames.append({'file_name':fn,'hf_image_path':ipath,'hf_label_path':lpath,'original_width':w,'original_height':h,'image_sha256':sha(dst_img),'label_sha256':sha(dst_lab),'objects':objects})
cellw=max(t.width for _,t in thumbs);cellh=max(t.height for _,t in thumbs);sheet=Image.new('RGB',(cellw*4,(cellh+30)*2));d=ImageDraw.Draw(sheet)
for idx,(fn,t) in enumerate(thumbs):
 x=(idx%4)*cellw;y=(idx//4)*(cellh+30);sheet.paste(t,(x,y));d.text((x+6,y+cellh+6),fn,fill='white')
sheet.save(OUT/'contact_sheet.jpg',quality=88)
manifest={'schema':'assetgraph-evidence-cube/v1','dataset':'SeaDronesSee Object Detection v2','source_repo':REPO,'license':'CC0-1.0','license_source':'Official SeaDronesSee repository declares dataset CC0 1.0; mirror preserves CC0','frames':frames,'sequence_note':'Contiguous numeric source IDs with coherent annotations. OD-v2 mirror does not expose persistent track IDs or clip provenance, so identity continuity is NOT claimed.','derivatives':{'annotated_dir':'annotated/','contact_sheet':'contact_sheet.jpg'},'distribution_status':'CLEAR_FOR_DEMO_DATA_CC0','sha256':{}}
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='cube_manifest.json':manifest['sha256'][str(p.relative_to(OUT))]=sha(p)
(OUT/'cube_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'frames':len(frames),'objects':sum(len(f['objects']) for f in frames),'distribution_status':manifest['distribution_status']},indent=2))