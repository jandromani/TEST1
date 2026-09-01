from __future__ import annotations
import hashlib,json,pathlib,requests,time,xml.etree.ElementTree as ET
from remotezip import RemoteZip
OUT=pathlib.Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
ARTICLE=26082217;N=128

def main():
 t=time.time();m=requests.get(f'https://api.figshare.com/v2/articles/{ARTICLE}',timeout=30).json();f=next(x for x in m['files'] if x['name']=='TRANSSET_v2.zip');u=f['download_url']
 with RemoteZip(u) as z:
  ims=sorted([n for n in z.namelist() if n.startswith('TRANSSET_v2/images/') and n.lower().endswith('.jpg')]);ranked=sorted(ims,key=lambda n:(hashlib.sha256(n.encode()).hexdigest(),n));chosen=ranked[:N];rows=[];classes={};total_boxes=0
  for im in chosen:
   stem=pathlib.PurePosixPath(im).stem; xml=f'TRANSSET_v2/label_xml/{stem}.xml'; ib=z.read(im); xb=z.read(xml); root=ET.fromstring(xb);objs=[]
   for o in root.findall('.//object'):
    name=(o.findtext('name') or '').strip(); bb=o.find('bndbox'); box=[float(bb.findtext(k)) for k in ('xmin','ymin','xmax','ymax')] if bb is not None else None;objs.append({'class':name,'bbox':box});classes[name]=classes.get(name,0)+1;total_boxes+=1
   rows.append({'image_member':im,'xml_member':xml,'filename_rank_hash':hashlib.sha256(im.encode()).hexdigest(),'image_sha256':hashlib.sha256(ib).hexdigest(),'xml_sha256':hashlib.sha256(xb).hexdigest(),'image_bytes':len(ib),'objects':objs})
 report={'schema':'assetgraph-evidence/transset-lockbox-v1','dataset':{'article_id':ARTICLE,'name':'TRANSSET','license':'CC BY 4.0','zip_name':f['name'],'zip_size':f['size'],'zip_md5':f.get('computed_md5') or f.get('supplied_md5')},'selection':{'rule':'sort all image member names by SHA256(filename), lexical tie-break; select first N','N':N,'uses_image_content':False,'uses_ground_truth':False,'uses_model_output':False},'selected_count':len(rows),'post_freeze_ground_truth_summary':{'classes':classes,'total_boxes':total_boxes},'items':rows,'lockbox_frozen':len(rows)==N,'elapsed_seconds':time.time()-t}
 (OUT/'transset_lockbox_128.json').write_text(json.dumps(report,indent=2));print(json.dumps({'selected':len(rows),'boxes':total_boxes,'classes':classes,'first':[{'image':x['image_member'],'rank_hash':x['filename_rank_hash']} for x in rows[:5]],'lockbox_frozen':report['lockbox_frozen']},indent=2))
if __name__=='__main__':main()
