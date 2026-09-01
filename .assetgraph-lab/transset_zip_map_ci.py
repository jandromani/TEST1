from __future__ import annotations
import collections,json,pathlib,requests,time
from remotezip import RemoteZip
OUT=pathlib.Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
ARTICLE=26082217

def main():
 t=time.time(); m=requests.get(f'https://api.figshare.com/v2/articles/{ARTICLE}',timeout=30).json(); f=next(x for x in m['files'] if x['name']=='TRANSSET_v2.zip'); url=f['download_url']
 with RemoteZip(url) as z:
  names=z.namelist(); ext=collections.Counter(pathlib.PurePosixPath(n).suffix.lower() or '<dir>' for n in names); tops=collections.Counter(n.split('/')[0] for n in names if n); second=collections.Counter('/'.join(n.split('/')[:2]) for n in names if '/' in n); samples=names[:100]
  text=[]
  for n in names:
   low=n.lower()
   if low.endswith(('readme.md','readme.txt','.json','.yaml','.yml')) and len(text)<12:
    try:
     b=z.read(n)
     if len(b)<5_000_000:text.append({'name':n,'bytes':len(b),'preview':b[:1200].decode('utf-8','replace')})
    except Exception as e:text.append({'name':n,'error':str(e)[:120]})
 report={'schema':'assetgraph-evidence/transset-remote-map-v1','article_id':ARTICLE,'file':{'id':f['id'],'name':f['name'],'size':f['size'],'md5':f.get('computed_md5') or f.get('supplied_md5'),'download_url':url},'member_count':len(names),'extensions':dict(ext),'top_level':dict(tops),'second_level':dict(second),'first_members':samples,'small_metadata_files':text,'lockbox_policy':{'training_allowed_before_first_score':False,'selection_must_use_structure_only':True},'map_pass':len(names)>1000 and ext.get('.jpg',0)>1000,'elapsed_seconds':time.time()-t}
 (OUT/'transset_remote_map.json').write_text(json.dumps(report,indent=2));print(json.dumps({'members':len(names),'extensions':dict(ext),'top_level':dict(tops),'second_level_top':second.most_common(20),'metadata':[x['name'] for x in text],'map_pass':report['map_pass']},indent=2))
if __name__=='__main__':main()
