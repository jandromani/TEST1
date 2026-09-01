from __future__ import annotations
import json,pathlib,re,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(parents=True,exist_ok=True)
DATASET='6snrjwcpkh';VERSION=4
urls=[
 f'https://api.data.mendeley.com/datasets/{DATASET}?version={VERSION}',
 f'https://api.data.mendeley.com/datasets/{DATASET}/files?version={VERSION}',
 f'https://api.data.mendeley.com/datasets/{DATASET}/folders?version={VERSION}',
 f'https://data.mendeley.com/datasets/{DATASET}/{VERSION}',
]
def req(url):
 r=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 AssetGraph Maturity Lab','Accept':'application/json,text/html,*/*'})
 try:
  with urllib.request.urlopen(r,timeout=60) as x:
   b=x.read();return {'status':x.status,'final_url':x.geturl(),'headers':dict(x.headers),'bytes':len(b),'text':b[:1000000].decode('utf-8','replace')}
 except urllib.error.HTTPError as e:
  b=e.read();return {'status':e.code,'final_url':url,'headers':dict(e.headers),'bytes':len(b),'text':b[:200000].decode('utf-8','replace')}
 except Exception as e:return {'status':None,'error':repr(e),'text':''}
raw={u:req(u) for u in urls};report={'schema':'assetgraph-debug/mendeley-uavobb-acquisition-v1','dataset':DATASET,'version':VERSION,'requests':{}}
for u,x in raw.items():
 t=x.get('text','');uuids=sorted(set(re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',t)))
 links=sorted(set(re.findall(r'https?://[^"\'<>\\ ]+',t)))
 report['requests'][u]={k:v for k,v in x.items() if k!='text'};report['requests'][u]['uuid_candidates']=uuids[:200];report['requests'][u]['downloadish_links']=[z for z in links if any(k in z.lower() for k in ('download','file','dataset'))][:200];report['requests'][u]['snippet']=t[:6000]
# if public snapshot JSON exposed files, normalize them.
for u,x in raw.items():
 try:
  d=json.loads(x.get('text',''))
  report['requests'][u]['json_top_type']=type(d).__name__
  report['requests'][u]['json']=d
 except Exception:pass
p=OUT/'mendeley_uavobb_acquisition_probe.json';p.write_text(json.dumps(report,indent=2));print(json.dumps({'statuses':{u:r['status'] for u,r in report['requests'].items()},'uuid_counts':{u:len(r.get('uuid_candidates',[])) for u,r in report['requests'].items()},'evidence':str(p)},indent=2))
