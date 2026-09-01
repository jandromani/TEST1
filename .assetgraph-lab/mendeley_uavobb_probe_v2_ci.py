from __future__ import annotations
import json,pathlib,re,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(parents=True,exist_ok=True)
DATASET='6snrjwcpkh';VERSION=4;PAGE=f'https://data.mendeley.com/datasets/{DATASET}/{VERSION}'
def req(url,method='GET',headers=None):
 h={'User-Agent':'Mozilla/5.0 AssetGraph Maturity Lab','Accept':'application/json,text/html,*/*'};h.update(headers or {})
 r=urllib.request.Request(url,headers=h,method=method)
 try:
  op=urllib.request.build_opener(urllib.request.HTTPRedirectHandler());x=op.open(r,timeout=60);b=x.read(2_000_000);return {'status':x.status,'final_url':x.geturl(),'headers':dict(x.headers),'bytes_read':len(b),'text':b.decode('utf-8','replace')}
 except urllib.error.HTTPError as e:
  b=e.read(500000);return {'status':e.code,'final_url':e.geturl(),'headers':dict(e.headers),'bytes_read':len(b),'text':b.decode('utf-8','replace')}
 except Exception as e:return {'status':None,'error':repr(e),'text':''}
page=req(PAGE);html=page.get('text','');uuids=sorted(set(re.findall(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',html)))
contexts={}
for u in uuids:
 pos=[];start=0
 while True:
  i=html.find(u,start)
  if i<0:break
  pos.append(html[max(0,i-700):min(len(html),i+900)]);start=i+len(u)
 contexts[u]=pos[:8]
endpoints={
 'legacy_public_snapshot':f'https://api.mendeley.com/datasets/{DATASET}?version={VERSION}',
 'data_snapshot':f'https://api.data.mendeley.com/datasets/{DATASET}?version={VERSION}',
 'zip_download_data':f'https://api.data.mendeley.com/datasets/{DATASET}/zip/file_downloaded?version={VERSION}',
 'zip_download_legacy':f'https://api.mendeley.com/datasets/{DATASET}/zip/file_downloaded?version={VERSION}',
}
results={k:req(v,headers={'Accept':'application/vnd.mendeley-public-dataset.1+json,application/json,*/*'}) for k,v in endpoints.items()}
file_tests={}
for u in uuids:
 for host in ('https://api.data.mendeley.com','https://api.mendeley.com'):
  url=f'{host}/datasets/{DATASET}/files/{u}/file_downloaded?version={VERSION}'
  r=req(url,headers={'Range':'bytes=0-31'});file_tests[url]={k:v for k,v in r.items() if k!='text'};file_tests[url]['snippet']=r.get('text','')[:300]
report={'schema':'assetgraph-debug/mendeley-uavobb-acquisition-v2','page':{k:v for k,v in page.items() if k!='text'},'uuid_candidates':uuids,'uuid_contexts':contexts,'endpoint_tests':{k:{kk:vv for kk,vv in r.items() if kk!='text'}|{'snippet':r.get('text','')[:1000]} for k,r in results.items()},'file_download_tests':file_tests}
# Try to find serialized file objects in HTML around common field names.
for pat in ['"files"','file_size','filename','fileUuid','file_uuid','Download All','6snrjwcpkh']:
 report.setdefault('pattern_contexts',{})[pat]=[];s=0
 while True:
  i=html.find(pat,s)
  if i<0:break
  report['pattern_contexts'][pat].append(html[max(0,i-1000):min(len(html),i+2500)]);s=i+len(pat)
  if len(report['pattern_contexts'][pat])>=10:break
p=OUT/'mendeley_uavobb_acquisition_probe_v2.json';p.write_text(json.dumps(report,indent=2));print(json.dumps({'uuids':uuids,'endpoints':{k:(r.get('status'),r.get('final_url')) for k,r in results.items()},'download_statuses':{u:r.get('status') for u,r in file_tests.items()},'evidence':str(p)},indent=2))
