from __future__ import annotations
import json,pathlib,urllib.request,urllib.error,zipfile,io,re,hashlib
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'evidence';OUT.mkdir(parents=True,exist_ok=True)
XML_URL='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13092195/fullTextXML'
SUPP_URL='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13092195/supplementaryFiles'
def req(url):
 r=urllib.request.Request(url,headers={'User-Agent':'AssetGraph-Maturity/0.7','Accept':'*/*'})
 try:
  with urllib.request.urlopen(r,timeout=180) as x:b=x.read();return {'status':x.status,'url':x.geturl(),'content_type':x.headers.get('Content-Type'),'bytes':len(b),'body':b}
 except urllib.error.HTTPError as e:return {'status':e.code,'url':e.geturl(),'content_type':e.headers.get('Content-Type'),'bytes':0,'body':e.read()}
 except Exception as e:return {'status':None,'url':url,'error':repr(e),'bytes':0,'body':b''}
def sha(b):return hashlib.sha256(b).hexdigest() if b else None
xml=req(XML_URL);supp=req(SUPP_URL);report={'schema':'assetgraph-debug/uavobb-europepmc-acquisition-v1','fulltext':{k:v for k,v in xml.items() if k!='body'},'supplementary':{k:v for k,v in supp.items() if k!='body'}}
report['fulltext']['sha256']=sha(xml['body']);report['supplementary']['sha256']=sha(supp['body'])
text=xml['body'].decode('utf-8','replace');links=sorted(set(re.findall(r'https?://[^\s"<>]+',text)))
report['fulltext']['google_drive_links']=[x.replace('&amp;','&') for x in links if 'drive.google.com' in x.lower()]
report['fulltext']['mendeley_links']=[x.replace('&amp;','&') for x in links if 'mendeley' in x.lower()]
report['fulltext']['data_availability_context']=[]
for needle in ('Original data','Google Drive','Data Availability'):
 s=0
 while True:
  i=text.find(needle,s)
  if i<0:break
  report['fulltext']['data_availability_context'].append(text[max(0,i-1200):i+2200]);s=i+len(needle)
if supp['body'].startswith(b'PK'):
 z=zipfile.ZipFile(io.BytesIO(supp['body']));names=z.namelist();report['supplementary']['zip_members']=[{'name':n,'size':z.getinfo(n).file_size,'compressed':z.getinfo(n).compress_size} for n in names]
 # inspect nested small metadata zips; do not recursively inflate giant members beyond listing
 for n in names:
  if n.lower().endswith(('.xml','.txt','.json','.html')) and z.getinfo(n).file_size<2_000_000:
   b=z.read(n);report['supplementary'].setdefault('text_snippets',{})[n]=b[:20000].decode('utf-8','replace')
else: report['supplementary']['prefix_hex']=supp['body'][:64].hex()
p=OUT/'uavobb_europepmc_acquisition_probe.json';p.write_text(json.dumps(report,indent=2));print(json.dumps({'fulltext_status':xml.get('status'),'fulltext_bytes':xml.get('bytes'),'google_drive_links':report['fulltext']['google_drive_links'],'supp_status':supp.get('status'),'supp_bytes':supp.get('bytes'),'supp_type':supp.get('content_type'),'members':report['supplementary'].get('zip_members',[])[:20],'evidence':str(p)},indent=2))
