from __future__ import annotations
import hashlib,json,re,requests,pathlib
OUT=pathlib.Path(__file__).resolve().parent/'evidence';OUT.mkdir(parents=True,exist_ok=True)
IDS=['19xJ98sQkmU9fuMPk74JkXxhKjgWAQYPm','1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7']
def magic(b):
 if b.startswith(b'PK\x03\x04'): return 'zip'
 if b.startswith(b'\x1f\x8b'): return 'gzip'
 if b.startswith(b'Rar!'): return 'rar'
 if b.startswith(b'\x89PNG'): return 'png'
 if b[:3]==b'\xff\xd8\xff': return 'jpeg'
 if b.lstrip().startswith(b'<'): return 'html/xml'
 return b[:16].hex()
def probe(fid):
 urls=[f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t',f'https://drive.google.com/uc?export=download&id={fid}&confirm=t'];attempts=[]
 for u in urls:
  try:
   with requests.get(u,headers={'User-Agent':'Mozilla/5.0','Range':'bytes=0-262143'},stream=True,timeout=60,allow_redirects=True) as r:
    first=b''
    for chunk in r.iter_content(65536):
     if chunk:first+=chunk
     if len(first)>=262144:break
    cd=r.headers.get('content-disposition','');ct=r.headers.get('content-type','');cr=r.headers.get('content-range');cl=r.headers.get('content-length');title=None
    if 'html' in ct.lower() or magic(first)=='html/xml':
     txt=first.decode('utf-8','ignore');m=re.search(r'<title>(.*?)</title>',txt,re.I|re.S);title=re.sub(r'\s+',' ',m.group(1)).strip() if m else None
    attempts.append({'requested_url':u,'status':r.status_code,'final_url':r.url,'content_type':ct,'content_disposition':cd,'content_range':cr,'content_length':cl,'first_bytes':len(first),'first_sha256':hashlib.sha256(first).hexdigest(),'magic':magic(first),'html_title':title})
  except Exception as e:attempts.append({'requested_url':u,'error':repr(e)})
 best=next((a for a in attempts if a.get('magic') not in (None,'html/xml') and a.get('status') in (200,206)),None);return {'file_id':fid,'attempts':attempts,'direct_binary_candidate':best}
def main():
 out={'schema':'assetgraph-acquisition/uavobb-google-drive-probe-v1','files':[probe(x) for x in IDS]};p=OUT/'uavobb_google_drive_probe.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
