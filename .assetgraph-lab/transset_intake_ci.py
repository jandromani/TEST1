from __future__ import annotations
import json, pathlib, requests, time
OUT=pathlib.Path(__file__).resolve().parent/'evidence';OUT.mkdir(exist_ok=True)
ARTICLE=26082217
API=f'https://api.figshare.com/v2/articles/{ARTICLE}'

def main():
    t=time.time();r=requests.get(API,timeout=30);r.raise_for_status();m=r.json();files=m.get('files',[]);rows=[]
    for f in files:
        u=f.get('download_url'); probe={}
        if u:
            try:
                q=requests.get(u,headers={'Range':'bytes=0-31'},allow_redirects=True,timeout=30)
                probe={'status':q.status_code,'range_requested':True,'content_range':q.headers.get('Content-Range'),'accept_ranges':q.headers.get('Accept-Ranges'),'content_length':q.headers.get('Content-Length'),'magic_hex':q.content[:16].hex(),'final_url':q.url}
            except Exception as e:probe={'error':type(e).__name__+':'+str(e)[:160]}
        rows.append({'id':f.get('id'),'name':f.get('name'),'size':f.get('size'),'md5':f.get('computed_md5') or f.get('supplied_md5'),'download_url':u,'probe':probe})
    lic=m.get('license') or {}; report={'schema':'assetgraph-evidence/transset-intake-v1','article_id':ARTICLE,'title':m.get('title'),'doi':m.get('doi'),'url_public_api':API,'license':lic,'published_date':m.get('published_date'),'modified_date':m.get('modified_date'),'files':rows,'total_files':len(rows),'total_bytes':sum(int(x.get('size') or 0) for x in files),'lockbox_policy':{'training_allowed_before_first_score':False,'first_use':'external coarse-vehicle generalization only'},'intake_pass':bool(files) and ('CC BY' in str(lic) or 'Creative Commons Attribution' in str(lic)),'elapsed_seconds':time.time()-t}
    (OUT/'transset_intake.json').write_text(json.dumps(report,indent=2)); print(json.dumps({'title':report['title'],'license':lic,'total_files':len(rows),'total_bytes':report['total_bytes'],'files':[{'name':x['name'],'size':x['size'],'range':x['probe'].get('content_range'),'status':x['probe'].get('status')} for x in rows],'intake_pass':report['intake_pass']},indent=2))
if __name__=='__main__':main()
