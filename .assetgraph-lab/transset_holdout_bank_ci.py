from __future__ import annotations
import hashlib, json, pathlib, time
import requests
from remotezip import RemoteZip

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
ARTICLE=26082217
BLOCK_SIZE=128
BLOCKS=8


def rank_hash(name:str)->str:
    return hashlib.sha256(name.encode('utf-8')).hexdigest()


def main():
    t=time.time()
    meta=requests.get(f'https://api.figshare.com/v2/articles/{ARTICLE}',timeout=30).json()
    zf=next(x for x in meta['files'] if x['name']=='TRANSSET_v2.zip')
    with RemoteZip(zf['download_url']) as rz:
        infos=[x for x in rz.infolist() if x.filename.lower().endswith(('.jpg','.jpeg','.png'))]
        ranked=sorted(infos,key=lambda x:(rank_hash(x.filename),x.filename))
        selected=ranked[:BLOCK_SIZE*BLOCKS]
        blocks=[]
        for b in range(BLOCKS):
            rows=[]
            for global_rank,info in enumerate(selected[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE],start=b*BLOCK_SIZE):
                rows.append({
                    'global_rank':global_rank,
                    'image_member':info.filename,
                    'rank_hash':rank_hash(info.filename),
                    'zip_crc32':f'{info.CRC:08x}',
                    'file_size':info.file_size,
                    'compress_size':info.compress_size,
                })
            blocks.append({'block_id':f'TRANSSET-H{b:02d}','start_rank':b*BLOCK_SIZE,'end_rank_exclusive':(b+1)*BLOCK_SIZE,'count':len(rows),'items':rows})
    report={
      'schema':'assetgraph-lockbox-bank/transset-v1',
      'dataset':{'name':'TRANSSET','article_id':ARTICLE,'license':'CC BY 4.0','zip_size':zf['size']},
      'selection':{
        'rule':'sort all image member names by SHA256(filename), then take contiguous blocks of 128',
        'uses_image_content':False,
        'uses_ground_truth':False,
        'uses_model_output':False,
        'uses_performance':False,
        'block_size':BLOCK_SIZE,
        'block_count':BLOCKS,
      },
      'policy':{
        'H00':'one-shot first external score for frozen Cycle11/Cycle14 profile',
        'future_blocks':'each future model promotion consumes at most one previously unopened block',
        'no_model_selection_on_consumed_block':True,
        'no_training_on_any_unopened_block':True,
      },
      'blocks':blocks,
      'frozen_before_first_external_score':True,
      'elapsed_seconds':time.time()-t,
    }
    p=OUT/'transset_holdout_bank.json';p.write_text(json.dumps(report,indent=2))
    print(json.dumps({'blocks':BLOCKS,'block_size':BLOCK_SIZE,'reserved_images':BLOCKS*BLOCK_SIZE,'first':blocks[0]['items'][0],'last':blocks[-1]['items'][-1]},indent=2))

if __name__=='__main__':main()
