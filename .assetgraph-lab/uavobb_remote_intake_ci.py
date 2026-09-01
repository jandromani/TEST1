from __future__ import annotations
import hashlib, io, json, pathlib, zipfile
from collections import Counter
from remotezip import RemoteZip
from PIL import Image

ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'evidence'; ASSET=ROOT/'assets'/'uavobb_intake'; OUT.mkdir(exist_ok=True); ASSET.mkdir(parents=True,exist_ok=True)
DATA_ID='1lPG2ZPxESXhsWbnrTn8ezIn_-1bH5IN7'; CODE_ID='19xJ98sQkmU9fuMPk74JkXxhKjgWAQYPm'
DATA_URL=f'https://drive.usercontent.google.com/download?id={DATA_ID}&export=download&confirm=t'; CODE_URL=f'https://drive.usercontent.google.com/download?id={CODE_ID}&export=download&confirm=t'

def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    import requests
    cb=requests.get(CODE_URL,timeout=60).content; cz=zipfile.ZipFile(io.BytesIO(cb)); code_members=[{'name':x.filename,'bytes':x.file_size} for x in cz.infolist()]
    rz=RemoteZip(DATA_URL); infos=rz.infolist(); names=[x.filename for x in infos]; exts=Counter(pathlib.PurePosixPath(n).suffix.lower() for n in names if not n.endswith('/')); roots=Counter(n.split('/')[0] for n in names)
    imgs=[x for x in infos if pathlib.PurePosixPath(x.filename).suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}]
    labels=[x for x in infos if pathlib.PurePosixPath(x.filename).suffix.lower() in {'.txt','.xml','.json','.csv'}]
    split_counts={k:sum(('/'+k+'/') in ('/'+n.lower().replace('\\','/')+'/') or n.lower().startswith(k+'/') for n in names) for k in ('train','val','valid','validation','test')}
    samples=[]
    for info in (imgs[:2]+labels[:3]):
        b=rz.read(info.filename); p=ASSET/pathlib.PurePosixPath(info.filename).name; p.write_bytes(b); row={'member':info.filename,'bytes':len(b),'sha256':sha(b),'compressed_bytes':info.compress_size}
        suf=p.suffix.lower()
        if suf in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}:
            try:
                with Image.open(io.BytesIO(b)) as im: row.update({'kind':'image','width':im.width,'height':im.height,'mode':im.mode})
            except Exception as e: row.update({'kind':'image','verify_error':repr(e)})
        else:
            txt=b[:4000].decode('utf-8','replace'); row.update({'kind':'label_or_metadata','preview':txt[:1200]})
        samples.append(row)
    rz.close()
    # Infer likely annotation layout only from names/content; do not transform dataset yet.
    lower=[n.lower() for n in names]; layout={'has_images_dir':any('/images/' in '/'+n for n in lower),'has_labels_dir':any('/labels/' in '/'+n for n in lower),'has_annotations_dir':any('/annotations/' in '/'+n for n in lower),'obb_tokens':sum('obb' in n for n in lower)}
    out={'schema':'assetgraph-acquisition/uavobb-remote-intake-v1','source':{'data_drive_id':DATA_ID,'data_url':DATA_URL,'declared_archive_bytes':601733395,'code_drive_id':CODE_ID,'code_zip_sha256':sha(cb),'code_zip_bytes':len(cb)},'archive':{'member_count':len(infos),'total_uncompressed_bytes':sum(x.file_size for x in infos),'total_compressed_member_bytes':sum(x.compress_size for x in infos),'extensions':dict(exts),'top_roots':roots.most_common(20),'split_name_counts':split_counts,'layout':layout,'first_members':names[:80]},'code_members':code_members,'sample_assets':samples,'training_factory_ready':bool(imgs and labels),'note':'Central directory and samples read by HTTP Range; full 601.7MB corpus was not downloaded.'}
    p=OUT/'uavobb_remote_intake.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({'member_count':len(infos),'extensions':dict(exts),'top_roots':roots.most_common(10),'splits':split_counts,'layout':layout,'code_members':code_members,'samples':samples,'ready':out['training_factory_ready']},indent=2))
if __name__=='__main__':main()
