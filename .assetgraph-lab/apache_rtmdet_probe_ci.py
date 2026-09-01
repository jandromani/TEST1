from __future__ import annotations
import hashlib, json, os, pathlib, platform, sys, time

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
PROTOCOL=ROOT/'cycle17_apache_probe_protocol.json'
MMR=pathlib.Path(os.environ['MMROTATE_DIR']).resolve()
CFG=MMR/'configs/rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py'
CKPT=pathlib.Path(os.environ['RTMDET_CKPT']).resolve()
IMG=pathlib.Path(os.environ['RTMDET_IMAGE']).resolve()
EXPECTED_COMMIT='3ff004eb21ea040455b5585db229edba4037f1bf'


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def run(cmd):
    import subprocess
    return subprocess.check_output(cmd,cwd=MMR,text=True).strip()


def main():
    t=time.time(); protocol=json.loads(PROTOCOL.read_text())
    source_commit=run(['git','rev-parse','HEAD'])
    license_text=(MMR/'LICENSE').read_text(errors='replace')
    source_ok=source_commit==EXPECTED_COMMIT
    license_ok=('Apache License' in license_text and 'Version 2.0' in license_text)

    import numpy as np
    import torch, torchvision, mmcv, mmdet, mmengine, mmrotate
    from mmdet.apis import inference_detector, init_detector
    from mmrotate.utils import register_all_modules
    versions={
      'python':sys.version.split()[0], 'numpy':np.__version__, 'torch':torch.__version__, 'torchvision':torchvision.__version__,
      'mmcv':mmcv.__version__, 'mmdet':mmdet.__version__, 'mmengine':mmengine.__version__, 'mmrotate':mmrotate.__version__
    }
    assert np.__version__=='1.26.4', versions
    register_all_modules()
    model=init_detector(str(CFG),str(CKPT),palette='dota',device='cpu')
    result=inference_detector(model,str(IMG))
    pred=result.pred_instances
    scores=pred.scores.detach().cpu().numpy().tolist()
    labels=pred.labels.detach().cpu().numpy().tolist()
    bboxes=pred.bboxes.detach().cpu().numpy().tolist()
    keep=[i for i,s in enumerate(scores) if float(s)>=.30]
    rows=[]
    for i in keep[:100]:
        rows.append({'score':float(scores[i]),'label':int(labels[i]),'bbox':list(map(float,bboxes[i]))})
    gates={
      'source_commit_exact':source_ok,
      'apache_license_detected':license_ok,
      'dependency_imports_pass':True,
      'checkpoint_load_pass':True,
      'cpu_inference_pass':True,
      'at_least_one_prediction_score_ge_0_30':len(keep)>0,
    }
    gates['apache_runtime_probe_pass']=all(gates.values())
    report={
      'schema':'assetgraph-evidence/apache-rtmdet-runtime-probe-v1',
      'protocol_sha256':sha256(PROTOCOL),
      'compatibility_fix':protocol['compatibility_fix'],
      'source':{
        'repository':'open-mmlab/mmrotate','commit':source_commit,'expected_commit':EXPECTED_COMMIT,
        'license_file_sha256':sha256(MMR/'LICENSE'),'license_detected':'Apache-2.0' if license_ok else 'UNKNOWN'
      },
      'model':{
        'name':'rotated_rtmdet_tiny-3x-dota','config':str(CFG.relative_to(MMR)),
        'checkpoint_sha256':sha256(CKPT),'checkpoint_bytes':CKPT.stat().st_size,
        'official_reported_dota_map':protocol['candidate']['official_reported_dota_map']
      },
      'input':{'image':IMG.name,'sha256':sha256(IMG),'bytes':IMG.stat().st_size},
      'environment':{'platform':platform.platform(),**versions},
      'inference':{
        'raw_prediction_count':len(scores),'score_ge_0_30_count':len(keep),
        'max_score':max(scores) if scores else 0.0,'first_predictions':rows
      },
      'gates':gates,
      'transset_accessed':False,
      'uavobb_training_performed':False,
      'onnx_claim':'NOT_TESTED',
      'elapsed_seconds':time.time()-t,
    }
    p=OUT/'cycle17_apache_rtmdet_probe.json';p.write_text(json.dumps(report,indent=2))
    print(json.dumps({'versions':versions,'detections_ge_030':len(keep),'max_score':report['inference']['max_score'],'gates':gates},indent=2))
    if not gates['apache_runtime_probe_pass']:
        raise SystemExit('Cycle17 Apache runtime probe gate failed')

if __name__=='__main__':main()
