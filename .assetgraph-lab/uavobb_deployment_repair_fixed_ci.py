from __future__ import annotations
import json, math, pathlib, shutil, time
import uavobb_deployment_repair_ci as C
import uavobb_forensic_ci as F

def parity_shared(onnx,root,nc):
    import numpy as np, torch, onnxruntime as ort
    from ultralytics.utils.nms import non_max_suppression
    net=C.MODEL.model.eval(); sess=ort.InferenceSession(str(onnx),providers=['CPUExecutionProvider']); inp=sess.get_inputs()[0].name
    tp=ptn=onn=0; rows=[]
    for im in sorted((root/'test'/'images').glob('*.jpg'))[:12]:
        a=F.preprocess_image(im,512)
        with torch.no_grad(): po=net(torch.from_numpy(a))
        if isinstance(po,(tuple,list)): po=po[0]
        if isinstance(po,(tuple,list)): po=po[0]
        oo=sess.run(None,{inp:a.astype('float32')})[0]
        A=non_max_suppression(po,C.CONF,.70,nc=nc,rotated=True,max_det=300)[0].detach().cpu().numpy()
        B=non_max_suppression(torch.from_numpy(oo),C.CONF,.70,nc=nc,rotated=True,max_det=300)[0].detach().cpu().numpy()
        ptn+=len(A); onn+=len(B); candidates=[]
        for i,x in enumerate(A):
            for j,y in enumerate(B):
                if int(x[5])!=int(y[5]): continue
                dc=math.hypot(float(x[0]-y[0]),float(x[1]-y[1])); ds=abs(float(x[2]-y[2]))+abs(float(x[3]-y[3])); da=abs(float(x[-1]-y[-1]))
                if dc<=2 and ds<=4 and da<=.03: candidates.append((dc+.2*ds+5*da,i,j))
        ua=set(); ub=set(); m=0
        for _,i,j in sorted(candidates):
            if i in ua or j in ub: continue
            ua.add(i); ub.add(j); m+=1
        tp+=m; rows.append({'image':im.name,'pt':len(A),'onnx':len(B),'matched':m})
    p=tp/onn if onn else 0; r=tp/ptn if ptn else 0
    return {'pt_total':ptn,'onnx_total':onn,'matched':tp,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'same_postprocess':True,'nms_import':'ultralytics.utils.nms.non_max_suppression','rows':rows}

def main():
    from ultralytics import YOLO
    if not F.PT.exists(): raise RuntimeError('Cycle11 checkpoint missing')
    t=time.time(); z,root=F.get_corpus(); N=F.class_names(root); C.MODEL=YOLO(str(F.PT))
    val=[C.evaluate(root,N,'valid',p) for p in C.PROFILES]
    chosen=C.choose(val)
    selected_test=C.evaluate(root,N,'test',chosen['profile'])
    baseline_profile=next(p for p in C.PROFILES if p['name']=='single1024')
    baseline_test=C.evaluate(root,N,'test',baseline_profile)
    onnx=pathlib.Path(C.MODEL.export(format='onnx',imgsz=512,opset=17,dynamic=False,simplify=False,device='cpu',verbose=False))
    parity=parity_shared(onnx,root,len(N)); small=(selected_test['size'].get('small') or {}).get('recall') or 0; ov=selected_test['other_vehicle_coarse']['recall']
    gates={'shared_postprocess_parity_pass':parity['f1']>=.98,'small_recall_pass':small>=.30,'fine_f1_nonregression_pass':selected_test['fine']['f1']>=baseline_test['fine']['f1']-.05,'other_vehicle_coarse_pass':ov>=.70}; gates['deployment_repair_pass']=all(gates.values())
    report={'schema':'assetgraph-evidence/uavobb-deployment-repair-v1','training_in_cycle14':False,'methodology_fix':'non-regression compares selected and baseline on the same frozen test split; shared NMS import updated for Ultralytics 8.4.137 API without changing thresholds or gates','dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':F.sha256(z)},'model':{'pt_sha256':F.sha256(F.PT),'onnx_sha256':F.sha256(onnx),'confidence':C.CONF},'selection_rule':'validation only: F1 >= single1024 F1-.05, P>=.50; maximize .55*smallR+.25*fineF1+.20*coarseF1','validation_profiles':val,'selected_profile':chosen['profile'],'frozen_test':selected_test,'frozen_test_baseline_single1024':baseline_test,'shared_postprocess_parity':parity,'hierarchical_semantics':{'stable_asset_class':'vehicle','fine_subtype_is_hypothesis':True},'gates':gates,'elapsed_seconds':time.time()-t}
    decision=C.decision(root,N,chosen['profile'],F.sha256(F.PT),F.sha256(onnx)); (C.OUT/'uavobb_deployment_repair.json').write_text(json.dumps(report,indent=2)); (C.OUT/'decision_object_cycle14.json').write_text(json.dumps(decision,indent=2)); shutil.copy2(onnx,C.OUT/'uavobb_deployment_repair.onnx'); (C.DIST/'assetgraph_frontier_v16_deployment_repair.html').write_text(C.html(report,decision)); print(json.dumps({'selected':chosen['profile'],'selected_test':selected_test,'baseline_test':baseline_test,'parity':parity,'gates':gates},indent=2))

if __name__=='__main__': main()
