from __future__ import annotations
import json, pathlib, shutil, time
import uavobb_deployment_repair_ci as C
import uavobb_forensic_ci as F

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
    parity=C.parity(onnx,root,len(N)); small=(selected_test['size'].get('small') or {}).get('recall') or 0; ov=selected_test['other_vehicle_coarse']['recall']
    gates={'shared_postprocess_parity_pass':parity['f1']>=.98,'small_recall_pass':small>=.30,'fine_f1_nonregression_pass':selected_test['fine']['f1']>=baseline_test['fine']['f1']-.05,'other_vehicle_coarse_pass':ov>=.70}; gates['deployment_repair_pass']=all(gates.values())
    report={'schema':'assetgraph-evidence/uavobb-deployment-repair-v1','training_in_cycle14':False,'methodology_fix':'non-regression compares selected and baseline on the same frozen test split','dataset':{'name':'UAV-OBB','license':'CC BY 4.0','archive_sha256':F.sha256(z)},'model':{'pt_sha256':F.sha256(F.PT),'onnx_sha256':F.sha256(onnx),'confidence':C.CONF},'selection_rule':'validation only: F1 >= single1024 F1-.05, P>=.50; maximize .55*smallR+.25*fineF1+.20*coarseF1','validation_profiles':val,'selected_profile':chosen['profile'],'frozen_test':selected_test,'frozen_test_baseline_single1024':baseline_test,'shared_postprocess_parity':parity,'hierarchical_semantics':{'stable_asset_class':'vehicle','fine_subtype_is_hypothesis':True},'gates':gates,'elapsed_seconds':time.time()-t}
    decision=C.decision(root,N,chosen['profile'],F.sha256(F.PT),F.sha256(onnx)); (C.OUT/'uavobb_deployment_repair.json').write_text(json.dumps(report,indent=2)); (C.OUT/'decision_object_cycle14.json').write_text(json.dumps(decision,indent=2)); shutil.copy2(onnx,C.OUT/'uavobb_deployment_repair.onnx'); (C.DIST/'assetgraph_frontier_v16_deployment_repair.html').write_text(C.html(report,decision)); print(json.dumps({'selected':chosen['profile'],'selected_test':selected_test,'baseline_test':baseline_test,'parity':parity,'gates':gates},indent=2))

if __name__=='__main__': main()
