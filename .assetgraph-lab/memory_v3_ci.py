from __future__ import annotations
import csv, json, math, statistics, hashlib, pathlib, urllib.request
from collections import defaultdict, Counter
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;ASSETS=ROOT/'assets'/'dut_v3';OUT=ROOT/'evidence';ASSETS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
COMMIT='80b8c746833664cd1e5244fccad79c7f2a7cbe31'
DEV=[('I01','intersection_01','intersection'),('I02','intersection_02','intersection'),('I03','intersection_03','intersection'),('I06','intersection_06','intersection'),('I11','intersection_11','intersection'),('I15','intersection_15','intersection'),('R01','roundabout_01','roundabout'),('R02','roundabout_02','roundabout'),('R05','roundabout_05','roundabout'),('R06','roundabout_06','roundabout'),('R08','roundabout_08','roundabout'),('R09','roundabout_09','roundabout')]
LOCKBOX=[('I04','intersection_04','intersection'),('I05','intersection_05','intersection'),('I07','intersection_07','intersection'),('I08','intersection_08','intersection'),('I09','intersection_09','intersection'),('R03','roundabout_03','roundabout'),('R04','roundabout_04','roundabout'),('R07','roundabout_07','roundabout'),('R10','roundabout_10','roundabout'),('R11','roundabout_11','roundabout')]
ALL_FEATURES=['ped_count','veh_count','ped_speed_early','veh_speed_early','direction_spread','vehicle_turn_index','density','tortuosity','veh_appears_late','early_min_cross','early_median_cross','early_cross_slope']
FAMILY_SETS=[['ped_count','veh_count','direction_spread','density','tortuosity','veh_appears_late'],['ped_count','veh_count','density','early_min_cross','early_median_cross','veh_appears_late'],ALL_FEATURES]
SPEED_SETS=[['veh_speed_early','vehicle_turn_index','veh_count','density','veh_appears_late'],['veh_speed_early','early_min_cross','early_cross_slope','veh_count','ped_count'],ALL_FEATURES]
CROSS_SETS=[['early_min_cross','early_median_cross','early_cross_slope','ped_count','veh_count','density','ped_speed_early','veh_speed_early','veh_appears_late'],['early_min_cross','early_median_cross','early_cross_slope','direction_spread','vehicle_turn_index','tortuosity'],ALL_FEATURES]
def dl(u,p):
 if not p.exists():urllib.request.urlretrieve(u,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mean(v):
 v=[x for x in v if x is not None];return statistics.fmean(v) if v else None
def ad(a,b):
 d=a-b
 while d>math.pi:d-=2*math.pi
 while d<-math.pi:d+=2*math.pi
 return d
def rows(path,kind):
 out=[]
 with open(path,newline='') as f:
  for r in csv.DictReader(f):
   fr=int(r['frame']);x=float(r['x_est']);y=float(r['y_est'])
   if kind=='ped':
    vx=float(r.get('vx_est',r.get('xv_est',0)) or 0);vy=float(r.get('vy_est',r.get('yv_est',0)) or 0);sp=math.hypot(vx,vy);hd=math.atan2(vy,vx)
   else:sp=float(r['vel_est']);hd=float(r['psi_est'])
   out.append({'id':f'{kind}:{r["id"]}','kind':kind,'frame':fr,'x':x,'y':y,'speed':sp,'heading':hd})
 return out
def cross_series(peds,vehs):
 pb=defaultdict(list);vb=defaultdict(list)
 for r in peds:pb[r['frame']].append(r)
 for r in vehs:vb[r['frame']].append(r)
 out=[]
 for f in sorted(set(pb)&set(vb)):
  ds=[math.hypot(p['x']-v['x'],p['y']-v['y']) for p in pb[f] for v in vb[f]]
  if ds:out.append((f,min(ds)))
 return out
def slope(vals):
 if len(vals)<3:return 0.
 xs=[x for x,_ in vals];ys=[y for _,y in vals];mx=mean(xs);my=mean(ys);den=sum((x-mx)**2 for x in xs) or 1.;return sum((x-mx)*(y-my) for x,y in vals)/den
def derive(mid,clip,fam):
 base=f'https://raw.githubusercontent.com/dongfang-steven-yang/vci-dataset-dut/{COMMIT}/data/trajectories_filtered';pp=ASSETS/f'{clip}_traj_ped_filtered.csv';vp=ASSETS/f'{clip}_traj_veh_filtered.csv';dl(f'{base}/{pp.name}',pp);dl(f'{base}/{vp.name}',vp)
 ped=rows(pp,'ped');veh=rows(vp,'veh');allr=ped+veh
 if not allr:raise RuntimeError(f'No observations {clip}')
 fs=sorted({r['frame'] for r in allr});lo,hi=fs[0],fs[-1];cut=lo+round((hi-lo)*.35);ep=[r for r in ped if r['frame']<=cut];ev=[r for r in veh if r['frame']<=cut];fp=[r for r in ped if r['frame']>cut];fv=[r for r in veh if r['frame']>cut];early=ep+ev
 by=defaultdict(list)
 for r in early:by[r['id']].append(r)
 headings=[];tort=[];turn=[]
 for rr in by.values():
  rr=sorted(rr,key=lambda x:x['frame']);dx=rr[-1]['x']-rr[0]['x'];dy=rr[-1]['y']-rr[0]['y'];headings.append(math.atan2(dy,dx));path=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(rr,rr[1:]));tort.append(path/max(math.hypot(dx,dy),.25))
  if rr[0]['kind']=='veh':turn += [abs(ad(b['heading'],a['heading'])) for a,b in zip(rr,rr[1:])]
 xs=[r['x'] for r in early];ys=[r['y'] for r in early];area=max(1.,(max(xs)-min(xs))*(max(ys)-min(ys))) if xs and ys else 1.;cx=mean([math.cos(x) for x in headings]) if headings else 1.;cy=mean([math.sin(x) for x in headings]) if headings else 0.;ecs=cross_series(ep,ev);fcs=cross_series(fp,fv);sentinel=max(10.,math.sqrt(area));early_vals=[d for _,d in ecs]
 feat={'ped_count':len({r['id'] for r in ep}),'veh_count':len({r['id'] for r in ev}),'ped_speed_early':mean([r['speed'] for r in ep]) or 0.,'veh_speed_early':mean([r['speed'] for r in ev]) or 0.,'direction_spread':1-math.hypot(cx,cy),'vehicle_turn_index':mean(turn) or 0.,'density':len({r['id'] for r in early})/area,'tortuosity':mean(tort) or 1.,'veh_appears_late':1. if not ev and fv else 0.,'early_min_cross':min(early_vals) if early_vals else sentinel,'early_median_cross':statistics.median(early_vals) if early_vals else sentinel,'early_cross_slope':slope(ecs)};target={'family':fam,'future_vehicle_speed':mean([r['speed'] for r in fv]),'future_min_cross':min([d for _,d in fcs]) if fcs else None}
 return {'id':mid,'clip':clip,'features':feat,'target':target,'cut_frame':cut,'input_hashes':{pp.name:sha(pp),vp.name:sha(vp)}}
def matrix(train,features):
 X=np.array([[m['features'][k] for k in features] for m in train],float);mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1.;return X,mu,sd
def zvec(m,features,mu,sd):return (np.array([m['features'][k] for k in features],float)-mu)/sd
def knn_predict(train,q,target,features,k,power):
 usable=[m for m in train if m['target'].get(target) is not None]
 if not usable:return None
 X,mu,sd=matrix(usable,features);ds=np.linalg.norm(X-zvec(q,features,mu,sd),axis=1);order=np.argsort(ds)[:min(k,len(usable))];vals=[(float(usable[i]['target'][target]),1/((float(ds[i])+.15)**power)) for i in order];return sum(v*w for v,w in vals)/sum(w for _,w in vals)
def knn_class(train,q,features,k,power):
 X,mu,sd=matrix(train,features);ds=np.linalg.norm(X-zvec(q,features,mu,sd),axis=1);order=np.argsort(ds)[:min(k,len(train))];score=defaultdict(float)
 for i in order:score[train[i]['target']['family']]+=1/((float(ds[i])+.15)**power)
 return sorted(score.items(),key=lambda x:(-x[1],x[0]))[0][0]
def ridge_predict(train,q,target,features,alpha):
 usable=[m for m in train if m['target'].get(target) is not None]
 if len(usable)<3:return mean([m['target'].get(target) for m in usable])
 X,mu,sd=matrix(usable,features);X=np.c_[np.ones(len(X)),(X-mu)/sd];y=np.array([m['target'][target] for m in usable],float);reg=np.eye(X.shape[1])*alpha;reg[0,0]=0.;beta=np.linalg.solve(X.T@X+reg,X.T@y);return float(np.r_[1.,zvec(q,features,mu,sd)]@beta)
def candidate_specs(target):
 if target=='family':
  specs=[{'kind':'majority'}]
  for si,fs in enumerate(FAMILY_SETS):
   for k in (2,3,4,5):
    for power in (1.,2.):specs.append({'kind':'knn_class','features':fs,'k':k,'power':power,'name':f'knn_class_s{si}_k{k}_p{power}'})
  return specs
 sets=SPEED_SETS if target=='future_vehicle_speed' else CROSS_SETS;direct='veh_speed_early' if target=='future_vehicle_speed' else 'early_min_cross';specs=[{'kind':'mean','name':'mean'}]
 for shrink in (.25,.5,.75,1.):specs.append({'kind':'direct','feature':direct,'shrink':shrink,'name':f'direct_{direct}_{shrink}'})
 for si,fs in enumerate(sets):
  for k in (2,3,4,5):
   for power in (1.,2.):specs.append({'kind':'knn','features':fs,'k':k,'power':power,'name':f'knn_s{si}_k{k}_p{power}'})
  for alpha in (.03,.1,.3,1.,3.,10.):specs.append({'kind':'ridge','features':fs,'alpha':alpha,'name':f'ridge_s{si}_a{alpha}'})
 return specs
def predict(spec,train,q,target):
 k=spec['kind']
 if target=='family':
  if k=='majority':
   c=Counter(m['target']['family'] for m in train);return sorted(c.items(),key=lambda x:(-x[1],x[0]))[0][0]
  return knn_class(train,q,spec['features'],spec['k'],spec['power'])
 if k=='mean':return mean([m['target'].get(target) for m in train])
 if k=='direct':
  mu=mean([m['target'].get(target) for m in train]);return spec['shrink']*q['features'][spec['feature']]+(1-spec['shrink'])*mu if mu is not None else None
 if k=='knn':return knn_predict(train,q,target,spec['features'],spec['k'],spec['power'])
 if k=='ridge':return ridge_predict(train,q,target,spec['features'],spec['alpha'])
def cv_loss(dev,target,spec):
 errs=[]
 for q in dev:
  truth=q['target'].get(target)
  if truth is None:continue
  p=predict(spec,[m for m in dev if m['id']!=q['id']],q,target)
  if p is not None:errs.append((p!=truth) if target=='family' else abs(float(p)-float(truth)))
 return mean(errs) if errs else float('inf')
def select_spec(dev,target):
 scored=[(cv_loss(dev,target,s),json.dumps(s,sort_keys=True),s) for s in candidate_specs(target)];scored.sort(key=lambda x:(x[0],x[1]));return scored[0][2],scored[:8]
def evaluate(dev,lock):
 selected={};cv={}
 for target in ('family','future_vehicle_speed','future_min_cross'):
  s,rank=select_spec(dev,target);selected[target]=s;cv[target]=[{'loss':a,'spec':c} for a,_,c in rank]
 baselines={'family':Counter(m['target']['family'] for m in dev).most_common(1)[0][0],'future_vehicle_speed':mean([m['target']['future_vehicle_speed'] for m in dev]),'future_min_cross':mean([m['target']['future_min_cross'] for m in dev])};cr=[m['target']['future_min_cross'] for m in dev if m['target']['future_min_cross'] is not None];th=statistics.median(cr) if cr else None;folds=[]
 for q in lock:
  f={'heldout':q['id'],'truth_family':q['target']['family'],'baseline_family':baselines['family'],'memory_family':predict(selected['family'],dev,q,'family')};f['family_baseline_correct']=f['baseline_family']==f['truth_family'];f['family_memory_correct']=f['memory_family']==f['truth_family']
  for target,prefix in [('future_vehicle_speed','speed'),('future_min_cross','cross')]:
   truth=q['target'].get(target);b=baselines[target];mp=predict(selected[target],dev,q,target)
   if truth is not None and b is not None and mp is not None:f[f'baseline_{prefix}_abs_error']=abs(b-truth);f[f'memory_{prefix}_abs_error']=abs(mp-truth);f[f'baseline_{prefix}_prediction']=b;f[f'memory_{prefix}_prediction']=mp;f[f'truth_{prefix}']=truth
  if th is not None and 'truth_cross' in f:
   truth=f['truth_cross']<th;f['baseline_close_correct']=(f['baseline_cross_prediction']<th)==truth;f['memory_close_correct']=(f['memory_cross_prediction']<th)==truth
  folds.append(f)
 def avg(k):return mean([f[k] for f in folds if k in f])
 metrics={'scenario_family_accuracy':{'baseline':avg('family_baseline_correct'),'memory':avg('family_memory_correct')},'future_vehicle_speed_mae_mps':{'baseline':avg('baseline_speed_abs_error'),'memory':avg('memory_speed_abs_error')},'future_min_ped_vehicle_distance_mae_m':{'baseline':avg('baseline_cross_abs_error'),'memory':avg('memory_cross_abs_error')},'future_close_interaction_accuracy':{'baseline':avg('baseline_close_correct'),'memory':avg('memory_close_correct')}};lift={}
 for k,v in metrics.items():
  b,m=v['baseline'],v['memory'];lift[k]=None if b is None or m is None else ((b-m)/max(abs(b),1e-9) if 'mae' in k else ((m-b)/max(abs(b),1e-9) if b else (1. if m>b else 0.)))
 vals=[x for x in lift.values() if x is not None];passed=bool(vals) and max(vals)>=.10 and min(vals)>=-.05;return {'selected_models':selected,'development_cv_top8':cv,'close_threshold_m':th,'metrics':metrics,'relative_lift':lift,'pass_rule':passed,'folds':folds}
dev=[derive(*m) for m in DEV];lock=[derive(*m) for m in LOCKBOX];result=evaluate(dev,lock);pack={'schema':'assetgraph-evidence/req008-memory-lift-v3-lockbox','request_id':'REQ-008','architecture':'target-specific selection on known development bank + frozen external lockbox','source_commit':COMMIT,'development_bank':[m['id'] for m in dev],'lockbox_bank':[m['id'] for m in lock],'development_missions':dev,'lockbox_missions':lock,'result':result,'leakage_controls':{'v1_v2_missions_only_used_for_model_selection':True,'lockbox_targets_never_used_for_model_selection':True,'all_query_features_first_35_percent_only':True,'future_targets_excluded_from_features':True,'model_selection_leave_one_out_on_development_bank':True}}
(OUT/'req008_memory_lift_v3.json').write_text(json.dumps(pack,indent=2));print(json.dumps(result['selected_models'],indent=2));print(json.dumps(result['metrics'],indent=2));print(json.dumps(result['relative_lift'],indent=2));print('PASS',result['pass_rule'])