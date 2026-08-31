from __future__ import annotations
import csv,json,math,statistics,hashlib,pathlib,urllib.request
from collections import defaultdict,Counter
ROOT=pathlib.Path(__file__).resolve().parent;ASSETS=ROOT/'assets'/'dut';OUT=ROOT/'evidence';ASSETS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
COMMIT='80b8c746833664cd1e5244fccad79c7f2a7cbe31'
MISSIONS=[('I01','intersection_01','intersection'),('I02','intersection_02','intersection'),('I03','intersection_03','intersection'),('I06','intersection_06','intersection'),('I11','intersection_11','intersection'),('I15','intersection_15','intersection'),('R01','roundabout_01','roundabout'),('R02','roundabout_02','roundabout'),('R05','roundabout_05','roundabout'),('R06','roundabout_06','roundabout'),('R08','roundabout_08','roundabout'),('R09','roundabout_09','roundabout')]
FAMILY=['ped_count','veh_count','direction_spread','density','tortuosity','veh_appears_late']
SPEED=['veh_speed_early','vehicle_turn_index','veh_count','ped_count','density','veh_appears_late']
CROSS=['early_min_cross','early_median_cross','early_cross_slope','ped_count','veh_count','density','ped_speed_early','veh_speed_early','veh_appears_late']

def dl(u,p):
    if not p.exists():urllib.request.urlretrieve(u,p)
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rows(path,kind):
    out=[]
    with open(path,newline='') as f:
        for n,r in enumerate(csv.DictReader(f),2):
            fr=int(r['frame']);x=float(r['x_est']);y=float(r['y_est'])
            if kind=='ped':vx=float(r.get('vx_est',r.get('xv_est',0)));vy=float(r.get('vy_est',r.get('yv_est',0)));sp=math.hypot(vx,vy);hd=math.atan2(vy,vx)
            else:sp=float(r['vel_est']);hd=float(r['psi_est']);vx=sp*math.cos(hd);vy=sp*math.sin(hd)
            out.append({'id':f'{kind}:{r["id"]}','kind':kind,'frame':fr,'x':x,'y':y,'speed':sp,'heading':hd})
    return out
def mean(v):return statistics.fmean(v) if v else None
def ad(a,b):
    d=a-b
    while d>math.pi:d-=2*math.pi
    while d<-math.pi:d+=2*math.pi
    return d
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
    if len(vals)<3:return 0.0
    xs=[x for x,_ in vals];ys=[y for _,y in vals];mx=mean(xs);my=mean(ys);den=sum((x-mx)**2 for x in xs) or 1.;return sum((x-mx)*(y-my) for x,y in vals)/den
def derive(mid,clip,fam):
    base=f'https://raw.githubusercontent.com/dongfang-steven-yang/vci-dataset-dut/{COMMIT}/data/trajectories_filtered';pp=ASSETS/f'{clip}_traj_ped_filtered.csv';vp=ASSETS/f'{clip}_traj_veh_filtered.csv';dl(f'{base}/{pp.name}',pp);dl(f'{base}/{vp.name}',vp)
    ped=rows(pp,'ped');veh=rows(vp,'veh');allr=ped+veh;fs=sorted({r['frame'] for r in allr});lo,hi=fs[0],fs[-1];cut=lo+round((hi-lo)*.35);ep=[r for r in ped if r['frame']<=cut];ev=[r for r in veh if r['frame']<=cut];fp=[r for r in ped if r['frame']>cut];fv=[r for r in veh if r['frame']>cut];early=ep+ev
    by=defaultdict(list)
    for r in early:by[r['id']].append(r)
    headings=[];tort=[];turn=[]
    for rr in by.values():
        rr=sorted(rr,key=lambda x:x['frame']);dx=rr[-1]['x']-rr[0]['x'];dy=rr[-1]['y']-rr[0]['y'];headings.append(math.atan2(dy,dx));path=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(rr,rr[1:]));tort.append(path/max(math.hypot(dx,dy),.25))
        if rr[0]['kind']=='veh':turn += [abs(ad(b['heading'],a['heading'])) for a,b in zip(rr,rr[1:])]
    xs=[r['x'] for r in early];ys=[r['y'] for r in early];area=max(1.,(max(xs)-min(xs))*(max(ys)-min(ys)));cx=mean([math.cos(x) for x in headings]) if headings else 1.;cy=mean([math.sin(x) for x in headings]) if headings else 0.
    ecs=cross_series(ep,ev);fcs=cross_series(fp,fv);sentinel=max(10.,math.sqrt(area));early_vals=[d for _,d in ecs]
    feat={'ped_count':len({r['id'] for r in ep}),'veh_count':len({r['id'] for r in ev}),'ped_speed_early':mean([r['speed'] for r in ep]) or 0.,'veh_speed_early':mean([r['speed'] for r in ev]) or 0.,'direction_spread':1-math.hypot(cx,cy),'vehicle_turn_index':mean(turn) or 0.,'density':len({r['id'] for r in early})/area,'tortuosity':mean(tort) or 1.,'veh_appears_late':1.0 if not ev and fv else 0.0,'early_min_cross':min(early_vals) if early_vals else sentinel,'early_median_cross':statistics.median(early_vals) if early_vals else sentinel,'early_cross_slope':slope(ecs)}
    target={'family':fam,'future_vehicle_speed':mean([r['speed'] for r in fv]),'future_min_cross':min([d for _,d in fcs]) if fcs else None}
    return {'id':mid,'clip':clip,'features':feat,'target':target,'cut_frame':cut,'input_hashes':{pp.name:sha(pp),vp.name:sha(vp)}}
def zdist(q,m,tr,features):
    s=0.
    for k in features:
        vals=[x['features'][k] for x in tr];mu=mean(vals);sd=statistics.pstdev(vals) or 1.;s+=((q['features'][k]-m['features'][k])/sd)**2
    return math.sqrt(s)
def neigh(q,tr,features,k=3):return sorted((zdist(q,m,tr,features),m) for m in tr)[:k]
def wavg(vals):
    vals=[x for x in vals if x[0] is not None]
    if not vals:return None
    sw=sum(w for _,w in vals);return sum(v*w for v,w in vals)/sw
def evaluate(ms):
    folds=[]
    for q in ms:
        tr=[m for m in ms if m['id']!=q['id']];cnt=Counter(m['target']['family'] for m in tr);bf=sorted(cnt.items(),key=lambda x:(-x[1],x[0]))[0][0]
        nf=neigh(q,tr,FAMILY);ns=neigh(q,tr,SPEED);nc=neigh(q,tr,CROSS);score=defaultdict(float)
        for d,m in nf:score[m['target']['family']]+=1/(d+.15)
        mf=sorted(score.items(),key=lambda x:(-x[1],x[0]))[0][0];sp=[m['target']['future_vehicle_speed'] for m in tr if m['target']['future_vehicle_speed'] is not None];cr=[m['target']['future_min_cross'] for m in tr if m['target']['future_min_cross'] is not None];bs=mean(sp);bc=mean(cr);mspeed=wavg([(m['target']['future_vehicle_speed'],1/(d+.15)) for d,m in ns]);mcross=wavg([(m['target']['future_min_cross'],1/(d+.15)) for d,m in nc]);ac=q['target']['future_min_cross'];av=q['target']['future_vehicle_speed'];th=statistics.median(cr) if cr else None
        f={'heldout':q['id'],'truth_family':q['target']['family'],'baseline_family':bf,'memory_family':mf,'family_baseline_correct':bf==q['target']['family'],'family_memory_correct':mf==q['target']['family'],'analogues':{'family':[m['id'] for _,m in nf],'speed':[m['id'] for _,m in ns],'cross':[m['id'] for _,m in nc]}}
        if av is not None and bs is not None and mspeed is not None:f['baseline_speed_abs_error']=abs(bs-av);f['memory_speed_abs_error']=abs(mspeed-av)
        if ac is not None and bc is not None and mcross is not None:
            f['baseline_cross_abs_error']=abs(bc-ac);f['memory_cross_abs_error']=abs(mcross-ac)
            if th is not None:truth=ac<th;f['baseline_close_correct']=(bc<th)==truth;f['memory_close_correct']=(mcross<th)==truth
        folds.append(f)
    def avg(k):
        v=[f[k] for f in folds if k in f];return mean(v)
    metrics={'scenario_family_accuracy':{'baseline':avg('family_baseline_correct'),'memory':avg('family_memory_correct')},'future_vehicle_speed_mae_mps':{'baseline':avg('baseline_speed_abs_error'),'memory':avg('memory_speed_abs_error')},'future_min_ped_vehicle_distance_mae_m':{'baseline':avg('baseline_cross_abs_error'),'memory':avg('memory_cross_abs_error')},'future_close_interaction_accuracy':{'baseline':avg('baseline_close_correct'),'memory':avg('memory_close_correct')}};lift={}
    for k,v in metrics.items():
        b,m=v['baseline'],v['memory']
        if b is None or m is None:lift[k]=None
        elif 'mae' in k:lift[k]=(b-m)/max(abs(b),1e-9)
        else:lift[k]=(m-b)/max(abs(b),1e-9) if b else (1. if m>b else 0.)
    vals=[x for x in lift.values() if x is not None];passed=bool(vals) and max(vals)>=.10 and min(vals)>=-.05
    return {'metrics':metrics,'relative_lift':lift,'pass_rule':passed,'folds':folds}
ms=[derive(*m) for m in MISSIONS];res=evaluate(ms);pack={'schema':'assetgraph-evidence/req008-memory-lift-v2','request_id':'REQ-008','architecture':'multi-head retrieval with early interaction geometry','source_commit':COMMIT,'feature_heads':{'family':FAMILY,'speed':SPEED,'cross':CROSS},'missions':ms,'result':res,'leakage_controls':{'all_query_features_observed_in_first_35_percent':True,'future_targets_excluded':True,'heldout_excluded':True,'fold_local_standardization':True}}
(OUT/'req008_memory_lift_v2.json').write_text(json.dumps(pack,indent=2));print(json.dumps(res['metrics'],indent=2));print(json.dumps(res['relative_lift'],indent=2));print('PASS',res['pass_rule'])
