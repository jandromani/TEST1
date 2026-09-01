from __future__ import annotations
import csv,json,math,statistics,hashlib,pathlib,urllib.request
from collections import defaultdict,Counter
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
ASSETS=ROOT/'assets'/'dut_v4'; OUT=ROOT/'evidence'; ASSETS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
COMMIT='80b8c746833664cd1e5244fccad79c7f2a7cbe31'
# 22 missions already exposed by v1-v3 become development only.
DEV_CLIPS=[
('I01','intersection_01','intersection'),('I02','intersection_02','intersection'),('I03','intersection_03','intersection'),('I04','intersection_04','intersection'),('I05','intersection_05','intersection'),('I06','intersection_06','intersection'),('I07','intersection_07','intersection'),('I08','intersection_08','intersection'),('I09','intersection_09','intersection'),('I11','intersection_11','intersection'),('I15','intersection_15','intersection'),
('R01','roundabout_01','roundabout'),('R02','roundabout_02','roundabout'),('R03','roundabout_03','roundabout'),('R04','roundabout_04','roundabout'),('R05','roundabout_05','roundabout'),('R06','roundabout_06','roundabout'),('R07','roundabout_07','roundabout'),('R08','roundabout_08','roundabout'),('R09','roundabout_09','roundabout'),('R10','roundabout_10','roundabout'),('R11','roundabout_11','roundabout')]
LOCKBOX=[('I10','intersection_10','intersection'),('I12','intersection_12','intersection'),('I13','intersection_13','intersection'),('I14','intersection_14','intersection'),('I16','intersection_16','intersection'),('I17','intersection_17','intersection')]
FEATURES=['ped_count','veh_count','ped_speed_early','veh_speed_early','direction_spread','vehicle_turn_index','density','tortuosity','veh_appears_late','early_min_cross','early_median_cross','early_cross_slope']
TARGETS=['family','future_vehicle_speed','future_min_cross','future_close']

def dl(url,path):
    if not path.exists(): urllib.request.urlretrieve(url,path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def mean(v):
    v=[x for x in v if x is not None]
    return statistics.fmean(v) if v else None
def ad(a,b):
    d=a-b
    while d>math.pi:d-=2*math.pi
    while d<-math.pi:d+=2*math.pi
    return d
def parse_rows(path,kind):
    out=[]
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            fr=int(r['frame']); x=float(r['x_est']); y=float(r['y_est'])
            if kind=='ped':
                vx=float(r.get('vx_est',0) or 0); vy=float(r.get('vy_est',0) or 0); sp=math.hypot(vx,vy); hd=math.atan2(vy,vx)
            else:
                sp=float(r['vel_est']); hd=float(r['psi_est'])
            out.append({'id':f'{kind}:{r["id"]}','kind':kind,'frame':fr,'x':x,'y':y,'speed':sp,'heading':hd})
    return out
def cross_series(peds,vehs):
    pb=defaultdict(list); vb=defaultdict(list)
    for r in peds: pb[r['frame']].append(r)
    for r in vehs: vb[r['frame']].append(r)
    out=[]
    for fr in sorted(set(pb)&set(vb)):
        ds=[math.hypot(p['x']-v['x'],p['y']-v['y']) for p in pb[fr] for v in vb[fr]]
        if ds: out.append((fr,min(ds)))
    return out
def slope(vals):
    if len(vals)<3:return 0.0
    xs=np.array([x for x,_ in vals],float); ys=np.array([y for _,y in vals],float)
    den=float(((xs-xs.mean())**2).sum())
    return float(((xs-xs.mean())*(ys-ys.mean())).sum()/den) if den>1e-12 else 0.0
def derive(mid,clip,family):
    base=f'https://raw.githubusercontent.com/dongfang-steven-yang/vci-dataset-dut/{COMMIT}/data/trajectories_filtered'
    pp=ASSETS/f'{clip}_traj_ped_filtered.csv'; vp=ASSETS/f'{clip}_traj_veh_filtered.csv'
    dl(f'{base}/{pp.name}',pp); dl(f'{base}/{vp.name}',vp)
    ped=parse_rows(pp,'ped'); veh=parse_rows(vp,'veh'); allr=ped+veh
    fs=sorted({r['frame'] for r in allr}); lo,hi=fs[0],fs[-1]; cut=lo+round((hi-lo)*.35)
    ep=[r for r in ped if r['frame']<=cut]; ev=[r for r in veh if r['frame']<=cut]; fp=[r for r in ped if r['frame']>cut]; fv=[r for r in veh if r['frame']>cut]
    early=ep+ev; by=defaultdict(list)
    for r in early:by[r['id']].append(r)
    headings=[]; tort=[]; turn=[]
    for rr in by.values():
        rr=sorted(rr,key=lambda x:x['frame']); dx=rr[-1]['x']-rr[0]['x'];dy=rr[-1]['y']-rr[0]['y']; headings.append(math.atan2(dy,dx))
        path=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(rr,rr[1:])); tort.append(path/max(math.hypot(dx,dy),.25))
        if rr[0]['kind']=='veh': turn += [abs(ad(b['heading'],a['heading'])) for a,b in zip(rr,rr[1:])]
    xs=[r['x'] for r in early];ys=[r['y'] for r in early];area=max(1.,(max(xs)-min(xs))*(max(ys)-min(ys))) if xs and ys else 1.
    cx=mean([math.cos(x) for x in headings]) if headings else 1.;cy=mean([math.sin(x) for x in headings]) if headings else 0.
    ecs=cross_series(ep,ev); fcs=cross_series(fp,fv); early_vals=[d for _,d in ecs]; sentinel=max(10.,math.sqrt(area))
    future_cross=min([d for _,d in fcs]) if fcs else None
    feat={'ped_count':len({r['id'] for r in ep}),'veh_count':len({r['id'] for r in ev}),'ped_speed_early':mean([r['speed'] for r in ep]) or 0.,'veh_speed_early':mean([r['speed'] for r in ev]) or 0.,'direction_spread':1-math.hypot(cx,cy),'vehicle_turn_index':mean(turn) or 0.,'density':len({r['id'] for r in early})/area,'tortuosity':mean(tort) or 1.,'veh_appears_late':1. if not ev and fv else 0.,'early_min_cross':min(early_vals) if early_vals else sentinel,'early_median_cross':statistics.median(early_vals) if early_vals else sentinel,'early_cross_slope':slope(ecs)}
    target={'family':family,'future_vehicle_speed':mean([r['speed'] for r in fv]),'future_min_cross':future_cross}
    return {'id':mid,'clip':clip,'features':feat,'target':target,'cut_frame':cut,'hashes':{pp.name:sha(pp),vp.name:sha(vp)}}
def enrich_close(ms,threshold):
    for m in ms:
        c=m['target']['future_min_cross'];m['target']['future_close']=None if c is None else bool(c<threshold)
def baseline(train,target):
    vals=[m['target'].get(target) for m in train if m['target'].get(target) is not None]
    if not vals:return None
    if target in ('family','future_close'):
        c=Counter(vals);return sorted(c.items(),key=lambda x:(-x[1],str(x[0])))[0][0]
    return mean(vals)
def zstats(train,features=FEATURES):
    X=np.array([[m['features'][k] for k in features] for m in train],float);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1.;return X,mu,sd
def knn(train,q,target,k=3,power=2.,features=FEATURES):
    usable=[m for m in train if m['target'].get(target) is not None]
    if not usable:return None
    X,mu,sd=zstats(usable,features);qv=(np.array([q['features'][x] for x in features])-mu)/sd;ds=np.linalg.norm((X-mu)/sd-qv,axis=1);inds=np.argsort(ds)[:min(k,len(ds))]
    if target in ('family','future_close'):
        score=defaultdict(float)
        for i in inds:score[usable[i]['target'][target]]+=1/((float(ds[i])+.1)**power)
        return sorted(score.items(),key=lambda x:(-x[1],str(x[0])))[0][0]
    ws=[1/((float(ds[i])+.1)**power) for i in inds];return sum(float(usable[i]['target'][target])*w for i,w in zip(inds,ws))/sum(ws)
def ridge(train,q,target,alpha=.3,features=FEATURES):
    usable=[m for m in train if m['target'].get(target) is not None]
    if len(usable)<4:return baseline(usable,target)
    X,mu,sd=zstats(usable,features);Z=(X-mu)/sd;Z=np.c_[np.ones(len(Z)),Z];y=np.array([float(m['target'][target]) for m in usable]);reg=np.eye(Z.shape[1])*alpha;reg[0,0]=0;beta=np.linalg.solve(Z.T@Z+reg,Z.T@y);qz=(np.array([q['features'][x] for x in features])-mu)/sd;return float(np.r_[1.,qz]@beta)
def candidates(target):
    out=[('baseline',{})]
    if target=='future_vehicle_speed':
        out += [('early_blend',{'w':w}) for w in (.4,.6,.75,.9)]
    if target in ('future_vehicle_speed','future_min_cross'):
        out += [('knn',{'k':k,'power':p}) for k in (2,3,4,5) for p in (1.,2.)]
        out += [('ridge',{'alpha':a}) for a in (.03,.1,.3,1.,3.,10.)]
    else:
        out += [('knn',{'k':k,'power':p}) for k in (2,3,4,5) for p in (1.,2.)]
    return out
def pred(spec,train,q,target):
    kind,args=spec
    if kind=='baseline':return baseline(train,target)
    if kind=='knn':return knn(train,q,target,**args)
    if kind=='ridge':return ridge(train,q,target,**args)
    if kind=='early_blend':
        b=baseline(train,target);e=q['features']['veh_speed_early'];return args['w']*e+(1-args['w'])*b
    raise ValueError(spec)
def loss(target,p,t):
    if p is None or t is None:return None
    return float(p!=t) if target in ('family','future_close') else abs(float(p)-float(t))
def cv_eval(dev,target,spec):
    rows=[]
    for q in dev:
        t=q['target'].get(target)
        if t is None:continue
        train=[m for m in dev if m['id']!=q['id']];b=baseline(train,target);p=pred(spec,train,q,target)
        rows.append({'id':q['id'],'baseline_loss':loss(target,b,t),'memory_loss':loss(target,p,t),'truth':t,'baseline':b,'memory':p})
    bl=mean([r['baseline_loss'] for r in rows]);ml=mean([r['memory_loss'] for r in rows])
    if target in ('family','future_close'):
        ba=1-bl if bl is not None else None;ma=1-ml if ml is not None else None;effect=None if ba is None else ((ma-ba)/max(abs(ba),1e-9) if ba else (1. if ma>ba else 0.))
    else: effect=None if bl is None else (bl-ml)/max(abs(bl),1e-9)
    return {'baseline_loss':bl,'memory_loss':ml,'relative_effect':effect,'rows':rows}
def choose(dev,target):
    scored=[]
    for s in candidates(target):
        e=cv_eval(dev,target,s);scored.append((e['memory_loss'] if e['memory_loss'] is not None else 1e99,json.dumps(s,sort_keys=True),s,e))
    scored.sort(key=lambda x:(x[0],x[1]));best=scored[0]
    # Selective promotion: memory must beat baseline by >=10% in development CV; otherwise abstain to baseline.
    promote=best[3]['relative_effect'] is not None and best[3]['relative_effect']>=.10 and best[2][0]!='baseline'
    return {'target':target,'candidate':best[2],'development_cv':best[3],'promote_memory':promote,'top5':[{'spec':x[2],'memory_loss':x[0],'relative_effect':x[3]['relative_effect']} for x in scored[:5]]}
def evaluate(dev,lock):
    decisions={t:choose(dev,t) for t in TARGETS}
    folds=[]
    for q in lock:
        f={'heldout':q['id']}
        for t in TARGETS:
            truth=q['target'].get(t)
            if truth is None:continue
            b=baseline(dev,t);d=decisions[t];raw=pred(d['candidate'],dev,q,t);used=raw if d['promote_memory'] else b
            f[t]={'truth':truth,'baseline':b,'raw_memory':raw,'served_prediction':used,'used_memory':d['promote_memory'],'baseline_loss':loss(t,b,truth),'served_loss':loss(t,used,truth)}
        folds.append(f)
    metrics={};effects={}
    for t in TARGETS:
        rs=[f[t] for f in folds if t in f]; bl=mean([r['baseline_loss'] for r in rs]); sl=mean([r['served_loss'] for r in rs])
        if t in ('family','future_close'):
            ba=None if bl is None else 1-bl;sa=None if sl is None else 1-sl;metrics[t]={'baseline_accuracy':ba,'selective_memory_accuracy':sa};effects[t]=None if ba is None else ((sa-ba)/max(abs(ba),1e-9) if ba else (1. if sa>ba else 0.))
        else:
            metrics[t]={'baseline_mae':bl,'selective_memory_mae':sl};effects[t]=None if bl is None else (bl-sl)/max(abs(bl),1e-9)
    vals=[v for v in effects.values() if v is not None];passed=bool(vals) and max(vals)>=.10 and min(vals)>=-.05
    return {'selection':decisions,'metrics':metrics,'relative_effect':effects,'pass_rule':passed,'folds':folds}
# Acquire development + lockbox, derive a development-only close threshold, then freeze.
dev=[derive(*m) for m in DEV_CLIPS]; lock=[derive(*m) for m in LOCKBOX]
dev_cross=[m['target']['future_min_cross'] for m in dev if m['target']['future_min_cross'] is not None];close_threshold=statistics.median(dev_cross)
enrich_close(dev,close_threshold);enrich_close(lock,close_threshold)
result=evaluate(dev,lock)
pack={'schema':'assetgraph-evidence/req008-selective-memory-v4','request_id':'REQ-008','source_commit':COMMIT,'protocol':{'development_bank':[x[0] for x in DEV_CLIPS],'fresh_lockbox':[x[0] for x in LOCKBOX],'visible_fraction':0.35,'promotion_rule':'target memory promoted only if development LOMO relative improvement >=10%; otherwise abstain/fallback baseline','final_rule':'at least one lockbox metric >=10% improvement and no metric worse than -5%'},'close_threshold_m_from_development_only':close_threshold,'development':dev,'lockbox':lock,'result':result,'leakage_controls':{'lockbox_not_used_for_model_or_gate_selection':True,'future_targets_not_in_query_features':True,'abstention_policy_frozen_before_lockbox':True}}
out=OUT/'req008_selective_memory_v4.json';out.write_text(json.dumps(pack,indent=2));print(json.dumps({'pass_rule':result['pass_rule'],'relative_effect':result['relative_effect'],'selection':{k:{'promote_memory':v['promote_memory'],'candidate':v['candidate'],'dev_effect':v['development_cv']['relative_effect']} for k,v in result['selection'].items()},'evidence':str(out)},indent=2))
