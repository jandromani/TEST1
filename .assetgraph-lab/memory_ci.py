from __future__ import annotations
import csv,json,math,statistics,hashlib,pathlib,urllib.request
from collections import defaultdict,Counter

ROOT=pathlib.Path(__file__).resolve().parent
ASSETS=ROOT/'assets'/'dut'; OUT=ROOT/'evidence'; ASSETS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
COMMIT='80b8c746833664cd1e5244fccad79c7f2a7cbe31'
MISSIONS=[('I01','intersection_01','intersection'),('I02','intersection_02','intersection'),('I03','intersection_03','intersection'),('I06','intersection_06','intersection'),('I11','intersection_11','intersection'),('I15','intersection_15','intersection'),('R01','roundabout_01','roundabout'),('R02','roundabout_02','roundabout'),('R05','roundabout_05','roundabout'),('R06','roundabout_06','roundabout'),('R08','roundabout_08','roundabout'),('R09','roundabout_09','roundabout')]
FEATURES=['ped_count','veh_count','ped_speed_early','veh_speed_early','direction_spread','vehicle_turn_index','density','tortuosity']

def download(url,p):
    if not p.exists(): urllib.request.urlretrieve(url,p)
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rows(path,kind):
    out=[]
    with open(path,newline='') as f:
        for n,r in enumerate(csv.DictReader(f),2):
            frame=int(r['frame']);x=float(r['x_est']);y=float(r['y_est'])
            if kind=='ped':
                vx=float(r.get('vx_est',r.get('xv_est',0)));vy=float(r.get('vy_est',r.get('yv_est',0)));speed=math.hypot(vx,vy);heading=math.atan2(vy,vx)
            else:
                speed=float(r['vel_est']);heading=float(r['psi_est']);vx=speed*math.cos(heading);vy=speed*math.sin(heading)
            out.append(dict(id=f'{kind}:{r["id"]}',kind=kind,frame=frame,x=x,y=y,speed=speed,heading=heading,vx=vx,vy=vy,row=n))
    return out
def mean(v):return statistics.fmean(v) if v else None
def ad(a,b):
    d=a-b
    while d>math.pi:d-=2*math.pi
    while d<-math.pi:d+=2*math.pi
    return d
def derive(mid,clip,family):
    pp=ASSETS/f'{clip}_traj_ped_filtered.csv';vp=ASSETS/f'{clip}_traj_veh_filtered.csv'
    base=f'https://raw.githubusercontent.com/dongfang-steven-yang/vci-dataset-dut/{COMMIT}/data/trajectories_filtered'
    download(f'{base}/{pp.name}',pp);download(f'{base}/{vp.name}',vp)
    ped=rows(pp,'ped');veh=rows(vp,'veh');allr=ped+veh;fs=sorted({r['frame'] for r in allr});lo,hi=fs[0],fs[-1];cut=lo+round((hi-lo)*.35)
    early=[r for r in allr if r['frame']<=cut];future=[r for r in allr if r['frame']>cut];ep=[r for r in early if r['kind']=='ped'];ev=[r for r in early if r['kind']=='veh'];fp=[r for r in future if r['kind']=='ped'];fv=[r for r in future if r['kind']=='veh']
    by=defaultdict(list)
    for r in early:by[r['id']].append(r)
    headings=[];tort=[];turn=[]
    for rr in by.values():
        rr=sorted(rr,key=lambda x:x['frame']);dx=rr[-1]['x']-rr[0]['x'];dy=rr[-1]['y']-rr[0]['y'];headings.append(math.atan2(dy,dx));path=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(rr,rr[1:]));net=math.hypot(dx,dy);tort.append(path/max(net,.25))
        if rr[0]['kind']=='veh':turn.extend(abs(ad(b['heading'],a['heading'])) for a,b in zip(rr,rr[1:]))
    xs=[r['x'] for r in early];ys=[r['y'] for r in early];area=max(1.,(max(xs)-min(xs))*(max(ys)-min(ys)))
    cx=statistics.fmean(math.cos(x) for x in headings) if headings else 1;cy=statistics.fmean(math.sin(x) for x in headings) if headings else 0
    pb=defaultdict(list);vb=defaultdict(list)
    for r in fp:pb[r['frame']].append(r)
    for r in fv:vb[r['frame']].append(r)
    mc=None
    for f,ps in pb.items():
        for p in ps:
            for v in vb.get(f,[]):
                d=math.hypot(p['x']-v['x'],p['y']-v['y']);mc=d if mc is None or d<mc else mc
    feat={'ped_count':len({r['id'] for r in ep}),'veh_count':len({r['id'] for r in ev}),'ped_speed_early':mean([r['speed'] for r in ep]) or 0.,'veh_speed_early':mean([r['speed'] for r in ev]) or 0.,'direction_spread':1-math.hypot(cx,cy),'vehicle_turn_index':mean(turn) or 0.,'density':len({r['id'] for r in early})/area,'tortuosity':mean(tort) or 1.}
    return {'id':mid,'clip':clip,'features':feat,'target':{'family':family,'future_vehicle_speed':mean([r['speed'] for r in fv]),'future_min_cross':mc},'cut_frame':cut,'input_hashes':{pp.name:sha(pp),vp.name:sha(vp)}}
def zdist(q,m,train):
    s=0.
    for k in FEATURES:
        vals=[x['features'][k] for x in train];mu=statistics.fmean(vals);sd=statistics.pstdev(vals) or 1.;s+=((q['features'][k]-m['features'][k])/sd)**2
    return math.sqrt(s)
def wavg(vals):
    vals=[x for x in vals if x[0] is not None]
    if not vals:return None
    sw=sum(w for _,w in vals);return sum(v*w for v,w in vals)/sw
def evaluate(ms,k=3):
    folds=[]
    for q in ms:
        tr=[x for x in ms if x['id']!=q['id']];cnt=Counter(x['target']['family'] for x in tr);basefam=sorted(cnt.items(),key=lambda x:(-x[1],x[0]))[0][0]
        pairs=sorted((zdist(q,m,tr),m) for m in tr)[:k];ws=[(m,1/(d+.15)) for d,m in pairs];fc=defaultdict(float)
        for m,w in ws:fc[m['target']['family']]+=w
        memfam=sorted(fc.items(),key=lambda x:(-x[1],x[0]))[0][0];sp=[x['target']['future_vehicle_speed'] for x in tr if x['target']['future_vehicle_speed'] is not None];cr=[x['target']['future_min_cross'] for x in tr if x['target']['future_min_cross'] is not None]
        bs=mean(sp);bc=mean(cr);mspeed=wavg([(m['target']['future_vehicle_speed'],w) for m,w in ws]);mcross=wavg([(m['target']['future_min_cross'],w) for m,w in ws]);th=statistics.median(cr) if cr else None;ac=q['target']['future_min_cross'];f={'heldout':q['id'],'truth_family':q['target']['family'],'baseline_family':basefam,'memory_family':memfam,'family_baseline_correct':basefam==q['target']['family'],'family_memory_correct':memfam==q['target']['family'],'analogues':[{'id':m['id'],'clip':m['clip'],'distance':d} for d,m in pairs]}
        av=q['target']['future_vehicle_speed']
        if av is not None and bs is not None and mspeed is not None:f['baseline_speed_abs_error']=abs(bs-av);f['memory_speed_abs_error']=abs(mspeed-av)
        if ac is not None and bc is not None and mcross is not None:
            f['baseline_cross_abs_error']=abs(bc-ac);f['memory_cross_abs_error']=abs(mcross-ac)
            if th is not None:
                truth=ac<th;f['baseline_close_correct']=(bc<th)==truth;f['memory_close_correct']=(mcross<th)==truth
        folds.append(f)
    def avg(k):
        v=[x[k] for x in folds if k in x];return statistics.fmean(v) if v else None
    metrics={'scenario_family_accuracy':{'baseline':avg('family_baseline_correct'),'memory':avg('family_memory_correct')},'future_vehicle_speed_mae_mps':{'baseline':avg('baseline_speed_abs_error'),'memory':avg('memory_speed_abs_error')},'future_min_ped_vehicle_distance_mae_m':{'baseline':avg('baseline_cross_abs_error'),'memory':avg('memory_cross_abs_error')},'future_close_interaction_accuracy':{'baseline':avg('baseline_close_correct'),'memory':avg('memory_close_correct')}}
    lift={}
    for k,v in metrics.items():
        b,m=v['baseline'],v['memory']
        if b is None or m is None:lift[k]=None
        elif 'mae' in k:lift[k]=(b-m)/max(abs(b),1e-9)
        else:lift[k]=(m-b)/max(abs(b),1e-9) if b else (1. if m>b else 0.)
    av=[x for x in lift.values() if x is not None];passed=bool(av) and max(av)>=.10 and min(av)>=-.05
    return {'metrics':metrics,'relative_lift':lift,'pass_rule':passed,'folds':folds}

missions=[derive(*m) for m in MISSIONS];res=evaluate(missions)
pack={'schema':'assetgraph-evidence/req008-memory-lift-v1','request_id':'REQ-008','dataset':'VCI-DUT','source_commit':COMMIT,'missions':missions,'result':res,'leakage_controls':{'family_not_in_features':True,'future_targets_not_in_features':True,'heldout_excluded':True,'fold_local_normalization':True}}
(OUT/'req008_memory_lift.json').write_text(json.dumps(pack,indent=2));print(json.dumps(res['metrics'],indent=2));print('LIFT',json.dumps(res['relative_lift'],indent=2));print('PASS',res['pass_rule'])
