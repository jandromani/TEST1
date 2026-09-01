from __future__ import annotations

import base64, hashlib, importlib.util, json, math, os, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
V4=ROOT/'build_intelligence_console_v4_model_lab_fixed2.py'
spec=importlib.util.spec_from_file_location('assetgraph_v4_for_v5',V4)
v4=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(v4)
V4_HTML=REPO/'dist'/'ASSETGRAPH_INTELLIGENCE_CONSOLE_v4_MODEL_LAB.html'
V4_MAN=REPO/'dist'/'assetgraph_console_v4_manifest.json'
OUT=REPO/'dist'/'ASSETGRAPH_INTELLIGENCE_CONSOLE_v5_IDENTITY_ARENA.html'
OUTMAN=REPO/'dist'/'assetgraph_console_v5_manifest.json'
WORK=ROOT/'assets'/'console_v5_identity_arena'; WORK.mkdir(parents=True,exist_ok=True)
DINO_COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def norm(v):
    v=np.asarray(v,dtype=np.float32);return v/max(float(np.linalg.norm(v)),1e-9)

def cos(a,b):return float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-9))

def data_from_html(s):
    m=re.search(r'const DATA=(.*?);\nlet missionIndex=',s,re.S)
    if not m:raise RuntimeError('DATA payload not found')
    return json.loads(m.group(1)),m

def replace_data(s,data):
    _,m=data_from_html(s); packed=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    return s[:m.start(1)]+packed+s[m.end(1):]

def decode(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(base64.b64decode(frame['image'].split(',',1)[1]));return path

def crop(path,box,pad=.18):
    with Image.open(path) as im:
        im=im.convert('RGB');w,h=im.size;x1,y1,x2,y2=map(float,box);bw=max(1.,x2-x1);bh=max(1.,y2-y1)
        x1=max(0,x1-bw*pad);y1=max(0,y1-bh*pad);x2=min(w,x2+bw*pad);y2=min(h,y2+bh*pad)
        return im.crop((int(x1),int(y1),max(int(x1)+1,int(x2)),max(int(y1)+1,int(y2)))).copy()

def handcrafted(path,box):return norm(v4.mod.crop_desc(path,box))

def geometry(box,w,h):
    x1,y1,x2,y2=map(float,box);bw=max(1.,x2-x1);bh=max(1.,y2-y1)
    return np.array([math.log(max((bw*bh)/(w*h),1e-9)),math.log(max(bw/bh,1e-9)),(x1+x2)/(2*w),(y1+y2)/(2*h)],dtype=np.float32)

def context(frame,bi):
    bs=frame.get('model_boxes',[]);b=bs[bi]['xyxy'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;diag=max(math.hypot(frame['width'],frame['height']),1.)
    radial=np.zeros(5,np.float32);angular=np.zeros(8,np.float32);same=0
    for j,o in enumerate(bs):
        if j==bi:continue
        q=o['xyxy'];dx=(q[0]+q[2])/2-cx;dy=(q[1]+q[3])/2-cy;d=math.hypot(dx,dy)/diag
        radial[min(4,int(d/.08))]+=1;ang=(math.atan2(dy,dx)+math.pi)/(2*math.pi);angular[min(7,int(ang*8))]+=1
        same+=str(o.get('class_name','')).lower()==str(bs[bi].get('class_name','')).lower()
    return norm(np.r_[radial,angular,float(same)])

def load_dino():
    import torch
    repo=Path(os.environ['DINOV2_DIR']);m=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=True);m.eval()
    ckdir=Path(torch.hub.get_dir())/'checkpoints';matches=sorted(ckdir.glob('*dinov2*vits14*.pth'))
    return m,(matches[0] if matches else None)

def embed(model,crops):
    import torch
    from torchvision import transforms
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))]);out=[]
    with torch.no_grad():
        for st in range(0,len(crops),24):
            x=torch.stack([tf(c) for c in crops[st:st+24]]);z=model.forward_features(x)['x_norm_clstoken'].cpu().numpy();out += [norm(v) for v in z]
    return out

def prepare(data):
    missions={m['id']:m for m in data['missions']};obs={};imgs=[];keys=[]
    for mid in ('M04','M07'):
        for fi,f in enumerate(missions[mid]['frames']):
            p=decode(f,WORK/mid/f'{fi:02d}.jpg')
            for bi,b in enumerate(f.get('model_boxes',[])):
                k=f'{mid}:{fi}:{bi}';r={'key':k,'mission':mid,'frame':fi,'bi':bi,'track_id':b.get('track_id'),'class_name':b.get('class_name'),'confidence':float(b.get('confidence',0)),'matched_gt_track':b.get('matched_gt_track'),'xyxy':list(map(float,b['xyxy']))}
                r['hand']=handcrafted(p,b['xyxy']);r['geo']=geometry(b['xyxy'],f['width'],f['height']);r['ctx']=context(f,bi);obs[k]=r;imgs.append(crop(p,b['xyxy']));keys.append(k)
    model,wp=load_dino();zz=embed(model,imgs)
    for k,z in zip(keys,zz):obs[k]['dino']=z
    evidence={'repo':'facebookresearch/dinov2','repo_commit':DINO_COMMIT,'model':'dinov2_vits14','license':'Apache-2.0 code and model weights','weight_sha256':sha(wp) if wp and wp.exists() else None,'weight_file':wp.name if wp else None}
    return obs,evidence

def proto(rows):
    return {'class_name':Counter(str(x.get('class_name','')) for x in rows).most_common(1)[0][0],'hand':norm(np.mean([x['hand'] for x in rows],0)),'dino':norm(np.mean([x['dino'] for x in rows],0)),'geo':np.mean([x['geo'] for x in rows],0),'ctx':norm(np.mean([x['ctx'] for x in rows],0))}

def groups(rows):
    g=defaultdict(list)
    for r in rows:g[f"T{r['track_id']}" if r.get('track_id') is not None else f"O{r['frame']}-{r['bi']}"] .append(r)
    out=[]
    for gid,rr in g.items():
        gt=[str(x['matched_gt_track']) for x in rr if x.get('matched_gt_track') is not None];maj=Counter(gt).most_common(1)[0][0] if gt else None
        out.append({'candidate_id':gid,'track_id':rr[0].get('track_id'),'class_name':Counter(str(x.get('class_name','')) for x in rr).most_common(1)[0][0],'confidence':float(np.mean([x['confidence'] for x in rr])),'observations':len(rr),'frames':sorted({x['frame'] for x in rr}),'eval_gt_track':int(maj) if maj and maj.isdigit() else maj,'gt_purity':sum(x==maj for x in gt)/len(gt) if gt else None,'hand':norm(np.mean([x['hand'] for x in rr],0)),'dino':norm(np.mean([x['dino'] for x in rr],0)),'geo':np.mean([x['geo'] for x in rr],0),'ctx':norm(np.mean([x['ctx'] for x in rr],0))})
    return out

def components(q,c):
    ds=math.exp(-abs(float(q['geo'][0]-c['geo'][0]))); ar=math.exp(-abs(float(q['geo'][1]-c['geo'][1])));pd=math.hypot(float(q['geo'][2]-c['geo'][2]),float(q['geo'][3]-c['geo'][3]));ps=math.exp(-2*pd)
    return {'hand':(cos(q['hand'],c['hand'])+1)/2,'dino':(cos(q['dino'],c['dino'])+1)/2,'geometry':.5*ds+.3*ar+.2*ps,'context':max(0,min(1,(cos(q['ctx'],c['ctx'])+1)/2))}

def rank(q,gallery,w):
    out=[]
    for c in gallery:
        if str(c['class_name']).lower()!=str(q['class_name']).lower():continue
        comp=components(q,c);score=sum(comp[k]*v for k,v in w.items());out.append({k:c[k] for k in ('candidate_id','track_id','class_name','confidence','observations','frames','eval_gt_track','gt_purity')}|{'score':float(score),'components':comp})
    out.sort(key=lambda x:(-x['score'],str(x['candidate_id'])));return out

ENGINES={'HANDCRAFTED_V4':{'hand':1.0},'DINOV2_SMALL':{'dino':1.0},'DINOV2_GEOMETRY':{'dino':.82,'geometry':.18},'DINOV2_CONTEXT':{'dino':.78,'context':.22},'FUSION':{'dino':.68,'hand':.10,'geometry':.08,'context':.14}}

def dev_tasks(obs):
    all04=[x for x in obs.values() if x['mission']=='M04' and x.get('matched_gt_track') is not None];by=defaultdict(list)
    for x in all04:by[str(x['matched_gt_track'])].append(x)
    tasks=[]
    for target,rr in by.items():
        rr=sorted(rr,key=lambda x:(x['frame'],x['bi']));fs=sorted({x['frame'] for x in rr})
        if len(fs)<2:continue
        cut=fs[max(0,len(fs)//2-1)];en=[x for x in rr if x['frame']<=cut];gal=[x for x in all04 if x['frame']>cut]
        if en and any(str(x.get('matched_gt_track'))==target for x in gal):tasks.append((target,proto(en),groups(gal)))
    return tasks

def dev_eval(tasks,w):
    rows=[]
    for target,q,g in tasks:
        rr=rank(q,g,w)
        if not rr:continue
        tr=next((i+1 for i,x in enumerate(rr) if str(x.get('eval_gt_track'))==target),None);margin=rr[0]['score']-(rr[1]['score'] if len(rr)>1 else 0);rows.append((rr[0]['score'],margin,tr==1))
    valid=len(rows);return {'queries':valid,'recall_at_1':sum(x[2] for x in rows)/max(valid,1),'rows':rows}

def calibrate(e):
    rows=e['rows']
    if not rows:return {'min_score':1.1,'min_margin':1.1,'policy':'ABSTAIN_ALL_NO_DEV'}
    ss=sorted({round(x[0],6) for x in rows});mm=sorted({round(x[1],6) for x in rows});best=None
    for s in [min(ss)-1e-6]+ss:
        for m in [min(mm)-1e-6]+mm:
            picked=[x for x in rows if x[0]>=s and x[1]>=m];fp=sum(not x[2] for x in picked);tp=sum(x[2] for x in picked);cand=(fp==0,tp,len(picked),-s,-m,s,m)
            if best is None or cand>best:best=cand
    return {'min_score':float(best[-2]),'min_margin':float(best[-1]),'development_confirmations':int(best[2]),'development_false_confirmations':0,'policy':'ZERO_FALSE_CONFIRMATIONS_ON_M04_DEVELOPMENT'}

def arena(data):
    obs,dino=prepare(data);tasks=dev_tasks(obs);dev={n:dev_eval(tasks,w) for n,w in ENGINES.items()};cal={n:calibrate(e) for n,e in dev.items()}
    a={str(x['matched_gt_track']) for x in obs.values() if x['mission']=='M04' and x.get('matched_gt_track') is not None};b={str(x['matched_gt_track']) for x in obs.values() if x['mission']=='M07' and x.get('matched_gt_track') is not None};targets=sorted(a&b,key=int)
    m04=[x for x in obs.values() if x['mission']=='M04'];gallery=groups([x for x in obs.values() if x['mission']=='M07']);eng={}
    for name,w in ENGINES.items():
        rows=[]
        for t in targets:
            en=[x for x in m04 if str(x.get('matched_gt_track'))==t]
            if not en:continue
            rr=rank(proto(en),gallery,w)
            if not rr:continue
            tr=next((i+1 for i,x in enumerate(rr) if str(x.get('eval_gt_track'))==t),None);margin=rr[0]['score']-(rr[1]['score'] if len(rr)>1 else 0);c=cal[name];dec='CONFIRMED' if rr[0]['score']>=c['min_score'] and margin>=c['min_margin'] else 'CANDIDATE';correct=dec=='CONFIRMED' and str(rr[0].get('eval_gt_track'))==t
            rows.append({'target_gt':int(t),'enrollment_observations':len(en),'target_rank':tr,'candidate_count':len(rr),'top1_margin':margin,'decision':dec,'confirmed_correct':correct,'top1':rr[0],'top5':rr[:5]})
        valid=[r for r in rows if r['target_rank'] is not None];conf=[r for r in rows if r['decision']=='CONFIRMED'];cc=[r for r in conf if r['confirmed_correct']]
        met={'queries':len(valid),'recall_at_1':sum(r['target_rank']==1 for r in valid)/max(len(valid),1),'recall_at_5':sum(r['target_rank']<=5 for r in valid)/max(len(valid),1),'mrr':float(np.mean([1/r['target_rank'] for r in valid])) if valid else 0,'confirmed_precision':len(cc)/max(len(conf),1),'coverage':len(conf)/max(len(rows),1),'false_confirmation_rate':(len(conf)-len(cc))/max(len(rows),1),'abstention_rate':1-len(conf)/max(len(rows),1)}
        eng[name]={'weights':w,'calibration':c,'development':{k:v for k,v in dev[name].items() if k!='rows'},'metrics':met,'rows':rows}
    base=eng['HANDCRAFTED_V4']['metrics'];best_name=sorted(eng,key=lambda n:(-eng[n]['metrics']['recall_at_1'],-eng[n]['metrics']['mrr'],eng[n]['metrics']['false_confirmation_rate'],n))[0];best=eng[best_name]['metrics'];dr=best['recall_at_1']-base['recall_at_1'];dm=best['mrr']-base['mrr'];prom=(dr>=.10 or dm>=.10) and best['confirmed_precision']>=.90 and best['false_confirmation_rate']<=.10
    return {'schema':'assetgraph-identity-arena/v1','stage':'C19_PERSISTENT_IDENTITY_ENGINE','principle':'GT may enroll/evaluate benchmark identities; GT is never an input to M07 candidate ranking.','ranking_leakage':{'gt_used_for_m07_ranking':False,'m07_gt_used_only_after_ranking':True,'abstention_thresholds_calibrated_on_m04_only':True},'dino':dino,'candidate_unit':'MODEL_TRACKLET_OR_SINGLETON','shared_gt_targets':[int(x) for x in targets],'shared_target_count':len(targets),'m07_candidate_units':len(gallery),'development_task_count':len(tasks),'engines':eng,'best_engine':best_name,'baseline_engine':'HANDCRAFTED_V4','promotion_gate':{'pass':prom,'best_engine':best_name,'recall_at_1_improvement_vs_baseline':dr,'mrr_improvement_vs_baseline':dm,'requirements':['R@1 +10pp OR MRR +0.10 vs handcrafted baseline','confirmed precision >= 0.90','false confirmation rate <= 0.10']},'commercial_contract':{'decision_states':['CONFIRMED','CANDIDATE','UNKNOWN'],'buyer_value':['ranked identity hypotheses','explicit abstention','per-candidate score decomposition','GT judge separated from runtime ranking','visible license boundary']}}

def pct(x):return f'{100*float(x):.1f}%'
def patch(s,data,A):
    data['identity_arena']=A;s=replace_data(s,data);s=s.replace('ASSETGRAPH INTELLIGENCE CONSOLE v4','ASSETGRAPH INTELLIGENCE CONSOLE v5',2).replace('INTELLIGENCE CONSOLE v4','INTELLIGENCE CONSOLE v5',2).replace('✦ SHOWCASE v4','✦ SHOWCASE v5',1)
    nav='<button data-page="failures">FAILURE ANALYSIS</button>';s=s.replace(nav,'<button data-page="identity">IDENTITY ARENA</button>'+nav,1)
    rows=''.join(f"<tr><td class='mono'>{n}</td><td>{pct(p['metrics']['recall_at_1'])}</td><td>{pct(p['metrics']['recall_at_5'])}</td><td>{p['metrics']['mrr']:.3f}</td><td>{next((r['target_rank'] for r in p['rows'] if str(r['target_gt'])=='386'),'—')}</td><td>{pct(p['metrics']['confirmed_precision'])}</td><td>{pct(p['metrics']['coverage'])}</td><td>{pct(p['metrics']['false_confirmation_rate'])}</td></tr>" for n,p in A['engines'].items())
    cards=[]
    for n,p in A['engines'].items():
        r=next((x for x in p['rows'] if str(x['target_gt'])=='386'),None)
        if not r:continue
        t=r['top1'];cards.append(f"<div class='v3card'><div class='pad'><span class='v3badge {'gt' if r['target_rank']==1 else 'exp'}'>{n}</span><h3>GT-386 rank {r['target_rank'] or 'MISS'} · {r['decision']}</h3><p>Top-1 {t['candidate_id']} → GT-{t.get('eval_gt_track')} · score {t['score']:.4f} · margin {r['top1_margin']:.4f}</p><div class='v3callout {'good' if r['target_rank']==1 else 'warn'}'>DINO {t['components']['dino']:.3f} · handcrafted {t['components']['hand']:.3f} · geometry {t['components']['geometry']:.3f} · context {t['components']['context']:.3f}</div><pre class='mono' style='white-space:pre-wrap;max-height:220px;overflow:auto'>{json.dumps(r['top5'],indent=2)}</pre></div></div>")
    base=A['engines']['HANDCRAFTED_V4']['metrics'];best=A['engines'][A['best_engine']]['metrics'];page=f'''<section class="v3page" id="v3-identity"><div class="v3eyebrow">C19 · PERSISTENT IDENTITY ENGINE · IDENTITY ARENA</div><div class="v3section"><h2>Rank, explain, or abstain.</h2><div class="v3sectionLead">The commercial boundary is not another detector: it is auditable identity resolution across time. M07 GT is hidden until every ranking is frozen.</div></div><div class="v3metrics"><div class="v3metric"><strong>{A['shared_target_count']}</strong><span>SHARED IDENTITIES</span></div><div class="v3metric"><strong>{A['m07_candidate_units']}</strong><span>CANDIDATE UNITS</span></div><div class="v3metric"><strong>{pct(base['recall_at_1'])}</strong><span>BASELINE R@1</span></div><div class="v3metric"><strong>{pct(best['recall_at_1'])}</strong><span>BEST R@1</span></div><div class="v3metric"><strong>{'PASS' if A['promotion_gate']['pass'] else 'HOLD'}</strong><span>PROMOTION</span></div></div><div class="v3callout {'good' if A['promotion_gate']['pass'] else 'warn'}"><b>{A['best_engine']}</b> · ΔR@1 {A['promotion_gate']['recall_at_1_improvement_vs_baseline']:+.3f} · ΔMRR {A['promotion_gate']['mrr_improvement_vs_baseline']:+.3f}. Green build ≠ automatic model promotion.</div><div class="v3section"><h2>Engine tournament</h2><table class="v3table"><thead><tr><th>ENGINE</th><th>R@1</th><th>R@5</th><th>MRR</th><th>GT-386 RANK</th><th>CONFIRM PREC.</th><th>COVERAGE</th><th>FALSE CONF.</th></tr></thead><tbody>{rows}</tbody></table></div><div class="v3section"><h2>GT-386 duel</h2><div class="v3grid">{''.join(cards)}</div></div><div class="v3section"><h2>Commercial evidence contract</h2><div class="v3product"><div class="v3productCol"><h3>CONFIRMED</h3><p>Automatic identity only when the frozen M04 calibration allows it.</p></div><div class="v3productCol"><h3>CANDIDATE</h3><p>Ranked hypothesis with score decomposition and provenance for analyst review.</p></div><div class="v3productCol"><h3>UNKNOWN</h3><p>Refuse to fabricate identity when evidence is insufficient.</p></div></div></div><div class="v3section"><div class="v3split"><div class="v3card"><div class="pad"><h3>Leakage controls</h3><pre class="mono">{json.dumps(A['ranking_leakage'],indent=2)}</pre></div></div><div class="v3card"><div class="pad"><h3>DINOv2 evidence</h3><pre class="mono">{json.dumps(A['dino'],indent=2)}</pre></div></div></div></div></section>'''
    marker='<section class="v3page" id="v3-failures"></section>';s=s.replace(marker,page+marker,1);s=s.replace('</head>','<style id="assetgraph-v5-style">#v3-identity .v3grid{grid-template-columns:repeat(3,minmax(0,1fr))}#v3-identity pre{font-size:9px;color:#8facb3;white-space:pre-wrap}@media(max-width:900px){#v3-identity .v3grid{grid-template-columns:1fr}}</style></head>',1);return s

def main():
    s=V4_HTML.read_text(encoding='utf-8');data,_=data_from_html(s);man=json.loads(V4_MAN.read_text());A=arena(data);man['v5']=A;OUT.write_text(patch(s,data,A),encoding='utf-8');OUTMAN.write_text(json.dumps(man,indent=2),encoding='utf-8');print(json.dumps({'html':str(OUT),'bytes':OUT.stat().st_size,'shared_targets':A['shared_target_count'],'candidate_units':A['m07_candidate_units'],'best_engine':A['best_engine'],'promotion':A['promotion_gate'],'metrics':{k:v['metrics'] for k,v in A['engines'].items()}},indent=2))
if __name__=='__main__':main()
