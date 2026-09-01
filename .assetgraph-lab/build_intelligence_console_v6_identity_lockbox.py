from __future__ import annotations

import importlib.util, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
V5=ROOT/'build_intelligence_console_v5_identity_arena.py'
spec=importlib.util.spec_from_file_location('assetgraph_v5_for_v6',V5)
v5=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(v5)

OUT=REPO/'dist'/'ASSETGRAPH_INTELLIGENCE_CONSOLE_v6_IDENTITY_LOCKBOX.html'
OUTMAN=REPO/'dist'/'assetgraph_console_v6_manifest.json'

COMPLEXITY={'HANDCRAFTED_V4':0,'DINOV2_SMALL':1,'DINOV2_GEOMETRY':2,'DINOV2_CONTEXT':2,'FUSION':4}


def gt_id(x):
    v=x.get('matched_gt_track')
    return None if v is None else str(v)

def target_ids(rows): return sorted({gt_id(x) for x in rows if gt_id(x) is not None},key=int)

def reciprocal(rank): return 0.0 if not rank else 1.0/rank

def dev_protocol(obs,name,w):
    rows=[x for x in obs.values() if x['mission']=='M04' and gt_id(x) is not None]
    frames=sorted({x['frame'] for x in rows}); cut=frames[max(0,len(frames)//2-1)]
    early=[x for x in rows if x['frame']<=cut]; late=[x for x in rows if x['frame']>cut]
    gallery=v5.groups(late); tasks=[]
    for t in target_ids(early):
        enroll=[x for x in early if gt_id(x)==t]
        if not enroll: continue
        full=v5.rank(v5.proto(enroll),gallery,w)
        tr=next((i+1 for i,x in enumerate(full) if str(x.get('eval_gt_track'))==t),None)
        if tr is None: continue
        pos_top=full[0]; pos_margin=pos_top['score']-(full[1]['score'] if len(full)>1 else 0.0)
        neg_gallery=[x for x in gallery if str(x.get('eval_gt_track'))!=t]
        neg=v5.rank(v5.proto(enroll),neg_gallery,w)
        neg_top=neg[0] if neg else None; neg_margin=(neg_top['score']-(neg[1]['score'] if len(neg)>1 else 0.0)) if neg else -1.0
        tasks.append({'target_gt':int(t),'target_rank':tr,'pos_score':pos_top['score'],'pos_margin':pos_margin,'pos_top_gt':pos_top.get('eval_gt_track'),'neg_score':neg_top['score'] if neg_top else -1.0,'neg_margin':neg_margin,'neg_top_gt':neg_top.get('eval_gt_track') if neg_top else None})
    candidates=[]
    for r in tasks:
        candidates.append((r['pos_score'],r['pos_margin'],str(r['pos_top_gt'])==str(r['target_gt']),'POS'))
        candidates.append((r['neg_score'],r['neg_margin'],False,'NEG_OPEN_SET'))
    scores=sorted({round(x[0],6) for x in candidates}); margins=sorted({round(x[1],6) for x in candidates})
    best=None
    for s in ([min(scores)-1e-6] if scores else [1.1])+scores:
        for m in ([min(margins)-1e-6] if margins else [1.1])+margins:
            accepted=[x for x in candidates if x[0]>=s and x[1]>=m]
            false=sum(not x[2] for x in accepted); tp=sum(x[2] for x in accepted); pos_total=max(sum(x[3]=='POS' for x in candidates),1)
            key=(false==0,tp,tp/pos_total,-s,-m)
            if best is None or key>best[0]: best=(key,s,m,false,tp)
    s=best[1] if best else 1.1; mg=best[2] if best else 1.1
    correct_scores=[r['pos_score'] for r in tasks if str(r['pos_top_gt'])==str(r['target_gt'])]
    unknown_floor=float(np.quantile(correct_scores,.10)-.02) if correct_scores else 1.1
    r1=sum(r['target_rank']==1 for r in tasks)/max(len(tasks),1); mrr=float(np.mean([reciprocal(r['target_rank']) for r in tasks])) if tasks else 0.0
    confirmed=[r for r in tasks if r['pos_score']>=s and r['pos_margin']>=mg]
    confirmed_correct=[r for r in confirmed if str(r['pos_top_gt'])==str(r['target_gt'])]
    neg_false=sum(r['neg_score']>=s and r['neg_margin']>=mg for r in tasks)
    return {'engine':name,'split':{'cut_frame_index':cut,'development_only':True},'queries':len(tasks),'closed_recall_at_1':r1,'closed_mrr':mrr,'confirm_threshold':{'min_score':float(s),'min_margin':float(mg),'unknown_score_floor':unknown_floor,'calibrated_without_M07':True},'confirmed_correct':len(confirmed_correct),'confirmed_coverage':len(confirmed)/max(len(tasks),1),'open_set_negative_tasks':len(tasks),'open_set_false_confirmations':int(neg_false),'tasks':tasks}

def select_engine(obs):
    dev={n:dev_protocol(obs,n,w) for n,w in v5.ENGINES.items()}
    def key(n):
        d=dev[n]
        return (-d['closed_recall_at_1'],-d['confirmed_coverage'],-d['closed_mrr'],d['open_set_false_confirmations'],COMPLEXITY[n],n)
    selected=sorted(dev,key=key)[0]
    return selected,dev

def decide(row,thr):
    score=row[0]['score']; margin=score-(row[1]['score'] if len(row)>1 else 0.0)
    if score>=thr['min_score'] and margin>=thr['min_margin']: return 'CONFIRMED',margin
    if score<thr['unknown_score_floor']: return 'UNKNOWN',margin
    return 'CANDIDATE',margin

def eval_engine(obs,name,w,dev):
    m04=[x for x in obs.values() if x['mission']=='M04' and gt_id(x) is not None]
    m07=[x for x in obs.values() if x['mission']=='M07']
    m04ids=set(target_ids(m04)); m07ids=set(target_ids(m07)); shared=sorted(m04ids&m07ids,key=int); absent=sorted(m04ids-m07ids,key=int); new=sorted(m07ids-m04ids,key=int)
    gallery=v5.groups(m07); rows=[]
    for t in sorted(m04ids,key=int):
        enroll=[x for x in m04 if gt_id(x)==t]; ranked=v5.rank(v5.proto(enroll),gallery,w)
        if not ranked: continue
        decision,margin=decide(ranked,dev['confirm_threshold']); tr=next((i+1 for i,x in enumerate(ranked) if str(x.get('eval_gt_track'))==t),None)
        expected='PRESENT' if t in m07ids else 'ABSENT_OPEN_SET'
        correct_confirm=decision=='CONFIRMED' and expected=='PRESENT' and str(ranked[0].get('eval_gt_track'))==t
        false_merge=decision=='CONFIRMED' and not correct_confirm
        rows.append({'target_gt':int(t),'expected':expected,'decision':decision,'target_rank':tr,'top1_margin':margin,'correct_confirmation':correct_confirm,'false_merge':false_merge,'top1':ranked[0],'top5':ranked[:5]})
    pos=[r for r in rows if r['expected']=='PRESENT' and r['target_rank'] is not None]; neg=[r for r in rows if r['expected']=='ABSENT_OPEN_SET']; conf=[r for r in rows if r['decision']=='CONFIRMED']; cc=[r for r in conf if r['correct_confirmation']]
    new_units=[{k:c.get(k) for k in ('candidate_id','track_id','class_name','confidence','observations','frames','eval_gt_track','gt_purity')} for c in gallery if c.get('eval_gt_track') is not None and str(c.get('eval_gt_track')) in new]
    metrics={'present_queries':len(pos),'absent_queries':len(neg),'recall_at_1':sum(r['target_rank']==1 for r in pos)/max(len(pos),1),'recall_at_5':sum(r['target_rank']<=5 for r in pos)/max(len(pos),1),'mrr':float(np.mean([reciprocal(r['target_rank']) for r in pos])) if pos else 0.0,'confirmed_precision':len(cc)/max(len(conf),1),'confirmed_coverage_present':sum(r['correct_confirmation'] for r in pos)/max(len(pos),1),'false_merge_rate_absent':sum(r['false_merge'] for r in neg)/max(len(neg),1),'open_set_safe_rate':sum(not r['false_merge'] for r in neg)/max(len(neg),1),'unknown_rate_absent':sum(r['decision']=='UNKNOWN' for r in neg)/max(len(neg),1),'candidate_rate_absent':sum(r['decision']=='CANDIDATE' for r in neg)/max(len(neg),1),'overall_false_merges':sum(r['false_merge'] for r in rows)}
    return {'engine':name,'weights':w,'frozen_development_protocol':dev,'shared_gt_targets':[int(x) for x in shared],'absent_gt_targets':[int(x) for x in absent],'new_gt_targets':[int(x) for x in new],'m07_candidate_units':len(gallery),'new_unenrolled_candidate_units':new_units,'metrics':metrics,'rows':rows}

def pct(x):return f'{100*float(x):.1f}%'
def patch(s,data,c19):
    data['identity_lockbox']=c19; s=v5.replace_data(s,data)
    s=s.replace('ASSETGRAPH INTELLIGENCE CONSOLE v5','ASSETGRAPH INTELLIGENCE CONSOLE v6',2).replace('INTELLIGENCE CONSOLE v5','INTELLIGENCE CONSOLE v6',2).replace('✦ SHOWCASE v5','✦ SHOWCASE v6',1)
    s=s.replace('<button data-page="identity">IDENTITY ARENA</button>','<button data-page="identity">IDENTITY ARENA</button><button data-page="lockbox">IDENTITY LOCKBOX</button>',1)
    e=c19['lockbox']; b=c19['baseline_lockbox']; m=e['metrics']; bm=b['metrics']; gate=c19['promotion_gate']
    rr=''.join(f"<tr><td>GT-{r['target_gt']}</td><td>{r['expected']}</td><td>{r['decision']}</td><td>{r['target_rank'] or '—'}</td><td>GT-{r['top1'].get('eval_gt_track')}</td><td>{r['top1']['score']:.4f}</td><td>{r['top1_margin']:.4f}</td><td>{'YES' if r['false_merge'] else 'NO'}</td></tr>" for r in e['rows'])
    devrows=''.join(f"<tr><td>{n}</td><td>{pct(d['closed_recall_at_1'])}</td><td>{d['closed_mrr']:.3f}</td><td>{pct(d['confirmed_coverage'])}</td><td>{d['open_set_false_confirmations']}</td><td>{'SELECTED' if n==c19['selected_engine'] else '—'}</td></tr>" for n,d in c19['development_tournament'].items())
    page=f'''<section class="v3page" id="v3-lockbox"><div class="v3eyebrow">C19.1 · OPEN-SET PERSISTENT IDENTITY · LOCKBOX</div><div class="v3section"><h2>Do not invent continuity.</h2><div class="v3sectionLead">Engine and thresholds are selected on M04 only. M07 stays unseen until the policy is frozen. Absent identities must not be auto-merged.</div></div><div class="v3metrics"><div class="v3metric"><strong>{c19['selected_engine']}</strong><span>FROZEN ENGINE</span></div><div class="v3metric"><strong>{pct(m['recall_at_1'])}</strong><span>LOCKBOX R@1</span></div><div class="v3metric"><strong>{pct(m['confirmed_coverage_present'])}</strong><span>SAFE AUTO-COVERAGE</span></div><div class="v3metric"><strong>{pct(m['open_set_safe_rate'])}</strong><span>OPEN-SET SAFE</span></div><div class="v3metric"><strong>{'PASS' if gate['pass'] else 'HOLD'}</strong><span>MINI LOCKBOX</span></div></div><div class="v3callout {'good' if gate['pass'] else 'warn'}"><b>{gate['status']}</b> · baseline R@1 {pct(bm['recall_at_1'])} → frozen R@1 {pct(m['recall_at_1'])}; false merges on absent identities {m['overall_false_merges']}. This is a mini-lockbox, not a product freeze.</div><div class="v3section"><h2>Development-only engine selection</h2><table class="v3table"><thead><tr><th>ENGINE</th><th>DEV R@1</th><th>DEV MRR</th><th>SAFE COVERAGE</th><th>OPEN-SET FALSE CONF.</th><th>STATUS</th></tr></thead><tbody>{devrows}</tbody></table></div><div class="v3section"><h2>M07 lockbox decisions</h2><table class="v3table"><thead><tr><th>MEMORY</th><th>EXPECTED</th><th>DECISION</th><th>TRUE RANK</th><th>TOP-1 GT AFTER JUDGING</th><th>SCORE</th><th>MARGIN</th><th>FALSE MERGE</th></tr></thead><tbody>{rr}</tbody></table></div><div class="v3section"><div class="v3product"><div class="v3productCol"><h3>Present memories</h3><p>{len(e['shared_gt_targets'])} enrolled identities really recur in M07. R@5 {pct(m['recall_at_5'])}; MRR {m['mrr']:.3f}.</p></div><div class="v3productCol"><h3>Absent memories</h3><p>GT-{', GT-'.join(map(str,e['absent_gt_targets']))} are not in M07. False merge rate {pct(m['false_merge_rate_absent'])}.</p></div><div class="v3productCol"><h3>New arrivals</h3><p>GT-{', GT-'.join(map(str,e['new_gt_targets']))} appears without prior enrollment and must remain a new/unlinked observation until evidence says otherwise.</p></div></div></div><div class="v3section"><div class="v3split"><div class="v3card"><div class="pad"><h3>Frozen policy</h3><pre class="mono">{json.dumps(e['frozen_development_protocol']['confirm_threshold'],indent=2)}</pre></div></div><div class="v3card"><div class="pad"><h3>Leakage controls</h3><pre class="mono">{json.dumps(c19['leakage_controls'],indent=2)}</pre></div></div></div></div></section>'''
    s=s.replace('<section class="v3page" id="v3-failures"></section>',page+'<section class="v3page" id="v3-failures"></section>',1)
    s=s.replace('</head>','<style id="assetgraph-v6-style">#v3-lockbox pre{font-size:9px;color:#8facb3;white-space:pre-wrap}#v3-lockbox .v3metric:first-child strong{font-size:14px}</style></head>',1)
    return s

def main():
    # V5 import reconstructs v4 assets; evaluate once with DINO features.
    base=v5.V4_HTML.read_text(encoding='utf-8'); data,_=v5.data_from_html(base); obs,dino=v5.prepare(data)
    selected,dev=select_engine(obs)
    lock=eval_engine(obs,selected,v5.ENGINES[selected],dev[selected])
    baseline=eval_engine(obs,'HANDCRAFTED_V4',v5.ENGINES['HANDCRAFTED_V4'],dev['HANDCRAFTED_V4'])
    lm=lock['metrics']; bm=baseline['metrics']; dr=lm['recall_at_1']-bm['recall_at_1']; dm=lm['mrr']-bm['mrr']
    passed=(dr>=.10 or dm>=.10) and lm['confirmed_precision']>=.90 and lm['confirmed_coverage_present']>=.50 and lm['false_merge_rate_absent']==0 and lm['open_set_safe_rate']==1
    c19={'schema':'assetgraph-persistent-identity/c19.1-open-set-lockbox','selected_engine':selected,'selection_rule':'M04 development only: maximize closed R@1, safe-confirm coverage, MRR; minimize open-set false confirmations; then prefer lower complexity. M07 is not consulted.','development_tournament':dev,'lockbox':lock,'baseline_lockbox':baseline,'dino':dino,'leakage_controls':{'engine_selected_without_M07':True,'thresholds_calibrated_without_M07':True,'M07_GT_not_used_for_ranking':True,'M07_GT_revealed_only_after_decisions':True,'open_set_negative_calibration_uses_M04_only':True},'promotion_gate':{'pass':passed,'status':'MINI_LOCKBOX_PASS_NOT_PRODUCT_FROZEN' if passed else 'HOLD','delta_recall_at_1_vs_handcrafted':dr,'delta_mrr_vs_handcrafted':dm,'requirements':['development-only engine selection','R@1 +10pp OR MRR +0.10 vs handcrafted','confirmed precision >=90%','safe auto-coverage >=50%','zero false merges on M07 absent identities'],'product_frozen':False,'next_required':'expanded multi-window/multi-video identity lockbox before freeze'}}
    data['identity_arena']={'note':'v5 exploratory tournament superseded for promotion decisions by C19.1 development-only selection.'}
    html=patch(base,data,c19); OUT.write_text(html,encoding='utf-8')
    man=json.loads(v5.V4_MAN.read_text()); man['v6']=c19; man['v5_exploratory_note']='C19 v5 compared multiple engines on M07; informative only. C19.1 supersedes its promotion claim with development-only selection.'; OUTMAN.write_text(json.dumps(man,indent=2),encoding='utf-8')
    print(json.dumps({'html':str(OUT),'bytes':OUT.stat().st_size,'selected_engine':selected,'development':{n:{k:d[k] for k in ('closed_recall_at_1','closed_mrr','confirmed_coverage','open_set_false_confirmations')} for n,d in dev.items()},'lockbox':lm,'baseline':bm,'shared':lock['shared_gt_targets'],'absent':lock['absent_gt_targets'],'new':lock['new_gt_targets'],'promotion':c19['promotion_gate']},indent=2))
if __name__=='__main__':main()
