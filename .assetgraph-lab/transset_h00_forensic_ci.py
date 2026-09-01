from __future__ import annotations
import collections, json, math, pathlib, re, statistics, time

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=True)
SCORE=ROOT/'evidence/c15e/transset_external_score.json'
LOCK=ROOT/'evidence/c15c/transset_lockbox_128.json'


def safe_div(a,b):return a/b if b else None

def quartile_cuts(vals):
    s=sorted(vals)
    def q(p):
        if not s:return 0
        x=(len(s)-1)*p;lo=int(math.floor(x));hi=int(math.ceil(x))
        return s[lo] if lo==hi else s[lo]*(hi-x)+s[hi]*(x-lo)
    return q(.25),q(.50),q(.75)

def main():
    t=time.time();score=json.loads(SCORE.read_text());lock=json.loads(LOCK.read_text())
    items={x['image_member']:x for x in lock['items']}
    rows=[]
    for r in score['per_image']:
        it=items[r['image_member']];areas=[];widths=[];heights=[];classes=[]
        for o in it['objects']:
            x0,y0,x1,y1=map(float,o['bbox']);w=max(0,x1-x0);h=max(0,y1-y0)
            areas.append(w*h);widths.append(w);heights.append(h);classes.append(o['class'])
        rows.append({**r,
          'median_gt_bbox_area_px2':statistics.median(areas) if areas else 0,
          'median_gt_bbox_width_px':statistics.median(widths) if widths else 0,
          'median_gt_bbox_height_px':statistics.median(heights) if heights else 0,
          'gt_classes':classes,
          'image_recall':safe_div(r['tp'],r['gt']),
          'image_precision':safe_div(r['tp'],r['pred']),
          'file_index':int(re.search(r'file_(\d+)',r['image_member']).group(1))
        })
    q1,q2,q3=quartile_cuts([r['median_gt_bbox_area_px2'] for r in rows])
    def aq(a):return 'Q1_smallest' if a<=q1 else 'Q2' if a<=q2 else 'Q3' if a<=q3 else 'Q4_largest'
    area=collections.defaultdict(lambda:{'images':0,'gt':0,'pred':0,'tp':0})
    bands=collections.defaultdict(lambda:{'images':0,'gt':0,'pred':0,'tp':0})
    class_presence=collections.defaultdict(lambda:{'images':0,'gt_instances':0,'tp_on_images':0,'gt_on_images':0})
    for r in rows:
        a=area[aq(r['median_gt_bbox_area_px2'])];a['images']+=1;a['gt']+=r['gt'];a['pred']+=r['pred'];a['tp']+=r['tp']
        b=bands[str((r['file_index']//500)*500)];b['images']+=1;b['gt']+=r['gt'];b['pred']+=r['pred'];b['tp']+=r['tp']
        cc=collections.Counter(r['gt_classes'])
        for c,n in cc.items():
            d=class_presence[c];d['images']+=1;d['gt_instances']+=n;d['tp_on_images']+=r['tp'];d['gt_on_images']+=r['gt']
    for d in list(area.values())+list(bands.values()):
        d['recall']=safe_div(d['tp'],d['gt']);d['precision']=safe_div(d['tp'],d['pred'])
    for d in class_presence.values():
        d['image_context_recall']=safe_div(d['tp_on_images'],d['gt_on_images'])
    any_iou=[r['mean_iou'] for r in rows if r['tp']>0 and r['mean_iou'] is not None]
    summary={
      'images':len(rows),
      'images_zero_predictions':sum(r['pred']==0 for r in rows),
      'images_zero_true_positives':sum(r['tp']==0 for r in rows),
      'images_any_true_positive':sum(r['tp']>0 for r in rows),
      'images_full_recall':sum(r['tp']==r['gt'] for r in rows),
      'under_count_images':sum(r['pred']<r['gt'] for r in rows),
      'equal_count_images':sum(r['pred']==r['gt'] for r in rows),
      'over_count_images':sum(r['pred']>r['gt'] for r in rows),
      'median_gt_per_image':statistics.median(r['gt'] for r in rows),
      'median_pred_per_image':statistics.median(r['pred'] for r in rows),
      'median_matched_iou_when_any':statistics.median(any_iou) if any_iou else None,
    }
    findings=[]
    if summary['images_zero_true_positives']/len(rows)>.5:findings.append('domain_shift_dominant: >50% images have zero matched detections')
    if summary['under_count_images']/len(rows)>.6:findings.append('recall_and_counting_failure_dominant: model under-counts on >60% images')
    if summary['median_matched_iou_when_any'] and summary['median_matched_iou_when_any']>.5:findings.append('localization_can_be_reasonable_when_detection_succeeds')
    report={
      'schema':'assetgraph-evidence/transset-h00-forensic-v1',
      'status':'CONSUMED_DIAGNOSTIC',
      'policy':{
        'H00_may_never_be_used_again_as_external_promotion_holdout':True,
        'no_reinference_performed':True,
        'inputs_only':['frozen H00 score artifact','frozen H00 lockbox ground truth'],
        'cycle16_was_precommitted_before_H00_score':True,
        'future_external_claims_require_H01_or_later':True
      },
      'aggregate_score':score['metrics'],
      'summary':summary,
      'gt_bbox_area_quartile_px2':{'cuts':[q1,q2,q3],'performance':dict(area)},
      'filename_index_band_diagnostic':dict(sorted(bands.items(),key=lambda x:int(x[0]))),
      'class_presence_context':dict(class_presence),
      'class_presence_caveat':'Context recall is NOT class recall because prediction-to-GT class assignment was not persisted in the one-shot scorer.',
      'findings':findings,
      'elapsed_seconds':time.time()-t
    }
    (OUT/'transset_h00_forensic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'summary':summary,'area':dict(area),'findings':findings},indent=2))

if __name__=='__main__':main()
