"""Create immutable stage allocations from completed inner-only evidence."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cnn_models import load_search_configs
from cnn_study import DEST,read,write,selection_key,sha

def frozen(path,obj):
    if path.exists() and read(path)!=obj:raise ValueError('Frozen plan already exists with different content')
    write(path,obj);print(str(path))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('a','b'));args=p.parse_args()
    configs=load_search_configs();byid={c['id']:c for c in configs}
    if args.stage=='a':
        reused={'textcnn_01','residual_01','dpcnn_01'}
        runs=[{'run_id':('timing_'+c['id']+'_s42') if c['id'] in reused else c['id']+'_screen_s42',
            'config_id':c['id'],'seed':42,'epochs':3,'fit_split':'screen_fit','cal_split':'inner_calibration','selection_split':'inner_selection'} for c in configs]
        frozen(ROOT/'configs/cnn_stage_a_v1.json',{'stage':'stage_a','allocation':'96 configs/16 families; 3 full screening-subset epochs; three equivalent timing runs reused',
            'accelerator_job_hour_allocations':{'stage_a_including_timing':3,'stage_b':10,'stage_c':20,'targeted_learning':6,'confirmation_scoring_profiling':3,'reserve':6},
            'total_accelerator_hours_ceiling':48,'heavy_cpu_hours_ceiling':24,'optional_vendor_attempts_ceiling':6,'optional_vendor_hours_ceiling':12,
            'usage_used_percent_cap':60,'runs':runs})
    else:
        records=[r for r in read(DEST/'stage_a_run_table.json') if r['complete']]
        families=sorted({c['family'] for c in configs});best=[];extras=[]
        for family in families:
            candidates=sorted([r for r in records if r['family']==family and not (family=='tcn' and byid[r['config_id']].get('direction')=='centered')],key=lambda r:(selection_key(r['metrics']['0.01']),r['parameters'],r['config_id']))
            if not candidates:raise ValueError('No viable representative for family '+family)
            best.append(candidates[0]);extras.append(candidates[1])
        extras=sorted(extras,key=lambda r:(selection_key(r['metrics']['0.01']),r['parameters'],r['config_id']))[:6]
        chosen=best+extras
        if not any(r['config_id']=='textcnn_01' for r in chosen):chosen.append(next(r for r in records if r['config_id']=='textcnn_01'))
        runs=[{'run_id':r['config_id']+'_fit_s42','config_id':r['config_id'],'seed':42,'epochs':12,
              'fit_split':'fit','cal_split':'inner_calibration','selection_split':'inner_selection','select_epoch':True} for r in chosen]
        frozen(ROOT/'configs/cnn_stage_b_v1.json',{'stage':'stage_b','selection_rule':'One per family, then six next representatives from distinct families; recall floor then FPR then recall then parameter count. Add the exact R1-style TextCNN anchor for adequately trained competence comparison. Centered TCN computations duplicate dilated configurations and are already covered there; retain causal/bidirectional TCN for diversity. One of 24 promotion slots reserved for a controlled structural replacement; existing configurations may receive targeted repairs within the separate learning allowance.',
            'stage_a_table_sha256':sha(DEST/'stage_a_run_table.json'),'runs':runs})
if __name__=='__main__':main()
