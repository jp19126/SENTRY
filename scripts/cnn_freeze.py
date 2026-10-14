"""Freeze inner-selected finalists and numerical diagnostics before development scoring."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cnn_study import DEST,read,write,sha
from cnn_models import load_search_configs
from cnn_artifacts import validate_prepared

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',required=True);p.add_argument('--quant-ids',nargs='+',required=True);p.add_argument('--extra-plan');p.add_argument('--epochs',type=int,default=12);p.add_argument('--epoch-policy',choices=('inner-selected','fixed'),default='inner-selected');p.add_argument('--common-epoch-pair',nargs=2,action='append',default=[]);a=p.parse_args()
    if len(a.ids)>8 or len(set(a.ids))!=len(a.ids):raise ValueError('Up to eight distinct finalist architectures')
    configs={c['id']:c for c in load_search_configs()};evidence=read(DEST/'stage_b_run_table.json')
    if a.extra_plan:
        extra=read(Path(a.extra_plan));configs.update({c['id']:c for c in extra.get('extra_configs',[])})
        evidence+=read(DEST/(extra['stage']+'_run_table.json'))
    available={r['config_id']:r for r in evidence if r['complete']}
    if any(cid not in available for cid in a.ids):raise ValueError('Finalist lacks completed adequate inner training')
    if not set(a.quant_ids)<=set(a.ids):raise ValueError('Quantization targets must be frozen finalists')
    for rep in {configs[cid]['representation'] for cid in a.ids}:
        validate_prepared(rep,('full_train','search_temporary_threshold','search_candidate_scoring','diagnostics'))
    epoch_map={cid:available[cid]['selected_epoch'] if a.epoch_policy=='inner-selected' else a.epochs for cid in a.ids}
    for pair in a.common_epoch_pair:
        if not set(pair)<=set(a.ids):raise ValueError('Common-epoch pair must be frozen finalists')
        for cid in pair:epoch_map[cid]=a.epochs
    runs=[{'run_id':cid+'_final_s'+str(seed),'config_id':cid,'seed':seed,'epochs':epoch_map[cid],'fit_split':'full_train',
        'cal_split':'inner_calibration','selection_split':'inner_selection','time':seed==42} for cid in a.ids for seed in (42,1337,2026)]
    # time is a profiling decision; the same exact spec is used by the train queue.
    path=ROOT/'configs/cnn_stage_c_v1.json'
    existing=read(path) if path.exists() else {}
    frozen={'stage':'stage_c','record':'frozen before any CNN canonical post-search scoring or diagnostic inspection',
        'frozen_unix':existing.get('frozen_unix',time.time()),'selection_rationale_sha256':sha(DEST/'finalist_selection.json'),
        'architecture_ids':a.ids,'quantization_config_ids':a.quant_ids,'seeds':[42,1337,2026],
        'selection_evidence_sha256':sha(DEST/'stage_b_run_table.json'),'inner_selected_results':[available[cid] for cid in a.ids],
        'extra_inner_plan_sha256':sha(Path(a.extra_plan)) if a.extra_plan else None,
        'extra_inner_results_sha256':sha(DEST/(extra['stage']+'_run_table.json')) if a.extra_plan else None,
        'data_protocol_sha256':sha(DEST/'prepared/data_protocol.json'),
        'prepared_representation_sha256':{rep:sha(DEST/'prepared'/rep/'summary.json') for rep in ('token','byte')},
        'raw_timing_source_sha256':sha(ROOT/'data/prepared/search_candidate_scoring.jsonl'),
        'finalist_configs':{cid:configs[cid] for cid in a.ids},
        'model_source_sha256':sha(ROOT/'scripts/cnn_models.py'),'epochs_by_architecture':epoch_map,
        'epoch_selection_policy':a.epoch_policy,'common_epoch_pairs':a.common_epoch_pair,'common_pair_fixed_epoch':a.epochs,
        'extra_configs':[configs[cid] for cid in a.ids if cid not in {c['id'] for c in load_search_configs()}],
        'arithmetic':['native CUDA FP32 TF32 disabled','static fake symmetric W8/inputA8 for three named configs; FP32 bias/norm/gates/accumulation'],
        'cudnn_plan_cache_limit':64,
        'cudnn_cache_limits':{cid:int(read(DEST/'runs'/available[cid]['run_id']/'run.json')['cudnn_plan_cache_limit']) for cid in a.ids},
        'cudnn_v8_disabled':{cid:int(read(DEST/'runs'/available[cid]['run_id']/'run.json').get('cudnn_v8_api_disabled','0')) for cid in a.ids},
        'quantization_calibration':'first512 canonical-full-training windows; peroutput/perembeddingrow weights and pertensor inputs maxabs/127',
        'postsearch_calibration':'search_temporary_threshold benign only; primary .01, secondary .005/.02, strict >',
        'postsearch_scoring':'batch-one native CUDA windows on search_candidate_scoring and frozen authored diagnostics, no further tuning',
        'timing':'seed42 B1 CUDA eager; same sixshortest/sixlongest benign docs, fivewarmups/twentyrepeats; coldprocess separate',
        'diagnostics_sha256':sha(DEST/'prepared/diagnostics_raw.jsonl'),'runs':runs}
    if path.exists() and read(path)!=frozen:raise ValueError('Existing freeze differs; no silent revision')
    write(path,frozen);print(str(path))
if __name__=='__main__':main()
