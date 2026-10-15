"""Materialize the prospective finite recipe matrix before any new fit."""
import argparse,json,math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--canonical-root',required=True);a=p.parse_args()
    path=Path(a.config);c=json.loads(path.read_text(encoding='utf-8'));root=Path(a.canonical_root)
    base='results/cnn_adoption_v1/prepared/token_r2/fit';expanded='results/kd_cnn_v2/prepared/expanded_fit'
    count=lambda folder:json.loads((root/folder/'manifest.json').read_text(encoding='utf-8'))['documents']
    base_steps=math.ceil(count(base)/32);expanded_steps=math.ceil(count(expanded)/32)
    common={'seed':42,'epochs':8,'learning_rate':0.0008,'weight_decay':0.01,'batch_documents':32,
        'microbatch_windows':128,'scheduler':'constant','warmup_fraction':0.,'gradient_clip':5.,'temperature':2.,
        'monitor_development':True,'scoring_batch_windows':128,'cpu_threads':1,
        'development_selection_root':'results/dual_track_v1/shared/data/prepared/new_selection',
        'development_calibration_root':'results/dual_track_v1/shared/data/prepared/new_development_calibration'}
    recipe=[]
    for arm in c['prospective_matrix']['arms']:
        expand=arm.endswith('expanded');r={**common,'run_id':arm+'_s42_dev','model_id':'residual64_rf13' if arm.startswith('rf13') else 'dilated64_rf127',
            'fit_root':expanded if expand else base,'label_weight':1. if arm.startswith('supervised') else .7,
            'logit_weight':0. if arm.startswith('supervised') else .3,'representation_weight':.1 if 'richer' in arm else 0.,
            'eval_every_steps':expanded_steps if expand or 'repeated' in arm else base_steps,
            'total_steps':8*(expanded_steps if expand or 'repeated' in arm else base_steps)}
        if r['logit_weight']:r['teacher_cache']='results/kd_cnn_v2/teacher_cache/'+('expanded_layer2_v1' if expand else 'base_layer2_v1')
        recipe.append(r)
    if c['recipes'] and c['recipes']!=recipe:raise ValueError('Recipe matrix already exists and differs')
    c['recipes']=recipe;c['shared_contract_sha256']='1dc93c47389aec8e823defebc578ed5d0d553c3df3c2045b5d9f6f781f1933b8'
    c['data_addendum']='results/kd_cnn_v2/data_addendum.json';c['phase']='development_matrix_frozen_before_fits'
    path.write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(recipe),'base_steps_per_pass':base_steps,'expanded_steps_per_pass':expanded_steps,'total_steps':sum(r['total_steps'] for r in recipe)}))


if __name__=='__main__':main()
