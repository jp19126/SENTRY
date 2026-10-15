"""Prepare a deterministic, non-scoring whole-BERT confirmation nomination.

Drafts never authorize scoring. Finalization requires both selected graph-route
qualifications and an explicit assertion that canonical hardware sources are
ready to snapshot. Confirmation arrays/JSONL/raw are hashed as opaque bytes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,SHARED,read,sha,write
from evaluate import ALPHAS,DATA_FILES,VARIANTS,ensure_directory,guard_path,target_descriptor
from profile_detector import graph_execution_identity

SCHEMA='sentry_bert_aie_v1_evaluation_freeze_v1'
SELECTED='w8a8_bf16_attention'

def relative(path):return str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\','/')

def role_identity(folder,protocol_hash,expected_documents,expected_labels):
    folder=guard_path(folder,'confirmation');manifest=read(folder/'manifest.json')
    if manifest['documents']!=expected_documents or manifest['document_labels']!=expected_labels:
        raise ValueError('Shared primary role counts/labels differ from the approved manifest')
    if manifest['data_protocol_sha256']!=protocol_hash:raise ValueError('Shared role protocol differs')
    if manifest['windowing']!={'length':256,'content_capacity':254,'overlap':64,'stride':190,'all_tails_complete':True}:
        raise ValueError('Window and complete-tail contract differs')
    expected=manifest['artifact_sha256']
    if not set(DATA_FILES)<=set(expected):raise ValueError('Incomplete prepared-role hash manifest')
    files={'manifest.json':sha(folder/'manifest.json')}
    for name,digest in expected.items():
        if Path(name).name!=name or (not name.endswith('.npy') and name!='documents.jsonl'):
            raise ValueError('Unsupported role artifact name')
        observed=sha(folder/name)
        if observed!=digest:raise ValueError('Prepared role artifact hash differs: '+name)
        files[name]=observed
    return {'directory':str(folder.resolve()),'files':files},manifest

def profile_nomination(path,models):
    profile=read(path)
    if not profile.get('complete') or profile.get('backend')!='cuda_graph':raise ValueError('Completed graph profile required: '+str(path))
    variant=profile['target']['variant']
    if variant not in models or profile['target']!=models[variant]:raise ValueError('Graph profile model/export descriptor differs')
    if profile['execution_identity']!=graph_execution_identity(variant):raise ValueError('Graph profiler execution source identity changed')
    quality=profile.get('execution_quality')
    if not quality or not quality.get('graph_bringup',{}).get('passed') or set(quality['roles'])!={'calibration','selection'}:
        raise ValueError('Both complete development roles and same-arithmetic graph bringup required')
    if not quality.get('reference_transfer_qualified') and profile.get('threshold_policy')!='execution_calibrated':
        raise ValueError('Changed numerical graph must have explicit own-calibrated policy')
    evidence={relative(path):sha(path)}
    for desc in quality['score_files'].values():
        if Path(desc['name']).name!=desc['name']:raise ValueError('Unsafe graph score filename')
        score_path=path.parent/desc['name']
        if sha(score_path)!=desc['sha256']:raise ValueError('Graph qualification score bytes changed')
        evidence[relative(score_path)]=desc['sha256']
    # Scores are deliberately not decoded here. Quality/threshold policies were
    # established by the completed development profile and remain explicit.
    return variant,profile['execution_identity'],{'profile_path':relative(path),'profile_sha256':sha(path),
        'evidence_files_sha256':evidence,'threshold_policy':profile['threshold_policy'],
        'reference_transfer_qualified':quality['reference_transfer_qualified'],
        'own_calibration_reported':True,'stage':'development_only'}

def build(args):
    configuration=read(ROOT/'configs/bert_aie_v1.json')
    primary=configuration['primary_confirmation_data']
    protocol=SHARED/'data/protocol.json';protocol_hash=sha(protocol)
    if protocol_hash!=primary['shared_protocol_sha256']:raise ValueError('Configured shared primary protocol changed')
    calibration_role=Path(args.calibration_role) if args.calibration_role else SHARED/primary['calibration']
    selection_role=Path(args.selection_role) if args.selection_role else SHARED/primary['selection']
    if calibration_role.resolve()!=(SHARED/'data/prepared/new_calibration').resolve() or selection_role.resolve()!=(SHARED/'data/prepared/new_confirmation').resolve():
        raise ValueError('Only the selected shared primary 1000/600 roles may be nominated')
    data={};role_manifests={}
    for role,folder,count,labels in (('calibration',calibration_role,1000,{'0':1000}),('selection',selection_role,600,{'0':400,'1':200})):
        data[role],role_manifests[role]=role_identity(folder,protocol_hash,count,labels)
    models={variant:target_descriptor(SimpleNamespace(variant=variant,calibration=args.calibration,
        export=args.export if variant==SELECTED else None)) for variant in VARIANTS}
    routes={};profiles={}
    for path in args.profile:
        variant,identity,evidence=profile_nomination(path,models)
        if variant in profiles:raise ValueError('Duplicate graph qualification for '+variant)
        routes[variant]=identity;profiles[variant]=evidence
    required=args.required_profile or ['source',SELECTED]
    pending=sorted(set(required)-set(profiles))
    if args.complete and pending:raise ValueError('Final freeze lacks required development graph qualifications: '+','.join(pending))
    if args.complete and not args.hardware_ready:raise ValueError('Final freeze requires --hardware-ready after canonical hardware source edits stop')
    stable=['scripts/bert_aie_v1/common.py','scripts/bert_aie_v1/evaluate.py','scripts/bert_aie_v1/analyze.py',
        'scripts/bert_aie_v1/model.py','scripts/bert_aie_v1/export.py','scripts/bert_aie_v1/qualify.py',
        'scripts/bert_aie_v1/profile_detector.py','scripts/bert_aie_v1/freeze_evaluation.py','scripts/r1_bert_cuda_graph.py',
        'gate_eval.py','configs/bert_aie_v1.json']
    sources={name:sha(ROOT/name) for name in stable}
    hardware={};build_sources={}
    if args.complete:
        for folder in (ROOT/'hardware/bert_aie_v1',ROOT/'scripts/bert_aie_v1'):
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix in ('.py','.cpp','.hpp','.h','.cfg','.md','.json'):
                    name=relative(path);digest=sha(path)
                    if name.startswith('hardware/'):hardware[name]=digest
                    else:build_sources[name]=digest
        # These canonical numerical/source identities and the exact compiled
        # comparison binary are required for the full C++ differential route.
        for name in ('scripts/bert_aie_v1/evaluate_cpp.py','hardware/bert_aie_v1/functional.cpp',
                     'hardware/bert_aie_v1/model.hpp','build/bert_aie_v1/common/bert_functional_stream'):
            sources[name]=sha(ROOT/name)
    artifacts={relative(SHARED/'contract.json'):sha(SHARED/'contract.json'),relative(protocol):protocol_hash,
        relative(args.calibration):sha(args.calibration),relative(args.export/'manifest.json'):sha(args.export/'manifest.json')}
    export_manifest=read(args.export/'manifest.json')
    for name,desc in export_manifest['files'].items():artifacts[relative(args.export/name)]=desc['sha256']
    if args.complete:
        binary='build/bert_aie_v1/common/bert_functional_stream';artifacts[binary]=sources[binary]
    for role,manifest in role_manifests.items():
        raw_path=SHARED/'data/raw'/('new_calibration.jsonl' if role=='calibration' else 'new_confirmation.jsonl')
        if sha(raw_path)!=manifest['raw_sha256']:raise ValueError('Opaque primary raw role bytes differ')
        artifacts[relative(raw_path)]=manifest['raw_sha256']
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
        path=ROOT/'checkpoints/floating/epoch_3'/name;artifacts[relative(path)]=sha(path)
    result={'schema':SCHEMA,'complete':bool(args.complete),'nomination_before_confirmation':True,
        'preparation_mode':'final' if args.complete else 'non-authorizing draft',
        'selected_variant':SELECTED,'models':models,'data':data,'execution_routes':routes,
        'profile_qualification':profiles,'pending_required_profiles':pending,
        'calibration_policy':{'alphas':list(ALPHAS),'primary':.01,'rule':'margin > threshold','aggregation':'max_window_margin'},
        'source_sha256':sources,'artifact_sha256':artifacts,
        'hardware_source_snapshot':{'included':bool(args.complete),'files_sha256':hardware,'canonical_scripts_sha256':build_sources,
            'note':'Final-only implementation provenance; generated build reports/logs excluded. Evaluation invalidation uses source_sha256, which includes functional.cpp/model.hpp/the compiled C++ binary. Unrelated PL pragmas and integration source hashes are separately recorded so subsequent build-only edits cannot silently invalidate or alter frozen numerical scoring.'},
        'selection_rationale':'Fixed matched checkpoint; native W8/A8 plus BF16 attention is nominated to isolate target-native numerical and residency mechanisms. Preserve FP32/source and W8/A8 FP32-attention controls. Development quality differences remain reported, with no confirmation-based format/threshold/seed rescue.',
        'evidence_boundary':'Numerical/model and supporting engineering comparison; no novel scientific contribution, physical performance, deployment safety or downstream LLM protection is established by this nomination.',
        'data_access':'Only manifests decoded. Confirmation arrays, document JSONL and raw JSONL hashed as opaque bytes; no model loaded for inference or scores computed.'}
    # Detect concurrent edits during preparation instead of issuing a mixed-state
    # final nomination. Future authorization also rechecks every frozen source.
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):raise ValueError('Canonical source changed while preparing freeze')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,default=ROOT/'results/bert_aie_v1/numerics/calibration_fit64.json')
    p.add_argument('--export',type=Path,default=ROOT/'results/bert_aie_v1/export/w8a8_bf16_attention')
    p.add_argument('--calibration-role',type=Path);p.add_argument('--selection-role',type=Path)
    p.add_argument('--profile',type=Path,action='append',default=[]);p.add_argument('--required-profile',action='append',choices=VARIANTS)
    p.add_argument('--complete',action='store_true');p.add_argument('--hardware-ready',action='store_true');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();ensure_directory(args.out.parent)
    if args.out.exists():raise ValueError('Preserve prior nomination; use a fresh output path')
    result=build(args);write(args.out,result)
    print(json.dumps({'schema':SCHEMA,'path':str(args.out),'sha256':sha(args.out),'complete':result['complete'],
        'models':list(result['models']),'graph_routes':list(result['execution_routes']),'pending_required_profiles':result['pending_required_profiles'],
        'hardware_sources_included':result['hardware_source_snapshot']['included']},indent=2))

if __name__=='__main__':main()
