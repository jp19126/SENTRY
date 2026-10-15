"""Bounded margin scoring; confirmation bytes require a prospective hash freeze.

No deployment/adoption gate is imposed. This functional numerical evaluation is
separate from AIE/RTL tests and physical measurements. Only root dispatches runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, read, rows, sha, write

DATA_FILES = ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')
ALPHAS = (.01, .005, .02, .05)
VARIANTS = ('source', 'fp32', 'w8a8_fp32_attention', 'w8a8_bf16_attention')

def ensure_directory(folder):
    """Create one existing-parent level at a time for Windows UNC shares."""
    folder=Path(folder).absolute()
    for path in reversed([folder,*folder.parents]):
        if str(path)!=path.anchor:path.mkdir(exist_ok=True)

def guard_path(folder, stage):
    folder = Path(folder).resolve()
    lowered = str(folder).replace('\\', '/').lower()
    if any(word in lowered for word in ('notinject', 'final_calibration', 'final_test', 'final-calibration', 'final-test')):
        raise ValueError('Protected final/NotInject input is excluded')
    if stage == 'development':
        if folder.name not in ('development_calibration', 'development_selection'):
            raise ValueError('Development accepts only named development role directories')
    elif 'cnn_adoption_v1' in lowered:
        raise ValueError('Old confirmation material is excluded from this task')
    return folder

def data_identity(folder):
    names=DATA_FILES+tuple(name for name in ('manifest.json','raw.jsonl') if (Path(folder)/name).exists())
    return {'directory': str(Path(folder).resolve()), 'files': {name: sha(Path(folder)/name) for name in names}}

def target_descriptor(args):
    from model import CHECKPOINT, CHECKPOINT_SHA256, CONFIG
    if sha(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('Matched source checkpoint changed')
    result = {'variant': args.variant, 'checkpoint_sha256': CHECKPOINT_SHA256,
              'config_sha256': sha(CONFIG), 'score': 'max_window_logit1_minus_logit0',
              'scale_calibration_sha256': None, 'export_manifest_sha256': None}
    if args.variant.startswith('w8'):
        if args.calibration is None: raise ValueError('Candidate requires FIT calibration identity')
        calibration = read(args.calibration)
        if calibration['data_role'] != 'fit': raise ValueError('Candidate scales must be FIT frozen')
        result['scale_calibration_sha256'] = sha(args.calibration)
    if args.export:
        manifest = read(Path(args.export)/'manifest.json')
        if manifest['variant'] != args.variant or manifest['model']['checkpoint_sha256'] != CHECKPOINT_SHA256:
            raise ValueError('Export model/format differs')
        if manifest['model']['calibration_sha256'] != result['scale_calibration_sha256']:
            raise ValueError('Export FIT calibration differs')
        result['export_manifest_sha256'] = sha(Path(args.export)/'manifest.json')
        for name, info in manifest['files'].items():
            if Path(name).name != name or sha(Path(args.export)/name) != info['sha256']:
                raise ValueError('Export stream changed: '+name)
    return result

def authorize(args, descriptor):
    """No data contents (even array headers) are opened until this returns."""
    folders = {key:guard_path(getattr(args,key+'_role'),args.stage) for key in ('calibration','selection')}
    if args.stage == 'development':
        if folders['calibration'].name!='development_calibration' or folders['selection'].name!='development_selection':
            raise ValueError('Development role assignment differs from its name')
        return folders, {key: data_identity(value) for key,value in folders.items()}, None
    if not args.freeze or not args.freeze_sha256 or sha(args.freeze) != args.freeze_sha256:
        raise ValueError('Confirmation requires the exact prospective freeze and SHA256')
    frozen = read(args.freeze)
    if not frozen.get('complete') or not frozen.get('nomination_before_confirmation'):
        raise ValueError('Completed prospective nomination required')
    if frozen.get('schema') != 'sentry_bert_aie_v1_evaluation_freeze_v1': raise ValueError('Wrong task freeze schema')
    if frozen['models'].get(args.variant) != descriptor: raise ValueError('Exact model/numerics/export were not nominated')
    if frozen.get('calibration_policy') != {'alphas':list(ALPHAS), 'primary':.01, 'rule':'margin > threshold', 'aggregation':'max_window_margin'}:
        raise ValueError('Frozen calibration policy differs')
    required = {'scripts/bert_aie_v1/evaluate.py','scripts/bert_aie_v1/model.py','scripts/bert_aie_v1/export.py',
                'scripts/bert_aie_v1/analyze.py','gate_eval.py'}
    if not required <= set(frozen['source_sha256']): raise ValueError('Required scoring/numerical sources not frozen')
    for name,digest in frozen['source_sha256'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != digest: raise ValueError('Frozen source changed: '+name)
    for key,folder in folders.items():
        expected = frozen['data'][key]
        if Path(expected['directory']).resolve() != folder or not set(DATA_FILES) <= set(expected['files']):
            raise ValueError('Exact role directory/artifacts were not frozen')
        for name,digest in expected['files'].items():
            if Path(name).name != name or sha(folder/name) != digest: raise ValueError('Frozen role bytes changed: '+name)
    return folders, frozen['data'], args.freeze_sha256

def load_prepared(folder):
    documents = list(rows(Path(folder)/'documents.jsonl'))
    ids = np.load(Path(folder)/'ids.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(Path(folder)/'masks.npy', mmap_mode='r', allow_pickle=False)
    starts = np.load(Path(folder)/'starts.npy', allow_pickle=False)
    if ids.ndim != 2 or ids.shape[1] != 256 or ids.shape != masks.shape:
        raise ValueError('Prepared inputs must be matching [windows,256] arrays')
    if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(masks.dtype,np.integer): raise ValueError('Integer token/mask arrays required')
    if starts.shape != (len(documents)+1,) or starts[0] != 0 or starts[-1] != len(ids) or np.any(np.diff(starts)<=0):
        raise ValueError('Every document must have its complete nonempty window interval')
    if len({d['document_id'] for d in documents}) != len(documents) or any(d['label'] not in (0,1) for d in documents):
        raise ValueError('Duplicate IDs or invalid labels')
    return {'documents':documents,'ids':ids,'masks':masks,'starts':starts}

def load_target(args):
    import torch
    from model import load_model, CHECKPOINT, CONFIG
    if args.variant == 'source':
        from transformers import BertConfig, BertForSequenceClassification
        model = BertForSequenceClassification(BertConfig.from_json_file(str(CONFIG))).eval()
        model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['state_dict'])
        return model.to(args.device)
    if args.export:
        from export import load_export
        return load_export(args.export,device=args.device)[0]
    scales = read(args.calibration)['input_scales'] if args.calibration else None
    return load_model(args.variant,scales=scales,device=args.device)[0]

def score_role(model, data, device, batch_windows, deadline):
    import torch
    margins = [];window_logits=[]
    with torch.inference_mode():
        for start in range(0,len(data['ids']),batch_windows):
            if time.monotonic()>deadline: raise TimeoutError('Scoring dispatch elapsed-time budget exhausted')
            stop = min(start+batch_windows,len(data['ids']))
            ids = torch.tensor(np.array(data['ids'][start:stop]),dtype=torch.long,device=device)
            mask = torch.tensor(np.array(data['masks'][start:stop]),dtype=torch.long,device=device)
            if torch.any(ids<0) or torch.any(ids>=30522) or torch.any((mask!=0)&(mask!=1)) or torch.any(mask.sum(-1)==0):
                raise ValueError('Invalid input IDs or key masks')
            logits = model(input_ids=ids,attention_mask=mask).logits
            values = (logits[:,1]-logits[:,0]).double().cpu().numpy()
            if not np.isfinite(values).all(): raise ValueError('Nonfinite logits')
            margins.extend(values.tolist())
            window_logits.extend(logits.detach().double().cpu().tolist())
    scored=[]
    for index,document in enumerate(data['documents']):
        low,high=map(int,data['starts'][index:index+2]); values=margins[low:high]
        digest=hashlib.sha256()
        for key in ('ids','masks'):
            value=np.ascontiguousarray(data[key][low:high]); digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        scored.append({**document,'window_count':high-low,'window_margins':values,'window_logits':window_logits[low:high],'margin':max(values),
                       'window_input_sha256':digest.hexdigest()})
    return scored

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('development','confirmation'),default='development')
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--calibration-role',type=Path); p.add_argument('--selection-role',type=Path)
    p.add_argument('--calibration',type=Path); p.add_argument('--export',type=Path)
    p.add_argument('--freeze',type=Path); p.add_argument('--freeze-sha256')
    p.add_argument('--out',type=Path); p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--batch-windows',type=int,default=16); p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--max-seconds',type=int,default=1800); p.add_argument('--describe',action='store_true')
    args=p.parse_args(); started=time.monotonic()
    if not 1<=args.batch_windows<=64 or not 1<=args.max_seconds<=1800 or args.cpu_threads<1: raise ValueError('Invalid bounded dispatch settings')
    descriptor=target_descriptor(args)
    if args.describe: print(json.dumps(descriptor,indent=2)); return
    if not args.out or not args.calibration_role or not args.selection_role: p.error('Scoring needs --out and both role directories')
    ensure_directory(args.out)
    if (args.out/'complete.json').exists(): raise ValueError('Preserve existing completed evaluation; choose a new output directory')
    folders,identities,freeze_hash=authorize(args,descriptor)
    import torch
    torch.set_num_threads(args.cpu_threads); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=load_target(args); outputs={}
    for role,folder in folders.items():
        data=load_prepared(folder)
        if role=='calibration' and args.stage=='confirmation' and any(d['label']!=0 for d in data['documents']):
            raise ValueError('Confirmation calibration role must be benign-only')
        outputs[role]=score_role(model,data,args.device,args.batch_windows,started+args.max_seconds)
        target=args.out/(role+'_scores.jsonl')
        target.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in outputs[role]),encoding='utf-8')
    from analyze import metrics, fit_threshold, roc_report
    benign=[r['margin'] for r in outputs['calibration'] if r['label']==0]
    thresholds={str(a):fit_threshold(benign,a) for a in ALPHAS}
    result={'schema':'sentry_bert_aie_v1_scores_v1','complete':True,'stage':args.stage,'target':descriptor,
            'data':identities,'freeze_sha256':freeze_hash,'score_files':{role:{'name':role+'_scores.jsonl','sha256':sha(args.out/(role+'_scores.jsonl'))} for role in outputs},
            'source_sha256':{f.name:sha(f) for f in (Path(__file__),HERE/'model.py',HERE/'export.py',HERE/'analyze.py')},
            'thresholds':thresholds,'primary_alpha':.01,'metrics':{key:metrics(outputs['selection'],value) for key,value in thresholds.items()},
            'roc':roc_report(outputs['selection']), 'elapsed_seconds':time.monotonic()-started,'device':args.device,
            'batch_windows':args.batch_windows,'cpu_threads':args.cpu_threads,'torch':torch.__version__,
            'boundary':'Functional numerical software; unsaturated logit1-logit0 margins, max over every document window; calibration uses benign rows only; no board/protection claim'}
    write(args.out/'complete.json',result)
    print(json.dumps({'complete':True,'out':str(args.out),'elapsed_seconds':result['elapsed_seconds'],'metrics':result['metrics']},indent=2))

if __name__=='__main__': main()
