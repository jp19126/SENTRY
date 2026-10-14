"""Common adoption scoring with explicit prospective calibration and frozen release.

Development never loads the new confirmation/calibration roles. Confirmation
requires an exact externally frozen nomination, format, priority and data identity.
Saved W8-QAT's historical threshold remains a distinct comparison, never a rescue.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

from cnn_adoption_train import (ROOT, DEST, Budget, identity, load_prepared,
    metrics, model_for, read, score, setup, sha, write)
from gate_eval import evaluate_documents

HISTORICAL_W8_THRESHOLD = 0.004557917360216379
FORMATS = ('fp32', 'w8a8_export', 'historical_w8_qat')


def path(value):
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def relative(value):
    return str(Path(value).relative_to(ROOT)).replace('\\', '/')


def calibration_policy(plan):
    policy = plan['calibration_policy']
    if (policy['primary_alpha'] != .005 or policy['secondary_alphas'] != [.01, .02]
            or policy['rule'] != 'score > threshold' or policy['procedure'] != 'gate_eval.fit_threshold'
            or policy['aggregation'] != 'max_window_risk'):
        raise ValueError('Adoption calibration/aggregation policy differs from approved prospective rule')
    return policy


def resolve_target(plan, run_id=None, reference_id=None, numerical_format=None):
    """Checkpoint/source metadata only; no evaluation rows are opened."""
    if bool(run_id) == bool(reference_id): raise ValueError('Exactly one run/reference required')
    identifier = run_id or reference_id
    if Path(identifier).name != identifier or identifier in ('.', '..'): raise ValueError('Unsafe evaluation ID')
    model_specs = plan.get('model_specs', plan.get('models', {}))
    if run_id:
        choices = [recipe for recipe in plan['runs'] if recipe['run_id'] == run_id]
        if len(choices) != 1: raise ValueError('Run absent/duplicated in plan')
        recipe = choices[0]; spec = model_specs[recipe['model_id']]
        folder = DEST / 'runs' / identifier
        run, done = read(folder / 'run.json'), read(folder / 'complete.json')
        if not done['complete'] or run['association']['recipe'] != recipe or run['association']['model_spec'] != spec:
            raise ValueError('Completed training does not match exact planned recipe/model')
        if done['association_sha256'] != run['association_sha256'] or identity(run['association']) != run['association_sha256']:
            raise ValueError('Training association changed')
        checkpoint = path(done['checkpoint'])
        checkpoint_identity = {'path': relative(checkpoint), 'sha256': sha(checkpoint),
                               'training_association_sha256': done['association_sha256']}
        if checkpoint_identity['sha256'] != done['checkpoint_sha256']: raise ValueError('Checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['association_sha256'] != done['association_sha256'] or state['recipe'] != recipe or state['model_spec'] != spec or state['epoch'] != done['selected_epoch']:
            raise ValueError('Checkpoint payload is associated with a different recipe/epoch/model')
        if run['association']['sources']['cnn_models'] != sha(ROOT / 'scripts/cnn_models.py'):
            raise ValueError('Model implementation differs from training')
        if spec['kind'] == 'bert' and run['association']['pretrained_policy'] != read(ROOT / 'configs/project.json')['models']['detector']:
            raise ValueError('Pretrained model policy differs from training')
        del state
        original_threshold = None
    else:
        reference = plan['references'][identifier]
        spec = reference.get('model_spec') or model_specs[reference['model_id']]
        checkpoint = path(reference['checkpoint']) if reference.get('checkpoint') else None
        original_threshold = reference.get('original_threshold')
        if spec['kind'] == 'cnn':
            if checkpoint is None or sha(checkpoint) != reference['checkpoint_sha256']:
                raise ValueError('Frozen CNN reference checkpoint absent/changed')
            checkpoint_identity = {'path': relative(checkpoint), 'sha256': reference['checkpoint_sha256']}
        elif spec['kind'] in {'historical_bert_w8', 'historical_bert_fp32'}:
            directory = ROOT / ('checkpoints/quantized/qat_w8_a8' if spec['kind'] == 'historical_bert_w8' else 'checkpoints/floating/epoch_3')
            expected = reference['checkpoint_files_sha256']
            actual = {name: sha(directory / name) for name in expected}
            required = {'config.json', 'quantization.json', 'quantized_state.pt'} if spec['kind'] == 'historical_bert_w8' else {'config.json', 'model.safetensors'}
            if not required <= set(expected) or actual != expected: raise ValueError('Historical BERT directory identity changed/incomplete')
            checkpoint_identity = {'path': relative(directory), 'files_sha256': actual}
            if spec['kind'] == 'historical_bert_w8':
                if original_threshold is not None and original_threshold != HISTORICAL_W8_THRESHOLD:
                    raise ValueError('Historical W8 original threshold cannot change')
                original_threshold = HISTORICAL_W8_THRESHOLD
            checkpoint = None
        else: raise ValueError('Reference must be frozen CNN or historical BERT')
    kind = spec['kind']
    fmt = numerical_format or ('historical_w8_qat' if kind == 'historical_bert_w8' else 'fp32')
    if fmt not in FORMATS or (kind == 'historical_bert_w8') != (fmt == 'historical_w8_qat'):
        raise ValueError('Format does not identify the actual model arithmetic')
    export_path = None; export_hash = None
    if fmt == 'w8a8_export':
        if kind != 'cnn': raise ValueError('Packed CNN export cannot represent BERT')
        exported = plan['exports'][identifier]
        export_path = path(exported['directory']) / 'manifest.json'
        export_hash = sha(export_path)
        if export_hash != exported['manifest_sha256']: raise ValueError('Export manifest changed')
        manifest = read(export_path)
        if manifest['checkpoint_sha256'] != checkpoint_identity['sha256']:
            raise ValueError('Export belongs to a different checkpoint')
        config = spec.get('config')
        if config and manifest['configuration'] != config: raise ValueError('Export configuration changed')
    descriptor = {'model_spec': spec, 'checkpoint_identity': checkpoint_identity,
                  'numerical_format': fmt, 'export_manifest_sha256': export_hash}
    return identifier, descriptor, checkpoint, export_path, original_threshold


def confirmation_authorization(plan, target_key, descriptor):
    """Validate frozen scientific choices BEFORE reading any held-out documents."""
    freeze_path = path(plan['confirmation_freeze'])
    if sha(freeze_path) != plan['confirmation_freeze_sha256']: raise ValueError('Confirmation freeze changed')
    frozen = read(freeze_path)
    for key in ('nomination', 'data_membership', 'calibration_policy', 'numerical_formats', 'priority_order', 'execution_policy'):
        if identity(frozen[key]) != frozen[key + '_sha256']: raise ValueError('Frozen ' + key + ' identity differs')
    if not frozen.get('complete') or not frozen.get('nomination_before_confirmation'):
        raise ValueError('Completed prospective nomination required')
    if frozen['nomination']['targets'].get(target_key) != descriptor:
        raise ValueError('Exact checkpoint/model/format target was not nominated')
    if frozen['numerical_formats'].get(target_key) != descriptor['numerical_format']:
        raise ValueError('Numerical format was not frozen')
    if frozen['calibration_policy'] != calibration_policy(plan): raise ValueError('Calibration rule differs from freeze')
    if frozen['priority_order'] != plan['priority_order'] or len(set(frozen['priority_order'])) != len(frozen['priority_order']):
        raise ValueError('Candidate priority differs/duplicates')
    if frozen['quality_gates'] != plan['quality_gates'] or identity(frozen['quality_gates']) != frozen['quality_gates_sha256']:
        raise ValueError('Readiness requirements changed')
    if frozen['prepared_root'] != plan['prepared_root']: raise ValueError('Prepared root differs from freeze')
    if frozen['execution_policy'] != plan['execution_policy']: raise ValueError('Scoring execution policy differs from freeze')
    for filename, expected in frozen['source_sha256'].items():
        if sha(ROOT / filename) != expected: raise ValueError('Frozen evaluation source changed: ' + filename)
    required_sources = {'scripts/cnn_adoption_evaluate.py', 'scripts/cnn_adoption_train.py', 'scripts/cnn_models.py', 'gate_eval.py'}
    if not required_sources <= set(frozen['source_sha256']): raise ValueError('Freeze lacks required source identities')
    if descriptor['numerical_format'] == 'w8a8_export' and not {'scripts/cnn_adoption_export.py', 'scripts/cnn_adoption_backend.py'} <= set(frozen['source_sha256']):
        raise ValueError('Freeze lacks packed-format/backend sources')
    if descriptor['model_spec']['kind'] == 'historical_bert_w8' and not {'gate_quant.py', 'scripts/r1_bert_int8.py'} <= set(frozen['source_sha256']):
        raise ValueError('Freeze lacks historical arithmetic loader sources')
    if set(frozen['data_membership']) != {'benign_calibration', 'confirmation'}:
        raise ValueError('Freeze lacks exact reserved data roles')
    return frozen


def verify_membership(folder, expected):
    """Hash opaque bytes after authorization; decode rows only after all match."""
    manifest_path = folder / 'manifest.json'
    if sha(manifest_path) != expected['manifest_sha256']: raise ValueError('Frozen data manifest changed')
    required = {'ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl'}
    if not required <= set(expected['artifact_sha256']): raise ValueError('Incomplete frozen membership files')
    for name, digest in expected['artifact_sha256'].items():
        if Path(name).name != name or sha(folder / name) != digest: raise ValueError('Frozen data bytes changed: ' + name)


def enrich_rows(scored, data, *, require_new_schema=False):
    expected = data['documents']
    if len(scored) != len(expected): raise ValueError('Scored membership length differs')
    required = ('source_group_id', 'source_dataset', 'genre', 'payload_group_id', 'mechanism_group_id', 'hard_benign', 'text_sha256')
    seen = set()
    for index, (row, document) in enumerate(zip(scored, expected)):
        if row['document_id'] in seen or row['document_id'] != document['document_id'] or row['label'] != document['label']:
            raise ValueError('Scored association or duplicate IDs')
        seen.add(row['document_id'])
        if require_new_schema and any(key not in document for key in required): raise ValueError('New-stage row lacks frozen association metadata')
        if require_new_schema and not isinstance(document['hard_benign'], bool): raise ValueError('Hard-benign flag must be Boolean')
        if 'text' in document and hashlib.sha256(document['text'].encode()).hexdigest() != document['text_sha256']:
            raise ValueError('Raw text SHA identity differs')
        low, high = map(int, data['starts'][index:index + 2])
        values = row['window_scores']
        if len(values) != high - low or not values or any(not np.isfinite(v) or not 0 <= v <= 1 for v in values) or row['risk_score'] != max(values):
            raise ValueError('Window association/max-risk aggregation differs')
        h = hashlib.sha256()
        for key in ('ids', 'masks'):
            value = np.ascontiguousarray(data[key][low:high])
            h.update(identity({'kind': key, 'dtype': str(value.dtype), 'shape': list(value.shape)}).encode()); h.update(value.tobytes())
        row.update(window_input_sha256=h.hexdigest(), window_count=high - low)
        for key in required: row.setdefault(key, document.get(key))
    return scored


def at_threshold(rows, threshold, gates):
    measured = evaluate_documents([r['risk_score'] for r in rows], [r['label'] for r in rows], threshold)
    measured['passes'] = measured['recall'] is not None and measured['fpr'] is not None and measured['recall'] >= .99 and measured['fpr'] <= .01
    hard = [r for r in rows if r['label'] == 0 and r.get('hard_benign') is True]
    hard_metric = evaluate_documents([r['risk_score'] for r in hard], [0] * len(hard), threshold)
    strata = {}
    for key in ('source_dataset', 'genre', 'mechanism_group_id', 'payload_group_id', 'probe_category', 'probe_topic'):
        for value in sorted({str(r[key]) for r in rows if r.get(key) is not None}):
            group = [r for r in rows if str(r.get(key)) == value]
            metric = evaluate_documents([r['risk_score'] for r in group], [r['label'] for r in group], threshold)
            metric.update(source_groups=len({r['source_group_id'] for r in group}),
                          attack_source_groups=len({r['source_group_id'] for r in group if r['label'] == 1}),
                          attack_payload_groups=len({r['payload_group_id'] for r in group if r['label'] == 1 and r.get('payload_group_id') is not None}),
                          payload_groups=len({r['payload_group_id'] for r in group if r.get('payload_group_id') is not None}))
            if key == 'mechanism_group_id':
                minimum_attacks = gates.get('minimum_mechanism_attacks')
                minimum_groups = gates.get('minimum_mechanism_attack_source_groups')
                adequate = minimum_attacks is not None and minimum_groups is not None and metric['attacks'] >= minimum_attacks and metric['attack_source_groups'] >= minimum_groups
                adequate = adequate and metric['attack_payload_groups'] >= gates.get('minimum_mechanism_payload_groups', 0)
                metric['adequately_populated'] = adequate
                metric['passes_mechanism_recall'] = metric['recall'] >= gates['minimum_mechanism_recall'] if adequate else None
            strata[key + ':' + value] = metric
    required_mechanisms = gates.get('required_mechanism_groups', [])
    for mechanism in required_mechanisms:
        key = 'mechanism_group_id:' + mechanism
        if key not in strata:
            strata[key] = {'documents': 0, 'attacks': 0, 'benign': 0, 'recall': None, 'fpr': None,
                           'source_groups': 0, 'attack_source_groups': 0, 'attack_payload_groups': 0, 'payload_groups': 0,
                           'adequately_populated': False, 'passes_mechanism_recall': None, 'status': 'Required stratum absent; unresolved'}
    statuses = [strata['mechanism_group_id:' + m]['passes_mechanism_recall'] for m in required_mechanisms]
    mechanism_pass = False if any(s is False for s in statuses) else None if not statuses or any(s is None for s in statuses) else True
    primary_genres = gates.get('primary_genre_values', [gates.get('primary_genre', 'email')])
    application = [r for r in rows if r.get('genre') in primary_genres]
    application_metric = evaluate_documents([r['risk_score'] for r in application], [r['label'] for r in application], threshold)
    application_pass = None if application_metric['recall'] is None or application_metric['fpr'] is None else application_metric['recall'] >= gates.get('minimum_primary_email_recall', .99) and application_metric['fpr'] <= gates.get('maximum_primary_email_fpr', .01)
    return {'overall': measured, 'hard_benign': hard_metric,
            'hard_benign_passes': hard_metric['fpr'] <= gates['maximum_hard_benign_fpr'] if hard else None,
            'required_mechanisms_pass': mechanism_pass, 'primary_application': application_metric,
            'primary_application_genres': primary_genres, 'primary_application_passes': application_pass, 'strata': strata}


def load_target(descriptor, checkpoint, export_path, device):
    if export_path:
        from cnn_adoption_export import load_export_reference
        model, manifest = load_export_reference(export_path, device='cpu')
        if manifest['checkpoint_sha256'] != descriptor['checkpoint_identity']['sha256']:
            raise ValueError('Packed loader checkpoint mismatch')
        if str(device).startswith('cuda'):
            from cnn_adoption_backend import native_from_export_reference
            model = native_from_export_reference(model, device=device)
        elif device != 'cpu': raise ValueError('Packed route supports explicit CPU/CUDA only')
        metadata = {'route': 'packed_W8_A8_checked_int32_CPU_reference' if device == 'cpu' else 'packed_W8_A8_native_CUDA_int32',
                    'export_manifest_sha256': sha(export_path), 'manifest': manifest,
                    'floating_exceptions': 'FP32 scale/rescale/bias, norms, nonlinearities, pooling, residuals and decision softmax'}
    else:
        model, metadata = model_for(descriptor['model_spec'], device, checkpoint)
        if descriptor['model_spec']['kind'] == 'historical_bert_w8':
            expected = descriptor['checkpoint_identity']['files_sha256']
            if any(expected.get(name) != digest for name, digest in metadata['checkpoint_files_sha256'].items()):
                raise ValueError('Historical loaded checkpoint differs from nominated files')
        metadata['route'] = descriptor['model_spec'].get('backend', 'native_fp32_tf32_disabled')
    return model.eval(), metadata


def evaluate(plan, identifier, descriptor, checkpoint, export_path, historical_threshold, stage, device, batch_windows, budget):
    policy = calibration_policy(plan); target_key = identifier + ':' + descriptor['numerical_format']
    prepared = path(plan['prepared_root'])
    if not prepared.resolve().is_relative_to((DEST / 'prepared').resolve()): raise ValueError('Only adoption prepared inputs are allowed')
    frozen = confirmation_authorization(plan, target_key, descriptor) if stage == 'confirmation' else None
    if frozen:
        actual_execution = {'device': str(device), 'batch_windows': batch_windows, 'torch': torch.__version__,
                            'cache_limit': int(os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT', '64')),
                            'cudnn_v8_api_disabled': int(os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0')), 'tf32': False}
        if actual_execution != frozen['execution_policy']: raise ValueError('Actual scoring backend/shape differs from freeze')
    roles = ('benign_calibration', 'confirmation') if frozen else ('development_calibration', 'development_selection')
    if frozen:
        for role in roles: verify_membership(prepared / role, frozen['data_membership'][role])
    data = {role: load_prepared(prepared / role, role, release_confirmation=bool(frozen)) for role in roles}
    if any(d['label'] != 0 for d in data[roles[0]]['documents']): raise ValueError('Threshold population must be benign-only')
    if stage == 'development':
        data['known_diagnostics'] = load_prepared(ROOT / 'results/cnn_study_v1/prepared/token/diagnostics', 'diagnostics')
    sources = {'scripts/cnn_adoption_evaluate.py': sha(Path(__file__)), 'scripts/cnn_adoption_train.py': sha(ROOT / 'scripts/cnn_adoption_train.py'),
               'scripts/cnn_models.py': sha(ROOT / 'scripts/cnn_models.py'), 'gate_eval.py': sha(ROOT / 'gate_eval.py')}
    for name in (('scripts/cnn_adoption_export.py', 'scripts/cnn_adoption_backend.py') if export_path else
                 ('gate_quant.py', 'scripts/r1_bert_int8.py') if descriptor['model_spec']['kind'] == 'historical_bert_w8' else ()):
        sources[name] = sha(ROOT / name)
    association = {'target': descriptor, 'calibration_policy': policy, 'quality_gates': plan['quality_gates'], 'stage': stage,
                   'data': {role: value['identity'] for role, value in data.items()}, 'source_sha256': sources,
                   'device': str(device), 'torch': torch.__version__, 'batch_windows': batch_windows,
                   'cache_limit': os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'),
                   'cudnn_v8_api_disabled': os.environ.get('TORCH_CUDNN_V8_API_DISABLED', '0'),
                   'historical_original_threshold': historical_threshold,
                   'confirmation_freeze_sha256': plan.get('confirmation_freeze_sha256') if frozen else None}
    folder = DEST / 'evaluation' / stage / identifier / descriptor['numerical_format']
    if (folder / 'complete.json').exists():
        saved = read(folder / 'complete.json')
        if saved['association'] != association: raise ValueError('Saved evaluation association differs; preserve old outcomes')
        for name, digest in saved['score_files_sha256'].items():
            if sha(folder / name) != digest: raise ValueError('Saved score file changed')
        return saved
    if folder.exists() and any(folder.iterdir()): raise ValueError('Partial evaluation already exists; preserve before retry')
    folder.mkdir(parents=True, exist_ok=True)
    model, backend = load_target(descriptor, checkpoint, export_path, device)
    scored = {}
    for role, value in data.items():
        scored[role] = enrich_rows(score(model, value, budget, batch_windows, device=device), value, require_new_schema=role != 'known_diagnostics')
    operating = metrics(scored[roles[0]], scored[roles[1]])
    descriptions = {alpha: at_threshold(scored[roles[1]], result['threshold'], plan['quality_gates']) for alpha, result in operating.items()}
    diagnostics = {alpha: at_threshold(scored['known_diagnostics'], result['threshold'], plan['quality_gates'])
                   for alpha, result in operating.items()} if 'known_diagnostics' in scored else {}
    original = None
    if historical_threshold is not None:
        original = {'threshold': historical_threshold, 'selection': at_threshold(scored[roles[1]], historical_threshold, plan['quality_gates']),
                    'known_diagnostics': at_threshold(scored['known_diagnostics'], historical_threshold, plan['quality_gates']) if 'known_diagnostics' in scored else None,
                    'scope': 'Historical unchanged threshold transfer; separate from prospective recalibration'}
    file_hashes = {}
    for role, values in scored.items():
        filename = role + '_scores.jsonl'; temporary = folder / (filename + '.tmp')
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in values), encoding='utf-8')
        os.replace(temporary, folder / filename); file_hashes[filename] = sha(folder / filename)
    result = {'complete': True, 'evaluation_id': identifier, 'target_key': target_key, 'association': association,
              'association_sha256': identity(association), 'backend': backend, 'metrics': operating,
              'operating_point_details': descriptions, 'known_diagnostics': diagnostics, 'historical_original_threshold': original,
              'score_files_sha256': file_hashes, 'policy_scope': 'One global benign-only threshold per model/seed/format. Secondary settings do not substitute for primary or permit another tuning loop.',
              'diagnostic_scope': 'Old known authored examples are descriptive, not an independent confirmation panel.',
              'command': sys.argv, 'python': sys.version, 'wall_seconds': time.monotonic() - budget.started,
              'memory_observations': budget.observations}
    write(folder / 'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-id'); group.add_argument('--reference-id')
    parser.add_argument('--stage', required=True, choices=('development', 'confirmation'))
    parser.add_argument('--device', default='cuda'); parser.add_argument('--batch-windows', type=int, default=1)
    parser.add_argument('--format', choices=FORMATS); parser.add_argument('--max-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.batch_windows < 1: raise ValueError('Positive scoring batch required')
    setup(42, args.device, 1); plan = read(args.plan)
    resolved = resolve_target(plan, args.run_id, args.reference_id, args.format)
    result = evaluate(plan, *resolved, args.stage, args.device, args.batch_windows, Budget(args.max_seconds))
    print(json.dumps({'complete': True, 'target': result['target_key'], 'metrics': result['metrics'],
                      'historical_original_threshold': result['historical_original_threshold']}))


if __name__ == '__main__': main()
