"""CPU-only NON-ADOPTION probability-saturation diagnostic on completed TextCNN.

Preserves primary scores/policy. Examines selected and optionally latest frozen
states on development calibration/selection only. No confirmation or fitting.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from cnn_adoption_train import ROOT, DEST, identity, inputs, load_prepared, model_for, read, setup, sha, write
from gate_eval import evaluate_documents, fit_threshold

ROLES = ('development_calibration', 'development_selection')
ALPHAS = (.005, .01, .02)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sigmoid64(margin):
    result = np.empty_like(margin, dtype=np.float64); positive = margin >= 0
    result[positive] = 1 / (1 + np.exp(-margin[positive]))
    exponential = np.exp(margin[~positive]); result[~positive] = exponential / (1 + exponential)
    return result


def operating(cal, selection, key):
    result = {}
    for alpha in ALPHAS:
        threshold = fit_threshold([r[key] for r in cal if r['label'] == 0], alpha)
        metric = evaluate_documents([r[key] for r in selection], [r['label'] for r in selection], threshold)
        metric.update(calibration_exact_threshold_ties=sum(r[key] == threshold for r in cal if r['label'] == 0),
                      selection_exact_threshold_ties=sum(r[key] == threshold for r in selection))
        result[str(alpha)] = metric
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', required=True); parser.add_argument('--run-id', required=True)
    parser.add_argument('--include-latest', action='store_true'); parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 600: raise ValueError('Probe bounded to at most600seconds')
    started = time.monotonic(); setup(42, 'cpu', 1)
    if torch.cuda.is_initialized(): raise ValueError('GPU must not be initialized')
    def budget():
        if time.monotonic() - started > args.max_seconds: raise RuntimeError('CPU probe ceiling reached')
    plan = read(args.plan); run_folder = DEST / 'runs' / args.run_id
    run, done, history = (read(run_folder / filename) for filename in ('run.json', 'complete.json', 'history.json'))
    if not done['complete'] or run['association']['model_spec']['kind'] != 'cnn' or run['association']['model_spec']['config_id'] != 'textcnn_01':
        raise ValueError('Completed TextCNN only')
    spec = run['association']['model_spec']
    prepared = ROOT / plan['prepared_root']
    if not prepared.resolve().is_relative_to(DEST.resolve()): raise ValueError('Adoption prepared data only')
    data = {role: load_prepared(prepared / role, role) for role in ROLES}
    for role in ROLES:
        if data[role]['identity'] != run['association'][role]: raise ValueError('Development population changed')
    checkpoints = [('selected', ROOT / done['checkpoint'], done['selected_epoch'])]
    if args.include_latest: checkpoints.append(('latest', run_folder / 'checkpoint_latest.pt', done['epochs_completed']))
    frozen = {name: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'epoch': epoch} for name, path, epoch in checkpoints}
    if frozen['selected']['sha256'] != done['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
    output = DEST / 'margin_probe' / args.run_id
    if (output / 'complete.json').exists(): raise ValueError('Preserve existing diagnostic; no overwrite')
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, checkpoint, epoch in checkpoints:
        budget(); model, _ = model_for(spec, 'cpu', checkpoint); model.eval()
        stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if stored['association_sha256'] != done['association_sha256'] or stored['epoch'] != epoch:
            raise ValueError('Checkpoint association/epoch mismatch')
        del stored
        scored = {}; files = {}
        for role in ROLES:
            a = data[role]; logits = []
            with torch.inference_mode():
                for index in range(len(a['ids'])):
                    budget(); value = model(**inputs(a, slice(index, index + 1), 'cpu')).logits
                    if not torch.isfinite(value).all(): raise ValueError('Nonfinite logit')
                    logits.append(value.cpu().numpy()[0])
            logits = np.asarray(logits, dtype=np.float32)
            margins32 = logits[:, 1] - logits[:, 0]
            margins64 = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
            probability32 = torch.from_numpy(logits).softmax(1)[:, 1].numpy()
            probability64 = sigmoid64(margins64)
            documents = []
            for di, document in enumerate(a['documents']):
                lo, hi = map(int, a['starts'][di:di + 2])
                documents.append({**document, 'max_margin_f32': float(margins32[lo:hi].max()),
                    'max_margin_f64': float(margins64[lo:hi].max()), 'risk_fp32': float(probability32[lo:hi].max()),
                    'risk_sigmoid_f64': float(probability64[lo:hi].max()),
                    'window_logits_fp32': logits[lo:hi].tolist(), 'window_margins_f32': margins32[lo:hi].tolist(),
                    'window_margins_f64': margins64[lo:hi].tolist(), 'window_risks_fp32': probability32[lo:hi].tolist(),
                    'window_risks_sigmoid_f64': probability64[lo:hi].tolist()})
            scored[role] = documents
            tensor_file = output / (name + '_' + role + '.npz')
            np.savez_compressed(tensor_file, logits_fp32=logits, margin_f32=margins32, margin_f64=margins64,
                                risk_fp32=probability32, risk_sigmoid_f64=probability64, starts=a['starts'])
            score_file = output / (name + '_' + role + '.jsonl')
            score_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in documents), encoding='utf-8')
            files[tensor_file.name], files[score_file.name] = sha(tensor_file), sha(score_file)
        cal, selection = (scored[role] for role in ROLES)
        variants = {key: operating(cal, selection, key) for key in ('risk_fp32', 'risk_sigmoid_f64', 'max_margin_f32', 'max_margin_f64')}
        saturation = {}
        for role, documents in scored.items():
            for label in (0, 1):
                subset = [r for r in documents if r['label'] == label]
                if subset:
                    saturation[role + ':label' + str(label)] = {'documents': len(subset),
                        'fp32_exact_one': sum(r['risk_fp32'] == 1 for r in subset), 'f64_exact_one': sum(r['risk_sigmoid_f64'] == 1 for r in subset),
                        'unique_fp32_document_scores': len({r['risk_fp32'] for r in subset}),
                        'unique_document_margins_f64': len({r['max_margin_f64'] for r in subset}),
                        'margin_range': [min(r['max_margin_f64'] for r in subset), max(r['max_margin_f64'] for r in subset)]}
        top = sorted(cal, key=lambda r: r['max_margin_f64'], reverse=True)[:25]
        top = [{key: row[key] for key in ('document_id', 'label', 'source_group_id', 'max_margin_f64', 'risk_fp32', 'risk_sigmoid_f64')} for row in top]
        cuda_history = next(record['selection_metrics'] for record in history if record['epoch'] == epoch)
        backend_comparison = {'saved_cuda_epoch_counts': cuda_history, 'cpu_fp32_own_calibration_counts': variants['risk_fp32'],
                              'note': 'CPU numerical route is measured separately; no CPU/GPU logit equality claim.'}
        if name == 'selected':
            deltas = []; cpu_at_cuda_threshold = {}; disagreements = {}; cuda_rows = {}
            for role, filename in zip(ROLES, ('calibration_scores.jsonl', 'selection_scores.jsonl')):
                path = run_folder / filename; cuda_rows[role] = rows(path); files['saved_cuda:' + filename] = sha(path)
                if len(cuda_rows[role]) != len(scored[role]): raise ValueError('Saved CUDA membership differs')
                for left, right in zip(scored[role], cuda_rows[role]):
                    if any(left[k] != right[k] for k in ('document_id', 'label', 'text_sha256', 'source_group_id')): raise ValueError('Saved CUDA association differs')
                    if len(left['window_risks_fp32']) != len(right['window_scores']): raise ValueError('Saved CUDA window count differs')
                    deltas.extend(abs(a - b) for a, b in zip(left['window_risks_fp32'], right['window_scores']))
            for alpha in ALPHAS:
                key = str(alpha); threshold = cuda_history[key]['threshold']
                cpu_at_cuda_threshold[key] = evaluate_documents([r['risk_fp32'] for r in selection], [r['label'] for r in selection], threshold)
                disagreements[key] = [r['document_id'] for r, old in zip(selection, cuda_rows[ROLES[1]]) if (r['risk_fp32'] > threshold) != (old['risk_score'] > threshold)]
            backend_comparison.update(maximum_absolute_window_risk_delta=max(deltas), unequal_window_risks=sum(delta != 0 for delta in deltas),
                                      window_risk_delta_exceeding_1e5=sum(delta > 1e-5 for delta in deltas),
                                      cpu_at_recorded_cuda_threshold=cpu_at_cuda_threshold, changed_selection_decision_ids=disagreements)
        else: backend_comparison['saved_cuda_window_risks'] = 'Unavailable for latest epoch; only its recorded history counts are compared.'
        reports.append({'checkpoint_role': name, 'checkpoint': frozen[name], 'diagnostic_operating_points': variants,
                        'saturation': saturation, 'top25_benign_calibration_by_margin': top,
                        'backend_comparison': backend_comparison, 'artifact_sha256': files})
        del model
    report = {'complete': True, 'run_id': args.run_id, 'NON_ADOPTION_DIAGNOSTIC': True,
              'purpose': 'Distinguish probability rounding/saturation from poor low-FPR logit-margin separation. Primary scores and policies are unchanged.',
              'checkpoints_frozen_before_inference': frozen, 'data_identity': {role: a['identity'] for role, a in data.items()},
              'reports': reports, 'command': sys.argv, 'python': sys.version, 'torch': torch.__version__, 'threads': 1,
              'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.monotonic() - started,
              'source_sha256': {'probe': sha(Path(__file__)), 'trainer_loader': sha(ROOT / 'scripts/cnn_adoption_train.py'), 'cnn_models': sha(ROOT / 'scripts/cnn_models.py'), 'gate_eval': sha(ROOT / 'gate_eval.py')},
              'limitations': ['CPU logits are FP32; computing their subtraction/sigmoid in FP64 does not establish full FP64 network inference.',
                  'Even FP64 sigmoid can round to1 at sufficiently large margins. Raw max margin retains ordering longer.',
                  'All alternative thresholds are diagnostic, fit separately using the SAME benign order statistic/alphas. They do not rescue adoption or overwrite recorded FP32 outcomes.',
                  'Current CPU/GPU differences are explicit. Latest-epoch CUDA window logits/risks were not saved, so equality there is untested.',
                  'Only development calibration/selection are used. No confirmation, reserved benign calibration, teacher, new labels, fitting or GPU inference.']}
    if report['cuda_initialized']: raise ValueError('CPU-only probe initialized CUDA')
    write(output / 'complete.json', report)
    print(json.dumps({'complete': True, 'wall_seconds': report['wall_seconds'], 'cuda_initialized': False,
                      'checkpoints': [{r['checkpoint_role']: {k: v['0.005'] for k, v in r['diagnostic_operating_points'].items()}} for r in reports]}))


if __name__ == '__main__': main()
