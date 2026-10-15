"""Bounded CPU-only direct-conversion development monitor; no model selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUNS = tuple(arm + '_s42_dev' for arm in ('supervised_original', 'logit_original',
    'logit_repeated_original', 'logit_expanded', 'richer_expanded', 'shuffled_richer_expanded'))
ROLES = ('new_development_calibration', 'new_selection')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value: raise ValueError('Immutable artifact differs: ' + str(path))
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def resolve(value, root):
    path = Path(value); return path if path.is_absolute() else root / path


def child(command, folder, name, remaining):
    environment = dict(os.environ); environment['CUDA_VISIBLE_DEVICES'] = ''
    began = time.monotonic(); log = folder / (name + '.log')
    dispatch = folder / (name + '.dispatch.json')
    if dispatch.exists():
        prior = read(dispatch)
        if prior['command'] != command or prior['returncode'] != 0 or not log.exists() or prior['log_sha256'] != sha(log):
            raise ValueError('Prior dispatch differs/failed; preserve and inspect: ' + str(dispatch))
        return prior
    if log.exists(): raise ValueError('Preserve previous dispatch log: ' + str(log))
    with log.open('x', encoding='utf-8') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
            timeout=max(1, min(1200, remaining())), env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    record = {'command': command, 'returncode': result.returncode, 'wall_seconds': time.monotonic() - began,
              'log': str(log), 'log_sha256': sha(log), 'device': 'cpu', 'CUDA_VISIBLE_DEVICES': ''}
    write(dispatch, record, True)
    if result.returncode: raise RuntimeError('CPU numerical dispatch failed: ' + str(log))
    return record


def score_identity(path):
    side = path.with_suffix('.complete.json'); complete = read(side)
    if not complete.get('complete') or complete['stage'] != 'development' or complete['role'] not in ROLES or complete['scores_sha256'] != sha(path):
        raise ValueError('Incomplete/mismatched development score artifact: ' + str(path))
    return {'path': str(path), 'sha256': sha(path), 'complete_sha256': sha(side), 'target': complete['target']}


def comparison(float_paths, integer_paths):
    import numpy as np
    from analysis import describe, paired, rows
    fc, fs = (rows(float_paths[role]) for role in ROLES)
    ic, ins = (rows(integer_paths[role]) for role in ROLES)
    for left, right in ((fc, ic), (fs, ins)):
        if [row['document_id'] for row in left] != [row['document_id'] for row in right]: raise ValueError('Float/integer document pairing differs')
        if any(a['text_sha256'] != b['text_sha256'] or len(a['window_margins']) != len(b['window_margins']) for a, b in zip(left, right)):
            raise ValueError('Float/integer text/window identities differ')
    fm, im = describe(fc, fs), describe(ic, ins)
    ft = fm['operating_points']['0.01']['overall']['threshold']; it = im['operating_points']['0.01']['overall']['threshold']
    float_decisions = np.array([row['margin'] > ft for row in fs])
    same = np.array([row['margin'] > ft for row in ins]); recalibrated = np.array([row['margin'] > it for row in ins])
    errors = np.array([abs(a - b) for x, y in zip(fs, ins) for a, b in zip(x['window_margins'], y['window_margins'])])
    doc_errors = np.array([abs(a['margin'] - b['margin']) for a, b in zip(fs, ins)])
    return {'float': fm, 'direct_integer': im, 'paired_integer_minus_float': paired(ins, fs, ic, fc, repeats=1000),
        'numerical_changes': {'window_margin_max_abs_error': float(errors.max()), 'window_margin_mean_abs_error': float(errors.mean()),
            'document_margin_max_abs_error': float(doc_errors.max()), 'document_margin_mean_abs_error': float(doc_errors.mean()),
            'float_threshold_margin': ft, 'integer_recalibrated_threshold_margin': it,
            'same_float_threshold_changes': int(np.sum(same != float_decisions)),
            'same_float_threshold_changed_ids': [row['document_id'] for row, changed in zip(fs, same != float_decisions) if changed],
            'recalibrated_threshold_changes': int(np.sum(recalibrated != float_decisions)),
            'recalibrated_threshold_changed_ids': [row['document_id'] for row, changed in zip(fs, recalibrated != float_decisions) if changed]},
        'interpretation': 'Direct conversion on selected development checkpoint only. Thresholds separately calibrated using the same frozen policy; same-float-threshold changes isolate conversion effects. No principal/QAT decision made.'}


def process_run(root, config, run_id, out, remaining):
    from finalize_recipes import source_artifact
    recipe = next(recipe for recipe in config['recipes'] if recipe['run_id'] == run_id)
    done = read(root / 'results/kd_cnn_v2/runs' / run_id / 'complete.json')
    source = source_artifact(root, config, run_id, 'checkpoint_selected.pt', done['selected_step'])
    destination = out / run_id; destination.mkdir(parents=True, exist_ok=True)
    completed = destination / 'complete.json'
    if completed.exists():
        result = read(completed)
        if result['source'] != source: raise ValueError('Numerical result belongs to a different selected checkpoint')
        for role in ROLES:
            for kind in ('float_scores', 'integer_scores'):
                now = score_identity(Path(result[kind][role]['path']))
                if now != result[kind][role]: raise ValueError('Prior score identity changed')
        return result
    export = root / 'results/kd_cnn_v2/exports' / (run_id + '_direct')
    fit = resolve(recipe['fit_root'], root); dispatches = []
    if (export / 'manifest.json').exists():
        manifest = read(export / 'manifest.json')
        if manifest['provenance']['checkpoint_sha256'] != source['checkpoint_sha256']:
            raise ValueError('Existing direct export belongs to different checkpoint')
        for file, field in (('model.bin', 'model_sha256'), ('model.npz', 'npz_sha256')):
            if sha(export / file) != manifest[field]: raise ValueError('Existing export payload changed')
        fit_manifest = read(fit / 'manifest.json')
        if manifest['provenance']['fit_identity']['manifest_sha256'] != sha(fit / 'manifest.json') or any(
            manifest['provenance']['fit_identity']['artifact_sha256'][name] != fit_manifest['artifact_sha256'][name]
            for name in ('ids.npy', 'masks.npy')):
            raise ValueError('Existing export calibration identity differs')
    else:
        command = [sys.executable, '-B', str(Path(__file__).with_name('numerical.py')), '--checkpoint', source['checkpoint'],
            '--calibration', str(fit), '--out', str(export), '--max-windows', '128']
        dispatches.append(child(command, destination, 'export_fit128', remaining))
    manifest = read(export / 'manifest.json')
    if manifest['provenance']['checkpoint_sha256'] != source['checkpoint_sha256']: raise ValueError('New export source mismatch')
    integer_id = run_id + '_direct'; float_id = run_id + '_selected_fp32'
    targets = {'targets': [
        {'id': float_id, 'kind': 'cnn_fp32', 'checkpoint': source['checkpoint'], 'checkpoint_sha256': source['checkpoint_sha256'], 'seed': 42},
        {'id': integer_id, 'kind': 'cnn_integer', 'export_dir': str(export), 'export_manifest_sha256': sha(export / 'manifest.json'), 'seed': 42}]}
    target_list = destination / 'targets.json'; write(target_list, targets, True)
    # Reuse the coordinator's already qualified supervised outputs, preserving
    # their original identities and avoiding redundant CPU execution.
    if run_id == 'supervised_original_s42_dev':
        float_paths = {role: root / 'results/kd_cnn_v2/development/anchors' / (run_id + '_' + role + '.jsonl') for role in ROLES}
        integer_paths = {role: root / 'results/kd_cnn_v2/development/numerical_initial' / integer_id / (role + '.jsonl') for role in ROLES}
    else:
        score_root = destination / 'scores'
        command = [sys.executable, '-B', str(Path(__file__).with_name('score_queue.py')), '--canonical-root', str(root),
            '--target-list', str(target_list), '--stage', 'development', '--out-root', str(score_root), '--device', 'cpu']
        dispatches.append(child(command, destination, 'score_development_cpu', remaining))
        float_paths = {role: score_root / float_id / (role + '.jsonl') for role in ROLES}
        integer_paths = {role: score_root / integer_id / (role + '.jsonl') for role in ROLES}
    float_identity = {role: score_identity(path) for role, path in float_paths.items()}
    integer_identity = {role: score_identity(path) for role, path in integer_paths.items()}
    for role in ROLES:
        if float_identity[role]['target']['checkpoint_sha256'] != source['checkpoint_sha256']:
            raise ValueError('Float comparison is not selected checkpoint')
        if integer_identity[role]['target']['export_manifest_sha256'] != sha(export / 'manifest.json'):
            raise ValueError('Integer comparison is not direct export')
    analysis = comparison(float_paths, integer_paths); analysis_path = destination / 'analysis.json'; write(analysis_path, analysis, True)
    result = {'complete': True, 'run_id': run_id, 'source': source, 'export_dir': str(export),
        'export_manifest_sha256': sha(export / 'manifest.json'), 'targets_sha256': sha(target_list),
        'float_scores': float_identity, 'integer_scores': integer_identity, 'analysis_sha256': sha(analysis_path),
        'float_primary': analysis['float']['operating_points']['0.01']['overall'],
        'integer_primary': analysis['direct_integer']['operating_points']['0.01']['overall'],
        'float_auc': analysis['float']['auc'], 'integer_auc': analysis['direct_integer']['auc'],
        'numerical_changes': analysis['numerical_changes'], 'dispatches': dispatches,
        'scope': 'CPU direct W8/A8 conversion and development scoring only; no confirmation/GPU/selection/QAT execution'}
    write(completed, result, True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical-root', type=Path, required=True); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True); parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--poll-seconds', type=float, default=45); parser.add_argument('--single-pass', action='store_true')
    parser.add_argument('--run-ids', nargs='+', help='Explicit registered RF127 seed42 development treatments; default is the frozen six-run set')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or not 0 < args.max_seconds <= 3600: raise ValueError('Bounded polling/wall time required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = args.canonical_root.resolve(); out = args.out_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not out.is_relative_to((root / 'results/kd_cnn_v2').resolve()): raise ValueError('Monitor outputs must remain task-owned')
    run_ids = tuple(args.run_ids) if args.run_ids else RUNS
    if len(set(run_ids)) != len(run_ids): raise ValueError('Duplicate explicit numerical run IDs')
    registered = {recipe['run_id']: recipe for recipe in read(args.config)['recipes']}
    for run_id in run_ids:
        recipe = registered.get(run_id, {})
        if recipe.get('model_id') != 'dilated64_rf127' or recipe.get('seed') != 42 or not recipe.get('monitor_development') or not run_id.endswith('_s42_dev'):
            raise ValueError('Numerical monitoring accepts explicit registered RF127 seed42 development treatments only')
    started = time.monotonic(); completed = {}; waiting = list(run_ids)
    def remaining(): return max(0., args.max_seconds - (time.monotonic() - started))
    while remaining() > 0:
        if (out / 'stop.request').exists() or (root / 'results/kd_cnn_v2/stop.request').exists(): break
        config = read(args.config)
        for run_id in run_ids:
            if run_id in completed: continue
            if not (root / 'results/kd_cnn_v2/runs' / run_id / 'complete.json').is_file(): continue
            if remaining() < 120: break
            result = process_run(root, config, run_id, out, remaining)
            completed[run_id] = result
            print(json.dumps({'event': 'numerical_development_complete', 'run_id': run_id,
                'float_auc': result['float_auc'], 'integer_auc': result['integer_auc'], 'float_primary': result['float_primary'],
                'integer_primary': result['integer_primary'], 'numerical_changes': result['numerical_changes']}), flush=True)
        waiting = [run for run in run_ids if run not in completed]
        status = {'complete': not waiting, 'completed_runs': list(completed), 'waiting_runs': waiting,
                  'wall_seconds': time.monotonic() - started, 'max_seconds': args.max_seconds, 'device': 'cpu',
                  'scope': 'Selected-checkpoint direct conversion on two development roles only; no candidate choice'}
        write(out / 'monitor_status.json', status)
        if not waiting or args.single_pass or remaining() <= args.poll_seconds: break
        time.sleep(min(args.poll_seconds, remaining()))
    print(json.dumps({'event': 'monitor_exit', 'completed': list(completed), 'waiting': waiting,
                      'wall_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__': main()
