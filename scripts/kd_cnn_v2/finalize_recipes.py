"""Freeze explicit coordinator choices and append bounded fixed-seed recipes.

No model selection, training, confirmation access or source edits occur here.
Default is a read-only preview. --write requires the exact current config digest.
Seed42 terminal/selected artifacts are reused; other seeds start from the same
PCA initializer, never a seed42 task-finetuned checkpoint.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

PRINCIPAL_ARMS = ('supervised_original', 'logit_original', 'logit_repeated_original', 'logit_expanded', 'richer_expanded')
OPTIONAL_ARMS = ('rf13_richer_expanded', 'shuffled_richer_expanded')
FINAL_SEEDS = (1337, 2026)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def resolve(path, canonical):
    path = Path(path)
    return path if path.is_absolute() else canonical / path


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Unsafe recipe ID')
    return value


def validate_recipe_ids(recipes):
    ids = [safe_id(recipe['run_id']) for recipe in recipes]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate recipe IDs')
    return {recipe['run_id']: recipe for recipe in recipes}


def final_recipe(parent, arm, seed, step, role='terminal'):
    """Pure recipe clone: fixed steps, no development reads, original schedule."""
    if parent['seed'] != 42 or not parent.get('monitor_development', True):
        raise ValueError('Expected seed42 development parent recipe')
    if not isinstance(step, int) or not 1 <= step <= parent['total_steps']:
        raise ValueError('Explicit final step outside parent training opportunity')
    if parent.get('initial_checkpoint') or parent.get('qat'):
        raise ValueError('Initial final matrix must start from declared common PCA initialization')
    result = copy.deepcopy(parent)
    result.update(run_id=f'{arm}_s{seed}_final_{role}', seed=seed, total_steps=step,
                  monitor_development=False, eval_every_steps=step)
    # Preserve the learning-rate schedule at a selected prefix even if a future
    # development parent uses linear decay. Constant/warmup0 is the current case.
    result['scheduler_steps'] = parent.get('scheduler_steps', parent['total_steps'])
    for name in ('development_root', 'development_selection_root', 'development_calibration_root', 'patience'):
        result.pop(name, None)
    return result


def proposal(config, principal_run, principal_step, include_rf13=False, include_shuffled=False):
    registry = validate_recipe_ids(config['recipes'])
    arms = list(PRINCIPAL_ARMS)
    if include_rf13: arms.append(OPTIONAL_ARMS[0])
    if include_shuffled: arms.append(OPTIONAL_ARMS[1])
    parents = {}
    for arm in arms:
        run = arm + '_s42_dev'
        if run not in registry: raise ValueError('Missing prescribed development parent: ' + run)
        parents[arm] = registry[run]
    if principal_run not in registry: raise ValueError('Unknown explicitly selected principal run')
    principal_arm = next((arm for arm in arms if principal_run == arm + '_s42_dev'), None)
    if principal_arm is None:
        # An explicitly nominated aligned RF127 richer scale treatment can be
        # the principal without altering the five frozen terminal causal arms.
        # Optional original RF13/shuffled arms still require their own flags.
        parent = registry[principal_run]
        if (principal_run in {arm + '_s42_dev' for arm in OPTIONAL_ARMS}
                or not principal_run.endswith('_s42_dev')
                or parent.get('model_id') != 'dilated64_rf127'
                or parent.get('representation_weight', 0) <= 0
                or 'shuffled' in parent.get('teacher_cache', '').lower()):
            raise ValueError('Additional principal must be explicitly nominated aligned RF127 richer parent; optional RF13/shuffled requires inclusion flag')
        principal_arm = principal_run[:-len('_s42_dev')]
    recipes, points = [], []
    for arm, parent in parents.items():
        for seed in FINAL_SEEDS:
            recipe = final_recipe(parent, arm, seed, parent['total_steps'])
            recipes.append(recipe)
        points.append({'arm': arm, 'role': 'terminal_causal_control', 'step': parent['total_steps'],
            'seed42_source_run': parent['run_id'], 'seed42_checkpoint_name': 'checkpoint_latest.pt',
            'final_recipe_ids': [f'{arm}_s{seed}_final_terminal' for seed in FINAL_SEEDS]})
    parent = registry[principal_run]
    if not isinstance(principal_step, int) or not 1 <= principal_step <= parent['total_steps']:
        raise ValueError('Principal step must be an explicit available parent prefix')
    if principal_step != parent['total_steps'] or principal_arm not in parents:
        for seed in FINAL_SEEDS: recipes.append(final_recipe(parent, principal_arm, seed, principal_step, 'selected'))
        selected_ids = [f'{principal_arm}_s{seed}_final_selected' for seed in FINAL_SEEDS]
        checkpoint = 'checkpoint_selected.pt' if principal_step != parent['total_steps'] else 'checkpoint_latest.pt'
    else:
        selected_ids = [f'{principal_arm}_s{seed}_final_terminal' for seed in FINAL_SEEDS]
        checkpoint = 'checkpoint_latest.pt'
    principal = {'arm': principal_arm, 'seed42_source_run': principal_run, 'step': principal_step,
                 'seed42_checkpoint_name': checkpoint, 'final_recipe_ids': selected_ids,
                 'selection': 'Explicit coordinator choice on development only; no seed or confirmation selection'}
    existing, additions = registry, []
    for recipe in recipes:
        if recipe['run_id'] in existing:
            if existing[recipe['run_id']] != recipe: raise ValueError('Existing final recipe identity differs: ' + recipe['run_id'])
        else: additions.append(recipe)
    validate_recipe_ids(config['recipes'] + additions)
    # Fixed causal edges must remain matched at terminal update counts.
    for left, right in (('supervised_original', 'logit_original'), ('logit_repeated_original', 'logit_expanded'), ('logit_expanded', 'richer_expanded')):
        if parents[left]['total_steps'] != parents[right]['total_steps']:
            raise ValueError('Terminal causal comparison update counts differ: ' + left + '/' + right)
    return additions, points, principal


def source_artifact(canonical, config, run_id, checkpoint_name, expected_step):
    """Validate complete source identity before reusing any trained checkpoint."""
    registry = validate_recipe_ids(config['recipes']); recipe = registry[run_id]
    folder = canonical / 'results/kd_cnn_v2/runs' / safe_id(run_id)
    complete, run = read(folder / 'complete.json'), read(folder / 'run.json')
    if not complete.get('complete') or complete['steps_completed'] != recipe['total_steps']:
        raise ValueError('Development source is not complete at predeclared terminal exposure: ' + run_id)
    association = run['association']
    if association['recipe'] != recipe or identity(association) != run['association_sha256'] or complete['association_sha256'] != run['association_sha256']:
        raise ValueError('Completed source recipe/run association differs: ' + run_id)
    model_config = config['model_specs'][recipe['model_id']]
    model_config = model_config.get('config', model_config)
    if association['model_config'] != model_config: raise ValueError('Source model identity differs')
    # This helper never imports training modules or initializes CUDA. CPU restore
    # only checks metadata and the source association; arrays remain unchanged.
    import torch
    checkpoint = folder / checkpoint_name
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if stored['association_sha256'] != run['association_sha256'] or stored['recipe'] != recipe or stored['step'] != expected_step or stored['model_config'] != model_config:
        raise ValueError('Chosen source checkpoint is not the explicit recipe/step: ' + str(checkpoint))
    if checkpoint_name == 'checkpoint_selected.pt' and complete['selected_step'] != expected_step:
        raise ValueError('Requested principal prefix was not retained; no interpolated checkpoint may be invented')
    if checkpoint_name == 'checkpoint_selected.pt' and sha(checkpoint) != complete['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed from completed digest')
    return {'run_id': run_id, 'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha(checkpoint),
        'step': expected_step, 'seed': recipe['seed'], 'recipe_sha256': identity(recipe),
        'run_record_sha256': sha(folder / 'run.json'), 'complete_record_sha256': sha(folder / 'complete.json'),
        'association_sha256': run['association_sha256'], 'source_state_reused_without_training': True}


def fit_accounting(config, additions, canonical):
    ids = set(validate_recipe_ids(config['recipes'] + additions))
    actual = canonical / 'results/kd_cnn_v2/runs'
    started = {folder.name for folder in actual.iterdir() if folder.is_dir() and (folder / 'run.json').is_file()} if actual.exists() else set()
    cap = min(28, int(config['resource_limits']['max_learning_runs']))
    union = ids | started
    if len(union) > cap: raise ValueError(f'Prospective matrix requires {len(union)} distinct planned/started fits, above cap {cap}')
    return {'cap': cap, 'planned_plus_started_distinct_fits': len(union), 'started_fits': len(started),
            'new_final_fits': len(additions), 'remaining_fit_slots': cap - len(union),
            'unregistered_started_run_ids': sorted(started - ids),
            'counting': 'Seed42 source reuse is not a new fit. All planned recipes and started task-run IDs count; optional QAT must fit remaining slots.'}


def histories(canonical):
    result = []
    folder = canonical / 'results/kd_cnn_v2/runs'
    if not folder.exists(): return result
    for run in sorted(folder.iterdir()):
        path = run / 'history.json'
        if not path.is_file(): continue
        rows = read(path)
        trace = [{'step': row['step'], 'fit_passes': row['fit_passes'], **row['mean_batch_loss'],
            'recall_at_nominal1pct': row.get('development', {}).get('0.01', {}).get('recall'),
            'observed_fpr': row.get('development', {}).get('0.01', {}).get('fpr'),
            'auc': row.get('development', {}).get('auc')} for row in rows]
        result.append({'run_id': run.name, 'complete': (run / 'complete.json').is_file(),
            'history_sha256': sha(path), 'trace': trace,
            'last_loss_relative_change': (trace[-1]['loss'] - trace[-2]['loss']) / abs(trace[-2]['loss']) if len(trace) > 1 and trace[-2]['loss'] else None,
            'interpretation_boundary': 'Descriptive executed learning trace; no automatic convergence threshold, candidate choice, or continuation authorization'})
    return result


def make_plan(config_path, canonical, out, principal_run, principal_step, include_rf13, include_shuffled):
    config = read(config_path); source_config_sha = sha(config_path)
    additions, points, principal = proposal(config, principal_run, principal_step, include_rf13, include_shuffled)
    accounting = fit_accounting(config, additions, canonical)
    sources = {}
    for point in points:
        key = point['seed42_source_run'] + ':terminal'
        sources[key] = source_artifact(canonical, config, point['seed42_source_run'], point['seed42_checkpoint_name'], point['step'])
        point['seed42_source_key'] = key
    principal_key = principal_run + (':terminal' if principal['seed42_checkpoint_name'] == 'checkpoint_latest.pt' else ':selected')
    if principal_key not in sources:
        sources[principal_key] = source_artifact(canonical, config, principal_run, principal['seed42_checkpoint_name'], principal_step)
    principal['seed42_source_key'] = principal_key
    choices = {'principal_run': principal_run, 'principal_step': principal_step, 'include_rf13': include_rf13, 'include_shuffled': include_shuffled}
    new_config = copy.deepcopy(config); new_config['recipes'].extend(additions)
    new_config['phase'] = 'final_learning_recipes_frozen_before_confirmation'
    new_config['final_learning_plan'] = {'path': str(out.resolve()), 'choices': choices,
        'terminal_recipe_ids': [run for point in points for run in point['final_recipe_ids']],
        'principal_recipe_ids': principal['final_recipe_ids'], 'seed42_reuse': sources}
    artifact = {'schema': 'kd_cnn_v2_final_learning_v1', 'immutable': True, 'before_new_confirmation': True,
        'choices': choices, 'source_config_sha256': source_config_sha,
        'output_config_sha256': hashlib.sha256(encoded(new_config)).hexdigest(),
        'common_contract_sha256': config['shared_contract_sha256'], 'seeds': [42, 1337, 2026],
        'terminal_causal_points': points, 'principal_selected_point': principal, 'seed42_source_artifacts': sources,
        'new_recipes': additions, 'fit_accounting': accounting, 'helper_sha256': sha(Path(__file__)),
        'selection_boundary': 'Coordinator supplies the principal run and retained step; this helper does not rank outcomes or choose a seed.',
        'final_execution': 'Train new fixed recipes without development monitoring, then freeze actual checkpoints/formats/threshold policy in evaluate.py before confirmation.',
        'qat_recipe_fields': {'required': ['run_id', 'model_id', 'seed', 'initial_checkpoint', 'initial_checkpoint_sha256', 'parent_run',
            'qat=true', 'qat_exponents', 'fit_root', 'total_steps', 'epochs', 'learning_rate', 'monitor_development'],
            'optional': ['teacher_cache', 'label_weight', 'logit_weight', 'representation_weight', 'temperature', 'scheduler', 'scheduler_steps',
                'warmup_fraction', 'batch_documents', 'microbatch_windows', 'gradient_clip', 'development_selection_root', 'development_calibration_root'],
            'exponent_boundary': 'Freeze FIT-calibrated exponents JSON; trainer binds contents in association. Export adapted model with the same exponents; do not recalibrate silently.',
            'training_head': 'Adaptation loads pure student only; richer head is newly initialized if representation_weight>0. Compare matched adaptation exposure and retain direct conversion.',
            'budget': 'At most two development adaptation recipes; fixed winning recipe across seeds only by explicit coordinator choice and within remaining fit/GPU budgets.'}}
    return artifact, new_config


def atomic_config(path, content, expected_sha):
    if sha(path) != expected_sha: raise ValueError('Config changed concurrently; no overwrite permitted')
    temporary = path.with_suffix(path.suffix + '.kd_final.tmp')
    if temporary.exists(): raise ValueError('Existing config candidate must be reviewed/preserved')
    temporary.write_bytes(content)
    if sha(path) != expected_sha:
        raise ValueError('Config changed while preparing candidate; original preserved, candidate left for review')
    os.replace(temporary, path)


def self_test():
    parents = []
    for arm in PRINCIPAL_ARMS + OPTIONAL_ARMS:
        parents.append({'run_id': arm + '_s42_dev', 'seed': 42, 'model_id': 'fixed', 'epochs': 8,
            'total_steps': 80 if arm in PRINCIPAL_ARMS[:2] else 88, 'monitor_development': True,
            'scheduler': 'constant', 'warmup_fraction': 0., 'learning_rate': .001,
            'development_selection_root': 'selection', 'development_calibration_root': 'calibration'})
    config = {'recipes': parents}
    original = copy.deepcopy(config)
    a, points, principal = proposal(config, 'richer_expanded_s42_dev', 44)
    assert len(a) == 12 and len(points) == 5 and principal['step'] == 44
    assert all(recipe['seed'] in FINAL_SEEDS and recipe['monitor_development'] is False and 'development_selection_root' not in recipe for recipe in a)
    b, points2, principal2 = proposal(config, 'richer_expanded_s42_dev', 88, True, True)
    assert len(b) == 14 and len(points2) == 7 and principal2['final_recipe_ids'] == ['richer_expanded_s1337_final_terminal', 'richer_expanded_s2026_final_terminal']
    extended = copy.deepcopy(config); extended['recipes'].extend(a)
    assert proposal(extended, 'richer_expanded_s42_dev', 44)[0] == []
    assert config == original
    for altered in (dict(config, recipes=config['recipes'] + [dict(a[0], learning_rate=.1)]),):
        try: proposal(altered, 'richer_expanded_s42_dev', 44)
        except ValueError: pass
        else: raise AssertionError('Changed recipe identity accepted')
    try: proposal(config, 'rf13_richer_expanded_s42_dev', 44)
    except ValueError: pass
    else: raise AssertionError('Implicit optional arm accepted')
    additional = copy.deepcopy(config)
    treatment = dict(parents[4], run_id='richer_weight3_expanded_s42_dev', model_id='dilated64_rf127', representation_weight=3.)
    additional['recipes'].append(treatment)
    extra, original_points, extra_principal = proposal(additional, treatment['run_id'], treatment['total_steps'])
    assert len(extra) == 12 and len(original_points) == 5
    assert extra_principal['final_recipe_ids'] == ['richer_weight3_expanded_s1337_final_selected', 'richer_weight3_expanded_s2026_final_selected']
    assert not any(point['arm'] == 'richer_weight3_expanded' for point in original_points)
    capped = copy.deepcopy(config); capped['resource_limits'] = {'max_learning_runs': 28}
    assert fit_accounting(capped, a, Path(__file__).parent / 'nonexistent_accounting_fixture')['planned_plus_started_distinct_fits'] == 19
    capped['recipes'].extend({'run_id': 'reserved' + str(index)} for index in range(8))
    try: fit_accounting(capped, b, Path(__file__).parent / 'nonexistent_accounting_fixture')
    except ValueError: pass
    else: raise AssertionError('Twenty-eight-fit ceiling exceeded without rejection')
    return {'passed': True, 'five_arms_plus_distinct_selected_new_fits': len(a), 'seven_terminal_arms_new_fits': len(b),
            'idempotent_recipe_proposal': True, 'original_recipes_unchanged': True,
            'fit_cap_and_changed_recipe_and_implicit_optional_arm_rejected': True,
            'explicit_extra_richer_principal_preserves_five_terminal_arms': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--canonical-root'); parser.add_argument('--out')
    parser.add_argument('--principal-run'); parser.add_argument('--principal-step', type=int)
    parser.add_argument('--include-rf13', action='store_true'); parser.add_argument('--include-shuffled', action='store_true')
    parser.add_argument('--write', action='store_true'); parser.add_argument('--config-sha256')
    parser.add_argument('--inspect-histories', action='store_true'); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test: print(json.dumps(self_test(), indent=2)); return
    if not args.canonical_root: parser.error('--canonical-root required')
    canonical = Path(args.canonical_root)
    if args.inspect_histories: print(json.dumps(histories(canonical), indent=2)); return
    if any(value is None for value in (args.config, args.out, args.principal_run, args.principal_step)):
        parser.error('--config, --out, --principal-run and --principal-step required')
    config_path, out = Path(args.config), Path(args.out)
    if args.write and (not args.config_sha256 or sha(config_path) != args.config_sha256):
        raise ValueError('--write requires exact current --config-sha256 to protect concurrent edits')
    if out.exists():
        existing = read(out)
        expected = {'principal_run': args.principal_run, 'principal_step': args.principal_step,
                    'include_rf13': args.include_rf13, 'include_shuffled': args.include_shuffled}
        if existing.get('schema') != 'kd_cnn_v2_final_learning_v1' or existing['choices'] != expected:
            raise ValueError('Existing immutable plan differs; preserve it')
        if sha(config_path) == existing['output_config_sha256']:
            print(json.dumps({'complete': True, 'reused_immutable_plan': str(out), 'fits': existing['fit_accounting']})); return
        if sha(config_path) != existing['source_config_sha256']:
            raise ValueError('Existing plan config source/output differs; explicit reconciliation required')
    artifact, new_config = make_plan(config_path, canonical, out, args.principal_run, args.principal_step, args.include_rf13, args.include_shuffled)
    if args.write:
        if out.exists() and read(out) != artifact: raise ValueError('Immutable final plan differs')
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(encoded(artifact))
        atomic_config(config_path, encoded(new_config), args.config_sha256)
    print(json.dumps({'write': args.write, 'plan': str(out), 'choices': artifact['choices'], 'fit_accounting': artifact['fit_accounting'],
        'new_recipe_ids': [recipe['run_id'] for recipe in artifact['new_recipes']],
        'source_config_sha256': artifact['source_config_sha256'], 'output_config_sha256': artifact['output_config_sha256']}, indent=2))


if __name__ == '__main__': main()
