"""Coordinator-only prospective freeze builder; never updates plan or scores data.

Use --plan --nomination --output after final checkpoints/formats are complete.
Nomination supplies exact targets, candidate_groups, final_recipes,
historical_anchor, priority_order, deployment_seed42, profiling_policy,
material_hardware_benefit, rationale, source_sha256 and backend_records.
Reserved manifests are decoded as metadata; document/score JSONL are hashed
as opaque bytes only. This helper introduces no additional approval stage.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from cnn_adoption_train import ROOT, DEST, identity, read, sha
from cnn_adoption_evaluate import calibration_policy, path, resolve_target, verify_membership
from cnn_adoption_profile import policy_for

SEEDS = {42, 1337, 2026}
BASE_SOURCES = {'scripts/cnn_adoption_freeze.py', 'scripts/cnn_adoption_evaluate.py',
    'scripts/cnn_adoption_train.py', 'scripts/cnn_models.py', 'gate_eval.py',
    'scripts/cnn_adoption_profile.py', 'scripts/cnn_adoption_compare.py',
    'scripts/r1_bert_cuda_graph.py', 'scripts/r1_challengers.py', 'configs/project.json'}


def require(condition, reason):
    if not condition: raise ValueError(reason)


def local(value, root=ROOT):
    result = Path(value); result = result if result.is_absolute() else root / result
    require(result.resolve().is_relative_to(root.resolve()), 'Artifact/source escapes workspace')
    return result


def selection_contract(plan, nomination):
    targets = nomination['targets']; groups = nomination['candidate_groups']; recipes = nomination['final_recipes']
    require(targets and groups and nomination['deployment_seed'] == 42, 'Explicit targets/groups and deployment seed42 required')
    require(isinstance(nomination['rationale'], str) and nomination['rationale'].strip(), 'Selection rationale required')
    for field in ('priority_order', 'profiling_policy', 'material_hardware_benefit', 'calibration_policy', 'quality_gates'):
        require(nomination[field] == plan[field], 'Nominated/planned policy differs: ' + field)
    require(nomination['material_hardware_benefit'], 'Explicit material hardware benefit required')
    calibration_policy(plan)
    gates = plan['quality_gates']
    require(gates['minimum_recall'] == .99 and gates['maximum_fpr'] == .01 and
        gates['maximum_hard_benign_fpr'] == .05 and gates['minimum_mechanism_recall'] == .95 and
        gates['minimum_primary_email_recall'] == .99 and gates['maximum_primary_email_fpr'] == .01 and
        gates['minimum_mechanism_attacks'] >= 100 and gates['minimum_mechanism_attack_source_groups'] >= 100 and
        gates['minimum_mechanism_payload_groups'] >= 20 and gates['primary_genre_values'] == ['natural_email'] and
        set(gates['required_mechanism_groups']) == {'role_delimiter_spoof', 'encoded_instruction', 'code_simulation'},
        'Approved primary/stress quality policy changed')
    policy = nomination['profiling_policy']
    args = SimpleNamespace(**{name: policy[name] for name in ('sample_per_stratum', 'long_stress_per_label', 'warmups', 'repeats', 'cpu_threads', 'cold_documents')})
    require(policy == policy_for(args) and args.sample_per_stratum >= 1 and args.long_stress_per_label >= 1 and
        args.warmups >= 1 and args.repeats >= 3 and args.cpu_threads in (1, 4) and args.cold_documents == 2,
        'Profiling policy differs from actual profiler or lacks repeats/long/cold cases')
    execution = plan['execution_policy']
    require(execution['device'] == 'cuda' and execution['batch_windows'] == 1 and execution['tf32'] is False,
        'Shared qualified CUDA batch-one FP32 scoring boundary required')
    seen = []; cnn_groups = []; bert_groups = []; run_ids = set()
    by_run = {r['run_id']: r for r in plan['runs']}
    require(len(by_run) == len(plan['runs']), 'Duplicate planned run IDs')
    for group_id, keys in groups.items():
        require(keys and len(set(keys)) == len(keys), 'Empty/duplicate candidate targets')
        require(all(key in targets for key in keys), 'Candidate references absent target')
        seen.extend(keys)
        specs = [targets[key]['model_spec'] for key in keys]
        require(all(spec == specs[0] for spec in specs), 'Candidate group mixes model specifications')
        kind = specs[0]['kind']; require(kind in ('cnn', 'bert'), 'Final group must be CNN or matched BERT')
        (cnn_groups if kind == 'cnn' else bert_groups).append(group_id)
        if kind == 'bert': require(specs[0].get('initialization', 'pretrained') == 'pretrained', 'Matched BERT must retain pretrained encoder')
        group_runs = {key.rsplit(':', 1)[0] for key in keys}
        require(len(group_runs) == 3 and all(run_id in recipes and run_id in by_run for run_id in group_runs), 'Each candidate requires exactly three explicit final recipes')
        selected = [recipes[run_id] for run_id in sorted(group_runs)]
        require({r['seed'] for r in selected} == SEEDS, 'Required three-seed coverage differs')
        for recipe in selected:
            require(recipe == by_run[recipe['run_id']] and recipe.get('monitor_development') is False and not recipe.get('patience', 0),
                'Exact fixed-epoch planned recipe required; no final early stopping')
            require(recipe['epochs'] > 0 and recipe.get('schedule_total_epochs', recipe['epochs']) >= recipe['epochs'], 'Explicit valid final epoch/scheduler horizon required')
        normalized = [{k: v for k, v in recipe.items() if k not in ('seed', 'run_id')} for recipe in selected]
        require(all(value == normalized[0] for value in normalized), 'Final recipes/horizons differ across seeds')
        for fmt in {targets[key]['numerical_format'] for key in keys}:
            require({key.rsplit(':', 1)[0] for key in keys if targets[key]['numerical_format'] == fmt} == group_runs,
                'Every nominated format must cover all three seeds')
        run_ids.update(group_runs)
    require(1 <= len(cnn_groups) <= 2 and len(bert_groups) == 1, 'Nominate one/two CNN candidates and exactly one matched BERT')
    require(set(recipes) == run_ids, 'Final recipe set differs from nominated runs')
    priority = nomination['priority_order']
    require(len(priority) == len(set(priority)) and set(priority) == set(cnn_groups), 'Priority must cover each CNN candidate exactly once')
    anchor = nomination['historical_anchor']
    require(anchor in targets and targets[anchor]['model_spec']['kind'] == 'historical_bert_w8' and
        targets[anchor]['numerical_format'] == 'historical_w8_qat', 'Exact historical W8 anchor required')
    require(len(seen) == len(set(seen)) and set(targets) == set(seen) | {anchor} and anchor not in seen,
        'Target/group/anchor association incomplete or duplicated')
    return run_ids


def source_contract(nomination):
    required = set(BASE_SOURCES)
    for descriptor in nomination['targets'].values():
        if descriptor['numerical_format'] == 'w8a8_export':
            required |= {'scripts/cnn_adoption_export.py', 'scripts/cnn_adoption_backend.py', 'scripts/check_cnn_adoption_backend.py'}
        if descriptor['model_spec']['kind'] == 'historical_bert_w8':
            required |= {'gate_quant.py', 'scripts/r1_bert_int8.py'}
        spec = descriptor['model_spec']
        if spec.get('embedding_initialization') or spec['kind'] == 'cnn' and 'pca' in spec.get('initialization', '').lower():
            required.add('scripts/cnn_adoption_embedding_init.py')
    if any(recipe.get('embedding_initialization') for recipe in nomination['final_recipes'].values()):
        required.add('scripts/cnn_adoption_embedding_init.py')
    sources = nomination['source_sha256']
    require(required <= set(sources), 'Nomination lacks scoring/profiling/conditional source closure: ' + ', '.join(sorted(required - set(sources))))
    for filename, digest in sources.items():
        require(sha(local(filename)) == digest, 'Nominated source changed: ' + filename)
    return sources


def backend_record(reference, descriptor, plan):
    file = local(reference['path']); require(file.resolve().is_relative_to(DEST.resolve()), 'Backend record outside adoption results')
    require(sha(file) == reference['sha256'], 'Backend qualification bytes changed')
    record = read(file); kind = reference['kind']
    if kind == 'development_evaluation':
        association = record['association']
        require(record['complete'] and association['stage'] == 'development' and association['target'] == descriptor and
            record['association_sha256'] == identity(association) and association['batch_windows'] == 1,
            'Development evaluator does not qualify exact target/format/B1')
        require(association['calibration_policy'] == plan['calibration_policy'] and association['quality_gates'] == plan['quality_gates'], 'Qualified evaluator policy differs')
        required = {'scripts/cnn_adoption_evaluate.py', 'scripts/cnn_adoption_train.py', 'scripts/cnn_models.py', 'gate_eval.py'}
        if descriptor['numerical_format'] == 'w8a8_export': required |= {'scripts/cnn_adoption_export.py', 'scripts/cnn_adoption_backend.py'}
        if descriptor['model_spec']['kind'] == 'historical_bert_w8': required |= {'gate_quant.py', 'scripts/r1_bert_int8.py'}
        require(required <= set(association['source_sha256']), 'Qualified evaluator omits scoring sources')
        for filename, digest in association['source_sha256'].items(): require(sha(local(filename)) == digest, 'Qualified scoring implementation changed')
        require(set(association['data']) == {'development_calibration', 'development_selection', 'known_diagnostics'}, 'Qualified evaluator data roles differ')
        for role in ('development_calibration', 'development_selection'):
            expected = plan['data_contract']['prepared_membership'][role]['artifact_sha256']
            actual = association['data'][role]
            require(actual['role'] == role and all(expected.get(name) == digest for name, digest in actual['files'].items()) and
                {'ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl'} <= set(actual['files']), 'Qualified evaluator used another development revision')
        execution = plan['execution_policy']
        require(association['device'] == execution['device'] and association['torch'] == execution['torch'] and
            int(association['cache_limit']) == execution['cache_limit'] and int(association['cudnn_v8_api_disabled']) == execution['cudnn_v8_api_disabled'],
            'Qualified evaluator execution boundary differs')
        require(set(record['score_files_sha256']) == {r + '_scores.jsonl' for r in ('development_calibration', 'development_selection', 'known_diagnostics')}, 'Unexpected development score roles')
        for name, digest in record['score_files_sha256'].items(): require(Path(name).name == name and sha(file.parent / name) == digest, 'Qualified score artifact changed')
    elif kind == 'native_integer':
        require(descriptor['numerical_format'] == 'w8a8_export' and record['passed'] and
            record['manifest_sha256'] == descriptor['export_manifest_sha256'] and
            record['checkpoint_sha256'] == descriptor['checkpoint_identity']['sha256'] and
            record['backend_source_sha256'] == sha(ROOT / 'scripts/cnn_adoption_backend.py') and
            record['source_sha256'] == sha(ROOT / 'scripts/check_cnn_adoption_backend.py') and
            record['backend']['fallback'] is False and record['torch'] == plan['execution_policy']['torch'] and
            record['dot_checks'] and all(row['exact_accumulator'] for row in record['dot_checks']) and
            record['output_checks'] and all(row['passed'] for row in record['output_checks']), 'Native packed arithmetic qualification differs or failed')
        require(local(record['fit_directory']).resolve() == local(plan['prepared_root']).resolve() / 'fit', 'Native qualification used a different fitting revision')
    elif kind == 'development_profile':
        association = record['association']
        require(record['complete'] and association['stage'] == 'development' and association['target'] == descriptor and
            association['policy'] == plan['profiling_policy'] and association['profiler_source_sha256'] == sha(ROOT / 'scripts/cnn_adoption_profile.py') and record['qualification']['passed'], 'Profile does not qualify exact target/policy/source')
    else: raise ValueError('Unknown backend record kind')
    return record


def construct(plan, nomination):
    require(not (DEST / 'evaluation/confirmation').exists() and not (DEST / 'profiling/confirmation').exists(), 'Confirmation output directory already exists; no retrospective freeze')
    run_ids = selection_contract(plan, nomination); sources = source_contract(nomination)
    records = nomination['backend_records']; require(set(records) == set(nomination['targets']), 'Every target requires explicit backend records')
    for key, nominated in nomination['targets'].items():
        identifier, fmt = key.rsplit(':', 1)
        resolved = resolve_target(plan, run_id=identifier if identifier in run_ids else None,
            reference_id=None if identifier in run_ids else identifier, numerical_format=fmt)
        require(resolved[1] == nominated, 'Actual completed target descriptor differs: ' + key)
        if identifier in run_ids:
            done = read(DEST / 'runs' / identifier / 'complete.json'); recipe = nomination['final_recipes'][identifier]
            require(done['epochs_completed'] == recipe['epochs'] and done['selected_epoch'] == recipe['epochs'], 'Final checkpoint did not complete exact fixed horizon')
            # Original training sources remain bound by completed association;
            # they need not equal current separately qualified scoring sources.
        if fmt == 'w8a8_export':
            from cnn_adoption_export import canonical_sha
            manifest_file = resolved[3]; manifest = read(manifest_file)
            payload = {k: v for k, v in manifest.items() if k != 'manifest_payload_sha256'}
            require(canonical_sha(payload) == manifest['manifest_payload_sha256'], 'Export payload identity differs')
            require(manifest['configuration'] == nominated['model_spec']['config'] and manifest['model_source_sha256'] == sources['scripts/cnn_models.py'], 'Export topology/source differs')
            for name, item in manifest['storage']['files'].items():
                require(Path(name).name == name and sha(manifest_file.parent / name) == item['sha256'] and (manifest_file.parent / name).stat().st_size == item['bytes'], 'Packed export bytes differ')
            fit_manifest = read(local(plan['prepared_root']) / 'fit/manifest.json')
            require(manifest['activation_calibration']['prepared_manifest'] == fit_manifest, 'Export scale calibration used different fitting inputs')
        require(records[key], 'Missing backend qualification')
        kinds = [reference['kind'] for reference in records[key]]
        require('development_evaluation' in kinds and (fmt != 'w8a8_export' or 'native_integer' in kinds), 'Exact development scoring/native arithmetic qualification required')
        for reference in records[key]: backend_record(reference, nominated, plan)
    prepared = local(plan['prepared_root']); require(prepared.resolve().is_relative_to((DEST / 'prepared').resolve()), 'Prepared data outside adoption revisions')
    expected = plan['data_contract']['prepared_membership']; membership = {}
    for role in ('benign_calibration', 'confirmation'):
        folder = prepared / role; verify_membership(folder, expected[role]); manifest = read(folder / 'manifest.json')
        require(manifest['role'] == role and manifest['artifact_sha256'] == expected[role]['artifact_sha256'], 'Reserved manifest role/artifact set differs')
        if role == 'benign_calibration': require(set(manifest['document_labels']) == {'0'}, 'Calibration metadata is not benign-only')
        raw = local(plan['raw_root']) / (role + '.jsonl')
        require(raw.resolve().is_relative_to(DEST.resolve()) and sha(raw) == expected[role]['raw_sha256'] == manifest['raw_sha256'], 'Reserved raw membership bytes differ')
        membership[role] = copy.deepcopy(expected[role])
    frozen = {'complete': True, 'nomination_before_confirmation': True, 'nomination': copy.deepcopy(nomination),
        'data_membership': membership, 'calibration_policy': plan['calibration_policy'], 'quality_gates': plan['quality_gates'],
        'numerical_formats': {key: target['numerical_format'] for key, target in nomination['targets'].items()},
        'priority_order': plan['priority_order'], 'execution_policy': plan['execution_policy'],
        'profiling_policy': plan['profiling_policy'], 'prepared_root': plan['prepared_root'], 'raw_root': plan['raw_root'],
        'source_sha256': sources, 'deployment_seed': 42, 'material_hardware_benefit': nomination['material_hardware_benefit'],
        'hardware_benefit_policy': plan['hardware_benefit'],
        'scope': 'Prospective explicit selection/artifact freeze only; reserved document and score JSONL were not decoded, no inference or new selection.'}
    for field in ('nomination', 'data_membership', 'calibration_policy', 'quality_gates', 'numerical_formats', 'priority_order', 'execution_policy', 'profiling_policy', 'material_hardware_benefit'):
        frozen[field + '_sha256'] = identity(frozen[field])
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--nomination', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); started = time.monotonic()
    output = local(args.output); require(output.resolve().is_relative_to(DEST.resolve()), 'Freeze output outside adoption results')
    require(output.suffix == '.json', 'Freeze output must be JSON')
    output.parent.mkdir(parents=True, exist_ok=True); require(not output.exists(), 'Freeze overwrite forbidden')
    frozen = construct(read(args.plan), read(args.nomination))
    frozen.update(command=sys.argv, plan_sha256=sha(args.plan), nomination_file_sha256=sha(args.nomination), wall_seconds=time.monotonic() - started)
    require(not (DEST / 'evaluation/confirmation').exists(), 'Confirmation appeared before freeze commit')
    with output.open('x', encoding='utf-8') as handle: handle.write(json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'confirmation_freeze': str(output.relative_to(ROOT)), 'confirmation_freeze_sha256': sha(output), 'targets': len(frozen['nomination']['targets']), 'wall_seconds': frozen['wall_seconds']}))


if __name__ == '__main__': main()
