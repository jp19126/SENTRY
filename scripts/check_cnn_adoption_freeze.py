"""Synthetic CPU freeze associations; invalid UTF8 reserved rows prove no decode."""
from __future__ import annotations
import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import patch
import cnn_adoption_freeze as freeze
import cnn_adoption_evaluate as evaluator
from cnn_adoption_compare import _validate_freeze
from cnn_adoption_train import ROOT, DEST, identity, read, sha, write
from cnn_adoption_profile import policy_for


def rejects(call):
    try: call()
    except (ValueError, KeyError): return
    raise AssertionError('Corrupt prospective contract accepted')


def main():
    started = time.monotonic(); directory = DEST / 'freeze_guard_fixture'
    prepared = directory / 'prepared/token'; raw = directory / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    membership = {}
    roles = ('development_calibration', 'development_selection', 'benign_calibration', 'confirmation')
    for role in roles:
        folder = prepared / role; folder.mkdir(parents=True, exist_ok=True)
        for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl'):
            (folder / name).write_bytes(b'\xff\xfe opaque synthetic bytes, not parseable arrays/JSONL')
        files = {name: sha(folder / name) for name in ('ids.npy', 'masks.npy', 'starts.npy', 'documents.jsonl')}
        rawfile = raw / (role + '.jsonl'); rawfile.write_bytes(b'\xff never decode reserved rows')
        manifest = {'role': role, 'artifact_sha256': files, 'raw_sha256': sha(rawfile), 'document_labels': {'0': 2} if 'calibration' in role else {'0': 1, '1': 1}}
        write(folder / 'manifest.json', manifest)
        membership[role] = {'manifest_sha256': sha(folder / 'manifest.json'), 'artifact_sha256': files, 'raw_sha256': sha(rawfile)}
    original = read(ROOT / 'configs/cnn_adoption_v1.json')
    profile = policy_for(SimpleNamespace(sample_per_stratum=2, long_stress_per_label=2, warmups=3, repeats=20, cpu_threads=1, cold_documents=2))
    plan = {'prepared_root': str(prepared), 'raw_root': str(raw), 'data_contract': {'prepared_membership': membership},
        'calibration_policy': original['calibration_policy'], 'quality_gates': original['quality_gates'],
        'hardware_benefit': original['hardware_benefit'], 'execution_policy': original['execution_policy'],
        'priority_order': ['synthetic_cnn'], 'profiling_policy': profile,
        'material_hardware_benefit': {'criterion': 'synthetic, no measured benefit claim'}, 'runs': []}
    targets = {}; groups = {'synthetic_cnn': [], 'synthetic_bert': []}; recipes = {}; records = {}
    for candidate, kind in (('synthetic_cnn', 'cnn'), ('synthetic_bert', 'bert')):
        spec = {'kind': kind, 'initialization': 'scratch' if kind == 'cnn' else 'pretrained'}
        for seed in (42, 1337, 2026):
            run_id = candidate + '_' + str(seed); key = run_id + ':fp32'
            recipe = {'run_id': run_id, 'model_id': candidate, 'seed': seed, 'epochs': 3,
                'schedule_total_epochs': 12, 'monitor_development': False, 'objective': 'document_max_bce'}
            plan['runs'].append(recipe); recipes[run_id] = recipe; groups[candidate].append(key)
            targets[key] = {'model_spec': spec, 'checkpoint_identity': {'path': 'synthetic.pt', 'sha256': 'a' * 64}, 'numerical_format': 'fp32', 'export_manifest_sha256': None}
            folder = directory / 'runs' / run_id; folder.mkdir(parents=True, exist_ok=True)
            write(folder / 'complete.json', {'epochs_completed': 3, 'selected_epoch': 3})
    anchor = 'synthetic_historical:historical_w8_qat'
    targets[anchor] = {'model_spec': {'kind': 'historical_bert_w8', 'backend': 'native_int8'},
        'checkpoint_identity': {'path': 'synthetic historical', 'files_sha256': {'dummy': 'b' * 64}}, 'numerical_format': 'historical_w8_qat', 'export_manifest_sha256': None}
    sources = {name: sha(ROOT / name) for name in freeze.BASE_SOURCES | {'gate_quant.py', 'scripts/r1_bert_int8.py'}}
    for key, descriptor in targets.items():
        folder = directory / 'development_records' / key.replace(':', '_'); folder.mkdir(parents=True, exist_ok=True)
        file_hashes = {}
        for role in ('development_calibration', 'development_selection', 'known_diagnostics'):
            name = role + '_scores.jsonl'; (folder / name).write_bytes(b'\xff opaque saved score bytes')
            file_hashes[name] = sha(folder / name)
        association = {'stage': 'development', 'target': descriptor, 'batch_windows': 1,
            'calibration_policy': plan['calibration_policy'], 'quality_gates': plan['quality_gates'],
            'source_sha256': sources, 'device': 'cuda', 'torch': plan['execution_policy']['torch'], 'cache_limit': '64', 'cudnn_v8_api_disabled': '0',
            'data': {role: {'role': role, 'files': membership[role]['artifact_sha256']} for role in ('development_calibration', 'development_selection')}}
        association['data']['known_diagnostics'] = {'role': 'diagnostics', 'files': {}}
        write(folder / 'complete.json', {'complete': True, 'association': association,
            'association_sha256': identity(association), 'score_files_sha256': file_hashes})
        records[key] = [{'kind': 'development_evaluation', 'path': str(folder / 'complete.json'), 'sha256': sha(folder / 'complete.json')}]
    nomination = {'targets': targets, 'candidate_groups': groups, 'final_recipes': recipes, 'historical_anchor': anchor,
        'deployment_seed': 42, 'rationale': 'Synthetic schema association check; no real nomination', 'source_sha256': sources, 'backend_records': records}
    for field in ('priority_order', 'profiling_policy', 'material_hardware_benefit', 'calibration_policy', 'quality_gates'):
        nomination[field] = plan[field]
    def resolve(plan, run_id=None, reference_id=None, numerical_format=None):
        identifier = run_id or reference_id
        return identifier, targets[identifier + ':' + numerical_format], None, None, None
    with patch.object(freeze, 'DEST', directory), patch.object(freeze, 'resolve_target', side_effect=resolve):
        frozen = freeze.construct(plan, nomination)
        for mutation in ('seed', 'priority', 'target', 'source', 'horizon', 'backend'):
            altered = copy.deepcopy(nomination)
            if mutation == 'seed': altered['final_recipes']['synthetic_cnn_2026']['seed'] = 42
            elif mutation == 'priority': altered['priority_order'] = ['synthetic_bert']
            elif mutation == 'target': altered['targets']['synthetic_cnn_42:fp32']['checkpoint_identity']['sha256'] = 'bad'
            elif mutation == 'source': altered['source_sha256']['scripts/cnn_models.py'] = 'bad'
            elif mutation == 'horizon': altered['final_recipes']['synthetic_cnn_42']['schedule_total_epochs'] = 1
            else: altered['backend_records']['synthetic_cnn_42:fp32'] = []
            rejects(lambda: freeze.construct(plan, altered))
        conditional = copy.deepcopy(nomination)
        conditional['final_recipes']['synthetic_cnn_42']['embedding_initialization'] = {'method': 'pca'}
        rejects(lambda: freeze.source_contract(conditional))
        confirmation_manifest = prepared / 'confirmation/manifest.json'; valid = read(confirmation_manifest)
        write(confirmation_manifest, {**valid, 'role': 'fit'})
        changed_plan = copy.deepcopy(plan); changed_plan['data_contract']['prepared_membership']['confirmation']['manifest_sha256'] = sha(confirmation_manifest)
        rejects(lambda: freeze.construct(changed_plan, nomination))
        write(confirmation_manifest, valid)
    # Actual consumer guards read the synthetic freeze, but no reserved rows.
    output = directory / 'synthetic_freeze.json'; write(output, frozen)
    consumer_plan = {**plan, 'confirmation_freeze': str(output), 'confirmation_freeze_sha256': sha(output)}
    assert evaluator.confirmation_authorization(consumer_plan, 'synthetic_cnn_42:fp32', targets['synthetic_cnn_42:fp32']) == frozen
    assert _validate_freeze(consumer_plan, ROOT)[0] == frozen
    for field in ('nomination', 'data_membership', 'source_sha256', 'priority_order'):
        corrupted = copy.deepcopy(frozen)
        if field == 'nomination': corrupted[field]['targets']['synthetic_cnn_42:fp32']['checkpoint_identity']['sha256'] = 'bad'
        elif field == 'data_membership': corrupted[field]['confirmation']['artifact_sha256']['ids.npy'] = 'bad'
        elif field == 'source_sha256': corrupted[field]['scripts/cnn_models.py'] = 'bad'
        else: corrupted[field] = ['synthetic_bert']
        write(output, corrupted); consumer_plan['confirmation_freeze_sha256'] = sha(output)
        rejects(lambda: evaluator.confirmation_authorization(consumer_plan, 'synthetic_cnn_42:fp32', targets['synthetic_cnn_42:fp32']))
    write(output, frozen)
    import torch
    assert not torch.cuda.is_initialized()
    report = {'complete': True, 'scope': 'Synthetic records and mocked completed CP resolution only; invalid UTF8 reserved/score bytes were hashed without decoding; no real freeze, config edit, reserved study rows, inference or GPU.',
        'checks': ['fixed three-seed/horizon/priority/target/backend association', 'actual evaluator and comparison freeze compatibility',
            'conditional PCA source closure', 'source/role/order/target/horizon corruption rejection', 'opaque membership and score hashing'],
        'targets': len(targets), 'cuda_initialized': False, 'wall_seconds': time.monotonic() - started,
        'source_sha256': {name: sha(ROOT / 'scripts' / name) for name in ('cnn_adoption_freeze.py', 'check_cnn_adoption_freeze.py')},
        'command': __import__('sys').argv}
    write(DEST / 'freeze_contract_checks.json', report); print(json.dumps(report))


if __name__ == '__main__': main()
