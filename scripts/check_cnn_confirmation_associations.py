"""CPU saved-score reconciliation after all frozen CNN confirmations pass checks.

No model loading, inference, threshold fitting or protected data is performed.
Canonical counts rely on the completed root checker; fake-quantization changes
and authored diagnostic counts are independently reconstructed by document ID.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/cnn_study_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    values = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    indexed = {row['document_id']: row for row in values}
    if len(indexed) != len(values):
        raise ValueError(f'Duplicate document IDs in {path}')
    return values, indexed


def counts(values, threshold):
    result = dict(tp=0, fn=0, fp=0, tn=0)
    for row in values:
        prediction = row['risk_score'] > threshold
        key = ('tp' if prediction else 'fn') if row['label'] else ('fp' if prediction else 'tn')
        result[key] += 1
    return result


def compact(metric):
    return {key: metric[key] for key in ('threshold', 'tp', 'fn', 'fp', 'tn', 'recall', 'fpr', 'passes')}


def main():
    freeze_path = ROOT / 'configs/cnn_stage_c_v1.json'
    table_path = DEST / 'confirmation_table.json'
    check_path = DEST / 'saved_score_check_confirmation.json'
    freeze, table, checked = read(freeze_path), read(table_path), read(check_path)
    expected = {spec['run_id']: spec for spec in freeze['runs']}
    records = {record['run_id']: record for record in table}
    if len(records) != len(table) or set(records) != set(expected) or len(records) != 24:
        raise ValueError('Complete unique frozen 24-run confirmation table is required')
    checked_runs = {item['run_id']: item for item in checked['checks']}
    if not checked['passed'] or checked['completed_runs_checked'] != 24 or set(checked_runs) != set(expected):
        raise ValueError('Passing full saved-score check is required')
    freeze_hash = sha(freeze_path)
    sources = {str(path.relative_to(ROOT)): sha(path) for path in
               (freeze_path, table_path, check_path, ROOT / 'scripts/cnn_models.py', ROOT / 'scripts/cnn_evaluate.py',
                ROOT / 'scripts/check_cnn_scores.py', ROOT / 'scripts/cnn_analyze.py', Path(__file__))}
    reviewed = []
    quantization = []
    by_architecture = {}
    discrepancies = []
    identity_fields = ('document_id', 'source_group_id', 'label', 'text_sha256', 'window_count', 'content_length')
    for spec in freeze['runs']:
        run_id, config_id = spec['run_id'], spec['config_id']
        record = records[run_id]
        run_folder = DEST / 'runs' / run_id
        confirmation_folder = DEST / 'confirmation' / run_id
        metadata, done = read(run_folder / 'run.json'), read(run_folder / 'complete.json')
        assert record['complete'] and record['freeze_sha256'] == freeze_hash
        assert metadata['spec'] == spec and record['config'] == metadata['config'] == freeze['finalist_configs'][config_id]
        assert record['seed'] == spec['seed']
        assert record['checkpoint_sha256'] == done['checkpoint_sha256'] == checked_runs[run_id]['checkpoint_sha256']
        assert record == read(confirmation_folder / 'complete.json')
        expected_variants = {'fp32', 'fake_w8a8'} if config_id in freeze['quantization_config_ids'] else {'fp32'}
        assert set(record['variants']) == expected_variants
        entry = {'run_id': run_id, 'config_id': config_id, 'seed': spec['seed'], 'epochs': spec['epochs'],
                 'checkpoint_sha256': record['checkpoint_sha256'], 'representation': record['config']['representation'],
                 'evaluator_source_sha256': record['evaluator_source_sha256'],
                 'training_runner_source_sha256': record['training_runner_source_sha256'],
                 'cache_limit': record['cudnn_plan_cache_limit'], 'cudnn_v8_api_disabled': record['cudnn_v8_api_disabled'],
                 'canonical': {}, 'diagnostics': {}}
        for variant, value in record['variants'].items():
            entry['canonical'][variant] = compact(value['metrics']['0.01'])
            diagnostic_path = confirmation_folder / (variant + '_diagnostics.jsonl')
            diagnostics, _ = rows(diagnostic_path)
            assert len(diagnostics) == 96 and sum(row['label'] for row in diagnostics) == 64
            threshold = value['metrics']['0.01']['threshold']
            diagnostic_counts = counts(diagnostics, threshold)
            assert all(diagnostic_counts[key] == value['diagnostics'][key] for key in diagnostic_counts)
            categories = sorted({row['probe_category'] for row in diagnostics})
            entry['diagnostics'][variant] = {
                **diagnostic_counts, 'documents': len(diagnostics), 'attacks': 64, 'benign': 32,
                'by_category': {category: counts([row for row in diagnostics if row['probe_category'] == category], threshold)
                                for category in categories}}
            sources[str(diagnostic_path.relative_to(ROOT))] = sha(diagnostic_path)
        if 'fake_w8a8' in record['variants']:
            floating_path = confirmation_folder / 'fp32_development.jsonl'
            quantized_path = confirmation_folder / 'fake_w8a8_development.jsonl'
            floating, floating_by_id = rows(floating_path)
            quantized, quantized_by_id = rows(quantized_path)
            assert floating_by_id.keys() == quantized_by_id.keys()
            ft = record['variants']['fp32']['metrics']['0.01']['threshold']
            qt = record['variants']['fake_w8a8']['metrics']['0.01']['threshold']
            changes, transitions = [], Counter()
            for q in quantized:
                f = floating_by_id[q['document_id']]
                assert all(f.get(key) == q.get(key) for key in identity_fields)
                before, after = f['risk_score'] > ft, q['risk_score'] > qt
                if before != after:
                    changes.append(q['document_id'])
                    if q['label']:
                        transitions['tp_to_fn' if before else 'fn_to_tp'] += 1
                    else:
                        transitions['fp_to_tn' if before else 'tn_to_fp'] += 1
            saved_changes = record['variants']['fake_w8a8']['changed_decision_ids']
            matched = changes == saved_changes
            if not matched:
                discrepancies.append({'run_id': run_id, 'kind': 'quantized_changed_decision_ids_mismatch',
                                      'recomputed': changes, 'saved': saved_changes})
            quantization.append({'run_id': run_id, 'documents': len(quantized),
                                 'same_order': [row['document_id'] for row in floating] == [row['document_id'] for row in quantized],
                                 'own_fp32_threshold': ft, 'own_fake8_threshold': qt,
                                 'changed_decision_ids_match_exactly': matched,
                                 'changed_decision_ids': changes, 'transition_counts':
                                 {key: transitions[key] for key in ('tp_to_fn', 'fn_to_tp', 'fp_to_tn', 'tn_to_fp')},
                                 'mixed_parameter_scale_payload_bytes': record['variants']['fake_w8a8']['formats']['_analytical_storage']['mixed_parameter_scale_payload_bytes']})
            for path in (floating_path, quantized_path):
                sources[str(path.relative_to(ROOT))] = sha(path)
        reviewed.append(entry)
        by_architecture.setdefault(config_id, []).append(entry)
    assert len(quantization) == 9
    summary = {}
    for config_id, entries in by_architecture.items():
        assert sorted(entry['seed'] for entry in entries) == sorted(freeze['seeds'])
        primary = [entry['canonical']['fp32'] for entry in entries]
        summary[config_id] = {'seeds': [entry['seed'] for entry in entries],
                              'fp32_pass_count': sum(metric['passes'] for metric in primary),
                              'fp32_all_seeds_pass': all(metric['passes'] for metric in primary),
                              'fp32_fp_by_seed': {str(entry['seed']): entry['canonical']['fp32']['fp'] for entry in entries},
                              'fp32_fn_by_seed': {str(entry['seed']): entry['canonical']['fp32']['fn'] for entry in entries},
                              'diagnostic_fp32_tp_by_seed': {str(entry['seed']): entry['diagnostics']['fp32']['tp'] for entry in entries},
                              'diagnostic_fp32_fp_by_seed': {str(entry['seed']): entry['diagnostics']['fp32']['fp'] for entry in entries}}
        if 'fake_w8a8' in entries[0]['canonical']:
            summary[config_id]['fake8_pass_count'] = sum(entry['canonical']['fake_w8a8']['passes'] for entry in entries)
            summary[config_id]['fake8_fp_by_seed'] = {str(entry['seed']): entry['canonical']['fake_w8a8']['fp'] for entry in entries}
            summary[config_id]['fake8_fn_by_seed'] = {str(entry['seed']): entry['canonical']['fake_w8a8']['fn'] for entry in entries}
            summary[config_id]['diagnostic_fake8_tp_by_seed'] = {str(entry['seed']): entry['diagnostics']['fake_w8a8']['tp'] for entry in entries}
            summary[config_id]['diagnostic_fake8_fp_by_seed'] = {str(entry['seed']): entry['diagnostics']['fake_w8a8']['fp'] for entry in entries}
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Complete frozen saved-score association review; canonical metrics rely on the passing root checker. Independent document-ID quantization transitions and authored diagnostic counts. CPU stdlib only; no inference or protected data.',
              'passed': not discrepancies, 'completed_runs_reviewed': len(reviewed), 'quantized_variants_reviewed': len(quantization),
              'source_sha256': sources, 'discrepancies': discrepancies, 'architectures': summary,
              'runs': reviewed, 'quantization_transitions': quantization,
              'limitations': [
                  'Canonical is permitted post-search development data, not protected final test, final calibration or NotInject; document classification does not establish end-to-end task protection.',
                  'Own benign-calibrated strict score > threshold decisions are compared; raw-score displacement is not itself a decision loss. No threshold is changed here.',
                  'Authored 96 diagnostics (64 attack/32 benign) are fixed constructed stress cases; topic/template dependence and population generalization remain unresolved.',
                  'Three fixed seeds give observed stability only. Existing source-group bootstrap conditions on fitted thresholds and does not include calibration, selection or seed-population uncertainty; zero observed misses do not imply zero population risk.',
                  'Local/context versus matched residual replacement uses common 12 epochs, matched seeds/groups and near-matched costs, but width, norm inputs, residual locations and dropout placements differ; no pure mechanism isolation or superiority follows.',
                  'Transfer epochs differ among ordinary finalists; token and byte units/window multiplicities differ. Architecture comparisons include their frozen training recipes and representations.',
                  'Confirmation uses uniform frozen cache64/v8-enabled CUDA FP32. Inner DP03 resource recovery used a different backend; bounded numerical checks did not prove full trajectory equivalence.',
                  'Fake W8/input-A8 uses FP32 accumulations, remaining parameters/nonlinear operations; its payload estimate is analytical and excludes layout/activation buffers. No native INT8 latency/memory or FPGA physical result is measured.',
                  'Warm batch-one timing covers 12 deterministic extreme-length benign documents with 20 repetitions. It excludes process startup; separate cold-model-load time is not a complete cold-process latency distribution.',
                  'Saved window/document probabilities reproduce counts and transitions; this review does not replay full tensor logits, training, quantized states or all CPU/CUDA cases.'
              ]}
    output = DEST / 'final_confirmation_review.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(json.dumps({'passed': result['passed'], 'runs': len(reviewed), 'quantized_variants': len(quantization),
                      'discrepancies': discrepancies, 'architectures': summary, 'output': str(output)}))


if __name__ == '__main__':
    main()
