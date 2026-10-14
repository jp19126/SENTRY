"""Read-only, CPU-only final error reconciliation after complete confirmation checks.

No model imports, inference, augmentation, threshold tuning, or protected files.
The only write is a fresh result JSON. Prerequisites are checked before opening
raw documents or score rows; partial confirmation cannot produce a final report.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from gate_eval import evaluate_documents, fit_threshold
from cnn_artifacts import validate_prepared
from cnn_confirm_queue import saved_confirmation

DEST = ROOT / 'results/cnn_study_v1'
SPLITS = ('search_temporary_threshold', 'search_candidate_scoring', 'diagnostics')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prerequisites(frozen, confirmations, score_check):
    expected = {s['run_id']: s for s in frozen['runs']}
    require(len(expected) == len(frozen['runs']), 'Duplicate frozen run IDs')
    wanted = {(cid, seed) for cid in frozen['architecture_ids'] for seed in frozen['seeds']}
    require(bool(wanted) and frozen.get('stage') == 'stage_c' and set(frozen['seeds']) == {42,1337,2026}, 'Expected nonempty frozen three-seed Stage C study')
    require({(s['config_id'], s['seed']) for s in expected.values()} == wanted and len(expected) == len(wanted), 'Incomplete frozen architecture/seed specification')
    require(len(confirmations) == len(expected) and {r['run_id'] for r in confirmations} == set(expected), 'All frozen confirmations must exist before this helper runs')
    require(all(r.get('complete') is True for r in confirmations), 'Confirmation is incomplete')
    require(score_check.get('passed') is True and score_check.get('stage') == 'confirmation', 'Complete independent confirmation saved-score check required')
    checks = score_check.get('checks', [])
    require(score_check.get('completed_runs_checked') == len(expected) and len(checks) == len(expected) and {r['run_id'] for r in checks} == set(expected) and all(r.get('passed') is True for r in checks), 'Saved-score check does not cover every frozen run')
    check_hashes = {r['run_id']: r['checkpoint_sha256'] for r in checks}
    require(all(check_hashes[r['run_id']] == r['checkpoint_sha256'] for r in confirmations), 'Checked checkpoint identities differ from confirmation')
    return expected


def index_unique(documents):
    mapping = {d['document_id']: d for d in documents}
    require(len(mapping) == len(documents), 'Duplicate document IDs')
    return mapping


def validate_rows(scored, prepared, raw):
    require(len(scored) == len(prepared), 'Scored document count differs')
    for actual, meta in zip(scored, prepared):
        for key in ('document_id', 'source_group_id', 'label', 'text_sha256'):
            require(actual[key] == meta[key], 'Score/prepared identity differs: ' + key)
        original = raw[actual['document_id']]
        raw_hash = hashlib.sha256(original['text'].encode('utf-8')).hexdigest()
        require(raw_hash == meta['text_sha256'] and original['label'] == meta['label'] and original['source_group_id'] == meta['source_group_id'], 'Raw/prepared identity differs')
        require(len(original['text'].encode('utf-8')) == meta['raw_utf8_bytes'], 'Raw byte length differs')
        scores = actual['window_scores']
        require(len(scores) == meta['window_count'] and bool(scores) and all(math.isfinite(s) and 0 <= s <= 1 for s in scores), 'Window counts/scores invalid')
        require(actual['risk_score'] == max(scores), 'Document maximum differs')


def errors(scored, threshold):
    return {'fp': {d['document_id'] for d in scored if d['label'] == 0 and d['risk_score'] > threshold},
            'fn': {d['document_id'] for d in scored if d['label'] == 1 and d['risk_score'] <= threshold}}


def metrics(scored, threshold):
    return evaluate_documents([d['risk_score'] for d in scored], [d['label'] for d in scored], threshold)


def strata(scored, threshold):
    result = {}
    for key in ('source_dataset', 'attack_type', 'attack_template_id', 'injection_position', 'probe_category', 'probe_topic'):
        result[key] = {str(value): metrics([d for d in scored if d.get(key) == value], threshold)
                       for value in sorted({d.get(key) for d in scored}, key=lambda x: str(x))}
    result['common_raw_utf8_length'] = {}
    for low, high in ((0,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,10**9)):
        group = [d for d in scored if low <= d['raw_utf8_bytes'] < high]
        result['common_raw_utf8_length'][f'{low}-{high-1}'] = metrics(group, threshold)
    result['representation_window_count'] = {}
    for low, high in ((1,2),(2,4),(4,8),(8,10**9)):
        result['representation_window_count'][f'{low}-{high-1}'] = metrics([d for d in scored if low <= d['window_count'] < high], threshold)
    return result


def overlap(named_sets):
    names = sorted(named_sets)
    counts = Counter(identity for values in named_sets.values() for identity in values)
    return {'run_ids': names, 'union_ids': sorted(counts), 'intersection_ids': sorted(set.intersection(*(named_sets[n] for n in names))) if names else [],
            'incidents': sum(counts.values()), 'membership_counts': dict(sorted(counts.items())),
            'membership_histogram': dict(Counter(str(n) for n in counts.values())),
            'pairwise': [{'left': a, 'right': b, 'intersection': len(named_sets[a] & named_sets[b]),
                          'union': len(named_sets[a] | named_sets[b]), 'intersection_ids': sorted(named_sets[a] & named_sets[b])}
                         for a, b in itertools.combinations(names, 2)]}


def description(doc):
    """Anonymous rule-based descriptions; never return email bodies or excerpts."""
    category = doc.get('probe_category')
    if category:
        return {'type': category, 'paraphrase': 'Frozen authored ' + category.replace('_', ' ') + ' recipe over a topic background; constructed role label, not a target-model outcome.'}
    if doc['label']:
        return {'type': 'constructed_insertion', 'paraphrase': 'External content contains a labeled inserted instruction; attack family and insertion position are reported separately.'}
    text = doc['text'].lower()
    if re.search(r'newsletter|news service|news release', text):
        kind, phrase = 'newsletter_or_reported_news', 'Business/news material reports events and may directly address the reader.'
    elif re.search(r'contract|agreement|legal|negotiat', text):
        kind, phrase = 'contract_or_legal_correspondence', 'Business correspondence discusses contractual or legal matters and requests follow-up.'
    elif re.search(r'meeting|invitation|attend|schedule', text):
        kind, phrase = 'coordination_or_invitation', 'A coordination message discusses attendance, scheduling or requested preparations.'
    elif re.search(r'server|network|software|printer', text):
        kind, phrase = 'technical_or_service_correspondence', 'Technical correspondence reports service, system or operational matters.'
    elif re.search(r'please|kindly|could you|can you', text):
        kind, phrase = 'ordinary_request_correspondence', 'Ordinary correspondence contains a direct request to its recipient.'
    else:
        kind, phrase = 'other_benign_correspondence', 'Canonical benign correspondence contains reported information or interpersonal communication.'
    return {'type': kind, 'paraphrase': phrase, 'quoted_format_cue': bool(re.search(r'(^|\n)>|"[^"\n]{3,}"', doc['text']))}


def quant_changes(floating, quantized, ft, qt):
    fmap, qmap = index_unique(floating), index_unique(quantized)
    require(set(fmap) == set(qmap), 'Numerical variants have different documents')
    changes = []
    for identity in fmap:
        f, q = fmap[identity], qmap[identity]
        before, after = f['risk_score'] > ft, q['risk_score'] > qt
        if before != after:
            kind = ('benign_new_false_positive' if after else 'benign_false_positive_recovered') if f['label'] == 0 else ('attack_miss_recovered' if after else 'attack_new_miss')
            changes.append({'document_id': identity, 'label': f['label'], 'transition': kind, 'fp32_score': f['risk_score'],
                            'fake_quantized_score': q['risk_score'], 'score_delta': q['risk_score']-f['risk_score'],
                            'before': before, 'after': after, 'source_group_id': f['source_group_id'],
                            'raw_utf8_bytes': f['raw_utf8_bytes'], 'probe_category': f.get('probe_category')})
    return {'floating_threshold': ft, 'quantized_own_threshold': qt, 'transition_counts': dict(Counter(x['transition'] for x in changes)), 'changes': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', default='configs/cnn_stage_c_v1.json')
    parser.add_argument('--score-check', default='results/cnn_study_v1/saved_score_check_confirmation.json')
    parser.add_argument('--output', default='results/cnn_study_v1/final_error_inspection.json')
    parser.add_argument('--max-examples', type=int, default=12)
    args = parser.parse_args()
    started = time.monotonic()
    output = (ROOT/args.output).resolve()
    require(output.is_relative_to(DEST.resolve()) and not output.exists(), 'Use a fresh output under the CNN study')
    require(0 <= args.max_examples <= 24, 'Bound examples to at most 24')
    freeze_path, check_path = (ROOT/args.freeze).resolve(), (ROOT/args.score_check).resolve()
    require(freeze_path.is_relative_to((ROOT/'configs').resolve()) and check_path.is_relative_to(DEST.resolve()), 'Only study configuration/check artifacts may be read')
    frozen, confirmations, checked = read(freeze_path), read(DEST/'confirmation_table.json'), read(check_path)
    expected = prerequisites(frozen, confirmations, checked)
    freeze_hash = digest(freeze_path)
    # All confirmation prerequisites are satisfied before any score/raw text read.
    require(digest(ROOT/'scripts/cnn_models.py') == frozen['model_source_sha256'], 'Frozen model source changed')
    protocol = read(DEST/'prepared/data_protocol.json')
    require(digest(DEST/'prepared/data_protocol.json') == frozen['data_protocol_sha256'], 'Frozen data protocol changed')
    require(digest(DEST/'prepared/diagnostics_raw.jsonl') == frozen['diagnostics_sha256'], 'Frozen diagnostic recipes changed')
    require(digest(ROOT/'data/prepared/search_candidate_scoring.jsonl') == frozen['raw_timing_source_sha256'], 'Frozen raw development source changed')
    for result in confirmations:
        saved = saved_confirmation(expected[result['run_id']], frozen, freeze_hash)
        require(saved == result, 'Confirmation table differs from its identity-checked saved record')
    prepared, artifact_checks = {}, {}
    for rep in {c['representation'] for c in frozen['finalist_configs'].values()}:
        require(digest(DEST/'prepared'/rep/'summary.json') == frozen['prepared_representation_sha256'][rep], 'Prepared representation changed')
        artifact_checks[rep] = validate_prepared(rep, SPLITS)
        prepared[rep] = {s: rows(DEST/'prepared'/rep/s/'documents.jsonl') for s in SPLITS}
    raw = {}
    for split in SPLITS:
        path = DEST/'prepared/diagnostics_raw.jsonl' if split == 'diagnostics' else ROOT/'data/prepared'/(split+'.jsonl')
        if split != 'diagnostics':
            require(digest(path) == protocol['source_files'][split]['sha256'], 'Canonical search source changed')
        raw[split] = index_unique(rows(path))
    require(not ({d['source_group_id'] for d in raw[SPLITS[0]].values()} & {d['source_group_id'] for d in raw[SPLITS[1]].values()}), 'Calibration/scoring source overlap')
    summaries, sets, quantization, examples = {}, defaultdict(dict), {}, defaultdict(list)
    for result in confirmations:
        rid, spec = result['run_id'], expected[result['run_id']]
        rep = result['config']['representation']
        variants = {}
        loaded = {}
        for variant, evidence in result['variants'].items():
            folder = DEST/'confirmation'/rid
            cal, sel, diag = [rows(folder/(variant+'_'+name+'.jsonl')) for name in ('calibration','development','diagnostics')]
            for scored, split in zip((cal,sel,diag), SPLITS):
                validate_rows(scored, prepared[rep][split], raw[split])
            threshold = fit_threshold([d['risk_score'] for d in cal if d['label'] == 0], .01)
            primary = metrics(sel, threshold)
            for key, value in primary.items():
                require(evidence['metrics']['0.01'][key] == value, 'Primary threshold/counts differ')
            dm = metrics(diag, threshold)
            require(dm == evidence['diagnostics'], 'Diagnostic counts differ')
            variants[variant] = {'primary': primary, 'development_strata': strata(sel,threshold), 'diagnostics': dm,
                                 'diagnostic_strata': strata(diag,threshold), 'development_errors': {}, 'diagnostic_errors': {}}
            loaded[variant] = (sel,diag,threshold)
            for name, scored, split in (('development',sel,SPLITS[1]),('diagnostic',diag,'diagnostics')):
                err = errors(scored, threshold)
                variants[variant][name+'_errors'] = {k: sorted(v) for k,v in err.items()}
                for kind, identities in err.items():
                    sets[(name,variant,kind)][rid] = identities
                    for identity in identities:
                        examples[(split,identity)].append({'run_id':rid,'seed':result['seed'],'variant':variant,'kind':kind})
        if 'fake_w8a8' in loaded:
            fsel,fdiag,ft = loaded['fp32'];qsel,qdiag,qt = loaded['fake_w8a8']
            quantization[rid] = {'development':quant_changes(fsel,qsel,ft,qt), 'diagnostics':quant_changes(fdiag,qdiag,ft,qt),
                                'interpretation':'Each numerical variant uses its own benign-only calibration threshold; decision changes include arithmetic and threshold movement. Fake quantization remains FP32 execution.'}
        summaries[rid] = {'config_id':result['config']['id'],'seed':result['seed'],'representation':rep,'checkpoint_sha256':result['checkpoint_sha256'],'variants':variants}
    unions = {':'.join(key): overlap(values) for key, values in sets.items()}
    seed_overlap = {}
    for cid in frozen['architecture_ids']:
        run_ids = [s['run_id'] for s in frozen['runs'] if s['config_id'] == cid]
        seed_overlap[cid] = {':'.join(key): overlap({rid: values[rid] for rid in run_ids if rid in values}) for key,values in sets.items() if any(rid in values for rid in run_ids)}
    selected_examples = []
    order = sorted(examples, key=lambda key:(-len(examples[key]), -len(raw[key[0]][key[1]]['text'].encode('utf-8')),key))
    # Reserve diagnostic examples so quotation/imperative/perturbation failures remain visible.
    diagnostics = [key for key in order if key[0] == 'diagnostics']
    canonical = [key for key in order if key[0] != 'diagnostics']
    probe_choices, seen_categories = [], set()
    for key in diagnostics:
        category = raw[key[0]][key[1]].get('probe_category')
        if category not in seen_categories and len(probe_choices) < args.max_examples//2:
            probe_choices.append(key); seen_categories.add(category)
    chosen = probe_choices + canonical[:args.max_examples-len(probe_choices)]
    if len(chosen) < args.max_examples:
        chosen += [key for key in order if key not in chosen][:args.max_examples-len(chosen)]
    for split, identity in chosen:
        d = raw[split][identity]
        selected_examples.append({'document_id':identity,'source_group_id':d['source_group_id'],'source_dataset':d['source_dataset'],
            'label':d['label'],'raw_utf8_bytes':len(d['text'].encode('utf-8')),'text_sha256':hashlib.sha256(d['text'].encode('utf-8')).hexdigest(),
            'attack_type':d.get('attack_type'),'attack_template_id':d.get('attack_template_id'),'injection_position':d.get('injection_position'),
            'description':description(d),'error_incidents':examples[(split,identity)]})
    report = {'complete':True,'observed_utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':freeze_hash,
        'independent_saved_score_check_sha256':digest(check_path),'helper_source_sha256':digest(Path(__file__)),
        'confirmation_table_sha256':digest(DEST/'confirmation_table.json'),'prepared_file_checks':artifact_checks,
        'runs':summaries,'error_unions':unions,'within_architecture_seed_overlaps':seed_overlap,'quantization_decision_changes':quantization,
        'illustrative_anonymous_examples':selected_examples,'description_method':'Rule-based anonymous cue descriptions, not exhaustive manual content annotation or evidence of detector causes; no raw excerpts are emitted.',
        'evidence_boundary':'Historically exposed post-search development sources; authored diagnostic topic/template variants, not independent natural/family tests. Own-calibration primary operating points; FP/FN overlap is descriptive and does not justify ensemble tuning. Source/template dependence and finite seeds prevent universal robustness/causal claims. No target-model outcome or FPGA advantage is measured.',
        'no_augmentation_or_tuning':True,'inference_calls':0,'protected_partitions_opened':False,'wall_seconds':time.monotonic()-started,'command':sys.argv}
    require(not output.exists(), 'Output appeared concurrently; preserve it')
    with output.open('x', encoding='utf-8') as f:
        f.write(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'complete':True,'runs':len(summaries),'output':str(output),'wall_seconds':report['wall_seconds']}))


if __name__ == '__main__':
    main()
