"""Bounded A/B campaign using the existing evolutionary adapter.

Default is a read-only prerequisite preview. --execute resumes committed work;
score/numerical approvals and complete common costs must already be qualified.
C requires an evidence-based method decision and is deliberately not defined here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import load_config, window_document
from search_precision import (CostTable, EvolutionAdapter, SCHEME, SPLITS,
                              load_score, precision_tuple, read_json,
                              readiness, recompute_metrics, resolved, rows,
                              validate_score_identity)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def relative(path):
    return str(path.relative_to(ROOT)).replace('\\', '/')


def map_name(precision, config):
    return ''.join(str(bit) for bit in precision_tuple(precision, config))


def tuple_tree(value):
    return tuple(tuple_tree(item) for item in value) if isinstance(value, list) else value


def pause_requested(destination):
    paths = [destination / 'pause.request'] + [ROOT / p for p in
             ('results/goal2/pause.request', 'results/goal3/pause.request')]
    return any(path.exists() for path in paths)


def paused(destination):
    if pause_requested(destination):
        raise RuntimeError('Pause marker present; campaign state is preserved.')


def protocol_snapshot(config):
    # Ordinary protocol/settings association; execution usage observations are mutable.
    return {key: deepcopy(config[key]) for key in
            ('seed', 'data', 'windowing', 'evaluation', 'training', 'quantization',
             'methods', 'hardware', 'measurement')}


def new_arm():
    return dict(initialized=False, generations=0, hardware_lookups=0,
                duplicate_only_batches=0, rng_state=None, consulted=[], pending=[],
                frozen_qat_shortlist=None, post_qat=None, active_precision_tuple=None)


def snapshot_adapter(adapter, state):
    state.update(initialized=adapter.initialized, generations=adapter.generations,
                 hardware_lookups=adapter.hardware_lookups,
                 duplicate_only_batches=adapter.duplicate_only_batches,
                 rng_state=adapter.rng.getstate(),
                 consulted=[dict(precision_tuple=key, candidate=value)
                            for key, value in adapter.consulted.items()],
                 frozen_qat_shortlist=adapter._qat_shortlist)


def restore_adapter(arm, state, config, costs, floating):
    adapter = EvolutionAdapter(arm, config, costs, floating['recall'],
                               floating_reference_identity=floating['identity'])
    for field in ('initialized', 'generations', 'hardware_lookups', 'duplicate_only_batches'):
        setattr(adapter, field, state[field])
    if state['rng_state'] is not None:
        adapter.rng.setstate(tuple_tree(state['rng_state']))
    for item in state['consulted']:
        candidate = item['candidate']
        key = precision_tuple(candidate['precision_map'], config)
        if tuple(item['precision_tuple']) != key or key in adapter.consulted:
            raise ValueError('Saved arm contains an inconsistent or duplicate logical map.')
        before = costs.lookups
        expected_profiles = costs.feasible_profiles(candidate['precision_map'])
        costs.lookups = before
        if expected_profiles != candidate['profiles'] or candidate['physical_rejection'] != (not expected_profiles):
            raise ValueError('Resumed physical feasibility differs from the common cost table.')
        score = candidate['score']
        if bool(expected_profiles) != (score is not None):
            raise ValueError('Resumed physical map lacks its required complete-network score.')
        if score is not None:
            fresh = load_score(resolved(score['source']), config, floating['recall'])
            if (fresh['precision_tuple'] != key or fresh['identity'] != score['identity']
                    or fresh['protection'] != score['protection']
                    or fresh['ordinary_0_5'] != score['ordinary_0_5']):
                raise ValueError('Resumed campaign score differs from its saved association.')
        adapter.consulted[key] = candidate
    if len(adapter.consulted) > config['methods']['max_unique_maps_per_method']:
        raise ValueError('Saved arm exceeds its logical-map allowance.')
    pending = [precision_tuple(value, config) for value in state['pending']]
    if len(pending) != len(set(pending)) or any(key in adapter.consulted for key in pending):
        raise ValueError('Saved pending batch repeats an already consulted map.')
    active = state.get('active_precision_tuple')
    if active is not None and (not pending or tuple(active) != pending[0]):
        raise ValueError('In-progress consultation differs from the first pending map.')
    if len(adapter.consulted) + len(pending) > config['methods']['max_unique_maps_per_method']:
        raise ValueError('Pending batch would exceed the logical-map allowance.')
    if state['frozen_qat_shortlist'] is not None:
        adapter._qat_shortlist = tuple(tuple(key) for key in state['frozen_qat_shortlist'])
        if (state['pending'] or len(adapter._qat_shortlist) > 2
                or len(adapter._qat_shortlist) != len(set(adapter._qat_shortlist))
                or any(key not in adapter.consulted for key in adapter._qat_shortlist)
                or len(adapter.consulted) != config['methods']['max_unique_maps_per_method']):
            raise ValueError('Frozen QAT shortlist is inconsistent with completed search work.')
    return adapter


def validate_bounds(config):
    methods = config['methods']
    limit = methods['max_unique_maps_per_method']
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('Logical allowance must be a positive integer at most 64 per arm.')
    if methods['qat_epochs'] != 1 or not 0 <= methods['max_shortlist_per_method'] <= 2:
        raise ValueError('Use the common one-epoch QAT allowance and at most two maps per arm.')


def score_work(folder):
    if (folder / 'progress.json').is_file():
        progress = read_json(folder / 'progress.json')
        return dict(committed_documents=sum(s['next_document'] for s in progress['splits'].values()),
                    committed_scoring_seconds=progress['scoring_seconds'])
    return dict(committed_documents=0, committed_scoring_seconds=0.0)



def batch_identity(config, precision, checkpoint):
    """B1 identity without a model import; actual backend version stays explicit."""
    from importlib.metadata import version
    run = read_json(ROOT / 'results/goal3/run.json')
    checkpoint = None if checkpoint is None else relative(resolved(checkpoint))
    identity = dict(checkpoint=run['selected_checkpoint'], qat=bool(checkpoint),
        windowing=config['windowing'], evaluation=config['evaluation'], data=config['data'],
        arithmetic_identity='bert_mini_ordered_fp32_frozen_quant_v1',
        execution_identity=f'torch_{version("torch")}__cpu_eager_fp32_threads4_tf32off_autocastoff_flushdenormalfalse',
        scheme_id=SCHEME, precision_map=precision, calibration_file=run['calibration_file'],
        frozen_scale_policy={k: run['format'][k] for k in ('weight_scale', 'activation_scale')})
    if checkpoint:
        identity['scored_checkpoint'] = checkpoint
    return dict(identity, precision_source='qat' if checkpoint else 'ptq', qat_checkpoint=checkpoint,
                storage_contract='decoded_signed_i8_codes_plus_real_packed_w4w8_v1')


def batch_proof_plan(config, precision=None, checkpoint=None, *, floating=False):
    """Read-only exact-proof reuse/preparation plan; no Torch or output writes."""
    from score_qualified_development import scoring_batch_policy
    size = config['methods'].get('ordered_scoring_batch_size', 1)
    if type(size) is not int or size not in (1, 8, 32):
        raise ValueError('Ordered campaign scoring batch size must be 1, 8 or 32.')
    if size == 1:
        return dict(batch_size=1, preparation_required=False, proof=None)
    bound = dict(config, _ordered_scoring_binding=dict(precision_map=precision, qat_checkpoint=checkpoint))
    canonical = floating or (set(precision.values()) == {8} and checkpoint is not None
                             and relative(resolved(checkpoint)) == 'checkpoints/quantized/qat_w8_a8')
    if canonical:
        _, proof = scoring_batch_policy(bound, floating=floating)
        return dict(batch_size=size, preparation_required=False, proof=relative(proof), canonical=True)
    from export_precision_qualification import BASE, validate_export, batching_state
    precision_tuple(precision, config)
    identity = batch_identity(config, precision, checkpoint)
    matches = [p.parent for p in sorted(BASE.glob('*/export.json'))
               if read_json(p).get('identity') == identity]
    if len(matches) > 1:
        raise ValueError('Multiple exact own-map exports exist; inspect them rather than creating another proof.')
    name = 'campaign_' + identity['precision_source'] + '_p' + map_name(precision, config)
    out = matches[0] if matches else BASE / name
    if out.exists() and not matches:
        raise ValueError('Deterministic own-map proof folder already contains different or uncommitted evidence.')
    count, positions, batches, seconds = 0, {8: 0, 32: 0}, 0, 0.0
    if matches:
        exports = validate_export(out)
        if exports.get('score_acceptance') is not None:
            raise ValueError('Own-map proof must not manufacture score approval.')
        if exports['identity'] != identity:
            raise ValueError('Own-map export identity changed during validation.')
        count = len(exports['completed_windows'])
        if count == 66 and not exports['complete']:
            raise ValueError('All66 B1 outputs exist but final commit is absent; inspect before automatic preparation.')
        committed = {(v['case_index'], v['window_index']) for v in exports['completed_windows']}
        for case in exports['selection']['cases']:
            for wi in range(case['window_count']):
                if (out / f"case{case['case_index']:02d}" / f'window{wi:03d}').exists() and (case['case_index'], wi) not in committed:
                    raise ValueError('Uncommitted B1 output requires inspection; automatic preparation cannot overwrite it.')
        if count == 66:
            _, _, evidence, _, positions = batching_state(out, exports)
            if evidence.get('active') is not None:
                raise ValueError('Uncommitted active batch requires inspection before a new proof forward.')
            if positions[size] < 66 and (out / 'batching' / f'b{size}' / f'batch{positions[size]//size:03d}').exists():
                raise ValueError('Uncommitted next batch output requires inspection before a new proof forward.')
            for record in evidence['batches']:
                if record['configured_batch_size'] == size:
                    batches += 1
                    seconds += record['forward_elapsed_seconds']
    ready = count == 66 and positions[size] == 66
    if ready:
        _, proof = scoring_batch_policy(bound)
        if proof.parent.parent != out:
            raise ValueError('Scoring proof differs from the exact export selected for reuse.')
    return dict(batch_size=size, preparation_required=not ready, proof=relative(out / 'batching/qualification.json'),
                output=relative(out), name=out.name, identity=identity, canonical=False,
                committed_b1_windows=count, committed_batch_windows=positions[size], committed_batches=batches,
                committed_instrumented_batch_forward_seconds=seconds,
                maximum_new_b1_windows=66-count, maximum_new_batches=(66-positions[size]+size-1)//size)


def proof_observation(plan):
    """Observe committed genuine work, including after an interrupted child."""
    out = resolved(plan['output'])
    export = read_json(out / 'export.json') if (out / 'export.json').is_file() else {}
    evidence = read_json(out / 'batching/qualification.json') if (out / 'batching/qualification.json').is_file() else {}
    batches = [v for v in evidence.get('batches', []) if v['configured_batch_size'] == plan['batch_size']]
    return dict(committed_b1_windows=len(export.get('completed_windows', [])),
                committed_batches=len(batches), committed_batch_windows=sum(v['actual_batch_size'] for v in batches),
                committed_instrumented_batch_forward_seconds=sum(v['forward_elapsed_seconds'] for v in batches))


def dispatch_proof(command, log, destination, task, state, state_path):
    """One bounded existing-exporter child; stop only that assigned process group."""
    if os.name != 'nt':
        raise RuntimeError('New own-map proof forwards require the established native Windows Python backend.')
    paused(destination)
    dispatch = dict(command=command, log=relative(log), state='dispatching',
                    started_utc=datetime.now(timezone.utc).isoformat(), timeout_seconds=3600)
    task['dispatches'].append(dispatch)
    save(state_path, state)
    started = time.monotonic()
    child = None
    try:
        with log.open('x', encoding='utf8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            dispatch.update(pid=child.pid, state='running')
            save(state_path, state)
            while child.poll() is None:
                paused(destination)
                if time.monotonic() - started > dispatch['timeout_seconds']:
                    raise RuntimeError('Bounded proof child timeout; preserve actual partial captures for inspection.')
                time.sleep(0.25)
            dispatch['returncode'] = child.returncode
            if child.returncode != 0:
                raise RuntimeError('Own-map proof exporter stopped; inspect its preserved actual output and log.')
            dispatch['state'] = 'returned'
    except BaseException as error:
        if child is not None and child.poll() is None:
            # Native exporter launches no workers; tree termination is confined to this new child.
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, timeout=30)
            child.wait(timeout=30)
        dispatch.update(state='stopped', error=str(error))
        raise
    finally:
        dispatch['elapsed_seconds'] = time.monotonic() - started
        save(state_path, state)


def prepare_batch_proof(config, state, state_path, precision=None, checkpoint=None, *, floating=False):
    plan = batch_proof_plan(config, precision, checkpoint, floating=floating)
    if plan['batch_size'] == 1 or plan.get('canonical'):
        return plan
    label = plan['output']
    baseline = {k: plan[k] for k in ('committed_b1_windows', 'committed_batches', 'committed_batch_windows',
                                    'committed_instrumented_batch_forward_seconds')}
    task = state.setdefault('batch_proof_work', {}).setdefault(label,
            dict(identity=plan['identity'], batch_size=plan['batch_size'], baseline=baseline, dispatches=[]))
    if task['identity'] != plan['identity'] or task['batch_size'] != plan['batch_size']:
        raise ValueError('Campaign proof work changed map/checkpoint/backend or selected batch size.')
    task.update(state='reused' if not plan['preparation_required'] else 'preparing', observed=baseline)
    save(state_path, state)
    if not plan['preparation_required']:
        return plan
    from search_precision import approved_score_identity
    approved = approved_score_identity(config)
    actual = dict(plan['identity'], execution_identity=plan['identity']['execution_identity']
                  + f"__document_score_batch{plan['batch_size']}_tail_actual")
    if any(actual.get(k) != v for k, v in approved.items()):
        raise ValueError('Native proof backend/arithmetic differs from the separately approved scoring identity.')
    destination = state_path.parent
    request_file = destination / 'batch_proof_requests' / (plan['name'] + '.json')
    if request_file.exists() and read_json(request_file) != precision:
        raise ValueError('Own-map proof request changed; preserve the existing request.')
    if not request_file.exists():
        save(request_file, precision)
    command = [sys.executable, str(ROOT / 'scripts/export_precision_qualification.py'),
               '--name', plan['name'], '--precision-map', str(request_file), '--source', plan['identity']['precision_source'],
               '--execute', '--batch-size', str(plan['batch_size'])]
    if checkpoint:
        command += ['--qat-checkpoint', relative(resolved(checkpoint))]
    try:
        for stage, flag, count in [('export', '--max-new-windows', plan['maximum_new_b1_windows']),
                                   ('batches', '--max-new-batches', plan['maximum_new_batches'])]:
            if not count:
                continue
            log = request_file.parent / (plan['name'] + f'_{stage}_{len(task["dispatches"])+1:03d}.log')
            dispatch_proof(command + ['--stage', stage, flag, str(count)], log, destination, task, state, state_path)
        plan = batch_proof_plan(config, precision, checkpoint)
        if plan['preparation_required']:
            raise RuntimeError('Bounded proof dispatch did not commit the required66-window own-map proof.')
        task['state'] = 'complete'
        return plan
    finally:
        task['observed'] = proof_observation(plan)
        state['accounting'] = accounting(state)
        save(state_path, state)


def mixed_review_sources():
    """Consumed arithmetic/input/checked-C source, using existing stable AST seams."""
    from export_precision_qualification import function_ast
    result = {p: dict(kind='text', text=(ROOT / p).read_text(encoding='utf-8-sig')) for p in
              ('gate_quant.py', 'gate_fpga_reference.py', 'gate_eval.py',
               'hardware/fixed_fp32_service.hpp', 'hardware/connected_fp32_bounds.hpp',
               'hardware/numerical_bridge_tb.cpp', 'scripts/run_mixed_exemplar_qualification.py')}
    for p, name in [('scripts/score_qualified_development.py', 'load_model'),
                    ('scripts/export_qualification_cases.py', 'validate_selection'),
                    ('scripts/run_precision_qualification.py', 'generate_precision_tb'),
                    ('scripts/run_qualification_cases.py', 'generate_bounds_tb')]:
        result[p + ':' + name] = dict(kind='ast', text=function_ast((ROOT / p).read_text(encoding='utf-8-sig'), name))
    return result


_mixed_review_context = None


def mixed_review_path(config, checkpoint=None):
    """Freeze a separate future QAT decision path before starting PTQ search.

    Absence of the optional key retains the historical shared-path contract.
    A declared but unresolved QAT path blocks only a requested QAT score.
    """
    methods = config['methods']
    key = ('mixed_qat_numerical_review_path'
           if checkpoint and 'mixed_qat_numerical_review_path' in methods
           else 'mixed_numerical_review_path')
    value = methods.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError('Mixed scoring requires methods.' + key + ' and an explicit complete checked-C coordinator decision.')
    return value


def validate_mixed_review(config, precision, checkpoint=None):
    """Neither batching nor inferred shared primitives can grant this decision."""
    if set(precision.values()) != {4, 8}:
        return None
    global _mixed_review_context
    precision_tuple(precision, config)
    value = mixed_review_path(config, checkpoint)
    path = resolved(value).resolve()
    relative(path)
    if not path.is_file():
        raise RuntimeError('Mixed numerical review file is absent; retained own-map batching proof is not numerical approval.')
    record = read_json(path)
    expected = dict(schema='sentry_campaign_mixed_numerical_review_v1', policy_id='bounded_connected_development_v2',
        coordinator_review=True, allow_development_scoring=True,
        approved_score_identity=config['methods']['approved_score_identity'],
        selected_checkpoint=read_json(ROOT / 'results/goal3/run.json')['selected_checkpoint'],
        selection=read_json(ROOT / 'results/goal4/numerical_bridge/development_cases_v1/selection.json'),
        source_snapshots=mixed_review_sources())
    identity = batch_identity(config, precision, checkpoint)
    expected.update(calibration=read_json(resolved(identity['calibration_file'])), frozen_scale_policy=identity['frozen_scale_policy'])
    if any(record.get(k) != v for k, v in expected.items()) or any(record.get(k) is not True for k in ('coordinator_review', 'allow_development_scoring')):
        raise ValueError('Mixed coordinator decision source/input/backend/calibration association differs.')
    from search_precision import approved_score_identity
    approved = approved_score_identity(config)
    size = config['methods'].get('ordered_scoring_batch_size', 1)
    execution = identity['execution_identity'] + (f'__document_score_batch{size}_tail_actual' if size != 1 else '')
    if approved != dict(arithmetic_identity=identity['arithmetic_identity'], execution_identity=execution):
        raise ValueError('Mixed review cannot substitute for the actual separately approved arithmetic/backend.')
    if record.get('scope') == 'exact_map_checkpoint':
        if record.get('precision_map') != precision or record.get('qat_checkpoint') != identity['qat_checkpoint']:
            raise ValueError('Mixed decision does not cover this exact requested map/checkpoint.')
    elif record.get('scope') == 'shared_fixed16_w4w8_primitives':
        covered = record.get('covered_sources')
        if (not isinstance(covered, list) or not covered or len(set(covered)) != len(covered)
                or not set(covered) <= {'ptq', 'qat'} or identity['precision_source'] not in covered
                or not isinstance(record.get('shared_scope_rationale'), str) or not record['shared_scope_rationale'].strip()):
            raise ValueError('Shared primitive coverage requires an explicit coordinator scope and scientific rationale.')
    else:
        raise ValueError('Unknown mixed numerical review scope.')
    if (not isinstance(record.get('reason'), str) or not record['reason'].strip()
            or not isinstance(record.get('evidence_paths'), list) or not record['evidence_paths']):
        raise ValueError('Concrete mixed numerical review reason and evidence are required.')
    for evidence in record['evidence_paths']:
        p = resolved(evidence).resolve(); relative(p)
        if not p.is_file() or p == path:
            raise ValueError('Missing mixed review evidence file.')
    out = resolved(record['exemplar_export']).resolve(); relative(out)
    qualification = out / 'bounds_v1/qualification.json'
    if relative(qualification) != record['exemplar_qualification']:
        raise ValueError('Mixed exemplar qualification must be its actual checked-C output.')
    exports = read_json(out / 'export.json'); state = read_json(qualification)
    if (exports['identity'] != record['exemplar_identity'] or state != record['exemplar_qualification_snapshot']
            or set(exports['precision_map'].values()) != {4, 8} or state.get('complete') is not True
            or state.get('active_window') is not None or len(state['windows']) != 66 or len(state.get('documents', [])) != 32
            or state.get('numerical_acceptance') is not None):
        raise ValueError('Actual complete mixed66-window/32-document exemplar association is required.')
    if record['scope'] == 'exact_map_checkpoint' and exports['identity'] != identity:
        raise ValueError('Exact mixed review exemplar differs from the requested map/checkpoint/backend.')
    if 'mixed_qat_numerical_review_path' in config['methods']:
        exemplar = exports['identity']
        if (exemplar.get('precision_source') != identity['precision_source']
                or exemplar.get('qat') is not identity['qat']):
            raise ValueError('Separated PTQ/QAT review requires an actual complete exemplar of the requested source.')
    for key in ('checkpoint', 'arithmetic_identity', 'execution_identity', 'scheme_id', 'calibration_file', 'frozen_scale_policy', 'windowing', 'evaluation', 'data'):
        if exports['identity'].get(key) != identity.get(key):
            raise ValueError('Mixed exemplar common numerical setting differs: ' + key)
    # Within one invocation, a previously checked exact record/qualification/source association
    # can be reused. Startup/resume always revalidates all66 actual C windows independently.
    if _mixed_review_context == (relative(path), record):
        return dict(path=relative(path), record=record,
                    saved_byte_validation=dict(validated_windows=66, complete=True, numerical_acceptance=None))
    # Existing inspector expects genuine Linux command paths. Its validate branch is read-only,
    # never takes a C-worker lock and never launches C/Torch; use that branch unchanged.
    code = ("import sys;from pathlib import Path;from types import SimpleNamespace;"
            "sys.path.insert(0,'scripts');import run_precision_qualification as q;"
            "out=q.ROOT/sys.argv[1];e=q.validate_export(out,complete=True);"
            "e=dict(e,parameter_folder=(out/'parameters').relative_to(q.ROOT).as_posix());"
            "state=q.read(out/'bounds_v1/qualification.json');"
            "use_adapter=any(v.get('recognized_mixed_exemplar') or v.get('mixed_adapter_dispatch') for v in state['windows']);"
            "from importlib import import_module;"
            "import_module('run_mixed_exemplar_qualification').validate_saved(out) if use_adapter else "
            "q.execute_locked(SimpleNamespace(stage='validate',reviewed_resolution=None),out,e,out/'bounds_v1',q.read(q.ROOT/'configs/project.json'),None);"
            "assert 'torch' not in sys.modules")
    command = ([sys.executable, '-c', code, relative(out)] if os.name != 'nt' else
               ['wsl', '-d', 'Ubuntu', '--cd', '/home/jp19126/Projects/SENTRY', '--', 'python3', '-c', code, relative(out)])
    checked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if checked.returncode != 0:
        raise ValueError('Independent complete mixed checked-C saved-byte validation failed: ' + checked.stderr[-4000:])
    result = json.loads(checked.stdout)
    if result != dict(validated_windows=66, complete=True, numerical_acceptance=None):
        raise ValueError('Mixed exemplar complete validator returned an unexpected result.')
    if read_json(path) != record or read_json(qualification) != state:
        raise ValueError('Mixed scientific decision/evidence changed during validation.')
    _mixed_review_context = (relative(path), deepcopy(record))
    return dict(path=relative(path), record=record, saved_byte_validation=result)


def bind_mixed_review(state, label, review):
    if review is None:
        return
    reviews = state.setdefault('mixed_numerical_reviews', {})
    prior = reviews.setdefault(review['path'], review)
    if prior != review:
        raise ValueError('Previously consumed mixed scientific review changed.')
    uses = state.setdefault('mixed_numerical_review_uses', {})
    if uses.setdefault(label, review['path']) != review['path']:
        raise ValueError('Previously scored mixed map changed scientific review path.')


def generate_score(folder, config, state, state_path, precision=None, checkpoint=None, *, floating=False):
    """Count actual committed score work independently of logical consultations."""
    from score_qualified_development import score_to_folder
    label = relative(folder)
    if not floating and set(precision.values()) == {4, 8} and (folder / 'candidate.json').is_file():
        review = validate_mixed_review(config, precision, checkpoint)
        bind_mixed_review(state, label, review)
        save(state_path, state)
    if (folder / 'candidate.json').is_file():
        if label in state['scoring_work']:
            state['scoring_work'][label].update(state='complete', observed=score_work(folder))
            save(state_path, state)
        return True
    prepare_batch_proof(config, state, state_path, precision, checkpoint, floating=floating)
    if not floating:
        review = validate_mixed_review(config, precision, checkpoint)
        bind_mixed_review(state, label, review)
    task = state['scoring_work'].setdefault(label, dict(stage='floating' if floating else
                ('qat' if checkpoint else 'ptq'), baseline=score_work(folder), dispatches=0))
    task['dispatches'] += 1
    task['state'] = 'dispatched'
    save(state_path, state)
    try:
        score_to_folder(folder, config, precision, checkpoint, floating=floating)
        task['state'] = 'complete'
    finally:
        task['observed'] = score_work(folder)
        state['accounting'] = accounting(state)
        save(state_path, state)
    return False


def floating_reference(folder, config, state, state_path):
    generate_score(folder, config, state, state_path, floating=True)
    record = read_json(folder / 'candidate.json')
    validate_score_identity(record['identity'], config, floating=True)
    metrics = recompute_metrics(folder, config)
    if (metrics['temporary_threshold'] != record['metrics']['primary']['threshold']
            or metrics['protection'] != record['metrics']['primary']['candidate_scoring']
            or metrics['ordinary_0_5'] != record['metrics']['ordinary_threshold_0_5']):
        raise ValueError('Qualified floating scores differ from their saved metrics.')
    return dict(identity=record['identity'], recall=metrics['protection']['recall'],
                metrics=metrics, source=relative(folder))


def ensure_cost(precision, costs, config, state, state_path):
    key = precision_tuple(precision, config)
    if key not in costs.lookup:
        # This is the actual common estimator, not a per-bit interpolation.
        from estimate_detector_cost import estimate_candidate_cost
        from freeze_goal4_cost_table import validate_frozen_cost_sources
        validate_frozen_cost_sources(costs.record, config, context=costs.frozen_source_context)
        row = estimate_candidate_cost(precision, config)
        if precision_tuple(row['precision_map'], config) != key:
            raise ValueError('Estimator returned a different candidate map.')
        costs.lookup[key] = row['profiles']
        before = costs.lookups
        try:
            costs.feasible_profiles(precision)
        except BaseException:
            del costs.lookup[key]
            raise
        finally:
            costs.lookups = before
        table_path = resolved(config['methods']['cost_table_path'])
        latest = read_json(table_path)
        if latest != costs.record:
            raise RuntimeError('Common cost table changed during campaign; resume after inspection.')
        validate_frozen_cost_sources(costs.record, config, context=costs.frozen_source_context)
        costs.record['candidates'].append(row)
        save(table_path, costs.record)
    row = next(row for row in costs.record['candidates']
               if precision_tuple(row['precision_map'], config) == key)
    name = map_name(precision, config)
    prior = state['cost_rows'].get(name)
    if prior is not None and prior != row:
        raise ValueError('Previously consulted candidate cost changed during resume.')
    state['cost_rows'][name] = row
    save(state_path, state)


def qat_identity(precision, checkpoint, config):
    run = read_json(ROOT / 'results/goal3/run.json')
    return dict(checkpoint=run['selected_checkpoint'], qat=True, precision_map=precision,
                scheme_id=SCHEME, calibration_file=run['calibration_file'],
                frozen_scale_policy={key: run['format'][key] for key in ('weight_scale', 'activation_scale')},
                windowing=config['windowing'], evaluation=config['evaluation'], data=config['data'],
                scored_checkpoint=relative(checkpoint), **config['methods']['approved_score_identity'])


def valid_qat_checkpoint(checkpoint, precision, config):
    """Only weights are reused; historical scores never acquire new identities."""
    if not (checkpoint / 'quantization.json').is_file():
        return False
    metadata = read_json(checkpoint / 'quantization.json')
    if 'training' not in metadata:
        return False
    validate_score_identity(qat_identity(precision, checkpoint, config), config, qat=True)
    training = metadata['training']
    for key in ('learning_rate', 'weight_decay'):
        if training[key] != config['training'][key]:
            raise ValueError('Completed QAT used a different common optimizer setting.')
    if training['optimizer'] != 'AdamW':
        raise ValueError('Completed QAT used a different common optimizer.')
    if training['effective_batch_documents'] != config['training']['effective_batch']:
        raise ValueError('Completed QAT used a different common document batch size.')
    if not all((checkpoint / name).is_file() for name in ('config.json', 'quantized_state.pt')):
        raise ValueError('Completed QAT metadata lacks its actual saved checkpoint.')
    return True


def train_shortlisted(precision, config, state, state_path, destination, args):
    """Dispatch the existing bounded one-epoch routine, with its resume checkpoint."""
    key = map_name(precision, config)
    checkpoint = destination / 'qat_checkpoints' / key
    if len(set(precision.values())) == 1:
        anchor = ROOT / 'checkpoints/quantized' / f'qat_w{next(iter(precision.values()))}_a8'
        if valid_qat_checkpoint(anchor, precision, config):
            state['qat_checkpoint_reuse'][key] = relative(anchor)
            save(state_path, state)
            return anchor
    if valid_qat_checkpoint(checkpoint, precision, config):
        if key in state['training_work']:
            state['training_work'][key].update(state='complete', training=read_json(checkpoint / 'quantization.json')['training'])
        state['qat_checkpoint_reuse'][key] = relative(checkpoint)
        save(state_path, state)
        return checkpoint
    paused(destination)
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from gate_quant import configure_exact_fp32, install_quantized_linears
    from quantization_sensitivity import qat_one_epoch
    run = read_json(ROOT / 'results/goal3/run.json')
    selection = {'checkpoint': run['selected_checkpoint']}
    calibration = read_json(resolved(run['calibration_file']))
    task = state['training_work'].setdefault(key, dict(checkpoint=relative(checkpoint), dispatches=0,
                precision_map=precision, device=args.training_device, microbatch_windows=args.batch_size))
    if task['device'] != args.training_device or task['microbatch_windows'] != args.batch_size:
        raise ValueError('QAT resume must retain its original device and microbatch setting.')
    resume = checkpoint / 'resume'
    if (resume / 'quantization.json').is_file():
        saved = read_json(resume / 'quantization.json')
        if (saved['precision_map'] != precision or saved['calibration'] != calibration
                or saved['format'] != run['format']
                or resolved(saved['from_floating_checkpoint']) != resolved(selection['checkpoint'])):
            raise ValueError('Partial QAT belongs to a different map/source/frozen format.')
    torch.set_num_threads(4)
    configure_exact_fp32()
    tokenizer = AutoTokenizer.from_pretrained(resolved(selection['checkpoint']), local_files_only=True, use_fast=True)
    train_windows = []
    for doc in rows(ROOT / 'data/prepared/train.jsonl'):
        if doc.get('split') != 'train':
            raise ValueError('QAT may read the assigned training partition only.')
        windows = window_document(tokenizer, doc['text'], doc['document_id'], doc['label'],
                                  doc.get('injection_spans'), config)
        if any(window['window_label'] is None for window in windows):
            raise ValueError('Unknown injection spans cannot be copied to window supervision.')
        train_windows.append(windows)
    model = AutoModelForSequenceClassification.from_pretrained(
            resolved(selection['checkpoint']), local_files_only=True).float().to(args.training_device)
    model = install_quantized_linears(model, calibration, precision)
    task['dispatches'] += 1
    task['state'] = 'dispatched'
    save(state_path, state)
    try:
        # Preserve its existing optimizer/RNG resume routine and all active pause markers.
        model, record = qat_one_epoch(model, tokenizer, train_windows, config,
                        SimpleNamespace(device=args.training_device, batch_size=args.batch_size),
                        key, destination / 'qat_training', checkpoint, calibration, selection,
                        pause_check=lambda: pause_requested(destination))
        task.update(state='complete', training=record)
        if not valid_qat_checkpoint(checkpoint, precision, config):
            raise RuntimeError('QAT dispatcher did not leave a valid completed checkpoint.')
    finally:
        if task['state'] != 'complete' and (resume / 'training_state.pt').is_file():
            partial = torch.load(resume / 'training_state.pt', map_location='cpu', weights_only=True)
            task['committed_training_seconds'] = partial['elapsed_seconds']
            task['committed_documents'] = partial['seen']
            task['state'] = 'partial'
        state['accounting'] = accounting(state)
        save(state_path, state)
        del model
    return checkpoint


def accounting(state):
    scoring = list(state['scoring_work'].values())
    actual = [task for task in scoring if task.get('observed', {}).get('committed_documents', 0)
              > task['baseline']['committed_documents']]
    training = list(state['training_work'].values())
    proofs = list(state.get('batch_proof_work', {}).values())
    return dict(logical_unique_maps={arm: len(value['consulted']) + int(value.get('active_precision_tuple') is not None)
                    for arm, value in state['arms'].items()},
                completed_consultations={arm: len(value['consulted']) for arm, value in state['arms'].items()},
                cheap_hardware_profile_lookups={arm: value['hardware_lookups'] for arm, value in state['arms'].items()},
                batch_proof_dispatches=sum(len(task['dispatches']) for task in proofs),
                proof_dispatch_elapsed_seconds=sum(dispatch.get('elapsed_seconds', 0.0)
                    for task in proofs for dispatch in task['dispatches']),
                new_committed_proof_b1_windows=sum(task.get('observed', task['baseline'])['committed_b1_windows']
                    - task['baseline']['committed_b1_windows'] for task in proofs),
                new_committed_proof_batches=sum(task.get('observed', task['baseline'])['committed_batches']
                    - task['baseline']['committed_batches'] for task in proofs),
                new_committed_instrumented_batch_forward_seconds=sum(task.get('observed', task['baseline'])['committed_instrumented_batch_forward_seconds']
                    - task['baseline']['committed_instrumented_batch_forward_seconds'] for task in proofs),
                proof_scope='Frozen66 B1 and own-map batch equivalence; capture time included; no additional logical map or numerical approval.',
                scoring_dispatches=sum(task['dispatches'] for task in scoring),
                scoring_jobs_with_committed_work=len(actual),
                new_committed_scoring_documents=sum(task['observed']['committed_documents']
                    - task['baseline']['committed_documents'] for task in actual),
                new_committed_scoring_seconds=sum(task['observed']['committed_scoring_seconds']
                    - task['baseline']['committed_scoring_seconds'] for task in actual),
                training_dispatches=sum(task['dispatches'] for task in training),
                completed_new_qat_epochs=sum(task['state'] == 'complete' for task in training),
                committed_training_seconds=sum(task.get('training', {}).get('seconds',
                    task.get('committed_training_seconds', 0.0)) for task in training),
                reused_qat_checkpoints=state['qat_checkpoint_reuse'],
                scope='Research scoring/training work; committed-work lower bound after abrupt termination; not inference latency.',
                C=None)


def execute(config, args):
    validate_bounds(config)
    missing = readiness(config)
    if missing:
        raise RuntimeError('Campaign unavailable: ' + '; '.join(missing))
    costs = CostTable(config)  # Validate the common table before any work or campaign writes.
    from freeze_goal4_cost_table import validate_frozen_cost_sources
    costs.frozen_source_context = validate_frozen_cost_sources(costs.record, config)
    destination = resolved(args.output)
    relative(destination)  # Keep campaign scores/checkpoints inside this repository.
    paused(destination)
    state_path = destination / 'campaign.json'
    if args.phase == 'qat' and not state_path.is_file():
        raise RuntimeError('QAT phase requires an existing campaign with a frozen shortlist.')
    state = read_json(state_path) if state_path.is_file() else dict(
        version=1, protocol=protocol_snapshot(config),
        cost_table_header={key: value for key, value in costs.record.items() if key != 'candidates'},
        arms={arm: new_arm() for arm in ('A', 'B')}, cost_rows={}, scoring_work={},
        training_work={}, batch_proof_work={}, mixed_numerical_reviews={}, mixed_numerical_review_uses={}, qat_checkpoint_reuse={}, floating=None, C=None,
        created_utc=datetime.now(timezone.utc).isoformat())
    if state['version'] != 1 or state['protocol'] != protocol_snapshot(config):
        raise ValueError('Campaign protocol changed; preserve the existing run instead of silently resuming.')
    if state['cost_table_header'] != {key: value for key, value in costs.record.items() if key != 'candidates'}:
        raise ValueError('Common cost-model/profile/cap metadata changed during resume.')
    for name, row in state['cost_rows'].items():
        if costs.lookup.get(precision_tuple(row['precision_map'], config)) != row['profiles']:
            raise ValueError('Previously consulted common costs changed during resume.')
    if args.phase == 'qat':
        requested = ('A', 'B') if args.arm == 'both' else (args.arm,)
        if any(state['arms'][arm]['frozen_qat_shortlist'] is None for arm in requested):
            raise RuntimeError('Complete bounded search before the requested QAT phase.')
    save(state_path, state)
    floating = floating_reference(destination / 'scores/floating', config, state, state_path)
    if state['floating'] is not None and state['floating'] != floating:
        raise ValueError('Floating reference changed during resume.')
    state['floating'] = floating
    save(state_path, state)
    completed = 0
    for arm in (('A', 'B') if args.arm == 'both' else (args.arm,)):
        arm_state = state['arms'][arm]
        adapter = restore_adapter(arm, arm_state, config, costs, floating)
        if args.phase in ('search', 'all'):
            while len(adapter.consulted) < config['methods']['max_unique_maps_per_method']:
                paused(destination)
                if args.max_new_maps is not None and completed >= args.max_new_maps:
                    state['accounting'] = accounting(state)
                    save(state_path, state)
                    return state
                if not arm_state['pending']:
                    arm_state['pending'] = adapter.propose()
                    snapshot_adapter(adapter, arm_state)
                    save(state_path, state)  # RNG and complete pending batch precede expensive work.
                precision = arm_state['pending'][0]
                ensure_cost(precision, costs, config, state, state_path)
                before = costs.lookups
                profiles = costs.feasible_profiles(precision)
                adapter.hardware_lookups += costs.lookups - before
                arm_state['active_precision_tuple'] = precision_tuple(precision, config)
                snapshot_adapter(adapter, arm_state)
                state['accounting'] = accounting(state)
                save(state_path, state)  # Charge this arm before its possibly interrupted score job.
                score = None
                reused = None
                if profiles:
                    folder = destination / 'scores/ptq' / map_name(precision, config)
                    reused = generate_score(folder, config, state, state_path, precision)
                    score = load_score(folder, config, floating['recall'])
                    score['score_reused'] = reused
                    if not reused:
                        work = state['scoring_work'][relative(folder)]
                        score['new_scoring_seconds'] = work['observed']['committed_scoring_seconds'] - work['baseline']['committed_scoring_seconds']
                adapter.consider(precision, score)
                arm_state['pending'].pop(0)
                arm_state['active_precision_tuple'] = None
                snapshot_adapter(adapter, arm_state)
                completed += 1
                state['accounting'] = accounting(state)
                save(state_path, state)
                print(json.dumps(dict(event='consulted', arm=arm,
                    logical_unique_maps=len(adapter.consulted), reused_score=reused)), flush=True)
            adapter.qat_shortlist()
            snapshot_adapter(adapter, arm_state)
            save(state_path, state)
        if args.phase in ('qat', 'all'):
            if args.max_new_maps is not None and completed >= args.max_new_maps:
                state['accounting'] = accounting(state)
                save(state_path, state)
                return state
            if adapter._qat_shortlist is None:
                raise RuntimeError('Complete bounded search before dispatching the frozen QAT shortlist.')
            scores = []
            for entry in adapter.qat_shortlist():
                paused(destination)
                precision = entry['candidate']['precision_map']
                checkpoint = train_shortlisted(precision, config, state, state_path, destination, args)
                folder = destination / 'scores/qat' / map_name(precision, config)
                generate_score(folder, config, state, state_path, precision, relative(checkpoint))
                score = load_score(folder, config, floating['recall'], qat=True)
                if resolved(score['identity']['scored_checkpoint']) != checkpoint:
                    raise ValueError('QAT score cache names a different checkpoint from the selected common run.')
                scores.append(score)
            arm_state['post_qat'] = adapter.post_qat_selection(scores)
            snapshot_adapter(adapter, arm_state)
            state['accounting'] = accounting(state)
            save(state_path, state)
    state['accounting'] = accounting(state)
    save(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--arm', choices=('A', 'B', 'both'), default='both')
    parser.add_argument('--phase', choices=('search', 'qat', 'all'), default='all')
    parser.add_argument('--output', default='results/goal5/campaign_ab_v1')
    parser.add_argument('--max-new-maps', type=int,
                        help='Stop after this many new logical consults in this invocation; preserve pending batches.')
    parser.add_argument('--training-device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if (args.max_new_maps is not None and args.max_new_maps < 1) or not 1 <= args.batch_size <= 32:
        parser.error('Use positive new-map limit and QAT microbatch 1..32.')
    config = load_config()
    if not args.execute:
        print(json.dumps(dict(execute=False, missing_prerequisites=readiness(config),
            arms=['A', 'B'] if args.arm == 'both' else [args.arm], phase=args.phase,
            output=args.output, logical_allowance_per_arm=config['methods']['max_unique_maps_per_method'],
            max_qat_shortlist_per_arm=config['methods']['max_shortlist_per_method'],
            qat_epochs=config['methods']['qat_epochs'], partitions=SPLITS,
            own_map_batch_preparation=dict(configured_batch_size=config['methods'].get('ordered_scoring_batch_size', 1),
                exporter='scripts/export_precision_qualification.py', maximum_b1_forwards_per_new_proof=66,
                maximum_batch_forwards={8:9, 32:3}, backend='Native sys.executable; existing CPU4 FP32 settings',
                mixed_numerical_review_path=config['methods'].get('mixed_numerical_review_path'),
                mixed_qat_numerical_review_path=config['methods'].get('mixed_qat_numerical_review_path'),
                mixed_numerical_review_required_before_score=True, acceptance=None),
            campaign_maps_consulted=0, new_scoring_calls=0, new_training_calls=0, C=None), indent=2))
        return 0
    try:
        state = execute(config, args)
    except (RuntimeError, ValueError) as error:
        print(json.dumps(dict(state='blocked_or_paused', reason=str(error), C=None)), file=sys.stderr)
        return 2
    print(json.dumps(dict(state='saved', campaign=relative(resolved(args.output) / 'campaign.json'),
                         accounting=state['accounting'], C=None), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
