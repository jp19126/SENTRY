"""Fixed L256 sequencer/control/mover simulation; arithmetic completion stubs only.

No new HLS synthesis, board project, NoC setup, physical allocation or inference.
--execute compiles existing control-register RTL and the new bounded integration.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from plan_detector_commands import create_plan, write_rtl_plan

ROOT = Path(__file__).resolve().parents[1]
FILES = ('gate_detector_sequencer.sv', 'gate_detector_control_tb.sv',
         'gate_data_mover.sv', 'gate_mover_axi_memory.sv')
BASE = 0x00000001FFFFF000  # Synthetic simulation address only; exercises a low32 carry.
W8 = 0xA55A
CASES = 13


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def prepare(plan, folder):
    write_rtl_plan(plan, folder)
    constants = []
    for name in ('token_ids', 'type_ids', 'attention_mask', 'input_scales'):
        constants.append(f"localparam integer {name.upper()}_OFFSET = {plan['buffers'][name]['offset']};")
    (folder / 'integration_fixture_constants.svh').write_text('\n'.join(constants) + '\n')

    def value(v):
        if not isinstance(v, dict):
            return v
        if 'arena_offset' in v:
            return BASE + v['arena_offset']
        if 'select_by_frozen_precision_group' in v:
            bits = 8 if W8 >> v['select_by_frozen_precision_group'] & 1 else 4
            return value(v['alternatives'][str(bits)])
        if 'frozen_precision_group' in v:
            return 8 if W8 >> v['frozen_precision_group'] & 1 else 4
        if 'local_input_scale_index' in v:
            return 0x3E800000 + v['local_input_scale_index']
        return v['f32_bits']

    expected = []
    for command in plan['commands']:
        if command['kind'] != 'service':
            continue
        record = [int(command['engine'] == 'fp32')]
        for write in command['register_writes'][:-1]:
            x = value(write['value'])
            record.append((x >> 32 if write.get('part') == 'high32' else x) & 0xFFFFFFFF)
        expected += record + [0] * (20 - len(record))
    assert len(expected) == 435 * 20
    (folder / 'expected_arguments.mem').write_text('\n'.join(f'{v:08x}' for v in expected) + '\n')


def report(record, path):
    lines = ['# Fixed L256 internal control and movement integration', '',
        'This simulation connects the new fixed sequencer and data mover to the actual generated integer and FP32 AXI-Lite register blocks. Explicit completion stubs replace arithmetic datapaths. It is control/layout evidence, not a full detector numerical run or physical DDR measurement.', '',
        f"Execution state: **{record['state']}**. Passed: **{record['passed']}**. Evidence: `{path.relative_to(ROOT).as_posix()}`.", '',
        'The frozen deployment configuration contains a model ID, W4/W8 group mask, exact FP32 threshold bits and arena base. Changes require reset and a matching cold load of weights/scales. The simulation address is deliberately synthetic and crosses a 32-bit pointer boundary; no address has been allocated on the board.', '',
        'The nominal memory mode is AXI32/address64, one outstanding transaction per direction, maximum 16 beats split at 4 KiB. First read data and write response are registered; read responses can sustain one beat per cycle. There is no extra DDR delay, arbitration or periodic throttling. This opt-in mode matches the previously used arithmetic harness edge convention; the mover-focused backpressure tests retain their distinct model.', '',
        '`inclusive_cycles` is the retained JSON field name for the elapsed sampled-edge difference: first result-valid cycle minus input valid/ready acceptance cycle, independent of result backpressure. No +1 is added to that elapsed time. It equals the DUT diagnostic `total_cycles` plus 1 because that state counter excludes the final observation boundary edge. Each stub interval is separately sampled from accepted ap_start while idle to ap_done, matching the existing arithmetic harness convention. The conditional control/layout term is this sampled elapsed difference minus the sum of those exact intervals. No assumed stub constant, polling gap or complete-detector latency is substituted.', '',
        'The DUT also exposes categories for AXI-Lite request/response states, interrupt-wait states and mover command/response states; these are diagnostic FSM occupancies, not independently additive replacements for the subtraction boundary.', '']
    if record.get('cases'):
        lines += ['| Case | Status | Sampled elapsed cycles | Stub intervals | Control/layout cycles | Read beats | Write beats |', '|---|---:|---:|---:|---:|---:|---:|']
        for row in record['cases']:
            lines.append(f"| {row['id']} | {row['status']} | {row['inclusive_cycles']} | {row['stub_interval_cycles']} | {row['control_layout_cycles']} | {row['read_beats']} | {row['write_beats']} |")
    lines += ['', 'Each complete schedule must issue 24 integer and 411 FP32 calls, 8,508 argument/start/IRQ-clear writes and 870 completion/status reads. Four IRQ-enable writes occur once during configuration and are excluded from warm-window counts. The TB checks every actual generated-register argument against the plan, both high and low pointer words, mixed group precision, local scales, source IDs, strict threshold equality, document maximum/reset, invalid IDs/masks/scales, address alignment/overflow and nonfinite final risk. Any error returns unusable status and requires reset.', '',
        'Traffic reconciliation: movement useful reads are 8,912,896 bytes and writes 9,043,968 bytes, totaling 17,956,864. Separately the four cached input reads consume 2,400 bytes and final risk consumes4, giving 17,959,268 mover-port bytes per window. Token/type IDs are 32-bit and validated before use; each is read once, mask bytes are packed four per 32-bit word, and scales are 24 words. Embedding and mask commands use those caches, so there is no repeated DDR ID read. Every layout element is a 32-bit word; the finite valid mask is literal negative zero and invalid mask is 0xff7fffff. The earlier 18,092,032-byte provisional ledger assumed 131,072 bytes of repeated FP32 base-mask reads and 4,096 bytes of wider ID reads. Removing those 135,168 bytes yields 17,956,864 movement bytes; adding the actual 2,404 cached-input/final-read bytes gives17,959,268, a 132,764-byte reduction from the old layout figure. Service-port traffic, command ROM fetches, host transfers and cold loading are outside these mover counters.', '',
        'No controller/mover synthesis, integrated timing closure, NoC/DDR contention, host latency, board measurement or broad detector numerical acceptance is established. Conditional addition of the control/layout term to the saved sequential service costs remains an analytical model, not observed complete execution.', '']
    if record.get('error'):
        lines += [f"Stopped reason: {record['error']}", '']
    (ROOT / 'reports/goal4_internal_integration.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / 'configs/project.json')
    plan = create_plan(config)
    if not args.execute:
        print(json.dumps({'execute': False, 'cases': CASES, 'commands': plan['counts'],
                          'arithmetic': 'explicit completion stubs', 'synthesis': False}, indent=2))
        return
    if os.name == 'nt':
        raise RuntimeError('Use configured WSL Python/tools.')
    pause = ROOT / 'results/goal2/pause.request'
    if pause.exists():
        raise RuntimeError('Pause marker present.')
    tools = Path(config['hardware']['hls']['vivado_executable']).parent
    tag = datetime.now(timezone.utc).isoformat().replace(':', '_').replace('.', '_').replace('+', '_')
    out = ROOT / 'results/goal4/internal_integration' / tag
    out.mkdir(parents=True)
    prepare(plan, out)
    for name in FILES:
        shutil.copyfile(ROOT / 'hardware' / name, out / name)
    controls = [plan['integer_builds'][-1]['control_source'], plan['fp32_build']['control_source']]
    for name in controls:
        shutil.copyfile(ROOT / name, out / Path(name).name)
    for name in (Path(__file__), ROOT / 'scripts/plan_detector_commands.py'):
        shutil.copyfile(name, out / name.name)
    (out / 'command_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (out / 'run.tcl').write_text('run all\nquit\n')
    record = dict(state='running', passed=False, recorded_utc=datetime.now(timezone.utc).isoformat(),
        arithmetic='explicit bounded completion stubs; no detector math', source_files=list(FILES),
        actual_control_sources=controls, synthesis_performed=False, physical_arena_allocated=False,
        clock_ns=5, memory_mode=dict(STALL=0, PIPELINED_READ=1, extra_DDR_delay=False),
        fixture_model_id='0xbf04010000000001', fixture_map='0xa55a', fixture_threshold='0x3b955a95',
        synthetic_arena_base=hex(BASE), commands=[])
    def save():
        (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    commands = [['xvlog', '--sv', *FILES, *[Path(name).name for name in controls]],
                ['xelab', '--debug', 'off', 'gate_detector_control_tb', '-s', 'gate_detector_control_sim'],
                ['xsim', 'gate_detector_control_sim', '-tclbatch', 'run.tcl']]
    env = os.environ.copy()
    env['PATH'] = str(tools) + os.pathsep + env.get('PATH', '')
    try:
        for command in commands:
            if pause.exists():
                raise RuntimeError('Pause marker present.')
            cmd = [str(tools / command[0]), *command[1:]]
            record['commands'].append(cmd)
            save()
            with (out / (command[0] + '.stdout.log')).open('w') as log:
                process = subprocess.Popen(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                begin = time.monotonic()
                while process.poll() is None:
                    if pause.exists() or time.monotonic() - begin > 900:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                        raise RuntimeError('Pause/900-second timeout at ' + command[0])
                    time.sleep(.25)
            if process.returncode:
                raise RuntimeError(command[0] + ' failed; inspect saved log.')
        text = (out / 'xsim.stdout.log').read_text(errors='replace')
        cases = [json.loads(line.split(' ', 1)[1]) for line in text.splitlines() if line.startswith('GATE_INTEGRATION_CASE ')]
        stubs = [json.loads(line.split(' ', 1)[1]) for line in text.splitlines() if line.startswith('GATE_INTEGRATION_STUB ')]
        if len(cases) != CASES or len({r['id'] for r in cases}) != CASES or 'PASS gate_detector_control 13 cases;' not in text:
            raise RuntimeError('Missing complete bounded integration evidence.')
        for index, case in enumerate(cases):
            measured = [s for s in stubs if s['case_index'] == index]
            if case['complete_window'] and len(measured) != 435:
                raise RuntimeError('Missing sampled service boundaries.')
            if any(s['latency_cycles'] != s['done_cycle'] - s['start_cycle'] for s in measured):
                raise RuntimeError('Inconsistent stub edge record.')
            if sum(s['latency_cycles'] for s in measured) != case['stub_interval_cycles']:
                raise RuntimeError('Stub sum differs from sampled boundaries.')
            if case['inclusive_cycles'] - case['stub_interval_cycles'] != case['control_layout_cycles']:
                raise RuntimeError('Control/layout subtraction differs.')
        record.update(state='returned', passed=True, cases=cases, sampled_stub_intervals=stubs,
                      complete_detector_latency_cycles=None, physical_measurement=False)
        save()
        report(record, out / 'run.json')
        print(json.dumps({'passed': True, 'evidence': str(out / 'run.json'), 'cases': cases}, indent=2))
    except Exception as exc:
        record.update(state='stopped', error=str(exc))
        save()
        report(record, out / 'run.json')
        raise


if __name__ == '__main__':
    main()
