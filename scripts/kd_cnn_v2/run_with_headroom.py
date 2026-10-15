"""Run the existing queue while requesting an exact checkpoint on low host RAM."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time


def available_bytes():
    if sys.platform != 'win32':
        raise RuntimeError('This host-RAM observer is Windows-only')
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_physical', 'available_physical', 'total_page', 'available_page',
             'total_virtual', 'available_virtual', 'available_extended')]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    if not 0 < value.available_physical <= value.total_physical:
        raise RuntimeError('Invalid host memory observation')
    return value.available_physical


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-ids', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--start-gib', type=float, default=4.5)
    p.add_argument('--stop-gib', type=float, default=3.0)
    a = p.parse_args()
    if not 0 < a.stop_gib < a.start_gib:
        p.error('Require positive stop floor below start headroom')
    stop = a.canonical_root / 'results/kd_cnn_v2/stop.request'
    if stop.exists():
        raise RuntimeError('Existing stop request must be resolved by coordinator')
    if a.out.exists():
        raise FileExistsError(a.out)
    free = available_bytes()
    if free < a.start_gib * 2**30:
        raise RuntimeError(f'Insufficient starting RAM: {free} bytes')
    a.out.mkdir(parents=True)
    command = [sys.executable, '-B', str(Path(__file__).with_name('run_queue.py')),
               '--canonical-root', str(a.canonical_root), '--config', str(a.config),
               '--run-ids', *a.run_ids, '--max-seconds-per-run', '1800']
    record = dict(command=command, start_available_bytes=free, start_gib=a.start_gib,
                  stop_gib=a.stop_gib, state='running', resource='cpu_observer',
                  boundary='Child queue owns GPU leases and accounts their costs; no vendor interference')
    def save():
        (a.out / 'record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    save()
    started = time.monotonic()
    with (a.out / 'queue.log').open('w', encoding='utf-8') as log, (a.out / 'memory.jsonl').open('w', encoding='utf-8') as observations:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = proc.pid
        save()
        try:
            while proc.poll() is None:
                free = available_bytes()
                observations.write(json.dumps({'unix': time.time(), 'available_bytes': free}) + '\n')
                observations.flush()
                if free < a.stop_gib * 2**30 and not stop.exists():
                    stop.write_text('Checkpoint requested by CNN host-RAM observer; no vendor process touched.\n', encoding='utf-8')
                    record['memory_stop_requested'] = True
                    save()
                time.sleep(1)
        except BaseException:
            if proc.poll() is None:
                stop.write_text('Checkpoint requested: CNN host-RAM observer interrupted.\n', encoding='utf-8')
            record.update(state='observer_interrupted', wall_seconds=time.monotonic()-started)
            save()
            raise
        record.update(exit_code=proc.returncode, state='completed' if proc.returncode == 0 else 'stopped_or_failed', wall_seconds=time.monotonic()-started)
        save()
    print(json.dumps(record))
    return proc.returncode


if __name__ == '__main__':
    sys.exit(main())
