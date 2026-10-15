"""Small cross-platform helpers for the bounded whole-BERT study."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = Path(os.environ.get('SENTRY_CANONICAL_ROOT', str(ROOT)))
SHARED = Path(os.environ.get('SENTRY_SHARED_ROOT', str(CANONICAL / 'results/dual_track_v1/shared')))
OUT = ROOT / 'results/bert_aie_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

@contextlib.contextmanager
def lock(name, timeout=0):
    """Atomic mkdir works for both UNC and WSL. Never steal an existing lock."""
    directory = SHARED / 'locks' / (name + '.lock')
    directory.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            if time.monotonic() - start >= timeout:
                raise RuntimeError(f'Shared resource occupied: {directory}')
            # Short released leases between sequential sibling runs must be
            # observable; atomic mkdir remains the sole ownership decision.
            time.sleep(.05)
    try:
        write(directory / 'owner.json', {'task':'bert_aie_v1','pid':os.getpid(),
              'host':socket.gethostname(),'created_unix':time.time(), 'resource':name})
        yield
    finally:
        (directory / 'owner.json').unlink(missing_ok=True)
        directory.rmdir()

def canonical_relative(path):
    return str(Path(path).relative_to(CANONICAL)).replace('\\', '/')
