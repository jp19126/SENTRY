"""Small cross-OS shared-device lock; no automatic stale-lock deletion."""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


class ResourceLock:
    def __init__(self, shared_root, resource="gpu0", task="kd_cnn_v2"):
        self.path = Path(shared_root) / "locks" / (resource + ".lock")
        self.owner = {"task": task, "pid": os.getpid(), "host": socket.gethostname(),
                      "started_unix": time.time(), "resource": resource}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            raise RuntimeError(f"Resource busy: {self.path}; inspect owner, never steal a live lock")
        try:
            (self.path / "owner.json").write_text(json.dumps(self.owner, indent=2), encoding="utf-8")
        except BaseException:
            self.path.rmdir()
            raise
        return self

    def __exit__(self, *exc):
        owner = self.path / "owner.json"
        if json.loads(owner.read_text(encoding="utf-8")) != self.owner:
            raise RuntimeError("Lock ownership changed; preserving directory")
        owner.unlink()
        self.path.rmdir()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-root", required=True)
    p.add_argument("--resource", default="gpu0")
    p.add_argument("--record", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("a subprocess command is required")
    record = Path(a.record)
    if record.exists():
        raise FileExistsError(record)
    record.parent.mkdir(parents=True, exist_ok=True)
    with ResourceLock(a.shared_root, a.resource) as lock:
        started = time.monotonic()
        value = {**lock.owner, "command": command, "state": "running"}
        record.write_text(json.dumps(value, indent=2), encoding="utf-8")
        try:
            result = subprocess.run(command, check=False)
            value.update(exit_code=result.returncode, state="completed" if result.returncode == 0 else "failed")
        except BaseException as error:
            value.update(state="interrupted", error=repr(error))
            raise
        finally:
            value["wall_seconds"] = time.monotonic() - started
            record.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
