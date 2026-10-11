#!/usr/bin/env python3
"""Check or safely replace the two SENTRY Markdown handoffs using strict UTF-8."""
import argparse
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("STATUS.md", "reports/board_handoff.md")
MAX_BYTES = 128 * 1024
MAX_LINE_BYTES = 8 * 1024
# Common UTF-8 bytes misread as Windows-1252/Latin-1. Valid UTF-8 alone
# cannot detect this: the demonstrated corrupted dash was valid UTF-8.
MOJIBAKE = re.compile(
    r"[\u00c2\u00c3\u00e2\u00f0]"
    r"[\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e"
    r"\u0192\u02c6\u02dc\u2013-\u201e\u2020-\u2022\u2026\u2030"
    r"\u2039\u203a\u20ac\u2122]"
)


def repo_path(value):
    path = ROOT / value
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("path must stay inside this repository")
    return path


def validate(data, label):
    if len(data) > MAX_BYTES:
        raise ValueError(f"{label}: exceeds {MAX_BYTES}-byte handoff limit")
    text = data.decode("utf-8", errors="strict")
    if not text.strip():
        raise ValueError(f"{label}: handoff must not be empty")
    if any(c in text for c in ("\x00", "\ufffd", "\ufeff")):
        raise ValueError(f"{label}: NUL, replacement character or BOM present")
    if "\r" in text.replace("\r\n", ""):
        raise ValueError(f"{label}: unsupported lone carriage return")
    if MOJIBAKE.search(text):
        raise ValueError(f"{label}: suspicious text-encoding corruption")
    if any(len(line) > MAX_LINE_BYTES for line in data.splitlines()):
        raise ValueError(f"{label}: line exceeds {MAX_LINE_BYTES}-byte limit")
    return text


def checked_read(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path.name}: exceeds {MAX_BYTES}-byte handoff limit")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    validate(data, path.name)
    return data


def update(name, candidate):
    if name not in TARGETS:
        raise ValueError("only the two declared handoffs may be replaced")
    target = repo_path(name)
    if target.is_symlink():
        raise ValueError("handoff target must not be a symlink")
    original = checked_read(target)
    mode = target.stat().st_mode & 0o777
    incoming = checked_read(repo_path(candidate))
    data = validate(incoming, "candidate").replace("\r\n", "\n").encode("utf-8")
    validate(data, "candidate")
    if data == original:
        print(f"unchanged {name}: {len(data)} bytes")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=target.name + ".",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        if checked_read(temporary) != data:
            raise ValueError("staged handoff readback differs from candidate")
        if checked_read(target) != original:
            raise ValueError("handoff changed during update; replacement refused")
        os.replace(temporary, target)
        temporary = None
        if checked_read(target) != data:
            raise ValueError("handoff readback differs after replacement")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"updated {name}: {len(data)} bytes, strict UTF-8/LF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="read-only check (default)")
    action.add_argument("--write", choices=TARGETS, help="replace one handoff")
    parser.add_argument("--from-file", help="UTF-8 candidate path inside the repository")
    args = parser.parse_args()
    if bool(args.write) != bool(args.from_file):
        parser.error("--write and --from-file must be supplied together")
    try:
        if args.write:
            update(args.write, args.from_file)
        else:
            for name in TARGETS:
                data = checked_read(repo_path(name))
                longest = max(map(len, data.splitlines()), default=0)
                print(f"checked {name}: {len(data)} bytes, longest line {longest} bytes")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"handoff refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
