"""Retrieve the one approved public checkpoint, with a fixed file allowlist.

No dataset, executable model code, access bypass, or remote inference is used.
The inspected PIGuard forward is reproduced using installed Transformers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1_challengers import resource_check, json_write

REPO = "leolee99/PIGuard"
REVISION = "dd78b24e330193a22d2293ac66922dd4f982f563"
FILES = ("README.md", "config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


def main():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_url
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/span_coupled_pilot_v1/public_reference")
    args = parser.parse_args()
    folder = (ROOT / args.output).resolve()
    if not folder.is_relative_to(ROOT / "results/span_coupled_pilot_v1"):
        raise ValueError("Output outside study")
    if folder.exists():
        raise FileExistsError("Preserve the existing retrieval attempt")
    before = resource_check(True)
    started = time.monotonic()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    sizes = {f.rfilename: f.size for f in info.siblings if f.rfilename in FILES}
    if info.sha != REVISION or info.gated or set(sizes) != set(FILES):
        raise ValueError("Pinned public reference metadata differs")
    if any(v is None for v in sizes.values()) or sum(sizes.values()) > 2 * 1024**3:
        raise ValueError("Checkpoint retrieval allowance exceeded")
    folder.mkdir(parents=True)
    # A workspace-local cache; the library never imports any downloaded module.
    record = dict(repository=REPO, revision=REVISION, license="MIT (model card)",
                  prompt_guard_2_access="401; no existing HF credential; no weights retrieved",
                  files={}, resources=[before], complete=False,
                  remote_code_executed=False, final_data_opened=False)
    json_write(folder / "retrieval.json", record)
    for name in FILES:
        if time.monotonic() - started > 1200:
            raise RuntimeError("Reference retrieval ceiling reached")
        # Direct bounded byte streaming avoids unsupported Windows locks on WSL
        # UNC filesystems. URLs contain only the pinned repository/revision/file.
        path = folder / name
        digest = hashlib.sha256()
        last = time.monotonic()
        received = 0
        with urllib.request.urlopen(hf_hub_url(REPO, name, revision=REVISION), timeout=30) as source, path.open("xb") as destination:
            for chunk in iter(lambda: source.read(1024**2), b""):
                received += len(chunk)
                if received > sizes[name]:
                    raise ValueError("Server exceeded pinned file size")
                destination.write(chunk)
                digest.update(chunk)
                if time.monotonic()-started > 1200:
                    raise RuntimeError("Reference retrieval ceiling reached")
                if (ROOT/"results/span_coupled_pilot_v1/pause.request").exists():
                    raise RuntimeError("Study pause requested")
                if time.monotonic()-last >= 5:
                    record["resources"].append(resource_check());last=time.monotonic()
        if path.stat().st_size != sizes[name]:
            raise ValueError("Retrieved file size mismatch")
        record["files"][name] = dict(bytes=sizes[name], sha256=digest.hexdigest())
        record["resources"].append(resource_check())
        json_write(folder / "retrieval.json", record)
        print(json.dumps({"retrieved": name, "bytes": sizes[name]}), flush=True)
    record.update(complete=True, wall_seconds=time.monotonic()-started)
    json_write(folder / "retrieval.json", record)


if __name__ == "__main__":
    main()
