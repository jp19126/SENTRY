"""Analysis-only in-memory membership adaptor for the frozen comparator.

The on-disk scoring/comparison source is never changed. Reserved score rows are
read only with explicit release and a complete independent sixteen-target audit.
--self-test and --opaque-freeze-check do not decode reserved document/score rows.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/cnn_adoption_v1"
ORIGINAL = ROOT / "scripts/cnn_adoption_compare.py"
PROVENANCE = DEST / "confirmation_comparison_adapter"
CORE = {"ids.npy", "masks.npy", "starts.npy", "documents.jsonl"}
OLD = 'if actual_data.get("files") != expected.get("artifact_sha256"):'
NEW = 'if not _analysis_recorded_files_match(actual_data.get("files"), expected.get("artifact_sha256")):'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def local(value, base=ROOT):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(p.resolve().is_relative_to(base.resolve()), "Path escapes permitted workspace")
    return p


def recorded_files_match(actual, expected):
    return (isinstance(actual, dict) and isinstance(expected, dict)
            and CORE <= set(actual) and CORE <= set(expected)
            and all(name in expected and value == expected[name] for name, value in actual.items()))


def preserve(p, content):
    if p.exists():
        require(p.read_bytes() == content, "Existing adaptor provenance differs; preserve prior attempt")
    else:
        with p.open("xb") as f:
            f.write(content)


def load_adaptor():
    original = ORIGINAL.read_bytes()
    source = original.decode("utf-8")
    require(source.count(OLD) == 1, "Comparator correction location changed")
    patched = source.replace(OLD, NEW)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    preserve(PROVENANCE / "comparator_original.py", original)
    diff = ''.join(difflib.unified_diff(source.splitlines(keepends=True), patched.splitlines(keepends=True),
                                      fromfile="frozen/cnn_adoption_compare.py", tofile="in-memory/membership-adaptor"))
    preserve(PROVENANCE / "membership_compatibility.diff", diff.encode())
    meta = {"original_source_sha256": hashlib.sha256(original).hexdigest(),
            "in_memory_logical_source_sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "diff_sha256": sha(PROVENANCE / "membership_compatibility.diff"),
            "exact_original_snapshot_sha256": sha(PROVENANCE / "comparator_original.py"),
            "correction": "Require four core scoring inputs; every recorded file must match a frozen artifact. Hash every frozen artifact and manifest before reserved rows.",
            "semantic_target_seed_format_guards": "Unmodified original comparator guards"}
    preserve(PROVENANCE / "provenance.json", (json.dumps(meta, indent=2) + "\n").encode())
    ns = {"__name__": "_analysis_only_frozen_comparator", "__file__": str(ORIGINAL),
          "_analysis_recorded_files_match": recorded_files_match}
    exec(compile(patched, str(ORIGINAL) + ":in-memory-analysis-adaptor", "exec"), ns)
    old_freeze = ns["_validate_freeze"]

    def all_membership_before_rows(plan, root):
        frozen, digest = old_freeze(plan, root)
        require(frozen["source_sha256"]["scripts/cnn_adoption_compare.py"] == meta["original_source_sha256"], "Adaptor original is not frozen comparator")
        prepared = local(frozen["prepared_root"], DEST / "prepared")
        for role, expected in frozen["data_membership"].items():
            require(role in ("benign_calibration", "confirmation"), "Unexpected reserved role")
            folder = prepared / role
            require(sha(folder / "manifest.json") == expected["manifest_sha256"], "Frozen manifest changed")
            require(CORE <= set(expected["artifact_sha256"]), "Frozen core inputs absent")
            for name, value in expected["artifact_sha256"].items():
                require(Path(name).name == name and sha(folder / name) == value, "Frozen artifact bytes changed: " + role + "/" + name)
        return frozen, digest

    ns["_validate_freeze"] = all_membership_before_rows
    return ns, meta


def synthetic(ns):
    full = {name: name + "-digest" for name in CORE | {"content_start.npy", "document_content_length.npy"}}
    core = {name: full[name] for name in CORE}
    require(recorded_files_match(core, full), "Valid four/six membership rejected")
    require(recorded_files_match(full, full), "Valid six/six membership rejected")
    missing = dict(core)
    missing.pop("ids.npy")
    changed = dict(core, **{"ids.npy": "corrupt"})
    extra = dict(core, **{"unknown.npy": "unknown"})
    for corrupt in (missing, changed, extra):
        require(not recorded_files_match(corrupt, full), "Corrupt recorded membership accepted")
    ns["_self_test"]()
    # Preserve the original historical seed guard: anchor path has no seed.
    ns["_pair_seed_agrees"]({"seed": 1337},
                           {"association": {"target": {"checkpoint_identity": {"path": "runs/local_final_s1337_r2/checkpoint_latest.pt"}}}},
                           {"association": {"target": {"checkpoint_identity": {"path": "checkpoints/quantized/qat_w8_a8"}}}})
    return {"complete": True, "synthetic_only": True, "reserved_rows_decoded": False,
            "corruptions_rejected": ["missing core file", "mismatched recorded hash", "extra unknown file"],
            "original_comparator_self_test": "passed", "unmodified_historical_seed_guard": "passed synthetic no-seed anchor path"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--opaque-freeze-check", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--pairs")
    parser.add_argument("--output")
    parser.add_argument("--saved-score-check")
    parser.add_argument("--confirmation-scores-released", action="store_true")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    start = time.monotonic()
    if not args.self_test and not args.opaque_freeze_check:
        require(args.confirmation_scores_released, "Reserved comparisons wait for explicit completed-sixteen-target release")
        require(args.saved_score_check and args.plan and args.pairs and args.output, "Plan/pairs/output/independent saved-score check required")
    ns, provenance = load_adaptor()
    if args.self_test:
        result = synthetic(ns)
    elif args.opaque_freeze_check:
        require(args.plan, "Plan required for opaque freeze integrity")
        frozen, digest = ns["_validate_freeze"](read(local(args.plan)), ROOT)
        result = {"complete": True, "opaque_membership_only": True, "reserved_rows_decoded": False,
                  "freeze_sha256": digest, "targets": len(frozen["nomination"]["targets"]),
                  "all_frozen_artifacts_and_manifests_hashed": True}
    else:
        output = local(args.output, DEST)
        require(not output.exists(), "Preserve existing analysis output")
        plan = read(local(args.plan))
        checked = read(local(args.saved_score_check, DEST))
        require(checked["complete"] is True and checked["targets_expected"] == checked["targets_passed"] == 16
                and checked["freeze_sha256"] == plan["confirmation_freeze_sha256"]
                and not checked["errors"], "Complete independent sixteen-target saved-score audit required")
        pairs = read(local(args.pairs, DEST))["pairs"]
        require(pairs and all(p.get("stage") == "confirmation" for p in pairs), "Confirmation-only pair manifest required")
        require(args.reps == 1000 and args.seed == 20261014, "Use declared fixed 1000-draw sensitivity comparison")
        summary = ns["compare"](args)
        result = read(output)
        result["analysis_only_adaptor"] = {**provenance, "saved_score_check_sha256": sha(local(args.saved_score_check, DEST)),
                                           "adaptor_sha256": sha(Path(__file__)), "all_frozen_input_bytes_verified_before_rows": True}
        result["adaptor_process_internal_wall_seconds"] = time.monotonic() - start
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        require(sha(ORIGINAL) == provenance["original_source_sha256"], "Frozen on-disk comparator changed")
        print(json.dumps(summary))
        return
    result.update(cpu_wall_seconds=time.monotonic() - start, provenance=provenance, adaptor_sha256=sha(Path(__file__)))
    if args.output:
        output = local(args.output, DEST)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("provenance",)}))


if __name__ == "__main__":
    main()
