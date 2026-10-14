"""Focused FIT-only check of CNN adoption calibration length strata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cnn_adoption_export import calibration_indices  # noqa: E402


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def length_bin(length: int) -> str:
    return "0-127" if length < 128 else "128-255" if length < 256 else "256-511" if length < 512 else "512+"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path,
                        default=ROOT / "results/cnn_adoption_v1/prepared/token_r2/fit")
    parser.add_argument("--windows", type=int, default=512)
    args = parser.parse_args()
    directory = args.calibration_dir if args.calibration_dir.is_absolute() else ROOT / args.calibration_dir
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("role") != "fit":
        raise AssertionError(f"Expected FIT role, found {manifest.get('role')!r}")
    documents_path = directory / "documents.jsonl"
    documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["document_id"]: row for row in documents}
    if any("content_tokens" not in row for row in documents):
        raise AssertionError("R2 prepared FIT metadata must provide content_tokens for every document")

    _, _, sample = calibration_indices(directory, args.windows)
    _, _, repeated = calibration_indices(directory, args.windows)
    if sample["calibration_id_sha256"] != repeated["calibration_id_sha256"]:
        raise AssertionError("Calibration selection is not deterministic")
    expected_strata = set()
    expected_bins = set()
    for record in sample["windows"]:
        source = by_id[record["document_id"]]
        token_count = int(source["content_tokens"])
        if record.get("content_tokens") != token_count:
            raise AssertionError(f"Selected record lost its R2 content_tokens value: {record['document_id']}")
        token_bin = length_bin(token_count)
        expected_bins.add(token_bin)
        genre = source.get("genre") or source.get("source_dataset") or "unspecified"
        expected_strata.add("|".join((str(source.get("source_dataset", "unspecified")), str(genre), token_bin)))
    if sorted(expected_strata) != sample["strata"]:
        raise AssertionError("Serialized calibration strata do not match selected FIT document metadata")
    if len(expected_bins) < 2:
        raise AssertionError(f"Sample did not exercise multiple content length bins: {sorted(expected_bins)}")

    print(json.dumps({
        "complete": True,
        "kind": "FIT-only calibration sampler metadata check; no model inference",
        "prepared_role": manifest["role"],
        "documents": len(documents),
        "selected_windows": sample["selected_windows"],
        "selected_source_groups": sample["source_group_count"],
        "length_bins_exercised": sorted(expected_bins),
        "strata_count": len(sample["strata"]),
        "calibration_id_sha256": sample["calibration_id_sha256"],
        "documents_jsonl_sha256": file_sha(documents_path),
        "deterministic_repeat": True,
        "boundary": "Reads prepared FIT metadata and arrays only; performs no forward pass and does not access evaluation, development, or confirmation partitions.",
    }, indent=2))


if __name__ == "__main__":
    main()
