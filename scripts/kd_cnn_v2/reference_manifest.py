"""Register byte-identical prior CNN controls without opening prior evaluations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

RUNS = (
    "local_context_03_final_s42_r2",
    "local_context_03_residual_replace_c98_final_s42_r2",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.canonical_root.resolve()
    destination = root / "results/kd_cnn_v2/references"
    output = destination / "manifest.json"
    if output.exists():
        raise FileExistsError(output)
    controls = []
    for run in RUNS:
        original = root / "results/cnn_adoption_v1/runs" / run / "checkpoint_latest.pt"
        copied = destination / run / "checkpoint_latest.pt"
        original_hash, copied_hash = sha(original), sha(copied)
        if original_hash != copied_hash or original.stat().st_size != copied.stat().st_size:
            raise ValueError(f"Reference bytes differ: {run}")
        checkpoint = torch.load(copied, map_location="cpu", weights_only=False)
        spec, recipe = checkpoint["model_spec"], checkpoint["recipe"]
        if spec["kind"] != "cnn" or recipe["run_id"] != run or recipe["seed"] != 42:
            raise ValueError(f"Unexpected reference checkpoint identity: {run}")
        if checkpoint["data_identity"]["role"] != "fit":
            raise ValueError(f"Reference FIT provenance differs: {run}")
        controls.append({
            "id": f"prior_{spec['config_id']}_s42",
            "kind": "cnn_fp32",
            "source_checkpoint": original.relative_to(root).as_posix(),
            "checkpoint": copied.relative_to(root).as_posix(),
            "checkpoint_sha256": copied_hash,
            "checkpoint_bytes": copied.stat().st_size,
            "source_run_id": run,
            "model_config_id": spec["config_id"],
            "model_spec_config": spec["config"],
            "original_fixed_seed": recipe["seed"],
            "original_parent_development_run": recipe.get("parent_development_run_id"),
            "original_training_data_role": checkpoint["data_identity"]["role"],
            "original_training_data_artifact_sha256": checkpoint["data_identity"]["files"],
            "original_objective": recipe["objective"],
            "original_kd_weight": recipe.get("distillation_weight"),
            "original_kd_temperature": recipe.get("temperature"),
            "original_teacher_checkpoint_sha256": recipe.get("teacher_checkpoint_sha256"),
            "original_embedding_initialization": recipe.get("embedding_initialization"),
        })
    manifest = {
        "schema": "kd_cnn_v2_prior_controls_v1",
        "purpose": "Fixed prior LocalContext and C98 controls on the same new panel; no retraining or checkpoint editing.",
        "controls": controls,
        "exposure_boundary": "Both checkpoints were trained on original r2 FIT and its fit-only teacher cache at fixed seed 42; the LocalContext checkpoint also uses prior raw-pretrained PCA embedding initialization, while C98 does not. Neither used the new fit_transfer data. Their original historical evaluations are not read or used to choose these controls. The new selection/calibration/confirmation source and authored payload ancestors are disjoint from old role assignments, subject to the documented teacher pretraining and mechanism limitations.",
        "reuse_rule": "Evaluate these byte-identical checkpoints under the same new-panel tokenization, document aggregation, calibration, and timing boundary as new kd_cnn_v2 candidates. Do not treat prior old-panel scores as new-panel outcomes.",
        "target_list_entries": [
            {"id": c["id"], "kind": c["kind"], "checkpoint": c["checkpoint"],
             "checkpoint_sha256": c["checkpoint_sha256"], "seed": c["original_fixed_seed"]}
            for c in controls
        ],
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "controls": [c["id"] for c in controls], "sha256": sha(output)}))


if __name__ == "__main__":
    main()
