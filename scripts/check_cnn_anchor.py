"""Focused CPU same-weight TextCNN/R1 equivalence and training-code audit.

Synthetic inputs only; no historical learning rerun, dataset, CUDA or checkpoint
loading is required. Architecture equality is distinct from training identity.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from cnn_models import build_model, load_search_configs
from r1_models import TokenCNN

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/cnn_study_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    config = next(c for c in load_search_configs() if c["id"] == "textcnn_01")
    assert config["embedding_dim"] == config["channels"] == 64
    assert config["kernel_sizes"] == [3, 5, 7] and config["pooling"] == "max"
    assert config["normalization"] == "none" and config["dropout"] == .2
    torch.manual_seed(173)
    reference = TokenCNN(30522, 0).cpu()
    anchor = build_model(config, 30522, 0).cpu()
    assert anchor.projection is None
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in anchor.parameters()) == 2015426
    reference_state = reference.state_dict()
    mapping = {"embedding.weight": "embedding.weight", "classifier.weight": "classifier.weight", "classifier.bias": "classifier.bias"}
    for i in range(3):
        for field in ("weight", "bias"):
            mapping[f"blocks.{i}.conv.{field}"] = f"convs.{i}.{field}"
    assert set(mapping) == set(anchor.state_dict()) and set(mapping.values()) == set(reference_state)
    anchor.load_state_dict({target: reference_state[source] for target, source in mapping.items()})
    assert all(torch.equal(anchor.state_dict()[target], reference_state[source]) for target, source in mapping.items())
    weight_hashes = {target: hashlib.sha256(anchor.state_dict()[target].numpy().tobytes()).hexdigest() for target in mapping}
    torch.manual_seed(907)
    recovered, tests = {}, []
    for index, lengths in enumerate(([1, 7, 17], [2, 3, 7, 31, 128, 256])):
        length = max(lengths)
        ids = torch.randint(1, 30522, (len(lengths), length))
        mask = torch.zeros_like(ids)
        for row, valid in enumerate(lengths):
            mask[row, :valid] = 1
        ids.masked_fill_(mask == 0, 0)
        for mode in ("eval", "train_same_dropout_rng"):
            reference.train(mode != "eval"); anchor.train(mode != "eval")
            with torch.no_grad():
                torch.manual_seed(145)
                left = reference(ids, mask, token_type_ids=torch.zeros_like(ids)).logits
                torch.manual_seed(145)
                right = anchor(ids, mask, token_type_ids=torch.zeros_like(ids)).logits
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
            tests.append({"case": index, "valid_lengths": lengths, "mode": mode,
                          "passed": True, "bitwise_equal": torch.equal(left, right),
                          "maximum_absolute_logit_difference": float((left - right).abs().amax())})
            recovered[f"case_{index}_{mode}_reference_logits"] = left.numpy()
            recovered[f"case_{index}_{mode}_anchor_logits"] = right.numpy()
        recovered[f"case_{index}_input_ids"] = ids.numpy()
        recovered[f"case_{index}_attention_mask"] = mask.numpy()
    reference.eval(); anchor.eval()
    with torch.no_grad():
        empty_ids = torch.zeros((2, 8), dtype=torch.long)
        empty_mask = torch.zeros_like(empty_ids)
        old_empty = reference(empty_ids, empty_mask).logits
        new_empty = anchor(empty_ids, empty_mask).logits
        new_zero_length = anchor(torch.empty((2, 0), dtype=torch.long)).logits
        torch.testing.assert_close(new_empty, anchor.classifier.bias.expand(2, -1), atol=0, rtol=0)
        torch.testing.assert_close(new_zero_length, new_empty, atol=0, rtol=0)
        assert torch.isfinite(new_empty).all()
        try:
            reference(torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.long))
        except RuntimeError as error:
            old_zero_length = {"accepted": False, "exception": type(error).__name__, "message": str(error)}
        else:
            old_zero_length = {"accepted": True}
    recovered["historical_empty_mask_logits"] = old_empty.numpy()
    recovered["anchor_empty_mask_logits"] = new_empty.numpy()
    output_tensors = DEST / "anchor_equivalence_tensors.npz"
    np.savez_compressed(output_tensors, **recovered)
    recipe = None
    tree = ast.parse((ROOT / "scripts/r1_challengers.py").read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "RECIPES" for target in node.targets):
            recipe = ast.literal_eval(node.value)["cnn"]
    assert recipe is not None
    current_anchor_records = []
    for folder in (DEST / "runs").glob("*textcnn_01*"):
        metadata = folder / "run.json"
        if metadata.exists():
            row = json.loads(metadata.read_text(encoding="utf-8"))
            current_anchor_records.append({"run_id": folder.name, "spec": row["spec"], "complete_record_present": (folder / "complete.json").exists(),
                                           "recorded_selection": row["selection"], "run_metadata_sha256": digest(metadata)})
    planned = json.loads((ROOT / "configs/cnn_stage_b_v1.json").read_text(encoding="utf-8"))
    planned_anchor = [s for s in planned["runs"] if s["config_id"] == "textcnn_01"]
    cache_check = DEST / "cudnn_cache_check.json"
    result = {"passed": True, "purpose": "same-weight model equivalence and inspected training differences; not a historical training reproduction or quality comparison",
              "device": "cpu", "threads": 1, "torch": torch.__version__, "python": sys.version, "synthetic_inputs_only": True,
              "seeds": {"reference_parameters": 173, "synthetic_inputs": 907, "controlled_dropout": 145},
              "config": config, "parameters_each": 2015426, "weight_mapping_anchor_to_reference": mapping,
              "matched_weight_sha256": weight_hashes, "numerical_tolerance_absolute_relative": [1e-6, 1e-6], "tests": tests,
              "architecture": "Both: learned vocabulary 30522 x 64 embedding; independent Conv1d 64->64 kernels 3/5/7, same centered padding, ReLU, valid-center global max each branch, concatenation width 192, dropout 0.2, Linear 192->2. New plain Conv wrappers explicitly pad/mask and concatenate features before channelwise max; this preserves the nonempty function with matched weights.",
              "empty_extension": {"historical_all_masked_logits_finite": bool(torch.isfinite(old_empty).all()),
                  "historical_zero_length": old_zero_length, "anchor_all_masked_logits_finite": True,
                  "anchor_zero_length_logits_equal_head_bias": True,
                  "explanation": "Historical all-masked branch maxima are -inf, producing nonfinite head outputs; zero-length convolution may raise. New model maps an empty summary to zero, returning finite head bias and accepts zero-length tensors. Ordinary tokenized empty text has CLS/SEP and is not an all-masked tensor."},
              "training_shared": {"objective": "Span-label window cross-entropy averaged within each document, then across actual document batch; same 1/(batch_documents*document_window_count) weight.",
                  "optimizer": "AdamW", "anchor_learning_rate": .001, "weight_decay": .01, "document_batch": 32,
                  "gradient_clip_norm": 5., "dropout": .2, "arithmetic": "FP32 native CUDA; TF32 disabled, deterministic algorithms enabled, cuDNN benchmark disabled in both runners"},
              "historical_recipe": recipe, "current_anchor_run_records": current_anchor_records, "stage_b_prespecified_anchor": planned_anchor,
              "training_differences": [
                  "Document order: historical random.Random(42+epoch).shuffle over Python list versus NumPy default_rng(seed+epoch).permutation. Identical integer seeds do not imply identical permutations.",
                  "Window microbatch: historical 64 versus study 128. Both sum the same document-balanced objective before one update, but floating reduction order/kernel shapes and training dropout draws can differ.",
                  "Data exposure: historical canonical train; current screening/StageB uses source-disjoint screen_fit/fit. Frozen finalists may use full canonical training only after selection. Historical checkpoints exposed to inner folds cannot serve as unexposed inner comparators.",
                  "Epoch/checkpoint policy: R1 fixed final epoch 10, no quality checkpoint selection. Study Stage A timing uses 3 final epochs; planned Stage B anchor uses 12 epochs and inner checkpoint selection. Frozen confirmation fits use their exact prespecified final epochs with no in-fit quality selection.",
                  "Cache: historical runner does not set cuDNN LRU capacity and historical metadata does not establish its launch value; current runner sets a default 64 before torch import. This is a resource control, not evidence by itself of bitwise training replay.",
                  "Study adds resume/spec/source/data identity guards, atomic checkpoint replacement and epoch-boundary Torch RNG restoration. Historical train starts fresh and stores RNG metadata without this resume path.",
                  "Same architecture and objective do not establish matching learned weights, optimization trajectories or reproduced historical quality; no learning campaign was run by this check."],
              "bounded_cache_numerical_evidence": {"path": str(cache_check.relative_to(ROOT)), "sha256": digest(cache_check),
                  "all_recorded_tensors_bitwise_equal": json.loads(cache_check.read_text())["all_tensors_bitwise_equal"],
                  "boundary": "Separate default-versus-64 diagnostic on recorded representative forward/loss/backward shapes only; does not establish whole training replay."} if cache_check.exists() else None,
              "tensor_artifact": {"path": str(output_tensors.relative_to(ROOT)), "sha256": digest(output_tensors)},
              "source_sha256": {name: digest(ROOT / "scripts" / name) for name in ("r1_models.py", "cnn_models.py", "r1_challengers.py", "cnn_study.py", "check_cnn_anchor.py")},
              "wall_seconds": time.perf_counter() - started}
    write(DEST / "anchor_equivalence_check.json", result)
    print(json.dumps({"passed": True, "tests": tests, "empty_extension": result["empty_extension"], "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
