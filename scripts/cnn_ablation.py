"""Generate a capacity-aware conventional replacement for local_context CNN.

This CPU-only helper does not fit models, select a winner, freeze a learning
recipe, change the registry, or infer FPGA performance. Width matching uses
only analytical parameter/MAC counts and retains the input representation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from check_cnn_models import check
from cnn_cost import registry_receptive_field
from cnn_models import build_model, load_search_configs, model_cost

ROOT = Path(__file__).resolve().parents[1]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def replacement_config(source, channels):
    result = copy.deepcopy(source)
    kernels, dilations = [], []
    for i in range(source["depth"]):
        kernels.extend((source["kernel_sizes"][0], source.get("context_kernel", 7), 1))
        dilations.extend((1, source.get("dilations", [1])[i % len(source.get("dilations", [1]))], 1))
    result.update(id=f"{source['id']}_residual_replace_c{channels}", family="residual",
                  channels=channels, depth=len(kernels), kernel_sizes=kernels,
                  dilations=dilations, gated=False)
    result.pop("context_kernel", None)
    result["ablation"] = {"source_config_id": source["id"],
                          "kind": "ordinary sequential residual replacement of local/context concatenation and fusion"}
    return result


def costs(config, vocab, pad, length):
    # Seed makes helper construction reproducible. Counts do not use learned
    # scores or labels and are unaffected by these initial random weights.
    torch.manual_seed(313)
    model = build_model(config, vocab, pad).cpu()
    result = model_cost(model, sequence_length=length, batch_size=1)
    del model
    return result


def difference(source, replacement):
    keys = ("parameters", "embedding_parameters", "nonembedding_parameters", "parameter_bytes_fp32",
            "parameter_bytes_ideal_int8", "conv_linear_macs", "largest_module_output_elements")
    return {key: {"source": source[key], "replacement": replacement[key],
                  "absolute_difference": replacement[key] - source[key],
                  "relative_difference": (replacement[key] - source[key]) / source[key] if source[key] else None}
            for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--pad-id", type=int, default=0)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--width-radius", type=int, help="Default ceil(25%% source width), minimum 8")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    configs = {c["id"]: c for c in load_search_configs()}
    source = configs[args.config_id]
    if source["family"] != "local_context":
        raise ValueError("This ablation helper accepts only local_context configurations")
    vocab = args.vocab_size if args.vocab_size is not None else (257 if source["representation"] == "byte" else 30522)
    length = args.sequence_length if args.sequence_length is not None else (1024 if source["representation"] == "byte" else 256)
    if length < 1:
        raise ValueError("sequence length must be positive")
    width = source["channels"]
    radius = args.width_radius if args.width_radius is not None else max(8, math.ceil(.25 * width))
    if not 1 <= radius <= width:
        raise ValueError("width radius must be between 1 and source width")
    target = costs(source, vocab, args.pad_id, length)
    trials = []
    for candidate_width in range(max(8, width - radius), width + radius + 1):
        candidate = replacement_config(source, candidate_width)
        estimate = costs(candidate, vocab, args.pad_id, length)
        p_error = abs(estimate["nonembedding_parameters"] / target["nonembedding_parameters"] - 1)
        m_error = abs(estimate["conv_linear_macs"] / target["conv_linear_macs"] - 1)
        trials.append({"channels": candidate_width, "nonembedding_parameters": estimate["nonembedding_parameters"],
                       "conv_linear_macs": estimate["conv_linear_macs"], "parameter_relative_error": p_error,
                       "mac_relative_error": m_error, "match_key": [max(p_error, m_error), p_error + m_error,
                                                                   abs(candidate_width - width), candidate_width]})
    chosen = min(trials, key=lambda row: row["match_key"])
    generated = replacement_config(source, chosen["channels"])
    matched = costs(generated, vocab, args.pad_id, length)
    source_rf, replacement_rf = registry_receptive_field(source), registry_receptive_field(generated)
    for field in ("max_receptive_field_tokens", "max_lookahead_tokens"):
        assert source_rf[field] == replacement_rf[field], field
    assert target["embedding_parameters"] == matched["embedding_parameters"]
    # Existing focused mask/gradient checker uses synthetic vocab257 and pad0.
    # It checks the generated topology at its selected width with real forward,
    # backward and AdamW update, not only the analytical matcher.
    numerical = check(generated)
    report = {"purpose": "capacity-aware conventional mechanism replacement proposal; no learned outcome or perfect isolation claim",
              "source_config": source, "source_config_identity_sha256": identity(source),
              "configuration": generated, "configuration_identity_sha256": identity(generated),
              "extra_configs": [generated], "source_cost": target, "replacement_cost": matched,
              "exact_count_differences": difference(target, matched),
              "source_receptive_field": source_rf, "replacement_receptive_field": replacement_rf,
              "matching": {"rule": "minimize largest absolute relative deviation of nonembedding parameters and Conv/Linear MACs; then sum, width change, width",
                           "candidate_width_min": max(8, width - radius), "candidate_width_max": width + radius,
                           "length": length, "batch": 1, "vocab_size": vocab, "pad_id": args.pad_id,
                           "selected": chosen, "trials": trials, "learned_scores_used": False},
              "preserved": ["embedding dimension and vocabulary", "representation", "maximum local receptive field and centered lookahead",
                            "token resolution", "evidence pooling type/top-k", "normalization type", "dropout probability", "training defaults"],
              "remaining_confounds": [
                  "Channel width may differ to approximate capacity; embedding storage remains identical but projected/channel feature dimensions change.",
                  "Concatenating local and contextual features followed by fusion is replaced with sequential residual temporal and pointwise convolutions.",
                  "Source local conv has no residual, contextual conv has a residual and fusion has an outer residual; replacement has a residual after every local/context/pointwise conv.",
                  "Both structures have three per-position normalization modules per original block, but placement and inputs differ.",
                  "Replacement pointwise block adds ReLU and dropout; source fusion uses ReLU or GLU followed by normalization and no fusion dropout.",
                  "GLU fusion is removed when source gated=True; Conv/Linear MAC matching excludes sigmoid, normalization, pooling, residual additions and data movement.",
                  "Parameter/MAC mismatch is reported exactly; largest single module tensor is not simultaneous activation storage or allocator peak.",
                  "Equal maximum receptive field does not imply equal effective receptive field or distributions of short skip paths.",
                  "Helper copies training defaults for use in a comparable fit; root must prespecify learning, seeds and comparison before dispatch."],
              "source_is_gated": source.get("gated", False), "numerical_mask_gradient_check": numerical,
              "numerical_check_vocabulary": {"vocab_size": 257, "pad_id": 0, "device": "cpu", "threads": 1},
              "evidence_kind": "analytical software tensor/operation counts plus synthetic CPU implementation check",
              "limitations": "No quality, latency, FPGA resource, energy or hardware advantage measurement; no learning recipe frozen or registry modified.",
              "python": sys.version, "torch": torch.__version__, "command": sys.argv,
              "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_source_sha256": hashlib.sha256((ROOT / "scripts/cnn_models.py").read_bytes()).hexdigest(),
              "wall_seconds": time.perf_counter() - started}
    folder = ROOT / "results/cnn_study_v1/ablations" / source["id"]
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("matcher.json", report), ("configuration.json", generated)):
        output = folder / name
        temp = output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temp.replace(output)
    print(json.dumps({"source": source["id"], "replacement": generated["id"], "selected": chosen,
                      "receptive_field": source_rf["max_receptive_field_tokens"], "lookahead": source_rf["max_lookahead_tokens"],
                      "numerical_check_passed": numerical["passed"], "wall_seconds": report["wall_seconds"], "folder": str(folder)}))


if __name__ == "__main__":
    main()
