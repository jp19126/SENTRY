"""CPU-only static W8/A8 CNN export and integer-accumulation reference.

This is a narrow research serializer for scripts/cnn_models.py, not a general
deployment runtime. Convolution/linear inputs and weights are signed symmetric
8-bit. Dot products use checked int32 accumulation (computed in int64 on CPU),
then FP32 rescale and FP32 bias. Embeddings use per-row W8; LayerNorm, pooling,
gates, activations, residual adds, masks and document/window aggregation remain
explicit FP32/reference operations. No FPGA or native-speed claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_models import build_model, model_cost  # noqa: E402

QMAX = 127
ALIGN = 64


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_run(run_dir: Path, checkpoint: Path | None):
    run_file = run_dir / "run.json"
    run = json.loads(run_file.read_text(encoding="utf-8-sig"))
    ckpt = checkpoint or run_dir / "checkpoint_latest.pt"
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = state.get("model_spec", {})
    config = run.get("config") or spec.get("config") or state.get("model_config")
    if config is None and isinstance(run.get("model_spec"), dict):
        config = run["model_spec"].get("config")
    if config is None:
        raise ValueError("No CNN config in run.json or checkpoint model_spec/model_config")
    model_kind = spec.get("kind", run.get("model_kind", "cnn"))
    if model_kind != "cnn":
        raise ValueError(f"Exporter is intentionally CNN-only; model_kind={model_kind!r}")
    rep = config.get("representation")
    vocab = 257 if rep == "byte" else 30522
    model = build_model(config, vocab, pad_id=0).cpu().eval()
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=True)
    return model, config, run, state, ckpt


def _stable_order(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calibration_indices(cal_dir: Path, max_windows: int):
    """Deterministic group/genre/length-stratified FIT-only window sample.

    Labels are read only as inert metadata and never consulted by selection or
    scale calculation. One window per original source group is selected before
    second windows, reducing repeated-variant domination.
    """
    docs_path = cal_dir / "documents.jsonl"
    ids_path, masks_path = cal_dir / "ids.npy", cal_dir / "masks.npy"
    starts_path, manifest_path = cal_dir / "starts.npy", cal_dir / "manifest.json"
    required = (docs_path, ids_path, masks_path, starts_path, manifest_path)
    if not all(p.exists() for p in required):
        raise FileNotFoundError(f"Calibration directory lacks required prepared FIT files: {cal_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if cal_dir.name not in {"fit", "full_train", "screen_fit"} and "fit" not in str(cal_dir).lower():
        raise ValueError("Activation calibration must come from an explicitly named FIT partition")
    docs = [json.loads(s) for s in docs_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    starts = np.load(starts_path, mmap_mode="r", allow_pickle=False)
    groups: dict[str, list[tuple[str, int, int, int]]] = {}
    for di, d in enumerate(docs):
        lo, hi = int(starts[di]), int(starts[di + 1])
        gid = d.get("source_group_id") or d["document_id"]
        length = int(d.get("content_token_count", d.get("content_length", 0)))
        length_bin = "0-127" if length < 128 else "128-255" if length < 256 else "256-511" if length < 512 else "512+"
        genre = d.get("genre") or d.get("source_dataset") or "unspecified"
        stratum = "|".join((str(d.get("source_dataset", "unspecified")), str(genre), length_bin))
        groups.setdefault(gid, []).append((stratum, di, lo, hi))
    by_stratum: dict[str, list[tuple[str, list[tuple[int, int, int]]]]] = {}
    for gid, rows in groups.items():
        stratum = rows[0][0]
        doc_rows = [(di, lo, hi) for _, di, lo, hi in rows]
        doc_rows.sort(key=lambda x: _stable_order(docs[x[0]]["document_id"]))
        by_stratum.setdefault(stratum, []).append((gid, doc_rows))
    # Each stratum sequence starts with one window from every source group,
    # followed by extra documents/windows. Round-robin among strata keeps small
    # genres represented before long-document tails dominate the calibration.
    stratum_sequences = {}
    for stratum, rows in by_stratum.items():
        rows.sort(key=lambda x: _stable_order(x[0]))
        first, extras = [], []
        for gid, group_docs in rows:
            di, lo, hi = group_docs[0]
            first.append((lo, di, 0))
            for other_di, other_lo, other_hi in group_docs[1:]:
                extras.extend((wi, other_di, 1) for wi in range(other_lo, other_hi))
            for wi in range(lo + 1, hi):
                extras.append((wi, di, 1))
        stratum_sequences[stratum] = first + extras
    strata = sorted(stratum_sequences, key=_stable_order)
    chosen: list[tuple[int, int, int]] = []
    chosen_docs: set[str] = set()
    for round_index in range(max((len(v) for v in stratum_sequences.values()), default=0)):
        for stratum in strata:
            seq = stratum_sequences[stratum]
            if round_index < len(seq):
                wi, di, selection_round = seq[round_index]
                chosen.append((wi, di, selection_round))
                chosen_docs.add(docs[di]["document_id"])
                if len(chosen) >= max_windows:
                    break
        if len(chosen) >= max_windows:
            break
    if not chosen:
        raise ValueError("No calibration windows selected")
    chosen.sort(key=lambda row: (row[1], row[0]))
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(masks_path, mmap_mode="r", allow_pickle=False)
    win_ids = [int(x[0]) for x in chosen]
    x = torch.as_tensor(np.asarray(ids[win_ids], dtype=np.int64))
    mask = torch.as_tensor(np.asarray(masks[win_ids], dtype=np.int64))
    records = []
    for wi, di, round_index in chosen:
        d = docs[di]
        records.append({k: d.get(k) for k in ("document_id", "source_group_id", "source_dataset", "genre", "text_sha256", "content_token_count", "content_length")}
                       | {"window_index": int(wi - int(starts[di])), "round": round_index})
    sources = {p.name: sha(p) for p in required}
    selection = {
        "partition_role": "FIT only; no calibration/scoring/confirmation inputs",
        "selection": "deterministic round-robin across source_dataset x genre(if present) x content-length strata; source-group grouped; labels not read for selection",
        "maximum_windows": max_windows,
        "selected_windows": len(records),
        "selected_documents": len(chosen_docs),
        "source_group_count": len({r["source_group_id"] or r["document_id"] for r in records}),
        "strata": sorted({"|".join((str(r["source_dataset"] or "unspecified"), str(r["genre"] or r["source_dataset"] or "unspecified"), ("0-127" if int(r.get("content_token_count") or r.get("content_length") or 0) < 128 else "128-255" if int(r.get("content_token_count") or r.get("content_length") or 0) < 256 else "256-511" if int(r.get("content_token_count") or r.get("content_length") or 0) < 512 else "512+"))) for r in records}),
        "prepared_manifest": manifest,
        "input_file_sha256": sources,
        "calibration_id_sha256": canonical_sha(records),
        "windows": records,
    }
    return x, mask, selection


def calibrate_input_scales(model: nn.Module, x: torch.Tensor, mask: torch.Tensor, batch: int = 32):
    maxima: dict[str, float] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            def observe(mod, inputs, key=name):
                value = inputs[0].detach().float()
                maxima[key] = max(maxima.get(key, 0.0), float(value.abs().max()))
            handles.append(module.register_forward_pre_hook(observe))
    torch.set_num_threads(1)
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            model(x[start:start + batch], mask[start:start + batch])
    for h in handles:
        h.remove()
    return {name: max(maxabs / QMAX, 1.0e-12) for name, maxabs in maxima.items()}, maxima


def quantize_weight(weight: torch.Tensor):
    axes = tuple(range(1, weight.ndim))
    scale = weight.detach().float().abs().amax(dim=axes, keepdim=True) / QMAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(weight.detach().float() / scale).clamp(-QMAX, QMAX).to(torch.int8)
    return codes.contiguous(), scale.contiguous()


def quantize_activation(x: torch.Tensor, scale: float):
    return torch.round(x.float() / scale).clamp(-QMAX, QMAX).to(torch.int8)


class IntegerConv1d(nn.Module):
    """CPU exact integer dot reference for unpadded grouped Conv1d calls."""
    def __init__(self, original: nn.Conv1d, input_scale: float,
                 weight_codes: torch.Tensor | None = None,
                 weight_scales: torch.Tensor | None = None,
                 bias_fp32: torch.Tensor | None = None):
        super().__init__()
        if original.stride != (1,) or original.padding != (0,) or original.padding_mode != "zeros":
            raise ValueError("Reference supports the CNN registry's stride-1, externally padded Conv1d")
        self.in_channels, self.out_channels = original.in_channels, original.out_channels
        self.kernel_size, self.dilation, self.groups = original.kernel_size[0], original.dilation[0], original.groups
        self.input_scale = float(input_scale)
        codes, scales = quantize_weight(original.weight) if weight_codes is None or weight_scales is None else (weight_codes, weight_scales)
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scales", scales.reshape(-1))
        bias = bias_fp32 if bias_fp32 is not None else original.bias.detach().float().clone() if original.bias is not None else torch.empty(0)
        self.register_buffer("bias_fp32", bias.float().clone())
        self.has_bias = original.bias is not None
        self.bound = (self.in_channels // self.groups) * self.kernel_size * QMAX * QMAX
        if self.bound > (1 << 31) - 1:
            raise OverflowError(f"Conv int32 worst-case bound exceeds signed int32: {self.bound}")

    def forward(self, x):
        qx = quantize_activation(x, self.input_scale).to(torch.int64)
        eff = (self.kernel_size - 1) * self.dilation + 1
        windows = qx.unfold(-1, eff, 1)[..., ::self.dilation]
        n, _, positions, _ = windows.shape
        cg, og = self.in_channels // self.groups, self.out_channels // self.groups
        outputs = []
        for g in range(self.groups):
            patches = windows[:, g * cg:(g + 1) * cg].permute(0, 2, 1, 3).reshape(n, positions, cg * self.kernel_size)
            weights = self.weight_codes[g * og:(g + 1) * og].reshape(og, -1).to(torch.int64)
            accum64 = torch.matmul(patches, weights.T)
            if bool((accum64.abs() > (1 << 31) - 1).any()):
                raise OverflowError("Observed signed int32 Conv accumulator overflow")
            outputs.append(accum64.to(torch.int32).to(torch.float32).transpose(1, 2))
        acc = torch.cat(outputs, dim=1)
        out = acc * (self.input_scale * self.weight_scales).view(1, -1, 1)
        if self.has_bias:
            out = out + self.bias_fp32.view(1, -1, 1)
        return out


class IntegerLinear(nn.Module):
    def __init__(self, original: nn.Linear, input_scale: float,
                 weight_codes: torch.Tensor | None = None,
                 weight_scales: torch.Tensor | None = None,
                 bias_fp32: torch.Tensor | None = None):
        super().__init__()
        self.in_features, self.out_features = original.in_features, original.out_features
        self.input_scale = float(input_scale)
        codes, scales = quantize_weight(original.weight) if weight_codes is None or weight_scales is None else (weight_codes, weight_scales)
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scales", scales.reshape(-1))
        bias = bias_fp32 if bias_fp32 is not None else original.bias.detach().float().clone() if original.bias is not None else torch.empty(0)
        self.register_buffer("bias_fp32", bias.float().clone())
        self.has_bias = original.bias is not None
        self.bound = self.in_features * QMAX * QMAX
        if self.bound > (1 << 31) - 1:
            raise OverflowError(f"Linear int32 worst-case bound exceeds signed int32: {self.bound}")

    def forward(self, x):
        qx = quantize_activation(x, self.input_scale).to(torch.int64)
        accum64 = torch.matmul(qx, self.weight_codes.to(torch.int64).T)
        if bool((accum64.abs() > (1 << 31) - 1).any()):
            raise OverflowError("Observed signed int32 Linear accumulator overflow")
        out = accum64.to(torch.int32).to(torch.float32) * (self.input_scale * self.weight_scales)
        if self.has_bias:
            out = out + self.bias_fp32
        return out


class IntegerEmbedding(nn.Module):
    def __init__(self, original: nn.Embedding,
                 weight_codes: torch.Tensor | None = None,
                 weight_scales: torch.Tensor | None = None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = original.num_embeddings, original.embedding_dim
        self.padding_idx = original.padding_idx
        codes, scales = quantize_weight(original.weight) if weight_codes is None or weight_scales is None else (weight_codes, weight_scales)
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scales", scales.reshape(-1))

    def forward(self, ids):
        scales = self.weight_scales[ids].unsqueeze(-1)
        return self.weight_codes[ids].to(torch.float32) * scales


def _parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else root
    key = parts[-1]
    if key.isdigit():
        return parent, key
    return parent, key


def _set_child(parent, key, child):
    if key.isdigit():
        parent[int(key)] = child
    else:
        setattr(parent, key, child)


def mapping_assessment(model: nn.Module, manifest: dict):
    """Shape/cost and target-contract comparison; explicitly analytical."""
    project = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    hw = project["hardware"]
    length = 256 if model.study_config.get("representation") != "byte" else 1024
    cnn_ops = model_cost(model, sequence_length=length, batch_size=1)
    cnn_profile = static_checkpoint_mapping(model.study_config, manifest["checkpoint_path"],
                                            manifest["checkpoint_sha256"], sequence_length=length)
    model_path = ROOT / "checkpoints/floating/epoch_3/model.safetensors"
    bert_cfg_path = ROOT / "checkpoints/floating/epoch_3/config.json"
    bert_cfg = json.loads(bert_cfg_path.read_text(encoding="utf-8"))
    try:
        from safetensors import safe_open
        with safe_open(str(model_path), framework="pt", device="cpu") as sf:
            bert_shapes = {k: tuple(sf.get_slice(k).get_shape()) for k in sf.keys()}
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect the saved BERT reference shape without loading data: {exc}") from exc
    bert_params = sum(int(np.prod(shape)) for shape in bert_shapes.values())
    bert_matrices = {k: shape for k, shape in bert_shapes.items() if len(shape) == 2}
    bert_fp32_params = sum(int(np.prod(shape)) for shape in bert_shapes.values() if len(shape) != 2)
    bert_matrix_weights = sum(int(np.prod(s)) for s in bert_matrices.values())
    bert_matrix_scales = sum(s[0] for s in bert_matrices.values())
    bert_dense = {k: s for k, s in bert_matrices.items() if "embeddings." not in k}
    bert_act_scales = len(bert_dense)
    bert_all_w8 = bert_matrix_weights + 4 * (bert_matrix_scales + bert_act_scales + bert_fp32_params)
    old_encoder = {k: s for k, s in bert_dense.items() if k.startswith("bert.encoder.layer.")}
    old_q_weights = sum(int(np.prod(s)) for s in old_encoder.values())
    old_q_scales = sum(s[0] for s in old_encoder.values())
    bert_existing_w8 = old_q_weights + 4 * old_q_scales + 4 * len(old_encoder) + 4 * (bert_params - old_q_weights)

    token_protocol = json.loads((ROOT / "results/cnn_study_v1/prepared/data_protocol.json").read_text(encoding="utf-8"))
    bert_h = int(bert_cfg["hidden_size"])
    bert_i = int(bert_cfg["intermediate_size"])
    bert_layers = int(bert_cfg["num_hidden_layers"])
    bert_linear_macs = bert_layers * length * (4 * bert_h * bert_h + 2 * bert_h * bert_i)
    bert_attention_macs = bert_layers * 2 * length * length * bert_h
    bert_macs = bert_linear_macs + bert_attention_macs + bert_h * bert_h + 2 * bert_h
    cnn_payload = int(manifest["storage"]["payload_bytes"]["combined"])
    payload_reduction_vs_matched = 1 - cnn_payload / bert_all_w8
    input_ids_bytes = length * 4
    input_mask_bytes = length
    # Existing protocol allocates one 64-byte request header and response.
    cnn_packet = 64 + input_ids_bytes + input_mask_bytes + 64
    bert_packet = 64 + length * 4 + length * 4 + length + 64
    uart_bps = 115200
    fp32_weight_bytes = int(cnn_ops["parameter_bytes_fp32"])
    channels = int(model.study_config.get("channels", 64))
    depth = int(model.study_config.get("depth", 1))
    target_total_memory_bits = hw["resource_limits"]["BLOCK_RAMS"] * 36 * 1024 + hw["resource_limits"]["ULTRA_RAMS"] * 288 * 1024
    family = model.study_config["family"]
    kernels = [int(k) for k in model.study_config.get("kernel_sizes", [3])]
    dilations = [int(d) for d in model.study_config.get("dilations", [1])]
    context_kernel = int(model.study_config.get("context_kernel", 7))
    if family == "textcnn":
        topology = {
            "family": family,
            "graph": f"Parallel centered width-{kernels} filter branches, {channels} channels per branch, then global masked pooling and one classifier.",
            "branch_schedule": "Branches may be time-multiplexed or concurrent; concurrent execution retains each branch output through pooling. No route is measured.",
        }
    elif family == "stacked":
        topology = {
            "family": family,
            "graph": f"{depth} serial centered convolution stages at {channels} channels with kernels {kernels}; global masked max pooling then classifier.",
            "branch_schedule": "Sequential feature maps can be reused stage by stage; per-stage FP32 rescale/bias and any intervening operations must be preserved or separately qualified.",
        }
    elif family == "local_context":
        norm_modules = cnn_profile["fp32_layernorm_parameter_count"] // max(1, 2 * channels)
        local_k = kernels[0]
        dilation = dilations[0]
        line_buffer = depth * ((local_k - 1) + (context_kernel - 1) * dilation) * channels
        feature_fp32 = length * channels * 4
        topology = {
            "family": family,
            "graph": (f"{depth} residual blocks; each applies local width-{local_k} ConvBlock, then centered context "
                      f"width-{context_kernel} dilation-{dilation} ConvBlock to the local output, concatenates both "
                      f"{channels}-channel maps, fuses 2C to C with width-1 Conv, applies tokenwise LayerNorm, then adds the block skip."),
            "gated": bool(model.study_config.get("gated", False)),
            "layernorm_module_count": norm_modules,
            "layernorm_affine_parameter_count": cnn_profile["fp32_layernorm_parameter_count"],
            "centered_feature_receptive_field_tokens": cnn_profile["receptive_field_tokens"],
            "centered_feature_lookahead_tokens": cnn_profile["centered_feature_lookahead_tokens"],
            "ideal_A8_delay_line_bytes_lower_bound": line_buffer,
            "one_full_FP32_C_map_bytes": feature_fp32,
            "branch_maps_plus_concat_FP32_if_fully_materialized_bytes": cnn_profile["activation_shape_bounds"]["local_context_branch_outputs_plus_concat_FP32_bytes_if_materialized"],
            "branch_maps_concat_block_input_and_fused_output_FP32_if_all_coexist_bytes": 6 * feature_fp32,
            "norm_and_fusion_schedule": "Each token's LayerNorm needs channel statistics; local/context branch values meet at concatenation/fusion. A streaming schedule may reduce whole-map retention, but no allocator/live-range proof exists.",
        }
    else:
        topology = {
            "family": family,
            "graph": f"{depth} serial centered residual stages at {channels} channels with kernels {kernels} and dilations {dilations}; global masked pooling then classifier.",
            "branch_schedule": "Each residual add must align the skip and transformed output; any LayerNorm/gate stages remain explicit FP32 exceptions in this reference.",
        }
    topology.update({
        "window_length": length,
        "complete_window_required": True,
        "global_pool_summary_FP32_bytes": cnn_profile["activation_shape_bounds"]["global_pool_summary_FP32_bytes"],
        "complete_document_retention": "One FP32 maximum-risk scalar per active document can be retained as window results arrive; the final decision still waits until every window has been processed.",
        "activation_shape_bounds": cnn_profile["activation_shape_bounds"],
        "liveness_boundary": "Analytical named tensor sizes and a schedule hypothesis only; not simultaneous allocation, target memory fit, bandwidth, timing, or synthesis evidence.",
    })
    report = {
        "kind": "analytical target-specific mapping hypothesis; no synthesis or physical measurement",
        "mapping_generator_sha256": sha(Path(__file__)),
        "candidate_export_manifest_object_sha256": canonical_sha(manifest),
        "candidate": {"run_id": manifest["run_id"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                      "representation": model.study_config["representation"], "window_length": length,
                      "parameters": cnn_ops["parameters"], "fp32_parameter_bytes": fp32_weight_bytes,
                      "exported_w8a8_payload_bytes": cnn_payload,
                      "payload_reduction_vs_fp32": 1 - cnn_payload / fp32_weight_bytes,
                      "conv_linear_macs_per_window": cnn_ops["conv_linear_macs"],
                      "analytical_complete_document_window_formula": cnn_profile["document_formula"],
                      "complete_document_score_retention": topology["complete_document_retention"]},
        "bert_shape_reference": {"checkpoint": "checkpoints/floating/epoch_3/model.safetensors",
            "checkpoint_sha256": sha(model_path), "config_sha256": sha(bert_cfg_path),
            "parameters": bert_params, "matrix_parameters": bert_matrix_weights,
            "fp32_vector_parameters": bert_fp32_params, "matrix_output_rows_for_scales": bert_matrix_scales,
            "linear_layers_with_A8_input_scales": bert_act_scales,
            "existing_encoder_only_W8_plus_FP32_rest_payload_bytes": bert_existing_w8,
            "existing_scope_boundary": "Shape-derived ideal packed payload: 24 encoder linears W8/A8, all embeddings/other parameters FP32; actual quantized checkpoint stores FP32 tensors/buffers, not a packed deployment file.",
            "hypothetical_all_2D_weights_W8_payload_bytes": bert_all_w8,
            "matched_payload_comparison": "CNN and BERT both count all 2-D weights W8 with per-output/embedding-row FP32 scales, all vector/bias parameters FP32, and static A8 scale per Conv/Linear input. BERT full model numerical equivalence at this broader quantization scope is unvalidated.",
            "cnn_payload_reduction_vs_hypothetical_all_matrix_w8": payload_reduction_vs_matched,
            "BERT_MACS_per_same_window": {"linear": bert_linear_macs, "attention_qk_and_av": bert_attention_macs,
                "pooler_classifier_approx": bert_h * bert_h + 2 * bert_h,
                "counting_boundary": "Linear plus QK/AV multiply-accumulates, no softmax/LN/embedding/mask costs; analytical for the saved 4-layer H=256 model."}},
        "work_comparison": {"cnn_macs_per_window": int(cnn_ops["conv_linear_macs"]), "bert_macs_per_window": int(bert_macs),
            "ratio_bert_over_cnn": float(bert_macs / max(1, cnn_ops["conv_linear_macs"])),
            "interpretation": "Operation-count hypothesis only; unlike operations, compiler scheduling and data movement are not modeled; no speedup inference."},
        "target": {"part": hw["part"], "board": hw["allocated_board"], "clock_objective_mhz": hw["clock_objective_mhz"],
            "resource_limits_70pct": hw["resource_limits"], "nominal_combined_BRAMECC_capacity_bytes": target_total_memory_bits // 8,
            "interface": hw["interface"], "hls_interfaces": "three m_axi bundles: activations, weights, outputs; AXI4-Lite control; project-specific integration pending for a CNN.",
            "shared_ddr_arena_bytes": 64 * 1024 * 1024,
            "storage_feasibility_boundary": "Exported payload is below nominal arena size and aggregate BRAM/URAM bit capacity, but this is not an allocation, banking, port, bandwidth, timing or free-resource proof."},
        "topology_schedule_hypothesis": topology,
        "shape_profile": cnn_profile,
        "transport": {"reused_BERT_packet_bytes_per_window": bert_packet,
            "hypothetical_CNN_packet_bytes_per_window_uint32_ids_mask_no_token_types": cnn_packet,
            "hypothetical_CNN_packet_reduction_vs_BERT": 1 - cnn_packet / bert_packet,
            "UART8N1_bits_per_byte": 10, "UART_baud": uart_bps,
            "hypothetical_CNN_serialization_ms_per_window": cnn_packet * 10 / uart_bps * 1000,
            "hypothetical_BERT_serialization_ms_per_window": bert_packet * 10 / uart_bps * 1000,
            "contract": "Reuses the recorded L256 header/response sizes and omits uint32 token_type_ids for CNN. No CNN command exists; document batching, per-window response, memory copy/cache and synchronization costs remain unvalidated."},
        "windowing": token_protocol.get("token", token_protocol.get("windowing", {})),
        "limitations": ["This profile is analytical for the checkpoint named in the export manifest; it does not establish model quality, activation-tail representativeness, new-fit calibration adequacy, real matched-BERT W8/A8 quality, or host-to-device complete-chain timing.",
            "No vendor build, physical device activity, power, throughput or resource utilization was performed."],
    }
    return report


def _align(buf: bytearray, boundary: int = ALIGN):
    buf.extend(b"\0" * ((-len(buf)) % boundary))


def _append_tensor(buf: bytearray, tensor: torch.Tensor, dtype: str):
    _align(buf)
    arr = tensor.detach().cpu().numpy()
    if dtype == "int8":
        arr = arr.astype(np.int8, copy=False)
    elif dtype == "float32":
        arr = arr.astype("<f4", copy=False)
    else:
        raise ValueError(dtype)
    offset = len(buf)
    raw = arr.tobytes(order="C")
    buf.extend(raw)
    return {"offset_bytes": offset, "length_bytes": len(raw), "shape": list(arr.shape), "dtype": dtype, "alignment_bytes": ALIGN}


def pack_export(model, input_scales, out_dir: Path, config, checkpoint: Path, run: dict, cal_record, maxima):
    int8_buf, scale_buf, fp32_buf = bytearray(), bytearray(), bytearray()
    tensor_map, ops = {}, []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.Embedding)):
            codes, weight_scales = quantize_weight(module.weight)
            is_embedding = isinstance(module, nn.Embedding)
            code_name = name + ".weight_codes"
            scale_name = name + ".weight_scales"
            tensor_map[code_name] = _append_tensor(int8_buf, codes, "int8")
            tensor_map[scale_name] = _append_tensor(scale_buf, weight_scales.reshape(-1), "float32")
            if not is_embedding:
                activation_name = name + ".input_scale_a8"
                tensor_map[activation_name] = _append_tensor(scale_buf, torch.tensor([input_scales[name]], dtype=torch.float32), "float32")
            bias = getattr(module, "bias", None)
            if bias is not None:
                tensor_map[name + ".bias_fp32"] = _append_tensor(fp32_buf, bias, "float32")
            kind = "embedding_lookup_w8_per_row" if is_embedding else "conv1d_w8a8_int32" if isinstance(module, nn.Conv1d) else "linear_w8a8_int32"
            k = module.embedding_dim if is_embedding else module.in_channels // module.groups * module.kernel_size[0] if isinstance(module, nn.Conv1d) else module.in_features
            ops.append({"name": name, "kind": kind, "input_scale": None if is_embedding else float(np.float32(input_scales[name])),
                        "accumulator": None if is_embedding else {"signed_bits": 32, "worst_case_abs_bound": k * QMAX * QMAX,
                            "CPU_checked_in_int64_then_narrowed": True},
                        "groups": getattr(module, "groups", None), "kernel_size": list(module.kernel_size) if isinstance(module, nn.Conv1d) else None,
                        "dilation": list(module.dilation) if isinstance(module, nn.Conv1d) else None,
                        "out_channels": getattr(module, "out_channels", None), "in_features": getattr(module, "in_features", None),
                        "weight_scale_axis": "per embedding row" if is_embedding else "per output channel"})
        elif isinstance(module, nn.LayerNorm):
            for pname, p in module.named_parameters(recurse=False):
                tensor_map[f"{name}.{pname}_fp32"] = _append_tensor(fp32_buf, p, "float32")
    # Include any remaining FP32 trainable parameters (e.g. unexpected future op).
    parameter_names = set(dict(model.named_parameters()))
    serialized = {name.removesuffix("_fp32") for name in tensor_map if name.endswith("_fp32")}
    quant_weight_names = {f"{name}.weight" for name, m in model.named_modules() if isinstance(m, (nn.Conv1d, nn.Linear, nn.Embedding))}
    for name, p in model.named_parameters():
        if name in quant_weight_names or name in serialized or name.endswith(".bias") and name.rsplit(".", 1)[0] in {op["name"] for op in ops}:
            continue
        tensor_map[name + "_fp32"] = _append_tensor(fp32_buf, p, "float32")
    for file_name, data in (("weights_w8.bin", int8_buf), ("scales_f32.bin", scale_buf), ("parameters_f32.bin", fp32_buf)):
        (out_dir / file_name).write_bytes(data)
    graph = {
        "model_class": type(model).__name__, "config": config,
        "sequence_operations": "As implemented in scripts/cnn_models.py: document tokenization/byte encoding and windowing are outside this graph; mask, temporal padding, residual/gated flow, downsampling, pooling, and document maximum retain current model semantics.",
        "integer_ops": ops,
        "fp32_operations": ["embedding dequantization", "all Conv/Linear bias additions", "LayerNorm and affine parameters", "ReLU/sigmoid/multiplication gates", "residual additions", "masked pooling/top-k/max/mean", "logits/softmax and window/document aggregation"],
        "liveness_note": "This op graph is static model structure. File sizes and per-op bounds do not establish activation liveness or FPGA allocation; see mapping report.",
    }
    write_json(out_dir / "op_graph.json", graph)
    files = {n: {"sha256": sha(out_dir / n), "bytes": (out_dir / n).stat().st_size} for n in ("weights_w8.bin", "scales_f32.bin", "parameters_f32.bin", "op_graph.json")}
    manifest = {
        "format_id": "sentry_cnn_static_w8a8_int32ref_v1",
        "model_kind": "cnn", "run_id": run.get("spec", {}).get("run_id") or run.get("run_id") or checkpoint.parent.name,
        "configuration": config, "configuration_sha256": canonical_sha(config),
        "checkpoint_path": str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint),
        "checkpoint_sha256": sha(checkpoint), "run_metadata_sha256": sha(checkpoint.parent / "run.json") if (checkpoint.parent / "run.json").exists() else None,
        "model_source_sha256": sha(ROOT / "scripts/cnn_models.py"),
        "activation_calibration": cal_record,
        "numerical": {"weights": "symmetric signed W8 [-127,127], nearest-even; per output-channel / per embedding row maxabs/127",
            "activations": "symmetric signed A8 [-127,127], nearest-even, static per tensor per Conv/Linear input; scale from selected FIT windows only",
            "accumulator": "signed int32; CPU dot computed in int64, observed and worst-case bounds checked before narrowing",
            "rescale": "FP32 accumulator * A8 scale * per-output W8 scale; FP32 bias after rescale",
            "FP32_exceptions": graph["fp32_operations"], "zero_point": 0, "rounding": "nearest-even torch.round", "saturation": "[-127,127]"},
        "storage": {"tensor_descriptors": tensor_map, "files": files,
            "payload_bytes": {"weights_w8": len(int8_buf), "scales_f32": len(scale_buf), "parameters_f32": len(fp32_buf),
                              "combined": len(int8_buf) + len(scale_buf) + len(fp32_buf)},
            "accounting": "Files include 64-byte padding before each tensor; no container/header/alignment overhead beyond listed files, no code/ROM/workspace/activations."},
        "activation_maxabs": maxima,
        "input_scales_a8": {name: float(np.float32(value)) for name, value in sorted(input_scales.items())},
        "evidence_boundary": "CPU numerical reference/export only. No native INT8 speed, Vivado/Vitis compile, resource utilization, clock closure, physical memory traffic or device measurement.",
    }
    manifest["manifest_payload_sha256"] = canonical_sha(manifest)
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _read_export_tensor(manifest_path: Path, manifest: dict, name: str) -> torch.Tensor:
    desc = manifest["storage"]["tensor_descriptors"].get(name)
    if desc is None:
        raise KeyError(f"Export has no tensor descriptor for {name}")
    dtype = desc.get("dtype")
    np_dtype = {"int8": np.dtype("i1"), "float32": np.dtype("<f4")}.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported exported dtype {dtype!r} for {name}")
    shape = tuple(int(x) for x in desc["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * np_dtype.itemsize
    offset, length = int(desc["offset_bytes"]), int(desc["length_bytes"])
    if offset < 0 or length != expected or offset % int(desc.get("alignment_bytes", 0)) != 0:
        raise ValueError(f"Invalid packed tensor descriptor for {name}: {desc}")
    filename = next((file_name for file_name, item in manifest["storage"]["files"].items()
                     if file_name in {"weights_w8.bin", "scales_f32.bin", "parameters_f32.bin"}
                     and name in manifest["storage"]["tensor_descriptors"]
                     and ((dtype == "int8" and file_name == "weights_w8.bin")
                          or (dtype == "float32" and (name.endswith("_fp32") or ".bias_fp32" in name)
                              and file_name == "parameters_f32.bin")
                          or (dtype == "float32" and not name.endswith("_fp32") and ".bias_fp32" not in name
                              and file_name == "scales_f32.bin"))), None)
    if filename is None:
        raise ValueError(f"Cannot determine packed file for tensor {name}")
    data_path = manifest_path.parent / filename
    raw = data_path.read_bytes()
    if offset + length > len(raw):
        raise ValueError(f"Packed tensor {name} exceeds {filename} bounds")
    array = np.frombuffer(raw, dtype=np_dtype, count=int(np.prod(shape)), offset=offset).reshape(shape).copy()
    return torch.from_numpy(array)


def load_export_reference(manifest_path: Path | str, device: str = "cpu"):
    """Rebuild the reference model from the immutable packed export payload.

    The checkpoint is used only to establish its frozen identity. All runtime
    quantized codes, scales, biases and floating-point exceptions are read from
    the exported files; this function never requantizes checkpoint weights.
    """
    if device != "cpu":
        raise ValueError("This reference loader is intentionally CPU-only")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_hash = manifest.pop("manifest_payload_sha256", None)
    if not payload_hash or canonical_sha(manifest) != payload_hash:
        raise ValueError("Export manifest payload SHA-256 mismatch")
    manifest["manifest_payload_sha256"] = payload_hash
    for filename, item in manifest["storage"]["files"].items():
        path = manifest_path.parent / filename
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha(path) != item["sha256"]:
            raise ValueError(f"Packed file missing or SHA/size mismatch: {filename}")
    # Validate descriptors once and reject overlaps or malformed alignment.
    by_file: dict[str, list[tuple[int, int, str]]] = {name: [] for name in manifest["storage"]["files"]}
    for name, desc in manifest["storage"]["tensor_descriptors"].items():
        tensor = _read_export_tensor(manifest_path, manifest, name)
        dtype = desc["dtype"]
        is_fp_parameter = name.endswith("_fp32") or ".bias_fp32" in name
        filename = "weights_w8.bin" if dtype == "int8" else "parameters_f32.bin" if is_fp_parameter else "scales_f32.bin"
        start, end = int(desc["offset_bytes"]), int(desc["offset_bytes"]) + int(desc["length_bytes"])
        by_file[filename].append((start, end, name))
    for filename, spans in by_file.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if previous[1] > current[0]:
                raise ValueError(f"Overlapping packed descriptors in {filename}: {previous[2]} / {current[2]}")
    checkpoint = ROOT / manifest["checkpoint_path"]
    if not checkpoint.is_file() or sha(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint identity no longer matches export manifest")
    source_path = ROOT / "scripts/cnn_models.py"
    if sha(source_path) != manifest["model_source_sha256"]:
        raise ValueError("CNN model source changed since export; refusing graph reconstruction")
    config = manifest["configuration"]
    vocab = 257 if config.get("representation") == "byte" else 30522
    model = build_model(config, vocab, pad_id=0).cpu().eval()
    parameters = dict(model.named_parameters())
    descs = manifest["storage"]["tensor_descriptors"]
    for name, param in parameters.items():
        packed_name = name + "_fp32"
        if packed_name in descs:
            value = _read_export_tensor(manifest_path, manifest, packed_name)
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"FP32 parameter shape mismatch: {name}")
            param.data.copy_(value)
    for name, module in list(model.named_modules()):
        if not name:
            continue
        parent, key = _parent(model, name)
        if isinstance(module, nn.Embedding):
            codes = _read_export_tensor(manifest_path, manifest, name + ".weight_codes")
            scales = _read_export_tensor(manifest_path, manifest, name + ".weight_scales")
            child = IntegerEmbedding(module, codes, scales)
        elif isinstance(module, nn.Conv1d):
            scale = manifest["input_scales_a8"][name]
            scale_blob = float(_read_export_tensor(manifest_path, manifest, name + ".input_scale_a8").item())
            if scale != scale_blob:
                raise ValueError(f"A8 scale manifest/blob mismatch for {name}")
            bias_name = name + ".bias_fp32"
            bias = _read_export_tensor(manifest_path, manifest, bias_name) if module.bias is not None else None
            child = IntegerConv1d(module, scale,
                _read_export_tensor(manifest_path, manifest, name + ".weight_codes"),
                _read_export_tensor(manifest_path, manifest, name + ".weight_scales"), bias)
        elif isinstance(module, nn.Linear):
            scale = manifest["input_scales_a8"][name]
            scale_blob = float(_read_export_tensor(manifest_path, manifest, name + ".input_scale_a8").item())
            if scale != scale_blob:
                raise ValueError(f"A8 scale manifest/blob mismatch for {name}")
            bias_name = name + ".bias_fp32"
            bias = _read_export_tensor(manifest_path, manifest, bias_name) if module.bias is not None else None
            child = IntegerLinear(module, scale,
                _read_export_tensor(manifest_path, manifest, name + ".weight_codes"),
                _read_export_tensor(manifest_path, manifest, name + ".weight_scales"), bias)
        else:
            continue
        _set_child(parent, key, child)
    return model.eval(), manifest


def _aligned_packed_bytes(lengths: list[int], alignment: int = ALIGN) -> int:
    cursor = 0
    for length in lengths:
        cursor += (-cursor) % alignment
        cursor += int(length)
    return cursor


def static_checkpoint_mapping(config: dict, checkpoint_path: Path | str,
                              expected_checkpoint_sha256: str,
                              sequence_length: int = 256) -> dict:
    """Shape-only profile for a frozen CNN checkpoint; no forward/inference.

    Counts exact tensor-shaped W8/A8 and FP32 export payloads including the
    serializer's 64-byte per-tensor alignment. MACs are derived from this
    registry's Conv1d/Linear shapes at the unchanged full sequence length.
    Only the four adoption-original topologies are currently supported, and
    the checkpoint is loaded solely for digest and state-dict shape checks.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_digest = sha(checkpoint_path)
    if checkpoint_digest != expected_checkpoint_sha256:
        raise ValueError(f"Frozen checkpoint SHA mismatch: {checkpoint_path}")
    family = config["family"]
    if family not in {"textcnn", "stacked", "local_context", "residual"}:
        raise ValueError(f"Static adoption profile is scoped to the four original topologies, got {family}")
    vocab = int(config.get("vocab_size", 30522))
    if "vocab_size" not in config:
        vocab = 30522 if config.get("representation", "token") == "token" else 257
    # Construction exposes the registered parameter shapes, but no module is
    # called. Preserve the caller's RNG state because initialization values are
    # irrelevant to this shape-only audit.
    with torch.random.fork_rng(devices=[]):
        model = build_model(config, vocab, pad_id=0).cpu().eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=True)
    modules = list(model.named_modules())
    named_parameters = dict(model.named_parameters())
    weight_lengths, scale_lengths, parameter_lengths = [], [], []
    quant_weight_names, quantized_modules = set(), []
    conv_macs = linear_macs = 0
    layer_norm_parameters = 0
    layer_norm_parameter_names: set[str] = set()
    module_rows = []
    for name, module in modules:
        if not name:
            continue
        if isinstance(module, nn.Embedding):
            weight = module.weight
            weight_lengths.append(weight.numel())
            scale_lengths.append(int(weight.shape[0]) * 4)
            quant_weight_names.add(name + ".weight")
            quantized_modules.append(name)
            module_rows.append({"name": name, "kind": "embedding_w8_per_row", "weight_shape": list(weight.shape),
                                "weight_bytes": weight.numel(), "weight_scale_bytes": int(weight.shape[0]) * 4})
        elif isinstance(module, nn.Conv1d):
            weight = module.weight
            weight_lengths.append(weight.numel())
            scale_lengths.extend((int(module.out_channels) * 4, 4))
            if module.bias is not None:
                parameter_lengths.append(int(module.bias.numel()) * 4)
            quant_weight_names.add(name + ".weight")
            quantized_modules.append(name)
            conv_macs += sequence_length * int(module.out_channels) * int(module.in_channels // module.groups) * int(module.kernel_size[0])
            module_rows.append({"name": name, "kind": "conv1d_w8a8_int32", "weight_shape": list(weight.shape),
                                "weight_bytes": weight.numel(), "weight_scale_bytes": int(module.out_channels) * 4,
                                "input_scale_bytes": 4, "bias_fp32_bytes": int(module.bias.numel()) * 4 if module.bias is not None else 0,
                                "groups": int(module.groups), "kernel": int(module.kernel_size[0]),
                                "counted_macs_per_window": sequence_length * int(module.out_channels) * int(module.in_channels // module.groups) * int(module.kernel_size[0])})
        elif isinstance(module, nn.Linear):
            weight = module.weight
            weight_lengths.append(weight.numel())
            scale_lengths.extend((int(module.out_features) * 4, 4))
            if module.bias is not None:
                parameter_lengths.append(int(module.bias.numel()) * 4)
            quant_weight_names.add(name + ".weight")
            quantized_modules.append(name)
            linear_macs += int(module.in_features) * int(module.out_features)
            module_rows.append({"name": name, "kind": "linear_w8a8_int32", "weight_shape": list(weight.shape),
                                "weight_bytes": weight.numel(), "weight_scale_bytes": int(module.out_features) * 4,
                                "input_scale_bytes": 4, "bias_fp32_bytes": int(module.bias.numel()) * 4 if module.bias is not None else 0,
                                "counted_macs_per_window": int(module.in_features) * int(module.out_features)})
        elif isinstance(module, nn.LayerNorm):
            norm_params = list(module.named_parameters(recurse=False))
            layer_norm_parameters += sum(param.numel() for _, param in norm_params)
            parameter_lengths.extend(param.numel() * 4 for _, param in norm_params)
            layer_norm_parameter_names.update(f"{name}.{pname}" for pname, _ in norm_params)
    for name, param in named_parameters.items():
        if name in quant_weight_names or name in layer_norm_parameter_names or any(name == module + ".bias" for module in quantized_modules):
            continue
        # All remaining trainable values are explicit FP32 exceptions. Avoid
        # double-counting LayerNorm and quantized-module biases already tallied.
        if any(name.startswith(module + ".") for module in quantized_modules):
            continue
        if param.numel():
            parameter_lengths.append(param.numel() * 4)
    weight_bytes = _aligned_packed_bytes(weight_lengths)
    scale_bytes = _aligned_packed_bytes(scale_lengths)
    parameter_bytes = _aligned_packed_bytes(parameter_lengths)
    params = sum(param.numel() for param in named_parameters.values())
    embedding_parameters = model.embedding.weight.numel()
    channels = int(config.get("channels", 64))
    embedding_dim = int(config.get("embedding_dim", 64))
    kernels = config.get("kernel_sizes", [3])
    if family == "textcnn":
        rf = max(int(k) for k in kernels)
    elif family == "stacked":
        rf = 1 + sum((int(kernels[i % len(kernels)]) - 1) * int(config.get("dilations", [1])[i % len(config.get("dilations", [1]))]) for i in range(int(config.get("depth", 1))))
    elif family == "local_context":
        local_k = int(kernels[0])
        context_k = int(config.get("context_kernel", 7))
        dilation = int(config.get("dilations", [1])[0])
        rf = 1 + int(config.get("depth", 1)) * ((local_k - 1) + (context_k - 1) * dilation)
    else:  # ordinary residual chain, all convolutions are centered in this config
        rf = 1 + sum((int(kernels[i % len(kernels)]) - 1) * int(config.get("dilations", [1])[i % len(config.get("dilations", [1]))]) for i in range(int(config.get("depth", 1))))
    lookahead = (rf - 1) // 2
    length = int(sequence_length)
    feature_map_a8 = length * channels
    feature_map_fp32 = feature_map_a8 * 4
    feature_map_int32 = feature_map_a8 * 4
    embedding_map_fp32 = length * embedding_dim * 4
    pooling_width = channels * (len(kernels) if family == "textcnn" else 1)
    payload = weight_bytes + scale_bytes + parameter_bytes
    return {
        "kind": "shape-only analytical profile; checkpoint loaded for identity and shape validation only; no forward/inference",
        "config": config, "config_sha256": canonical_sha(config),
        "checkpoint": str(checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "model_source_sha256": sha(ROOT / "scripts/cnn_models.py"),
        "representation": config.get("representation", "token"), "sequence_length": length,
        "parameters": params, "embedding_parameters": embedding_parameters,
        "fp32_parameter_payload_bytes": params * 4,
        "packed_payload_bytes": {"weights_w8_aligned": weight_bytes, "scales_f32_aligned": scale_bytes,
                                 "parameters_f32_aligned": parameter_bytes, "combined": payload,
                                 "reduction_vs_fp32": 1 - payload / (params * 4)},
        "macs_per_full_window": {"conv1d": conv_macs, "linear": linear_macs, "total": conv_macs + linear_macs,
                                 "counting_boundary": "Conv1d and Linear MACs from exact registered shapes at L; excludes embedding lookup, norms, activation, gates, masks, pooling, aggregation and transfer."},
        "receptive_field_tokens": rf, "centered_feature_lookahead_tokens": lookahead,
        "global_window_decision": "Every selected topology applies global pooling before its classifier; a complete window is required even when the local convolutional feature lookahead is smaller.",
        "activation_shape_bounds": {"one_full_A8_C_channel_map_bytes": feature_map_a8,
                                    "one_full_FP32_C_channel_map_bytes": feature_map_fp32,
                                    "one_full_int32_C_channel_bytes": feature_map_int32,
                                    "one_full_FP32_embedding_output_map_bytes": embedding_map_fp32,
                                    "textcnn_parallel_branch_A8_maps_bytes_if_materialized": 3 * feature_map_a8 if family == "textcnn" else None,
                                    "textcnn_parallel_branch_FP32_maps_bytes_if_materialized": 3 * feature_map_fp32 if family == "textcnn" else None,
                                    "textcnn_concatenated_A8_map_bytes_if_materialized": 3 * feature_map_a8 if family == "textcnn" else None,
                                    "textcnn_concatenated_FP32_map_bytes_if_materialized": 3 * feature_map_fp32 if family == "textcnn" else None,
                                    "local_context_two_branch_A8_concat_bytes_if_materialized": 2 * feature_map_a8 if family == "local_context" else None,
                                    "local_context_two_branch_FP32_concat_bytes_if_materialized": 2 * feature_map_fp32 if family == "local_context" else None,
                                    "local_context_branch_outputs_plus_concat_FP32_bytes_if_materialized": 4 * feature_map_fp32 if family == "local_context" else None,
                                    "residual_skip_plus_output_FP32_maps_bytes_if_materialized": 2 * feature_map_fp32 if family == "residual" else None,
                                    "global_pool_summary_FP32_bytes": pooling_width * 4,
                                    "boundary": "Per-tensor shape bounds only. These examples are not simultaneous live allocations, implementation buffers, BRAM/URAM allocation, or upper bounds on all workspace."},
        "embedding_access": {"w8_table_bytes": embedding_parameters, "per_row_f32_scale_bytes": int(model.embedding.num_embeddings) * 4,
                             "access_pattern": "Token-indexed row gather plus per-row scale; bank/port/NoC contention and cache behavior are unmeasured."},
        "document_formula": {"content_capacity": 254, "overlap": 64, "stride": 190,
                             "windows_for_n_content_tokens": "max(1, ceil(max(0,n-254)/190)+1)",
                             "macs_for_n_tokens": f"{conv_macs + linear_macs} * windows_for_n_content_tokens",
                             "note": "All tails are evaluated; no early document decision before all windows because document risk is max-window risk."},
        "transport": {"ids_and_mask_bytes_per_window": length * 5,
                      "hypothetical_per_window_packet_bytes": 64 + length * 5 + 64,
                      "existing_BERT_packet_bytes_same_length": 64 + length * 4 + length * 4 + length + 64,
                      "packet_scope": "Existing protocol arithmetic only: 64-byte request and response headers, uint32 IDs, uint8 mask; CNN omits BERT token types. No CNN packet contract or target performance is measured."},
        "operator_inventory": module_rows,
        "fp32_layernorm_parameter_count": layer_norm_parameters,
    }


def original_static_mapping(plan_path: Path | str = ROOT / "configs/cnn_adoption_v1.json") -> dict:
    """Build reproducible analytical records keyed to the four frozen originals."""
    plan_path = Path(plan_path)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    model_specs = plan.get("model_specs", plan.get("models", {}))
    result = {"kind": "preliminary shape/accounting evidence; not a fit, inference, numerical-equivalence, FPGA implementation, or physical measurement",
              "execution_record": {"command": "python -B scripts/cnn_adoption_export.py --static-originals-only",
                  "device": "CPU; no model forward/inference", "python": sys.version,
                  "torch": torch.__version__, "platform": platform.platform(),
                  "exporter_sha256": sha(Path(__file__)),
                  "model_source_sha256": sha(ROOT / "scripts/cnn_models.py")},
              "plan_sha256": sha(plan_path), "tokenizer_data_contract": {"length": 256, "content_capacity": 254,
                  "overlap": 64, "stride": 190, "window_formula": "max(1, ceil(max(0,n-254)/190)+1)",
                  "tokenizer_runs_on": "Desktop host; existing A72 supervisor receives already-windowed IDs and masks. Streaming tokenization is unimplemented and unmeasured."},
              "profiles": {}}
    for ref_id in ("original_textcnn_01", "original_stacked_01", "original_local_context_03", "original_local_context_03_residual_replace_c98"):
        ref = plan["references"][ref_id]
        spec = model_specs[ref["model_id"]]
        if spec.get("kind") != "cnn":
            raise ValueError(f"Unexpected non-CNN spec for {ref_id}")
        profile = static_checkpoint_mapping(spec["config"], ref["checkpoint"], ref["checkpoint_sha256"], 256)
        profile["reference_id"] = ref_id
        result["profiles"][ref_id] = profile
    # Same accounting recipe for the comparison model: every 2-D matrix W8,
    # per-row FP32 scale (including embedding rows), static A8 input scale for
    # every non-embedding matrix, and all non-matrix parameters FP32.
    bert_path = ROOT / "checkpoints/floating/epoch_3/model.safetensors"
    bert_config_path = ROOT / "checkpoints/floating/epoch_3/config.json"
    bert_config = json.loads(bert_config_path.read_text(encoding="utf-8"))
    from safetensors import safe_open
    with safe_open(str(bert_path), framework="pt", device="cpu") as sf:
        bert_shapes = {key: tuple(sf.get_slice(key).get_shape()) for key in sf.keys()}
    matrix_shapes = {key: shape for key, shape in bert_shapes.items() if len(shape) == 2}
    linear_shapes = {key: shape for key, shape in matrix_shapes.items() if "embeddings." not in key}
    vector_shapes = {key: shape for key, shape in bert_shapes.items() if len(shape) != 2}
    bert_weight_bytes = _aligned_packed_bytes([int(np.prod(shape)) for shape in matrix_shapes.values()])
    bert_scale_bytes = _aligned_packed_bytes([shape[0] * 4 for shape in matrix_shapes.values()] + [4 for _ in linear_shapes])
    bert_parameter_bytes = _aligned_packed_bytes([int(np.prod(shape)) * 4 for shape in vector_shapes.values()])
    bert_payload = bert_weight_bytes + bert_scale_bytes + bert_parameter_bytes
    h, inner, layers = int(bert_config["hidden_size"]), int(bert_config["intermediate_size"]), int(bert_config["num_hidden_layers"])
    L = 256
    bert_macs = layers * L * (4 * h * h + 2 * h * inner) + layers * 2 * L * L * h + h * h + 2 * h
    result["precision_matched_bert_shape_route"] = {
        "checkpoint": "checkpoints/floating/epoch_3/model.safetensors",
        "checkpoint_sha256": sha(bert_path), "config_sha256": sha(bert_config_path),
        "parameters": sum(int(np.prod(shape)) for shape in bert_shapes.values()),
        "matrix_parameter_count": sum(int(np.prod(shape)) for shape in matrix_shapes.values()),
        "matrix_tensor_count": len(matrix_shapes), "matrix_output_rows_for_scales": sum(shape[0] for shape in matrix_shapes.values()),
        "linear_matrix_count_for_A8_input_scales": len(linear_shapes),
        "nonmatrix_FP32_parameter_count": sum(int(np.prod(shape)) for shape in vector_shapes.values()),
        "packed_payload_bytes_64B_tensor_aligned": {"weights_w8": bert_weight_bytes, "scales_f32": bert_scale_bytes,
                                                     "parameters_f32": bert_parameter_bytes, "combined": bert_payload},
        "payload_format": "All 2-D weights W8; per-first-dimension-row FP32 scales including embedding rows; one FP32 static A8 input scale per non-embedding 2-D weight; all non-2-D weights/biases/norm parameters FP32. Each tensor starts at 64-byte alignment.",
        "MACs_per_L256_window": bert_macs,
        "MAC_counting_boundary": "Four-layer BERT shape-derived dense linears plus QK/AV; excludes softmax, LayerNorm, embedding lookup, masks and transfer. This payload/operation route has no new numerical-equivalence result.",
        "limitations": "Shape-derived hypothetical matched-format BERT accounting, not an exported or numerically qualified checkpoint in this broader all-matrix W8 scope."
    }
    for profile in result["profiles"].values():
        family = profile["config"]["family"]
        candidate_payload = profile["packed_payload_bytes"]["combined"]
        profile["precision_matched_bert_comparison"] = {
            "bert_payload_bytes_64B_tensor_aligned": bert_payload,
            "candidate_payload_reduction_vs_matched_bert": 1 - candidate_payload / bert_payload,
            "bert_macs_over_candidate_macs": bert_macs / max(1, profile["macs_per_full_window"]["total"]),
            "boundary": "Both are shape/format accounting only; reduction does not imply equivalent numerical quality, host-chain latency, target fit, or speedup."
        }
        windows = []
        for n_tokens in (254, 255, 444, 445, 634, 635):
            n_windows = max(1, int(np.ceil(max(0, n_tokens - 254) / 190)) + 1)
            windows.append({"content_tokens": n_tokens, "windows": n_windows,
                            "candidate_conv_linear_macs": n_windows * profile["macs_per_full_window"]["total"],
                            "host_ID_plus_mask_input_bytes": n_windows * 256 * 5,
                            "hypothetical_per_window_request_response_bytes": n_windows * (64 + 256 * 5 + 64),
                            "hypothetical_one_batch_request_response_bytes": 128 + n_windows * 256 * 5})
        profile["document_examples"] = windows
        C = int(profile["config"].get("channels", 64))
        width = int(profile["config"].get("embedding_dim", 64))
        profile["dataflow_and_retention"] = {
            "first_stage": "The selected desktop host holds document text, applies the existing tokenizer and constructs 256-ID windows with 254-content capacity and 64-token overlap. It sends uint32 IDs and uint8 masks by UART to the A72 supervisor, which validates/copies the request before PL submission. Neither A72 nor PL has a mapped tokenizer in the recorded architecture.",
            "embedding_to_first_A8": "Embedding rows are gathered as W8 plus a per-row FP32 scale. The CPU reference dequantizes to FP32 before the first Conv requantizes with its static A8 input scale. Each integer-dot Conv/Linear then rescales to FP32 and adds FP32 bias; the next Conv quantizes its FP32 input to A8. Fusing these boundaries into narrower fixed-point streaming operators would require a separate numerical check.",
            "window_schedule": "Process one complete window at a time, reuse intermediate activation storage, emit two logits or a risk scalar, then release window activations. Full global pooling means no window score before the full window; the document max-risk result is unavailable until every window has completed.",
            "topology_specific_PL_hypothesis": {
                "textcnn": "Three independent centered width-3/5/7 Conv branches at 64 channels. PL may run branches in parallel or time-multiplex a convolution engine across branch weights; parallel execution retains multiple branch outputs, while serialized execution reduces compute concurrency but still needs the three branch summaries for the final 192-value max-pooled classifier input.",
                "stacked": "Two centered width-3, 64-channel stages. This is the shortest sequential path and has two-token feature lookahead; a fused PL streaming schedule is the simplest candidate to test, provided embedding dequantize/requantize and per-stage FP32 bias/rescale boundaries are preserved or separately qualified.",
                "local_context": "Two residual local-context blocks, each with local width-3, context width-15, and 2C-to-C fusion; six tokenwise LayerNorms overall. The local/context branch values must meet at fusion; a fully materialized route can hold two C-wide maps plus a 2C concat, while streaming/branch scheduling changes live storage and needs an explicit schedule.",
                "residual": "Six sequential 98-channel residual convolutions with kernels 3/15/1/3/15/1 and six tokenwise LayerNorms. A simple chain has fewer simultaneous branches than local_context but each residual add needs skip/output alignment; width-15 stages set long local buffers and 16-token lookahead."
            }.get(family),
            "document_retention": {"host_ID_sequence_if_materialized_bytes": "4*n content tokens, excluding tokenizer/raw-text storage and special tokens",
                                   "rolling_window_token_IDs_if_streamable": 254 * 4,
                                   "device_running_document_risk_FP32_bytes": 4,
                                   "output_per_window_if_host_aggregates_logits_bytes": 2 * 4,
                                   "output_per_window_if_device_aggregates_risk_bytes": 4,
                                   "streaming_tokenizer_status": "A bounded-overlap host token stream is a schedule option only; tokenizer chunk equivalence and an actual stream implementation are unverified."},
            "retained_weights": {"embedding_W8_plus_row_scales_bytes": int(profile["embedding_access"]["w8_table_bytes"] + profile["embedding_access"]["per_row_f32_scale_bytes"]),
                                 "complete_packed_payload_bytes": candidate_payload,
                                 "target_route": "Shared PS/PL DDR/NoC weights bundle is the least-assumptive source; BRAM/URAM caching or banking is only a future schedule question."},
            "activation_boundaries": profile["activation_shape_bounds"],
            "quality_of_bound": "The listed full-map sizes are allocations for named hypothetical materializations; line buffers/rolling summaries are lower-bound scheduling ideas. Neither set proves simultaneous peak, BRAM/URAM fit, DSP/LUT use, 200 MHz closure, NoC bandwidth, or physical performance."
        }
        profile["transport"]["uart_8n1_at_115200"] = {
            "hypothetical_CNN_per_window_packet_bytes": 64 + 256 * 5 + 64,
            "hypothetical_BERT_per_window_packet_bytes": 64 + 256 * 9 + 64,
            "CNN_per_window_serialization_ms": (64 + 256 * 5 + 64) * 10 / 115200 * 1000,
            "BERT_per_window_serialization_ms": (64 + 256 * 9 + 64) * 10 / 115200 * 1000,
            "one_batched_CNN_document_serialization_ms_for_N_windows": "(128 + 1280*N) * 10 / 115200 * 1000",
            "separate_request_response_serialization_ms_for_N_windows": "N * 1408 * 10 / 115200 * 1000",
            "boundary": "UART serialization arithmetic only. The CNN packet, batching, response timing, tokenization, PS copy/cache/mailbox and PL processing are not implemented or measured."
        }
    project = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    hw = project["hardware"]
    result["target_contract"] = {"part": hw["part"], "board": hw["allocated_board"],
        "clock_objective_mhz": hw["clock_objective_mhz"], "resource_limits_70pct": hw["resource_limits"],
        "nominal_combined_bram_uram_bytes": int((hw["resource_limits"]["BLOCK_RAMS"] * 36 * 1024 + hw["resource_limits"]["ULTRA_RAMS"] * 288 * 1024) // 8),
        "interface": hw["interface"], "shared_ddr_arena_bytes": 64 * 1024 * 1024,
        "host_and_PL_boundary": "Existing selected architecture is a standalone A72 supervisor with a PL detector. Host/PS tokenizer/windowing and request serialization remain on the host side; the PL-facing contract currently has three AXI4 m_axi bundles (activation, weights, output) plus AXI4-Lite. There is no CNN command or CNN graph integration in that contract.",
        "AIE_boundary": "No AIE route or XRT runtime is recorded in the configured project; this profile does not assume AI Engine mapping or availability.",
        "physical_boundary": "Configured target/capacity/AXI catalog only. No CNN build, resource use, allocation, port/bandwidth proof, physical clock/power, or latency measurement is present."}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static-originals-only", action="store_true",
                    help="Write shape-only profiles for the four frozen CNN originals; loads checkpoint tensors for identity/shape checks but performs no forward/inference")
    ap.add_argument("--plan", type=Path, default=ROOT / "configs/cnn_adoption_v1.json")
    ap.add_argument("--static-output", type=Path, default=ROOT / "results/cnn_adoption_v1/original_static_mapping.json")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--calibration-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--max-calibration-windows", type=int, default=512)
    ap.add_argument("--reference-windows", type=int, default=4)
    args = ap.parse_args()
    if args.static_originals_only:
        profile = original_static_mapping(args.plan)
        static_output = args.static_output if args.static_output.is_absolute() else ROOT / args.static_output
        write_json(static_output, profile)
        print(json.dumps({"complete": True, "kind": profile["kind"], "output": str(static_output),
                          "profiles": {key: {"checkpoint_sha256": value["checkpoint_sha256"],
                              "parameters": value["parameters"], "packed_payload_bytes": value["packed_payload_bytes"]["combined"],
                              "macs_per_window": value["macs_per_full_window"]["total"],
                              "receptive_field_tokens": value["receptive_field_tokens"]}
                              for key, value in profile["profiles"].items()}}, indent=2))
        return
    if args.run_dir is None or args.calibration_dir is None:
        ap.error("--run-dir and --calibration-dir are required unless --static-originals-only is selected")
    started = time.monotonic()
    if args.max_calibration_windows < 1 or args.reference_windows < 1:
        raise ValueError("Window counts must be positive")
    torch.set_num_threads(1)
    model, config, run, checkpoint_state, checkpoint = load_run(args.run_dir, args.checkpoint)
    x, mask, cal_record = calibration_indices(args.calibration_dir, args.max_calibration_windows)
    input_scales, maxima = calibrate_input_scales(model, x, mask)
    run_id = run.get("spec", {}).get("run_id") or args.run_dir.name
    out_dir = args.out_dir or ROOT / "results/cnn_adoption_v1/exports" / run_id
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_path = out_dir / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        same_identity = (previous.get("checkpoint_sha256") == sha(checkpoint)
                         and previous.get("configuration_sha256", canonical_sha(previous.get("configuration"))) == canonical_sha(config)
                         and previous.get("activation_calibration", {}).get("calibration_id_sha256") == cal_record["calibration_id_sha256"])
        if not same_identity:
            raise ValueError("Refusing to reuse export directory for a different checkpoint/config/calibration identity; use a new run_id/output folder")
    fp_model = model.eval()
    manifest = pack_export(model, input_scales, out_dir, config, checkpoint, run, cal_record, maxima)
    int_model, _ = load_export_reference(out_dir / "manifest.json", device="cpu")
    ncheck = min(args.reference_windows, len(x))
    with torch.inference_mode():
        ref = fp_model(x[:ncheck], mask[:ncheck]).logits.float()
        got = int_model(x[:ncheck], mask[:ncheck]).logits.float()
    delta = (got - ref).abs()
    agreement = int((got.argmax(-1) == ref.argmax(-1)).sum())
    check = {
        "complete": True, "kind": "CPU-only integer dot-product reference; no labels or thresholds",
        "windows": ncheck, "logit_shape": list(ref.shape), "max_abs_logit_error": float(delta.max()),
        "mean_abs_logit_error": float(delta.mean()), "argmax_agreement_windows": agreement,
        "argmax_changed_windows": ncheck - agreement,
        "input_id_sha256": canonical_sha(x[:ncheck].tolist()), "mask_sha256": canonical_sha(mask[:ncheck].tolist()),
        "scope": "Reference vectors are selected FIT windows only; no evaluation/scoring/confirmation data.",
    }
    # Reference evidence binds the exact packed bytes actually loaded above.
    check["manifest_sha256"] = sha(out_dir / "manifest.json")
    write_json(out_dir / "reference_check.json", check)
    write_json(out_dir / "activation_calibration.json", cal_record)
    mapping = mapping_assessment(model, manifest)
    write_json(out_dir / "mapping_assessment.json", mapping)
    write_json(out_dir / "execution.json", {
        "complete": True, "device": "CPU", "torch": torch.__version__,
        "python": sys.version, "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
        "wall_seconds": time.monotonic() - started, "command": sys.argv,
        "exporter_sha256": sha(Path(__file__)), "checkpoint_sha256": manifest["checkpoint_sha256"],
        "calibration_id_sha256": cal_record["calibration_id_sha256"],
        "reference_window_count": ncheck, "cuda_execution": False,
        "packed_loader": True,
        "vendor_or_physical_activity": False,
    })
    print(json.dumps({"complete": True, "export": str(out_dir), "checkpoint_sha256": manifest["checkpoint_sha256"],
                      "windows_calibrated": cal_record["selected_windows"], "reference": check,
                      "payload_bytes": manifest["storage"]["payload_bytes"],
                      "matched_bert_payload_reduction": mapping["bert_shape_reference"]["cnn_payload_reduction_vs_hypothetical_all_matrix_w8"],
                      "mapping_file": str(out_dir / "mapping_assessment.json")}, indent=2))


if __name__ == "__main__":
    main()
