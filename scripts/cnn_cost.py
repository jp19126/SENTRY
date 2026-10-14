"""Transparent analytical cost helpers for configuration-driven 1-D CNNs.

The module reports parameter bytes and module-level MAC/activation estimates
from a real shaped forward pass. These are software-derived counts, not FPGA
resource, latency, power, or synthesis estimates. Functional operators and
framework fusion may not appear in module hooks. Supply explicit receptive
field paths when the model is branched, pooled, or otherwise non-sequential.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


def conv_path_summary(ops: Sequence[Mapping[str, int]], *, padding: str = "centered") -> dict[str, int | str]:
    """Compute RF, token jump, and lookahead for one ordered path.

    Each operation has ``kernel`` and optional ``dilation`` and ``stride``.
    Pooling can use the same keys. ``padding`` is either ``centered`` (symmetric
    same-padding) or ``causal``. For parallel paths, report each and take max.
    """
    if padding not in {"centered", "causal", "right"}:
        raise ValueError("padding must be 'centered', 'causal', or 'right'")
    rf, jump, lookahead = 1, 1, 0
    for op in ops:
        kernel = int(op["kernel"])
        dilation = int(op.get("dilation", 1))
        stride = int(op.get("stride", 1))
        if min(kernel, dilation, stride) < 1:
            raise ValueError("kernel, dilation, and stride must be positive")
        effective = (kernel - 1) * dilation + 1
        if padding == "centered":
            # MaskedConv in cnn_models.py pads left=floor(E-1)/2 and
            # right=ceil(E-1)/2, so even kernels inspect one extra future tap.
            lookahead += ((effective - 1 + 1) // 2) * jump
        elif padding == "right":
            # Nonoverlapping pair max-pool groups [2j, 2j+1].
            lookahead += (effective - 1) * jump
        rf += (effective - 1) * jump
        jump *= stride
    return {"receptive_field_tokens": rf, "output_token_stride": jump,
            "lookahead_tokens": 0 if padding == "causal" else lookahead,
            "padding_policy": padding}


def registry_receptive_field(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return receptive-field summary for one ``cnn_models.py`` registry config.

    The reported local RF excludes the final global evidence-pooling span.
    For a bidirectional TCN, the backward path contributes future context up to
    RF-1 tokens. This follows the current registry topology; recheck if it changes.
    """
    family = str(config["family"])
    kernels = [int(k) for k in config.get("kernel_sizes", [3])]
    dilations = [int(d) for d in config.get("dilations", [1])]
    depth = int(config.get("depth", 3))
    if family == "textcnn":
        summaries = [conv_path_summary([{"kernel": k}], padding="centered") for k in kernels]
        return {"family": family, "input_unit": "token", "paths": summaries,
                "max_receptive_field_tokens": max(s["receptive_field_tokens"] for s in summaries),
                "max_lookahead_tokens": max(s["lookahead_tokens"] for s in summaries),
                "global_pooling": True}
    if family == "inception":
        max_kernel = max(kernels)
        path = [{"kernel": max_kernel} for _ in range(depth)]
        paths = [conv_path_summary(path, padding="centered")]
    elif family == "local_context":
        context_kernel = int(config.get("context_kernel", 7))
        path = []
        for i in range(depth):
            path.extend(({"kernel": kernels[0]},
                         {"kernel": context_kernel, "dilation": dilations[i % len(dilations)]}))
        paths = [conv_path_summary(path, padding="centered")]
    elif family == "dense":
        # The maximum-span path passes through each dense layer in sequence.
        paths = [conv_path_summary([{"kernel": kernels[i % len(kernels)]} for i in range(depth)], padding="centered")]
    elif family in {"dpcnn", "byte_hierarchy"}:
        path = []
        downsample_after = set(range(1, depth - 1, 2))
        for i in range(depth):
            path.append({"kernel": kernels[i % len(kernels)],
                         "dilation": dilations[i % len(dilations)]})
            if i in downsample_after:
                if family == "dpcnn":
                    path.append({"kernel": 3, "stride": 1})
                    path.append({"kernel": 2, "stride": 2})
                else:
                    path.append({"kernel": 2, "stride": 2})
        # Pool3 uses centered padding; pair-pooling is right-aligned. Calculate
        # the mixed sequence explicitly so stride changes propagate correctly.
        rf, jump, lookahead = 1, 1, 0
        for i in range(depth):
            k, d = kernels[i % len(kernels)], dilations[i % len(dilations)]
            eff = (k - 1) * d + 1
            lookahead += ((eff - 1 + 1) // 2) * jump
            rf += (eff - 1) * jump
            if i in downsample_after:
                if family == "dpcnn":
                    lookahead += jump
                    rf += 2 * jump
                lookahead += jump
                rf += jump
                jump *= 2
        paths = [{"receptive_field_tokens": rf, "output_token_stride": jump,
                  "lookahead_tokens": lookahead, "padding_policy": "registry-pool-mixed"}]
    else:
        layer_count = len(dilations) if family == "idcnn" else depth
        path = [{"kernel": kernels[i % len(kernels)],
                 "dilation": dilations[i % len(dilations)]} for i in range(layer_count)]
        if family == "idcnn":
            path = path * int(config.get("repeats", 3))
        direction = str(config.get("direction", "causal"))
        policy = "causal" if family == "tcn" and direction in {"causal", "bidirectional"} else "centered"
        base = conv_path_summary(path, padding=policy)
        if family == "tcn" and direction == "bidirectional":
            base["max_lookahead_tokens"] = int(base["receptive_field_tokens"]) - 1
        paths = [base]
    return {"family": family, "input_unit": "byte" if family == "byte_hierarchy" else "token", "paths": paths,
            "max_receptive_field_tokens": max(int(p["receptive_field_tokens"]) for p in paths),
            "max_lookahead_tokens": max(int(p.get("max_lookahead_tokens", p["lookahead_tokens"])) for p in paths),
            "global_pooling": True}


def document_window_count(content_tokens: int, *, window_tokens: int = 254,
                          overlap_tokens: int = 64) -> int:
    """Tail-preserving count; defaults match BERT L256 with two special tokens."""
    if content_tokens < 0 or window_tokens < 1 or not 0 <= overlap_tokens < window_tokens:
        raise ValueError("invalid token count/window/overlap")
    stride = window_tokens - overlap_tokens
    return max(1, 1 + math.ceil(max(0, content_tokens - window_tokens) / stride))


def estimate_conv1d(input_length: int, in_channels: int, out_channels: int,
                   kernel: int, *, dilation: int = 1, stride: int = 1,
                   groups: int = 1, padding: str = "same", bytes_per_value: int = 4) -> dict[str, int | str]:
    """Analytical one-layer Conv1d work and simple streaming storage bound.

    ``padding`` accepts ``same`` (ceil(L/stride) outputs), ``valid``, or
    ``causal`` (same output length as input for stride 1). The line-buffer
    term stores the span between taps; it is not an FPGA RAM estimate.
    """
    if min(input_length, in_channels, out_channels, kernel, dilation, stride,
           groups, bytes_per_value) < 1:
        raise ValueError("dimensions, groups, and bytes_per_value must be positive")
    if in_channels % groups or out_channels % groups:
        raise ValueError("groups must divide both channel counts")
    if padding not in {"same", "valid", "causal"}:
        raise ValueError("padding must be same, valid, or causal")
    effective = (kernel - 1) * dilation + 1
    if padding == "valid":
        out_length = max(0, (input_length - effective) // stride + 1)
    else:
        out_length = math.ceil(input_length / stride)
    macs = out_length * out_channels * (in_channels // groups) * kernel
    weights = out_channels * (in_channels // groups) * kernel
    line_buffer = max(0, effective - 1) * in_channels * bytes_per_value
    centered_lookahead = 0 if padding == "causal" else ((effective - 1 + 1) // 2)
    return {"input_length": input_length, "output_length": out_length,
            "effective_kernel": effective, "groups": groups, "weight_parameters": weights,
            "macs": macs, "activation_output_bytes": out_length * out_channels * bytes_per_value,
            "input_line_buffer_bytes": line_buffer, "lookahead_positions": centered_lookahead,
            "padding": padding,
            "assumption": "one output MAC per input-weight product; bias, activation, normalization, pooling, alignment, and control work excluded"}


def estimate_registry_config(config: Mapping[str, Any], *, vocab_size: int,
                             sequence_length: int, batch_size: int = 1,
                             activation_bytes: int = 4) -> dict[str, Any]:
    """Count a current ``cnn_models.py`` config without importing Torch.

    This structural count mirrors its Conv1d/Linear hook MAC convention and
    unique parameter count, including embedding, bias, LayerNorm, and head
    parameters. It assumes every supplied position is valid. It is useful when
    model execution is unavailable, and is not a measured runtime profile.
    Recheck this mapping whenever the registry topology changes.
    """
    if min(vocab_size, sequence_length, batch_size, activation_bytes) < 1:
        raise ValueError("vocabulary, sequence length, batch size, and byte width must be positive")
    family = str(config["family"])
    emb = int(config.get("embedding_dim", 64))
    width = int(config.get("channels", 64))
    depth = int(config.get("depth", 3))
    kernels = [int(k) for k in config.get("kernel_sizes", [3])]
    dilations = [int(d) for d in config.get("dilations", [1])]
    norm = str(config.get("normalization", "layer"))
    if norm not in {"layer", "none"}:
        raise ValueError(f"unsupported normalization: {norm}")
    params = vocab_size * emb
    macs = 0
    max_module_output = batch_size * sequence_length * emb
    max_conv_delay = 0
    max_conv_lookahead = 0
    def conv(cin: int, cout: int, kernel: int, *, dilation: int = 1,
             groups: int = 1, length: int = sequence_length,
             causal: bool = False, count_params: bool = True,
             calls: int = 1) -> None:
        nonlocal params, macs, max_module_output, max_conv_delay, max_conv_lookahead
        if cin % groups or cout % groups:
            raise ValueError(f"invalid groups={groups} for {cin}->{cout}")
        if count_params:
            params += cout * (cin // groups) * kernel + cout
        macs += calls * batch_size * length * cout * (cin // groups) * kernel
        max_module_output = max(max_module_output, batch_size * length * cout)
        span = (kernel - 1) * dilation
        max_conv_delay = max(max_conv_delay, span * cin * activation_bytes)
        if not causal:
            max_conv_lookahead = max(max_conv_lookahead, (span + 1) // 2)

    def layer_norm(channels: int, *, count_params: bool = True) -> None:
        nonlocal params
        if norm == "layer" and count_params:
            params += 2 * channels

    # StudyCNN projects every non-TextCNN family to its configured width.
    if emb != width or family != "textcnn":
        conv(emb, width, 1)

    length = sequence_length
    pool_delay = 0
    if family == "textcnn":
        for kernel in kernels:
            conv(width, width, kernel)
        output_width = width * len(kernels)
    elif family == "inception":
        for _ in range(depth):
            for kernel in kernels:
                conv(width, width, kernel)
            conv(width * len(kernels), width, 1)
            layer_norm(width)
        output_width = width
    elif family == "local_context":
        context_kernel = int(config.get("context_kernel", 7))
        gated = bool(config.get("gated", False))
        for i in range(depth):
            conv(width, width, kernels[0])
            layer_norm(width)
            conv(width, width, context_kernel,
                 dilation=dilations[i % len(dilations)])
            layer_norm(width)
            conv(width * 2, width * (2 if gated else 1), 1)
            layer_norm(width)
        output_width = width
    elif family == "dense":
        growth = int(config.get("growth", width // 2))
        for i in range(depth):
            conv(width + i * growth, growth, kernels[i % len(kernels)])
            layer_norm(growth)
        conv(width + depth * growth, width, 1)
        output_width = width
    else:
        is_idcnn = family == "idcnn"
        schedule_depth = len(dilations) if is_idcnn else depth
        repeats = int(config.get("repeats", 3)) if is_idcnn else 1
        directions = 2 if family == "tcn" and config.get("direction") == "bidirectional" else 1
        gated = family == "gated" or bool(config.get("gated", False))
        separable = family == "depthwise"
        groups = int(config.get("groups", 4)) if family == "grouped" else 1
        downsample_after = set(range(1, depth - 1, 2)) if family in {"dpcnn", "byte_hierarchy"} else set()
        for direction_i in range(directions):
            length = sequence_length
            for repeat_i in range(repeats):
                for i in range(schedule_depth):
                    kernel = kernels[i % len(kernels)]
                    dilation = dilations[i % len(dilations)]
                    causal = family == "tcn" and config.get("direction", "causal") != "centered"
                    if separable:
                        count_params = repeat_i == 0 or direction_i > 0
                        conv(width, width, kernel, dilation=dilation, groups=width, length=length,
                             count_params=count_params)
                        conv(width, width * (2 if gated else 1), 1, length=length,
                             count_params=count_params)
                    else:
                        cout = width * (2 if gated else 1)
                        conv(width, cout, kernel, dilation=dilation, groups=groups,
                             length=length, causal=causal,
                             count_params=(repeat_i == 0 or direction_i > 0))
                    layer_norm(width, count_params=(repeat_i == 0 or direction_i > 0))
                    if i in downsample_after:
                        if family == "dpcnn":
                            pool_delay = max(pool_delay, 2 * width * activation_bytes)
                            length = (length + 1) // 2
                        else:
                            pool_delay = max(pool_delay, width * activation_bytes)
                            length = (length + 1) // 2
        output_width = width * directions

    pool_kind = str(config.get("pooling", "max"))
    topk = int(config.get("topk", 4))
    if pool_kind == "attention":
        conv(output_width, 1, 1, length=length)
    pool_multiplier = 2 if pool_kind == "max_mean" else topk if pool_kind == "topk" else 1
    head_in = output_width * pool_multiplier
    params += head_in * 2 + 2
    macs += batch_size * head_in * 2
    max_module_output = max(max_module_output, batch_size * 2)

    rf = registry_receptive_field(config)
    lookahead = int(rf["max_lookahead_tokens"])
    if family == "tcn" and config.get("direction") == "centered":
        max_conv_lookahead = lookahead
    return {
        "id": str(config.get("id", "")), "family": family,
        "input_unit": "byte" if family == "byte_hierarchy" else "token",
        "sequence_length": sequence_length, "batch_size": batch_size,
        "parameters": params, "embedding_parameters": vocab_size * emb,
        "nonembedding_parameters": params - vocab_size * emb,
        "parameter_bytes_fp32": params * 4,
        "parameter_bytes_ideal_int8": params,
        "embedding_table_bytes_fp32": vocab_size * emb * 4,
        "embedding_table_bytes_ideal_int8": vocab_size * emb,
        "embedding_logical_gather_bytes_fp32": sequence_length * emb * 4 * batch_size,
        "conv_linear_macs": macs,
        "largest_single_module_output_elements": max_module_output,
        "largest_single_module_output_bytes_fp32": max_module_output * 4,
        "max_single_conv_input_delay_bytes_fp32": max_conv_delay // activation_bytes * 4,
        "max_single_conv_input_delay_bytes_int8": max_conv_delay // activation_bytes,
        "downsample_pool_delay_bytes_fp32": pool_delay // activation_bytes * 4,
        "global_pool_summary_bytes_fp32": head_in * 4,
        "local_receptive_field": rf["max_receptive_field_tokens"],
        "receptive_field_unit": "byte" if family == "byte_hierarchy" else "token",
        "lookahead_positions": lookahead,
        "global_pooling": True,
        "document_cost_note": "per-window counts; multiply MACs and embedding gathers by actual tail-preserving window count; static parameters once; aggregation/preprocessing excluded",
        "estimator": "pure-Python structural equivalent of cnn_models.model_cost Conv1d/Linear MAC convention; not executed Torch hooks or a runtime measurement",
    }


def scale_window_profile(profile: Mapping[str, Any], window_count: int) -> dict[str, Any]:
    """Expand per-window work to a complete document while retaining static storage."""
    if window_count < 1:
        raise ValueError("window_count must be positive")
    result = dict(profile)
    module_macs = profile.get("module_macs", {})
    embedding_rows = sum(int(layer.get("lookup_count") or 0) for layer in profile.get("module_details", [])
                         if layer.get("type") == "Embedding")
    embedding_gather_bytes = sum(int(layer.get("active_gather_bytes") or 0) for layer in profile.get("module_details", [])
                                 if layer.get("type") == "Embedding")
    result["document_windows"] = window_count
    result["document_module_macs"] = {k: int(v) * window_count for k, v in module_macs.items()}
    result["document_module_output_bytes_sum"] = int(profile.get("module_output_bytes_sum", 0)) * window_count
    result["document_embedding_lookup_count"] = embedding_rows * window_count
    result["document_embedding_logical_gather_bytes"] = embedding_gather_bytes * window_count
    result["parameter_bytes_by_dtype"] = dict(profile.get("parameter_bytes_by_dtype", {}))
    result["document_rule"] = "static parameters loaded once; per-window work and outputs summed; window aggregation, transport, and text preprocessing remain separate"
    return result


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return tuple(int(x) for x in shape) if shape is not None else None


def _tensor_bytes(value: Any) -> int:
    if not hasattr(value, "numel") or not hasattr(value, "element_size"):
        return 0
    return int(value.numel()) * int(value.element_size())


def profile_cnn(model: Any, input_ids: Any, attention_mask: Any, *,
                token_type_ids: Any = None,
                forward_fn: Callable[[Any, Any, Any], Any] | None = None,
                input_unit: str = "token",
                receptive_field_paths: Sequence[Sequence[Mapping[str, int]]] | None = None,
                padding_policy: str = "centered") -> dict[str, Any]:
    """Profile one supplied batch with hooks and return analytical counts.

    ``forward_fn(model, input_ids, attention_mask)`` can adapt a model with a
    nonstandard call signature. The model is temporarily put in eval mode and
    its prior training state is restored. This executes exactly one forward;
    it does not score labels or update the model.
    """
    import torch
    from torch import nn

    if input_unit not in {"token", "character", "byte", "chunk"}:
        raise ValueError("input_unit must be token, character, byte, or chunk")
    modules = list(model.modules())
    handles = []
    totals: dict[str, int] = defaultdict(int)
    layers: list[dict[str, Any]] = []
    largest_output = 0

    def hook_for(name: str, module: Any) -> Callable[..., None]:
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal largest_output
            in_shape = _shape(inputs[0]) if inputs else None
            out_shape = _shape(output)
            out_bytes = _tensor_bytes(output)
            largest_output = max(largest_output, out_bytes)
            record: dict[str, Any] = {"name": name, "type": type(module).__name__,
                                      "input_shape": in_shape, "output_shape": out_shape,
                                      "output_bytes": out_bytes}
            if isinstance(module, nn.Conv1d):
                if out_shape and len(out_shape) == 3:
                    macs = out_shape[0] * out_shape[2] * module.out_channels * (
                        module.in_channels // module.groups) * module.kernel_size[0]
                else:
                    macs = 0
                weights = module.weight.numel()
                totals["convolution_macs"] += int(macs)
                record.update(macs=int(macs), parameters=int(weights + (module.bias.numel() if module.bias is not None else 0)),
                              kernel=list(module.kernel_size), dilation=list(module.dilation),
                              stride=list(module.stride), groups=module.groups,
                              weight_bytes=int(module.weight.numel() * module.weight.element_size()))
            elif isinstance(module, nn.Linear):
                macs = int(output.numel() * module.in_features) if hasattr(output, "numel") else 0
                totals["linear_macs"] += macs
                record.update(macs=macs, parameters=int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)))
            elif isinstance(module, nn.Embedding):
                ids = inputs[0] if inputs else None
                valid = None
                if ids is not None and hasattr(ids, "numel"):
                    valid = int(ids.numel())
                    if module.padding_idx is not None:
                        valid = int((ids != module.padding_idx).sum().item())
                record.update(lookup_count=valid,
                              table_parameters=int(module.weight.numel()),
                              table_bytes=int(module.weight.numel() * module.weight.element_size()),
                              active_gather_bytes=(None if valid is None else valid * module.embedding_dim * module.weight.element_size()))
            elif isinstance(module, (nn.MaxPool1d, nn.AvgPool1d)):
                record.update(kernel=module.kernel_size, stride=module.stride)
            layers.append(record)
        return hook

    for name, module in model.named_modules():
        if name and isinstance(module, (nn.Conv1d, nn.Linear, nn.Embedding,
                                        nn.MaxPool1d, nn.AvgPool1d)):
            handles.append(module.register_forward_hook(hook_for(name, module)))

    module_training_states = [(module, bool(module.training)) for module in modules]
    model.eval()
    try:
        with torch.inference_mode():
            if forward_fn is None:
                if token_type_ids is None:
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    output = model(input_ids=input_ids, attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
            else:
                output = forward_fn(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in module_training_states:
            module.training = training

    params = list(model.parameters())
    total_parameters = sum(int(p.numel()) for p in params)
    parameter_bytes_by_dtype: dict[str, int] = defaultdict(int)
    for p in params:
        parameter_bytes_by_dtype[str(p.dtype)] += int(p.numel() * p.element_size())
    embedding_params = 0
    embedding_bytes_by_dtype: dict[str, int] = defaultdict(int)
    seen = set()
    for module in modules:
        if isinstance(module, nn.Embedding) and id(module.weight) not in seen:
            seen.add(id(module.weight))
            embedding_params += int(module.weight.numel())
            embedding_bytes_by_dtype[str(module.weight.dtype)] += int(module.weight.numel() * module.weight.element_size())

    rf = None
    if receptive_field_paths is not None:
        rf = [conv_path_summary(path, padding=padding_policy) for path in receptive_field_paths]
    ids_shape = _shape(input_ids) or ()
    mask_shape = _shape(attention_mask) or ()
    return {
        "scope": "one supplied forward; per-window analytical profile",
        "input_unit": input_unit,
        "batch_shape_input_ids": ids_shape,
        "batch_shape_attention_mask": mask_shape,
        "valid_input_positions": int(attention_mask.sum().item()) if hasattr(attention_mask, "sum") else None,
        "parameter_count_unique": total_parameters,
        "parameter_bytes_by_dtype": dict(parameter_bytes_by_dtype),
        "embedding_parameters": embedding_params,
        "embedding_bytes_by_dtype": dict(embedding_bytes_by_dtype),
        "module_macs": {"conv1d": totals["convolution_macs"], "linear": totals["linear_macs"],
                        "total_counted": totals["convolution_macs"] + totals["linear_macs"]},
        "module_output_bytes_sum": sum(layer["output_bytes"] for layer in layers),
        "largest_single_module_output_bytes": largest_output,
        "module_details": layers,
        "receptive_field_paths": rf,
        "document_cost_rule": "sum per-window MAC/embedding gather costs over every tail-preserving window; max-window risk aggregation is additional scalar compare work",
        "limitations": [
            "Hooks count registered Conv1d and Linear modules only; functional ops, custom kernels, activations, normalization, comparisons and host preprocessing are excluded from MAC totals.",
            "Output-byte sum is not peak live memory; residual branches, reuse, fusion, padding masks and allocator work need topology-aware accounting.",
            "Parameter and MAC counts do not predict FPGA LUTs, BRAMs, DSPs, clock, power, throughput, or end-to-end latency.",
            "Supply every meaningful path in receptive_field_paths for branched/residual models; characterize pooling and document aggregation separately.",
            "Embedding active_gather_bytes is a logical row-payload count, not off-chip traffic; cache reuse and memory placement are unspecified."
        ]
    }
