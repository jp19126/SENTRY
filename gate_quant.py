"""Goal 3 fixed-scale Brevitas arithmetic for the 16 BERT encoder precision groups.

This is software integer emulation with STE gradients, not accelerated INT4.
Only encoder linear inputs/weights are quantized. All other operations stay FP32.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

GROUP_SUFFIXES = OrderedDict([
    ("qkv", ("attention.self.query", "attention.self.key", "attention.self.value")),
    ("attention_output", ("attention.output.dense",)),
    ("ffn_input", ("intermediate.dense",)),
    ("ffn_output", ("output.dense",)),
])
SCHEME_ID = "static_symmetric_encoder_linear_w4w8a8_fp32_other_v1"


def configure_exact_fp32() -> dict:
    """Do not initialize a CUDA context; disable lower precision matmul globally."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "autocast": False,
    }


def signed_limits(bits: int) -> tuple[int, int]:
    if bits not in (4, 8):
        raise ValueError("The shared search permits only signed W4/W8 and A8.")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def quantize_codes(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """Independent reference: nearest-even, zero point zero, full-range saturation."""
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("Quantization scales must be finite and strictly positive.")
    low, high = signed_limits(bits)
    return torch.round(x.float() / scale.float()).clamp(low, high).to(torch.int64)


def int_linear_reference(
    input_codes: torch.Tensor, weight_codes: torch.Tensor, weight_bits: int = 8
) -> torch.Tensor:
    """Exact CPU reference; compute int64 then require the declared int32 result."""
    if input_codes.device.type != "cpu" or weight_codes.device.type != "cpu":
        raise ValueError("Integer reference runs on CPU; no accelerated GPU claim.")
    if input_codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Integer reference requires integer input codes.")
    if weight_codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Integer reference requires integer weight codes.")
    low, high = signed_limits(weight_bits)
    if bool(((input_codes < -128) | (input_codes > 127)).any()):
        raise ValueError("Input codes exceed signed A8.")
    if bool(((weight_codes < low) | (weight_codes > high)).any()):
        raise ValueError("Weight codes exceed the declared precision.")
    result = input_codes.to(torch.int64) @ weight_codes.to(torch.int64).T
    if bool(((result < -(1 << 31)) | (result > (1 << 31) - 1)).any()):
        raise OverflowError("Accumulator overflow: the declared int32 format is inadequate.")
    return result.to(torch.int32)


def w8_atomic_reference(input_codes: torch.Tensor, weight_codes: torch.Tensor) -> torch.Tensor:
    """Signed high nibble plus unsigned low nibble; exact CPU reference."""
    if input_codes.device.type != "cpu" or weight_codes.device.type != "cpu":
        raise ValueError("Atomic arithmetic reference is CPU only.")
    weights = weight_codes.to(torch.int64)
    if bool(((weights < -128) | (weights > 127)).any()):
        raise ValueError("W8 codes must be within [-128, 127].")
    high = torch.div(weights, 16, rounding_mode="floor")
    low = weights - 16 * high
    return (16 * (input_codes.to(torch.int64) @ high.T)
            + input_codes.to(torch.int64) @ low.T)


def group_names() -> list[str]:
    return [f"layer{layer}.{group}" for layer in range(4) for group in GROUP_SUFFIXES]


def normalize_precision_map(bits: int | Sequence[int] | Mapping[str, int]) -> OrderedDict:
    names = group_names()
    if isinstance(bits, int):
        values = [bits] * len(names)
    elif isinstance(bits, Mapping):
        if set(bits) != set(names):
            raise ValueError("Precision-map keys must match the sixteen fixed groups.")
        values = [int(bits[name]) for name in names]
    else:
        values = list(bits)
    if len(values) != 16 or any(value not in (4, 8) for value in values):
        raise ValueError("Exactly sixteen W4/W8 decisions are required.")
    return OrderedDict(zip(names, values))


def encoder_linears(model: nn.Module):
    cfg = model.config
    if (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.intermediate_size) != (4, 256, 4, 1024):
        raise ValueError("This adapter is scoped to the configured BERT-Mini architecture.")
    for layer in range(4):
        for group, suffixes in GROUP_SUFFIXES.items():
            for suffix in suffixes:
                path = f"bert.encoder.layer.{layer}.{suffix}"
                module = model.get_submodule(path)
                if not isinstance(module, (nn.Linear, GateQuantLinear)):
                    raise TypeError(f"{path} is not a supported linear module.")
                yield f"layer{layer}.{group}", path, module


class GateQuantLinear(nn.Module):
    """Fixed-scale IntQuant codes, exact FP32 integer dot, FP32 rescale and bias."""

    def __init__(self, linear: nn.Linear, input_scale: float, bits: int = 8):
        super().__init__()
        from brevitas.core.quant import IntQuant
        if linear.weight.dtype != torch.float32:
            raise ValueError("Load the floating checkpoint in FP32 before conversion.")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # Every partial sum has magnitude <= K*128*128. FP32 represents every
        # integer through 2**24, so this is also an exact int32 accumulator emulator.
        if self.in_features * 128 * 128 > (1 << 24):
            raise ValueError("K exceeds the exact integer-valued FP32 dot-product bound.")
        self.weight = linear.weight
        self.bias = linear.bias
        device = self.weight.device
        maximum = self.weight.detach().abs().amax(dim=1, keepdim=True)
        for width in (4, 8):
            positive_maximum = signed_limits(width)[1]
            scale = torch.where(maximum > 0, maximum / positive_maximum, torch.ones_like(maximum))
            self.register_buffer(f"weight_scale_{width}", scale)
        if not input_scale > 0:
            raise ValueError("Input scale must be positive.")
        self.register_buffer("input_scale", torch.tensor(float(input_scale), device=device))
        self.register_buffer("zero_point", torch.tensor(0.0, device=device))
        self.register_buffer("activation_width", torch.tensor(8.0, device=device))
        self.register_buffer("weight_width", torch.tensor(float(bits), device=device))
        self.quantizer = IntQuant(narrow_range=False, signed=True, input_view_impl=nn.Identity())
        self.weight_bits = bits
        self.execution_mode = "ste_emulation"
        self.collect_saturation = False
        self.saturation_count = 0
        self.activation_count = 0
        self.set_bits(bits)

    def set_bits(self, bits: int) -> None:
        signed_limits(bits)
        self.weight_bits = int(bits)
        self.weight_width.fill_(float(bits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            raise ValueError("The arithmetic contract requires FP32 input and no AMP/autocast.")
        if torch.is_autocast_enabled(x.device.type):
            raise RuntimeError("Autocast must be disabled for exact integer emulation.")
        scale = getattr(self, f"weight_scale_{self.weight_bits}")
        if self.collect_saturation:
            with torch.no_grad():
                rounded = torch.round(x / self.input_scale)
                self.saturation_count += int(((rounded < -128) | (rounded > 127)).sum())
                self.activation_count += x.numel()
        if self.execution_mode == "integer_reference":
            a = quantize_codes(x, self.input_scale, 8)
            w = quantize_codes(self.weight, scale, self.weight_bits)
            accumulator = int_linear_reference(a, w, self.weight_bits).float()
        elif self.execution_mode == "ste_emulation":
            a = self.quantizer.to_int(self.input_scale, self.zero_point, self.activation_width, x)
            w = self.quantizer.to_int(scale, self.zero_point, self.weight_width, self.weight)
            with torch.autocast(device_type=x.device.type, enabled=False):
                accumulator = F.linear(a, w, None)
        else:
            raise ValueError(f"Unknown execution mode {self.execution_mode!r}.")
        # Keep operation order identical in emulation, CPU integer reference and export.
        result = accumulator * (self.input_scale * scale.squeeze(1))
        if self.bias is not None:
            result = result + self.bias
        return result


def calibrate_inputs(model: nn.Module, batches: Iterable[Mapping[str, torch.Tensor]]) -> dict:
    """Observe FP-model train-only batches, excluding padding; caller records source IDs.

    Data ownership is explicit: callers must supply training windows, never search,
    final calibration or test documents. No precision candidate recalibrates scales.
    """
    stats = {path: {"abs_max": 0.0, "values": 0} for _, path, _ in encoder_linears(model)}
    handles = []
    active = {"mask": None}

    def hook_for(path):
        def observe(module, inputs):
            value = inputs[0].detach()
            mask = active["mask"]
            if value.ndim == 3 and mask is not None:
                value = value[mask.bool()]
            stats[path]["abs_max"] = max(stats[path]["abs_max"], float(value.abs().max()))
            stats[path]["values"] += value.numel()
        return observe

    was_training = model.training
    model.eval()
    count = 0
    try:
        for _, path, module in encoder_linears(model):
            if isinstance(module, GateQuantLinear):
                raise ValueError("Calibrate the floating checkpoint exactly once before replacement.")
            handles.append(module.register_forward_pre_hook(hook_for(path)))
        with torch.inference_mode():
            for batch in batches:
                active["mask"] = batch.get("attention_mask")
                model(**batch)
                count += int(batch["input_ids"].shape[0])
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not count or any(not value["values"] for value in stats.values()):
        raise ValueError("Calibration requires nonempty real training windows.")
    return {
        "scheme_id": SCHEME_ID, "window_count": count,
        "input_scale_rule": "valid-token max_abs/127; zero-only tensors use scale 1",
        "modules": {
            path: data | {"input_scale": data["abs_max"] / 127.0 if data["abs_max"] else 1.0}
            for path, data in stats.items()
        },
    }


def install_quantized_linears(model, calibration: Mapping, bits=8, copy_model=False):
    if calibration["scheme_id"] != SCHEME_ID:
        raise ValueError("Calibration belongs to a different numerical scheme.")
    configure_exact_fp32()
    model = copy.deepcopy(model) if copy_model else model
    precision = normalize_precision_map(bits)
    modules = list(encoder_linears(model))
    for group, path, linear in modules:
        if isinstance(linear, GateQuantLinear):
            raise ValueError("This model has already been quantized.")
        parent_path, child_name = path.rsplit(".", 1)
        replacement = GateQuantLinear(
            linear, float(calibration["modules"][path]["input_scale"]), precision[group])
        setattr(model.get_submodule(parent_path), child_name, replacement)
    return model


def set_precision_map(model, bits) -> OrderedDict:
    precision = normalize_precision_map(bits)
    for group, _, module in encoder_linears(model):
        if not isinstance(module, GateQuantLinear):
            raise ValueError("Install quantized linears before setting the precision map.")
        module.set_bits(precision[group])
    return precision


def set_execution_mode(model, mode: str) -> None:
    if mode not in ("ste_emulation", "integer_reference"):
        raise ValueError("Execution mode must be explicit.")
    for _, _, module in encoder_linears(model):
        if not isinstance(module, GateQuantLinear):
            raise ValueError("Model is not quantized.")
        module.execution_mode = mode


def weight_numerical_error(model) -> dict:
    total = {}
    for group, path, module in encoder_linears(model):
        if not isinstance(module, GateQuantLinear):
            raise ValueError("Weight error requires installed quantized linears.")
        scale = getattr(module, f"weight_scale_{module.weight_bits}")
        restored = quantize_codes(module.weight, scale, module.weight_bits).float() * scale
        squared = (restored - module.weight.detach()).square()
        total[path] = {"group": group, "weight_bits": module.weight_bits,
                       "elements": squared.numel(), "squared_error_sum": float(squared.sum()),
                       "mean_squared_error": float(squared.mean())}
    return total


def logical_storage_estimate(model) -> dict:
    """Logical packed payload only; the saved training checkpoint remains FP32."""
    total_parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    selected_fp32_bytes = 0
    selected_packed_bytes = 0
    active_scale_bytes = 0
    for _, _, module in encoder_linears(model):
        if not isinstance(module, GateQuantLinear):
            raise ValueError("Logical storage estimate requires a selected quantized map.")
        selected_fp32_bytes += module.weight.numel() * module.weight.element_size()
        selected_packed_bytes += module.weight.numel() * module.weight_bits // 8
        active_scale_bytes += 4 * (module.out_features + 1)
    return {
        "logical_selected_packed_weight_bytes": selected_packed_bytes,
        "fixed_FP32_nonsearch_parameter_bytes": total_parameter_bytes - selected_fp32_bytes,
        "logical_active_FP32_scale_bytes": active_scale_bytes,
        "logical_payload_bytes": total_parameter_bytes - selected_fp32_bytes + selected_packed_bytes + active_scale_bytes,
        "saved_parameter_representation_bytes": total_parameter_bytes,
        "scope": "analytical packed-weight payload estimate, excluding file/container/transport/alignment overhead",
        "packed_runtime_checkpoint_created": False,
        "saved_checkpoint_representation": "FP32 parameters plus quantization buffers; actual file sizes reported separately",
    }


def save_quantized_checkpoint(model, directory: str | Path, calibration: Mapping, metadata: Mapping):
    """Save quant buffers with updated QAT weights; reload through this module."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(target)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()},
               target / "quantized_state.pt")
    precision = {}
    for group, _, module in encoder_linears(model):
        precision[group] = module.weight_bits
    record = dict(metadata) | {
        "scheme_id": SCHEME_ID, "precision_map": precision, "calibration": dict(calibration),
        "format": format_record(),
    }
    (target / "quantization.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_quantized_checkpoint(directory: str | Path, device="cpu"):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    path = Path(directory)
    record = json.loads((path / "quantization.json").read_text(encoding="utf-8"))
    model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(path))
    model = install_quantized_linears(model, record["calibration"], record["precision_map"])
    # Includes the original frozen scales; never regenerate them from QAT weights.
    state = torch.load(path / "quantized_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device), record


def format_record() -> dict:
    import brevitas
    return {
        "scheme_id": SCHEME_ID, "brevitas": brevitas.__version__,
        "quantizer": "brevitas.core.quant.IntQuant",
        "search_scope": "24 encoder linears in 16 fixed groups; W4/W8 and A8 inputs",
        "weight_scale": "static FP-checkpoint row max_abs/(2**(b-1)-1), separately W4/W8",
        "activation_scale": "static training valid-token max_abs/127, common across maps/QAT",
        "zero_tensor_scale": 1.0, "zero_point": 0,
        "rounding": "nearest even via Brevitas RoundSte/torch.round",
        "clipping": "full signed range [-2**(b-1), 2**(b-1)-1]",
        "accumulator": "signed int32; exact integer-valued FP32 emulation requires K<=1024",
        "rescale": "FP32 accumulator * (input_scale * per-row weight_scale), then FP32 bias",
        "other_operations": "fixed FP32 embeddings, attention matmuls, residuals, LayerNorm, GELU, softmax, pooler, classifier",
        "qat": "STE for quantized codes; scales frozen; same one-epoch allowance",
        "performance_claim": "software integer emulation, not accelerated INT4/INT8 timing",
        "fp32_settings": configure_exact_fp32(),
    }
