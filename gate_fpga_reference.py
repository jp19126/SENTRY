"""Optional ordered FP32 reference for fixed quantized or floating BERT-Mini.

Inference only. Historical gate_quant/evaluator behavior is not changed.
Fixed-FP reductions/grouping follow panel_cache_batched_attention_v1; floating
encoder projections extend the same full-K order in software only. Torch nonlinear
functions and division are not proven equivalent to HLS C math or synthesized IP.
Only CPU eager FP32, B=1 and L=256 have a scheduled bring-up; other devices/batches
remain unvalidated even though independent leading batch dimensions are retained.
"""
from dataclasses import dataclass

import torch
from torch import nn

from gate_quant import GateQuantLinear, encoder_linears, quantize_codes

ARITHMETIC_IDENTITY = 'bert_mini_ordered_fp32_frozen_quant_v1'
FLOATING_ARITHMETIC_IDENTITY = 'bert_mini_ordered_fp32_floating_v1'
PRIMITIVE_IDENTITY = 'ordered_cpu_eager_fp32_saved_service_order_v1'
SERVICE_REVISION = 'panel_cache_batched_attention_v1'
LIBRARY_CAVEAT = 'Torch exp/erf/tanh/sqrt/div are unverified versus HLS C math and RTL IP.'
L, D, H, HD, CAP = 256, 256, 4, 64, 32768


def scalar(value, like):
    return torch.tensor(value, dtype=torch.float32, device=like.device)


def require_fp32(x):
    if x.dtype != torch.float32 or torch.is_autocast_enabled(x.device.type):
        raise ValueError('Ordered reference requires eager FP32 with autocast disabled.')


def ordered_dot(x, weight, bias=None):
    """X[...,M,K], weights[...,N,K]: separate FP32 products, ascending K sum."""
    require_fp32(x)
    require_fp32(weight)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError('DOT reduction dimensions differ.')
    leading = torch.broadcast_shapes(x.shape[:-2], weight.shape[:-2])
    result = torch.zeros((*leading, x.shape[-2], weight.shape[-2]), dtype=torch.float32, device=x.device)
    for k in range(x.shape[-1]):
        product = torch.mul(x[..., :, k:k+1], weight[..., :, k].unsqueeze(-2))
        result = torch.add(result, product)
    if bias is not None:
        result = torch.add(result, bias)
    return result


def ordered_layer_norm(x, gamma, beta):
    """Biased two-pass centered variance, exactly the declared width256 order."""
    require_fp32(x)
    if x.shape[-1] != D:
        raise ValueError('Ordered LayerNorm is fixed to width256.')
    total = torch.zeros_like(x[..., 0])
    for k in range(D):
        total = torch.add(total, x[..., k])
    mean = torch.div(total, scalar(256, x))
    square_sum = torch.zeros_like(total)
    for k in range(D):
        centered = torch.add(x[..., k], torch.neg(mean))
        product = torch.mul(centered, centered)
        square_sum = torch.add(square_sum, product)
    variance = torch.div(square_sum, scalar(256, x))
    denominator = torch.sqrt(torch.add(variance, scalar(1e-12, x)))
    centered = torch.add(x, torch.neg(mean.unsqueeze(-1)))
    normalized = torch.div(centered, denominator.unsqueeze(-1))
    product = torch.mul(normalized, gamma)
    return torch.add(product, beta)


def ordered_softmax(x, mask, scale=1.0):
    """Divide, mask, ascending maximum, exp, ascending sum, divide last axis."""
    require_fp32(x)
    divided = torch.div(x, scalar(scale, x))
    scores = torch.add(divided, mask)
    maximum = torch.full_like(scores[..., 0], -torch.finfo(torch.float32).max)
    for k in range(scores.shape[-1]):
        maximum = torch.where(scores[..., k] > maximum, scores[..., k], maximum)
    exponentials = torch.exp(torch.add(scores, torch.neg(maximum.unsqueeze(-1))))
    total = torch.zeros_like(maximum)
    for k in range(scores.shape[-1]):
        total = torch.add(total, exponentials[..., k])
    return torch.div(exponentials, total.unsqueeze(-1))


def ordered_gelu(x):
    require_fp32(x)
    argument = torch.div(x, scalar(1.4142135623730951, x))
    half = torch.mul(x, scalar(.5, x))
    nonlinear = torch.erf(argument)
    return torch.mul(half, torch.add(scalar(1, x), nonlinear))


def ordered_embedding_add(word, token_type, position):
    require_fp32(word)
    return torch.add(torch.add(word, token_type), position)


def ordered_residual_add(x, residual):
    require_fp32(x)
    return torch.add(x, residual)


def stream_rows(operation, x, *values):
    """Retain service row chunks; batch dimensions are independent, not reduced."""
    step = CAP // x.shape[-1]
    parts = []
    for start in range(0, x.shape[-2], step):
        stop = min(x.shape[-2], start + step)
        # Aligned row tensors are sliced; one-dimensional parameters broadcast.
        args = [value[..., start:stop, :] if value.ndim >= 2 else value for value in values]
        parts.append(operation(x[..., start:stop, :], *args))
    return torch.cat(parts, dim=-2)


def ordered_attention(query, key, value, key_mask):
    """Fixed four heads, 128-query rows,64-key panels, ascending QK and AV sums."""
    require_fp32(query)
    if query.shape[-2:] != (L, D) or key.shape != query.shape or value.shape != query.shape:
        raise ValueError('Ordered attention is fixed to [...,256,256].')
    context = torch.empty_like(query)
    for head in range(H):
        lo, hi = head * HD, (head + 1) * HD
        keys = key[..., :, lo:hi].contiguous()
        transposed_value = value[..., :, lo:hi].transpose(-1, -2).contiguous()
        for start in range(0, L, CAP // L):
            stop = min(L, start + CAP // L)
            queries = query[..., start:stop, lo:hi].contiguous()
            tiles = [ordered_dot(queries, keys[..., offset:offset+64, :]) for offset in range(0, L, 64)]
            scores = torch.cat(tiles, dim=-1)
            mask = key_mask.unsqueeze(-2).expand_as(scores).contiguous()
            probabilities = ordered_softmax(scores, mask, 8.0)
            context[..., start:stop, lo:hi] = ordered_dot(probabilities, transposed_value)
    return context


@dataclass
class OrderedBertOutput:
    logits: torch.Tensor
    risk: torch.Tensor
    intermediates: dict[str, torch.Tensor] | None = None


class OrderedBertMini(nn.Module):
    """Opt-in inference wrapper sharing the loaded model's parameters and scales.

    encoder_mode='quantized' is the unchanged default. Explicit 'floating' uses
    ordinary nn.Linear weights with ascending full-K FP32 product/add reductions
    and bias last, without quantization or a native/fused linear kernel. Independent
    outputs are vectorized; FFN-output K=1024 is a software reference operation,
    not a claim that the bounded physical FP32 DOT service accepts that shape.
    The floating mode has its own identity and makes no 411-service-call claim.

    No checkpoint conversion, parameter/scale updates, training, or automatic
    evaluator registration. Future document scoring must explicitly use
    predict_risk rather than the historical evaluator's native softmax.
    """
    arithmetic_identity = ARITHMETIC_IDENTITY
    primitive_identity = PRIMITIVE_IDENTITY
    library_caveat = LIBRARY_CAVEAT

    def __init__(self, model, *, encoder_mode='quantized'):
        super().__init__()
        if encoder_mode not in ('quantized', 'floating'):
            raise ValueError("Explicit encoder mode must be 'quantized' or 'floating'.")
        cfg = model.config
        if (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size,
                cfg.layer_norm_eps, cfg.hidden_act, cfg.num_labels) != (4, 256, 4, 1024, 1e-12, 'gelu', 2):
            raise ValueError('Ordered wrapper is scoped to the configured BERT-Mini classifier.')
        if getattr(cfg, 'position_embedding_type', 'absolute') != 'absolute' or cfg.is_decoder:
            raise ValueError('Only absolute-position encoder BERT is supported.')
        if model.training or any(module.training for module in model.modules()):
            raise ValueError('Call eval() on the existing model before wrapping it.')
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError('Loaded parameters must remain FP32.')
        linears = list(encoder_linears(model))
        if len(linears) != 24:
            raise ValueError('The fixed four-layer encoder must contain exactly24 selected linears.')
        for _, _, module in linears:
            if encoder_mode == 'quantized':
                if not isinstance(module, GateQuantLinear) or module.collect_saturation:
                    raise ValueError('Require existing quantized linears without mutating saturation counters.')
            elif type(module) is not nn.Linear:
                raise ValueError('Floating mode requires ordinary nn.Linear at all24 encoder sites.')
        self.encoder_mode = encoder_mode
        self.arithmetic_identity = (ARITHMETIC_IDENTITY if encoder_mode == 'quantized'
                                    else FLOATING_ARITHMETIC_IDENTITY)
        self.model = model
        self.config = cfg
        self.training = False

    def train(self, mode=True):
        if mode:
            raise RuntimeError('OrderedBertMini is inference-only; training is unsupported.')
        return super().train(False)

    @staticmethod
    @torch.inference_mode()
    def predict_risk(logits):
        if logits.shape[-1] != 2:
            raise ValueError('Two-class logits are required.')
        return ordered_softmax(logits, torch.zeros_like(logits))[..., 1]

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, token_type_ids=None, *, capture_intermediates=False):
        if self.training or self.model.training:
            raise RuntimeError('Ordered reference requires evaluation mode.')
        if input_ids.ndim != 2 or input_ids.shape[1] != L or input_ids.shape[0] < 1:
            raise ValueError('Expected [batch,256] token IDs; other lengths are not implemented.')
        if attention_mask.shape != input_ids.shape or bool(((attention_mask != 0) & (attention_mask != 1)).any()):
            raise ValueError('Expected matching binary attention mask.')
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if token_type_ids.shape != input_ids.shape:
            raise ValueError('Token type IDs must match token IDs.')
        if input_ids.device != self.model.bert.embeddings.word_embeddings.weight.device:
            raise ValueError('Inputs and loaded model must share a device.')
        if torch.is_autocast_enabled(input_ids.device.type):
            raise RuntimeError('Autocast must be disabled.')
        if input_ids.device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError('Disable TF32 for exact integer code products; CUDA remains unvalidated.')
        if torch.compiler.is_compiling():
            raise RuntimeError('Eager operation boundaries are required; compilation is unsupported.')
        traces = {} if capture_intermediates else None

        def capture(name, value):
            if traces is not None:
                traces[name] = value.detach().clone()
            return value

        def linear(name, module, x):
            if self.encoder_mode == 'quantized':
                if not isinstance(module, GateQuantLinear) or module.collect_saturation:
                    raise ValueError('Quantized module/counter contract changed.')
                if traces is not None:
                    capture(name + '_a8', quantize_codes(x, module.input_scale, 8).to(torch.int8))
                result = module(x)
            else:
                if type(module) is not nn.Linear:
                    raise ValueError('Floating linear module contract changed.')
                result = ordered_dot(x, module.weight, module.bias)
            if name in ('layer1_query', 'layer1_key', 'layer1_value'):
                capture(name + '_output', result)
            return result

        def norm(x, module):
            return stream_rows(ordered_layer_norm, x, module.weight, module.bias)

        embeddings = self.model.bert.embeddings
        positions = torch.arange(L, dtype=torch.long, device=input_ids.device).reshape(1, L)
        word = embeddings.word_embeddings(input_ids)
        token_type = embeddings.token_type_embeddings(token_type_ids)
        position = embeddings.position_embeddings(positions).expand_as(word)
        hidden = stream_rows(ordered_embedding_add, word, token_type, position)
        hidden = capture('embedding_ln', norm(hidden, embeddings.LayerNorm))
        mask = torch.mul(torch.sub(scalar(1, hidden), attention_mask.float()), scalar(torch.finfo(torch.float32).min, hidden))
        for index, block in enumerate(self.model.bert.encoder.layer):
            prefix = 'layer' + str(index)
            query, key, value = [linear(prefix + '_' + name, getattr(block.attention.self, name), hidden)
                                 for name in ('query', 'key', 'value')]
            context = ordered_attention(query, key, value, mask)
            if index == 1:
                capture('layer1_attention_context', context)
            projected = linear(prefix + '_attention_output', block.attention.output.dense, context)
            normalized = norm(stream_rows(ordered_residual_add, projected, hidden), block.attention.output.LayerNorm)
            intermediate = linear(prefix + '_ffn_input', block.intermediate.dense, normalized)
            activated = stream_rows(ordered_gelu, intermediate)
            output = linear(prefix + '_ffn_output', block.output.dense, activated)
            hidden = capture(prefix, norm(stream_rows(ordered_residual_add, output, normalized), block.output.LayerNorm))
        pooler = self.model.bert.pooler.dense
        tiles = [ordered_dot(hidden[:, :1], pooler.weight[o:o+64], pooler.bias[o:o+64]) for o in range(0, D, 64)]
        pooled = torch.tanh(torch.cat(tiles, dim=-1))
        classifier = self.model.classifier
        logits = capture('logits', ordered_dot(pooled, classifier.weight, classifier.bias).squeeze(-2))
        risk = capture('risk', self.predict_risk(logits))
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(risk).all()):
            raise ValueError('Ordered reference produced nonfinite logits/risk.')
        return OrderedBertOutput(logits, risk, traces)
