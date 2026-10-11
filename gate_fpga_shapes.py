"""Explicit CPU ordered BERT-Mini references for sequence lengths128 and512.

The existing L256 wrapper/source and GateQuant arithmetic remain unchanged.
This wrapper shares the loaded model and frozen scales; no model conversion or
retraining. Full-L attention retains ascending K,64-key panels and query slabs
min(128,CAP//L). A synthetic RTL shape check does not qualify this full model.
Torch nonlinear caveats and software-only floating K1024 remain unchanged.
No batch/length numerical or device timing acceptance is implied by this source.
"""
import torch
from torch import nn

from gate_quant import GateQuantLinear, quantize_codes
from gate_fpga_reference import (
    OrderedBertMini, OrderedBertOutput, D, H, HD, CAP,
    ordered_dot, ordered_softmax, ordered_layer_norm, ordered_gelu,
    ordered_embedding_add, ordered_residual_add, stream_rows, scalar, require_fp32,
    ARITHMETIC_IDENTITY, FLOATING_ARITHMETIC_IDENTITY,
)

SUPPORTED_LENGTHS = (128, 512)


def ordered_attention_shape(query, key, value, key_mask, sequence_length):
    """One joint full-L attention reduction; independent leading batches."""
    require_fp32(query)
    length = sequence_length
    if type(length) is not int or length not in SUPPORTED_LENGTHS:
        raise ValueError('Use the unchanged base wrapper for L256.')
    if query.shape[-2:] != (length, D) or key.shape != query.shape or value.shape != query.shape:
        raise ValueError('Attention operands must match the explicit [...,L,256] shape.')
    if key_mask.shape != query.shape[:-1]:
        raise ValueError('Expected one full-length additive key mask per batch.')
    require_fp32(key)
    require_fp32(value)
    require_fp32(key_mask)
    context = torch.empty_like(query)
    step = min(128, CAP // length)
    for head in range(H):
        lo, hi = head * HD, (head + 1) * HD
        keys = key[..., :, lo:hi].contiguous()
        transposed_value = value[..., :, lo:hi].transpose(-1, -2).contiguous()
        for start in range(0, length, step):
            stop = min(length, start + step)
            queries = query[..., start:stop, lo:hi].contiguous()
            tiles = [ordered_dot(queries, keys[..., offset:offset+64, :])
                     for offset in range(0, length, 64)]
            scores = torch.cat(tiles, dim=-1)
            mask = key_mask.unsqueeze(-2).expand_as(scores).contiguous()
            probabilities = ordered_softmax(scores, mask, 8.0)
            context[..., start:stop, lo:hi] = ordered_dot(probabilities, transposed_value)
    return context


class OrderedBertMiniShapes(OrderedBertMini):
    """Inference-only opt-in128/512 wrapper; shares every existing parameter.

    sequence_length is mandatory and fixed per instance. encoder_mode selects
    unchanged GateQuant modules or ordinary floating linears with ascending
    full-K FP32 products/additions. Callers retain the established CPU settings,
    checkpoint, precision map and scales, and must qualify actual outputs before
    assigning score-identity acceptance. Independent batches remain unreduced.
    """
    def __init__(self, model, *, sequence_length, encoder_mode='quantized'):
        if type(sequence_length) is not int or sequence_length not in SUPPORTED_LENGTHS:
            raise ValueError('Explicit sequence_length must be128 or512; use the base wrapper for256.')
        super().__init__(model, encoder_mode=encoder_mode)
        if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
            raise ValueError('Length-specific reference requires existing CPU FP32 parameters.')
        available = model.bert.embeddings.position_embeddings.num_embeddings
        if available < sequence_length:
            raise ValueError('Loaded position embeddings do not cover the requested full length.')
        self.sequence_length = sequence_length
        self.source_arithmetic_identity = (ARITHMETIC_IDENTITY if encoder_mode == 'quantized'
                                           else FLOATING_ARITHMETIC_IDENTITY)
        self.arithmetic_identity = ('bert_mini_ordered_fp32_frozen_quant' if encoder_mode == 'quantized'
                                    else 'bert_mini_ordered_fp32_floating') + f'_l{sequence_length}_v1'
        self.shape_qualification = None

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, token_type_ids=None, *, capture_intermediates=False):
        if self.training or self.model.training:
            raise RuntimeError('Ordered reference requires evaluation mode.')
        length = self.sequence_length
        if input_ids.device.type != 'cpu':
            raise ValueError('Length-specific reference requires CPU eager FP32.')
        if input_ids.dtype != torch.long:
            raise ValueError('Token IDs must use torch.long.')
        if input_ids.ndim != 2 or input_ids.shape[1] != length or input_ids.shape[0] < 1:
            raise ValueError(f'Expected [batch,{length}] token IDs for this explicit shape wrapper.')
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
        positions = torch.arange(length, dtype=torch.long, device=input_ids.device).reshape(1, length)
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
            context = ordered_attention_shape(query, key, value, mask, length)
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
