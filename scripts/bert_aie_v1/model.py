"""Readable matched BERT inference and the explicit W8/A8 mixed precision variant.

This is a functional numerical reference, not evidence of hardware execution.
All nonlinear operations/residuals and pooler/head remain FP32. Encoder linear
inputs use FIT-frozen symmetric scalar scales; weights use per-output scales.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'results/cnn_adoption_v1/runs/bert_matched_lr5e5_supervised_s42_r2/checkpoint_selected.pt'
CHECKPOINT_SHA256 = '95025bfc0e29163efea613d556d7da0b516ec1bb6c6fac9d9c4c9c6f1f373525'
CONFIG = ROOT / 'checkpoints/huggingface/models--google--bert_uncased_L-4_H-256_A-4/snapshots/387825ce42dbb39b87911cdf8e383ee3b25184f8/config.json'
VARIANTS = ('fp32', 'w8a8_fp32_attention', 'w8a8_bf16_attention')
SUFFIXES = ('attention.self.query', 'attention.self.key', 'attention.self.value',
            'attention.output.dense', 'intermediate.dense', 'output.dense')
LINEARS = tuple(f'bert.encoder.layer.{i}.{s}' for i in range(4) for s in SUFFIXES)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def quantize(x, scale):
    """Ties-to-even, symmetric signed codes [-127,127], no zero point."""
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8)

def integer_dot(x, weight):
    """Exact int8 [M,K] x [O,K] -> int32 [M,O]; no float emulation."""
    shape = x.shape[:-1]
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    if matrix.device.type == 'cuda':
        # CUDA int_mm requires aligned M/N. Padding is removed after the dot.
        m, k = matrix.shape; o = weight.shape[0]
        kp=max(32,math.ceil(k/8)*8);mp=max(32,math.ceil(m/8)*8);op=max(32,math.ceil(o/8)*8)
        xp = F.pad(matrix, (0, kp-k, 0, mp-m))
        wp = F.pad(weight, (0, kp-k, 0, op-o))
        result = torch._int_mm(xp, wp.t().contiguous())[:m, :o]
    else:
        result = matrix.to(torch.int32) @ weight.to(torch.int32).t()
    return result.reshape(*shape, weight.shape[0])

def bf16_matmul(a, b, chunk=64):
    """BF16 operands, FP32 dot accumulation; 64-wide services summed in order.

    Software expresses the arithmetic contract. Target compiler accumulation
    order can differ and requires measured differential qualification.
    """
    a = a.to(torch.bfloat16).float(); b = b.to(torch.bfloat16).float()
    result = torch.zeros((*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), a.shape[-2], b.shape[-1]), device=a.device)
    for start in range(0, a.shape[-1], chunk):
        result = result + a[..., start:start+chunk] @ b[..., start:start+chunk, :]
    return result

class ReadableBert(nn.Module):
    def __init__(self, state, config, variant='fp32', scales=None, observe=False):
        super().__init__()
        if variant not in VARIANTS: raise ValueError('Unknown numerical variant')
        self.config = config; self.variant = variant; self.observe = observe
        self.observed = {}; self.saturation = {}; self.scales = dict(scales or {})
        self.weights = nn.ParameterDict({k.replace('.', '__'): nn.Parameter(v.detach().float(), requires_grad=False) for k, v in state.items()})
        self._source_keys = tuple(state)
        if variant != 'fp32':
            if set(self.scales) != set(LINEARS): raise ValueError('Need all 24 FIT-frozen input scales')
            for name in LINEARS:
                w = self.w(name + '.weight')
                scale = w.abs().amax(1).clamp_min(torch.finfo(torch.float32).tiny) / 127
                self.register_buffer(name.replace('.', '__') + '__codes', quantize(w, scale[:, None]))
                self.register_buffer(name.replace('.', '__') + '__scales', scale)
                if not math.isfinite(self.scales[name]) or self.scales[name] <= 0: raise ValueError('Invalid input scale')
        self.eval()

    def w(self, key): return self.weights[key.replace('.', '__')]
    def source_state(self): return {k: self.w(k).detach() for k in self._source_keys}
    def norm(self, x, name):
        return F.layer_norm(x, (256,), self.w(name+'.weight'), self.w(name+'.bias'), self.config['layer_norm_eps'])
    def linear(self, x, name):
        if name in LINEARS and self.observe:
            self.observed[name] = max(self.observed.get(name, 0.), float(x.detach().abs().max()))
        if self.variant == 'fp32' or name not in LINEARS:
            return F.linear(x, self.w(name+'.weight'), self.w(name+'.bias'))
        scale = self.scales[name]
        if self.observe:
            previous = self.saturation.get(name, [0, 0])
            self.saturation[name] = [previous[0] + int((x.abs() / scale >= 127.5).sum()), previous[1] + x.numel()]
        codes = getattr(self, name.replace('.', '__')+'__codes')
        ws = getattr(self, name.replace('.', '__')+'__scales')
        acc = integer_dot(quantize(x, scale), codes)
        # Bias is added after FP32 dequantization, never rounded into int32.
        return (acc.float() * scale) * ws + self.w(name+'.bias')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, return_hidden=False):
        b, n = input_ids.shape
        if n > 256 or n == 0: raise ValueError('Supported window length is 1..256')
        if attention_mask is None: attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None: position_ids = torch.arange(n, device=input_ids.device)[None, :]
        x = F.embedding(input_ids, self.w('bert.embeddings.word_embeddings.weight'))
        x = x + F.embedding(token_type_ids, self.w('bert.embeddings.token_type_embeddings.weight'))
        x = x + F.embedding(position_ids, self.w('bert.embeddings.position_embeddings.weight'))
        x = self.norm(x, 'bert.embeddings.LayerNorm')
        additive_mask = (1.0-attention_mask[:,None,None,:].float()) * torch.finfo(torch.float32).min
        hidden = [x] if return_hidden else None
        mm = bf16_matmul if self.variant == 'w8a8_bf16_attention' else torch.matmul
        for i in range(4):
            p = f'bert.encoder.layer.{i}'
            q, k, v = [self.linear(x, p+'.attention.self.'+name).view(b,n,4,64).transpose(1,2) for name in ('query','key','value')]
            scores = mm(q, k.transpose(-1,-2)) / 8.0 + additive_mask
            probabilities = torch.softmax(scores, dim=-1)
            context = mm(probabilities, v).transpose(1,2).contiguous().view(b,n,256)
            x = self.norm(x+self.linear(context, p+'.attention.output.dense'), p+'.attention.output.LayerNorm')
            intermediate = F.gelu(self.linear(x, p+'.intermediate.dense'), approximate='none')
            x = self.norm(x+self.linear(intermediate, p+'.output.dense'), p+'.output.LayerNorm')
            if return_hidden: hidden.append(x)
        pooled = torch.tanh(self.linear(x[:,0], 'bert.pooler.dense'))
        logits = self.linear(pooled, 'classifier')
        return SimpleNamespace(logits=logits, hidden_states=hidden)

def load_model(variant='fp32', scales=None, device='cpu', checkpoint=CHECKPOINT, observe=False):
    if variant == 'fp32_reference': variant = 'fp32'
    if sha(checkpoint) != CHECKPOINT_SHA256: raise ValueError('Matched checkpoint identity changed')
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    expected = {'hidden_size':256,'intermediate_size':1024,'num_attention_heads':4,'num_hidden_layers':4,'hidden_act':'gelu','layer_norm_eps':1e-12}
    if any(config[k] != v for k,v in expected.items()): raise ValueError('Unsupported backbone semantics')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model = ReadableBert(state, config, variant, scales, observe).to(device)
    return model, {'checkpoint':str(checkpoint), 'checkpoint_sha256':CHECKPOINT_SHA256, 'variant':variant, 'config':config,
                   'source_config_sha256':sha(CONFIG),'model_source_sha256':sha(Path(__file__)),
                   'attention':'bidirectional key padding; FP32 minimum additive mask', 'dropout':'disabled evaluation',
                   'weight_codes':'symmetric per output RNE [-127,127]', 'linear_accumulator':'INT32 exact',
                   'nonlinear':'FP32 exact erf GELU, LayerNorm eps1e-12, softmax, tanh', 'pooler_classifier':'FP32'}

def inventory(model):
    tensors = [{'name':k,'shape':list(v.shape),'dtype':str(v.dtype),'elements':v.numel(),'bytes':v.numel()*v.element_size()} for k,v in model.source_state().items()]
    n,d,f,layers = 256,256,1024,4
    per_layer = {'qkv':3*n*d*d,'attention_output':n*d*d,'qk_and_pv':2*n*n*d,'feed_forward':2*n*d*f}
    return {'parameters':tensors,'parameter_elements':sum(t['elements'] for t in tensors),'parameter_bytes_fp32':sum(t['bytes'] for t in tensors),
            'matrix_macs_per_layer':per_layer,'encoder_macs_per_window':layers*sum(per_layer.values()),
            'pooler_classifier_macs':d*d+d*2, 'window_length':n, 'layers':layers,'heads':4,'head_dim':64,
            'operators':[{'name':'word_embedding','parameter_shape':[30522,256],'output':[1,n,d]},
                         {'name':'position_embedding','parameter_shape':[512,256],'output':[1,n,d]},
                         {'name':'token_type_embedding','parameter_shape':[2,256],'output':[1,n,d]},
                         {'name':'embedding_sum_then_LayerNorm','shape':[1,n,d],'epsilon':1e-12},
                         {'name':'Q/K/V_each_per_layer','input':[1,n,d],'weight':[d,d],'bias':[d],'output':[1,4,n,64]},
                         {'name':'QK_per_layer','left':[1,4,n,64],'right':[1,4,64,n],'output':[1,4,n,n],'scale':0.125},
                         {'name':'masked_softmax_per_layer','shape':[1,4,n,n],'axis':-1,'mask':'keys only; bidirectional'},
                         {'name':'PV_per_layer','left':[1,4,n,n],'right':[1,4,n,64],'output':[1,n,d]},
                         {'name':'attention_output_per_layer','weight':[d,d],'bias':[d],'residual_then_LayerNorm':True},
                         {'name':'FF1_per_layer','input':[1,n,d],'weight':[f,d],'bias':[f],'output':[1,n,f],'activation':'erf GELU'},
                         {'name':'FF2_per_layer','weight':[d,f],'bias':[d],'output':[1,n,d],'residual_then_LayerNorm':True},
                         {'name':'pooler','input':[1,d],'weight':[d,d],'bias':[d],'activation':'tanh','selection':'CLS index0'},
                         {'name':'classifier','input':[1,d],'weight':[2,d],'bias':[2],'output':[1,2]}],
            'excluded_from_macs':'embedding/LN/softmax/GELU/residual and host/document operations; analytical not measured',
            'activation_live_fp32_bytes':{'hidden':n*d*4,'qkv':3*n*d*4,'attention_scores':4*n*n*4,'ff_intermediate':n*f*4}}
