"""Approved span_coupled_pilot_v1 models; no data access or job dispatch.

The SoPa comparator is a bank of two-transition max-plus WFSAs with
learned self-loops and epsilon transitions, not a renamed coupled model.
"""
from types import SimpleNamespace
import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("coupled", "independent", "span_mlp", "sopa")
H, LEFT, RIGHT, R = 58, 28, 29, 32


def ste_quant(x, step, lo, hi):
    q = torch.clamp(torch.round(x / step), lo, hi) * step
    return x + (q - x).detach()


def pow2_exp(x, dimensions=None):
    peak = x.detach().abs().amax(dim=dimensions) if dimensions else x.detach().abs().max()
    return torch.ceil(torch.log2(torch.clamp(peak / 127, min=2.0**-30))).to(torch.int32)


def masks_for(ids, masks, content_start, document_content_length):
    """BERT windows have exactly CLS, contiguous content, SEP, then padding."""
    p = torch.arange(ids.shape[1], device=ids.device)[None, :]
    lengths = masks.sum(1).long() - 2
    content = (p >= 1) & (p < lengths[:, None] + 1)
    g = content_start[:, None] + p - 1
    end = content_start + lengths
    eligible = content & ((g - 31 >= content_start[:, None]) | (content_start[:, None] == 0))
    eligible &= ((g + 32 < end[:, None]) | (end[:, None] == document_content_length[:, None]))
    return content, eligible, lengths


class SpanGuard(nn.Module):
    def __init__(self, arm, vocab_size=30522, pad_id=0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.vocab_size, self.pad_id = arm, vocab_size, pad_id
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=pad_id)
        self.convs = nn.ModuleList(nn.Conv1d(64, 64, k, padding=k//2) for k in (3, 5, 7))
        self.proj = nn.Linear(192, 64)
        self.anchor = nn.Parameter(torch.zeros(192))
        self.bias = nn.Parameter(torch.zeros(()))
        if arm == "sopa":
            self.epsilon = nn.Parameter(torch.full((2, 16), -2.))
            self.pattern_head = nn.Parameter(torch.randn(16) * .02)
        else:
            self.lambdas = nn.Parameter(torch.full((2, R), .95))
            if arm == "span_mlp":
                self.mlp1 = nn.Linear(64, 32)
                self.mlp2 = nn.Linear(32, 1, bias=False)
            else:
                self.head_a = nn.Parameter(torch.randn(R) * .02)
                self.head_b = nn.Parameter(torch.randn(R) * .02)
                self.head_c = nn.Parameter(torch.randn(R) * .005)
        self.register_buffer("emb_exp", torch.zeros((), dtype=torch.int32))
        self.register_buffer("conv_exp", torch.zeros(192, dtype=torch.int32))
        self.register_buffer("proj_exp", torch.zeros(64, dtype=torch.int32))
        self.register_buffer("z_exp", torch.zeros((), dtype=torch.int32))
        self.register_buffer("observed_z_max", torch.zeros(()))
        self.register_buffer("quant_frozen", torch.tensor(False))
        self.qat = False
        self.observe = False

    def freeze_quantization(self):
        self.emb_exp.copy_(pow2_exp(self.embedding.weight))
        self.conv_exp.copy_(torch.cat([pow2_exp(c.weight, (1, 2)) for c in self.convs]))
        self.proj_exp.copy_(pow2_exp(self.proj.weight, (1,)))
        self.z_exp.copy_(torch.ceil(torch.log2(torch.clamp(self.observed_z_max / 127, min=2.**-30))).int())
        self.quant_frozen.fill_(True)

    def qweight(self, x, exponent, output=False):
        if not self.qat:
            return x
        step = torch.pow(2., exponent.float())
        if output:
            step = step.reshape([-1] + [1] * (x.ndim - 1))
        return ste_quant(x, step, -127, 127)

    def q12(self, x):
        return ste_quant(x, 2.**-12, -2147483647, 2147483647) if self.qat else x

    def encode(self, input_ids, content):
        weight = self.qweight(self.embedding.weight, self.emb_exp)
        x = F.embedding(input_ids, weight, padding_idx=self.pad_id) * content[:, :, None]
        outputs = []
        for n, conv in enumerate(self.convs):
            ew = self.conv_exp[n*64:(n+1)*64]
            w = self.qweight(conv.weight, ew, True)
            b = conv.bias
            if self.qat:
                b = ste_quant(b, torch.pow(2., (self.emb_exp+ew).float()), -2.**62, 2.**62)
            outputs.append(F.relu(F.conv1d(x.transpose(1, 2), w, b, padding=conv.padding)))
        z = torch.cat(outputs, 1).transpose(1, 2) * content[:, :, None]
        if self.observe:
            self.observed_z_max.copy_(torch.maximum(self.observed_z_max, z.detach().max()))
        if self.qat:
            z = ste_quant(z, torch.pow(2., self.z_exp.float()), 0, 127)
        return z

    def projection(self, z, content):
        w = self.qweight(self.proj.weight, self.proj_exp, True)
        b = self.proj.bias
        if self.qat:
            b = ste_quant(b, torch.pow(2., (self.z_exp+self.proj_exp).float()), -2.**62, 2.**62)
        p = F.linear(z, w, b)
        if self.arm != "sopa":
            p = torch.clamp(p, -8, 8)
        return self.q12(p) * content[:, :, None]

    def paired_states(self, p):
        padded = F.pad(p, (0, 0, LEFT, RIGHT))
        state = torch.zeros_like(p)
        lam = self.lambdas.clamp(0, 1).reshape(64)
        if self.qat:
            lam = ste_quant(lam, 1/256, 0, 256)
        for age in range(H):
            previous = self.q12(state * lam)
            state = torch.clamp(previous + padded[:, age:age+p.shape[1]], -8, 8)
        return state[..., :R], state[..., R:]

    def coupled_head(self, a, b, z):
        anchor = (z * self.q12(self.anchor)).sum(-1)
        if self.arm == "span_mlp":
            hidden = F.relu(F.linear(torch.cat((a,b), -1), self.q12(self.mlp1.weight), self.q12(self.mlp1.bias)))
            hidden = self.q12(hidden)
            core = F.linear(hidden, self.q12(self.mlp2.weight)).squeeze(-1)
        else:
            product = self.q12(a*b)
            core = (a*self.q12(self.head_a)+b*self.q12(self.head_b)+product*self.q12(self.head_c)).sum(-1)
        return self.q12(core)+self.q12(anchor)+self.q12(self.bias)

    def sopa_score(self, p, z, content):
        # 16 two-edge patterns; each has two consuming transitions, two loops,
        # and epsilon alternatives, using the max-plus Viterbi semiring.
        nonlinear_input=ste_quant(p,1/128,-2048,2048) if self.qat else p
        main = torch.tanh(nonlinear_input[..., :32]).reshape(*p.shape[:2], 2, 16)
        loops = F.logsigmoid(nonlinear_input[..., 32:]).reshape(*p.shape[:2], 2, 16)
        eps = F.logsigmoid(self.epsilon)
        if self.qat:
            main, loops, eps = self.q12(main), self.q12(loops), self.q12(eps)
        s1 = eps[0].expand(p.shape[0], -1)
        s2 = (eps[0]+eps[1]).expand(p.shape[0], -1)
        pooled = s2
        for t in range(p.shape[1]):
            n1 = torch.maximum(torch.maximum(s1+loops[:,t,0], main[:,t,0]), eps[0])
            n2 = torch.maximum(torch.maximum(s2+loops[:,t,1], s1+main[:,t,1]), n1+eps[1])
            s1 = torch.where(content[:,t,None], n1, s1)
            s2 = torch.where(content[:,t,None], n2, s2)
            pooled = torch.maximum(pooled, s2)
        global_z = z.masked_fill(~content[...,None], -1e30).max(1).values
        global_z = torch.where(content.any(1)[:,None], global_z, torch.zeros_like(global_z))
        return self.q12((pooled*self.q12(self.pattern_head)).sum(-1)) + self.q12((global_z*self.q12(self.anchor)).sum(-1)) + self.q12(self.bias)

    def forward(self, input_ids, attention_mask, content_start, document_content_length, head_swap=False):
        content, eligible, lengths = masks_for(input_ids, attention_mask, content_start, document_content_length)
        z = self.encode(input_ids, content)
        p = self.projection(z, content)
        if self.arm == "sopa":
            logit = self.sopa_score(p, z, content)
            anchor_logits = None
        else:
            a, b = self.paired_states(p)
            if self.arm == "independent" or head_swap:
                def pool(x):
                    y = x.masked_fill(~eligible[...,None], -1e30).max(1).values
                    return torch.where(eligible.any(1)[:,None], y, torch.zeros_like(y))
                logit = self.coupled_head(pool(a), pool(b), pool(z))
                anchor_logits = None
            else:
                anchor_logits = self.coupled_head(a, b, z).masked_fill(~eligible, -1e30)
                logit = anchor_logits.max(1).values
        no_anchor = ~eligible.any(1) if self.arm != "sopa" else ~content.any(1)
        empty = document_content_length == 0
        # Finite constant training CE for no-eligible-anchor windows. Scoring
        # explicitly emits zero risk; empty documents instead use head bias.
        logit = torch.where(no_anchor, logit*0-16, logit)
        logit = torch.where(empty, self.q12(self.bias).expand_as(logit), logit)
        return SimpleNamespace(logits=torch.stack((torch.zeros_like(logit),logit),1),
            risk=torch.where(no_anchor & ~empty, torch.zeros_like(logit), torch.sigmoid(logit)),
            anchor_logits=anchor_logits, eligible=eligible, no_anchor=no_anchor, empty=empty)


def model_metadata(model):
    all_count = sum(p.numel() for p in model.parameters())
    return dict(arm=model.arm, vocab_size=model.vocab_size, pad_id=model.pad_id,
        H=H, R=R, context=[-LEFT,RIGHT], footprint=[-31,32], kernels=[3,5,7],
        embedding_width=64, convolution_channels_each=64, parameters=all_count,
        nonembedding_parameters=all_count-model.embedding.weight.numel(),
        sopa_topology="16 two-transition max-plus WFSAs, loops and epsilon; full-window pooling" if model.arm=="sopa" else None)
