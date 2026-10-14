"""Small CNN registry for the development-only architecture study.

All temporal operations mask their outputs. Normalization is per-position,
never across sequence/batch positions. Padding therefore cannot create evidence
or affect normalization. Empty documents map to a zero summary plus head bias.
These are conventional model structures, not claims of algorithmic novelty.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ("textcnn", "stacked", "residual", "dpcnn", "dilated", "idcnn",
            "tcn", "depthwise", "grouped", "large_kernel", "gated",
            "inception", "local_context", "dense", "pooling", "byte_hierarchy")


def apply_mask(x, mask):
    return x.masked_fill(~mask, 0)


class MaskedConv(nn.Module):
    """Explicit same-length or causal convolution, including even kernels."""
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1,
                 groups=1, causal=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel,
                              dilation=dilation, groups=groups)
        extent = (kernel - 1) * dilation
        self.left = extent if causal else extent // 2
        self.right = 0 if causal else extent - self.left

    def forward(self, x, mask):
        x = self.conv(F.pad(apply_mask(x, mask), (self.left, self.right)))
        return apply_mask(x, mask), mask


class TokenNorm(nn.Module):
    def __init__(self, channels, kind="layer"):
        super().__init__()
        if kind not in ("layer", "none"):
            raise ValueError("normalization must be layer or none")
        self.norm = nn.LayerNorm(channels) if kind == "layer" else nn.Identity()

    def forward(self, x, mask):
        return apply_mask(self.norm(x.transpose(1, 2)).transpose(1, 2), mask)


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel=3, dilation=1, norm="layer",
                 dropout=.1, residual=False, gated=False, causal=False,
                 groups=1, separable=False):
        super().__init__()
        self.residual, self.gated = residual, gated
        self.temporal = MaskedConv(channels, channels if separable else
                                  channels * (2 if gated else 1), kernel,
                                  dilation, channels if separable else groups, causal)
        self.mixing = MaskedConv(channels, channels * (2 if gated else 1), 1) if separable else None
        self.norm = TokenNorm(channels, norm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y, _ = self.temporal(x, mask)
        if self.mixing is not None:
            y, _ = self.mixing(y, mask)
        if self.gated:
            a, b = y.chunk(2, dim=1)
            y = apply_mask(a * torch.sigmoid(b), mask)
        else:
            y = apply_mask(F.relu(y), mask)
        y = self.dropout(self.norm(y, mask))
        return apply_mask(x + y if self.residual else y, mask), mask


def downsample(x, mask):
    """Nonoverlapping masked max pairs, retaining an odd-length final token."""
    if x.shape[-1] % 2:
        x = F.pad(x, (0, 1))
        mask = F.pad(mask, (0, 1), value=False)
    new_mask = mask.reshape(mask.shape[0], 1, -1, 2).any(dim=-1)
    y = x.masked_fill(~mask, float("-inf")).reshape(x.shape[0], x.shape[1], -1, 2).amax(-1)
    return apply_mask(y, new_mask), new_mask


class InceptionBlock(nn.Module):
    def __init__(self, channels, kernels, norm, dropout, residual=True):
        super().__init__()
        self.branches = nn.ModuleList(MaskedConv(channels, channels, k) for k in kernels)
        self.fuse = MaskedConv(channels * len(kernels), channels, 1)
        self.norm, self.dropout = TokenNorm(channels, norm), nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, mask):
        branches = [apply_mask(F.relu(conv(x, mask)[0]), mask) for conv in self.branches]
        y, _ = self.fuse(torch.cat(branches, dim=1), mask)
        y = self.dropout(self.norm(apply_mask(F.relu(y), mask), mask))
        return apply_mask(x + y if self.residual else y, mask), mask


class LocalContextBlock(nn.Module):
    def __init__(self, channels, kernel, context_kernel, dilation, norm, dropout, gated):
        super().__init__()
        self.local = ConvBlock(channels, kernel, norm=norm, dropout=dropout)
        self.context = ConvBlock(channels, context_kernel, dilation, norm, dropout, True)
        self.fuse = MaskedConv(2 * channels, channels * (2 if gated else 1), 1)
        self.gated, self.norm = gated, TokenNorm(channels, norm)

    def forward(self, x, mask):
        local, _ = self.local(x, mask)
        context, _ = self.context(local, mask)
        y, _ = self.fuse(torch.cat((local, context), dim=1), mask)
        if self.gated:
            a, b = y.chunk(2, dim=1)
            y = a * torch.sigmoid(b)
        else:
            y = F.relu(y)
        return apply_mask(x + self.norm(y, mask), mask), mask


class DenseStack(nn.Module):
    def __init__(self, channels, growth, depth, kernel, norm, dropout):
        super().__init__()
        kernels = [kernel] if isinstance(kernel, int) else kernel
        self.layers = nn.ModuleList(MaskedConv(channels + i * growth, growth, kernels[i % len(kernels)]) for i in range(depth))
        self.norms = nn.ModuleList(TokenNorm(growth, norm) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.compress = MaskedConv(channels + growth * depth, channels, 1)

    def forward(self, x, mask):
        for conv, norm in zip(self.layers, self.norms):
            y, _ = conv(x, mask)
            y = self.dropout(norm(apply_mask(F.relu(y), mask), mask))
            x = apply_mask(torch.cat((x, y), dim=1), mask)
        x, _ = self.compress(x, mask)
        return apply_mask(F.relu(x), mask), mask


class EvidencePool(nn.Module):
    def __init__(self, channels, kind="max", topk=4):
        super().__init__()
        if kind not in ("max", "mean", "max_mean", "topk", "topk_mean", "attention"):
            raise ValueError(f"unknown pooling {kind}")
        if topk < 1:
            raise ValueError("topk must be positive")
        self.kind, self.topk = kind, topk
        self.output_dim = channels * (2 if kind == "max_mean" else topk if kind == "topk" else 1)
        self.attention = nn.Conv1d(channels, 1, 1) if kind == "attention" else None

    def forward(self, x, mask):
        count = mask.sum(-1).clamp_min(1)
        valid = mask.any(-1)
        maximum = x.masked_fill(~mask, float("-inf")).amax(-1).masked_fill(~valid, 0)
        mean = apply_mask(x, mask).sum(-1) / count
        if self.kind == "max":
            return maximum
        if self.kind == "mean":
            return mean
        if self.kind == "max_mean":
            return torch.cat((maximum, mean), dim=1)
        if self.kind == "attention":
            score = self.attention(apply_mask(x, mask)).masked_fill(~mask, -1e30)
            weights = apply_mask(torch.softmax(score, dim=-1), mask)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            return (x * weights).sum(-1)
        ranked = x.masked_fill(~mask, float("-inf"))
        if x.shape[-1] < self.topk:
            ranked = F.pad(ranked, (0, self.topk - x.shape[-1]), value=float("-inf"))
        values = ranked.topk(self.topk, dim=-1).values
        values = values.masked_fill(~torch.isfinite(values), 0)
        if self.kind == "topk":
            return values.flatten(1)
        return values.sum(-1) / count.clamp_max(self.topk)


class StudyCNN(nn.Module):
    def __init__(self, config, vocab_size, pad_id=0):
        super().__init__()
        self.study_config = dict(config)
        self.pad_id = pad_id
        family = config["family"]
        if family not in FAMILIES:
            raise ValueError(f"unknown CNN family {family}")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must belong to supplied vocabulary")
        emb, width = int(config.get("embedding_dim", 64)), int(config.get("channels", 64))
        depth = int(config.get("depth", 3))
        kernels = config.get("kernel_sizes", [3])
        dilations = config.get("dilations", [1])
        if min(emb, width, depth, *kernels, *dilations) < 1:
            raise ValueError("dimensions, kernels, depth and dilations must be positive")
        norm, dropout = config.get("normalization", "layer"), config.get("dropout", .1)
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.projection = MaskedConv(emb, width, 1) if emb != width or family != "textcnn" else None
        self.blocks = nn.ModuleList()
        self.backward_blocks = nn.ModuleList()
        self.shared_repeats = 1
        self.downsample_after = set()
        output_width = width
        if family == "textcnn":
            # Three independently pooled temporal branches, as in the R1 reference.
            self.blocks.extend(MaskedConv(width, width, k) for k in kernels)
            output_width *= len(kernels)
        elif family == "dense":
            self.blocks.append(DenseStack(width, config.get("growth", width // 2), depth, kernels, norm, dropout))
        elif family == "inception":
            self.blocks.extend(InceptionBlock(width, kernels, norm, dropout) for _ in range(depth))
        elif family == "local_context":
            self.blocks.extend(LocalContextBlock(width, kernels[0], config.get("context_kernel", 7),
                                                 dilations[i % len(dilations)], norm, dropout,
                                                 config.get("gated", False)) for i in range(depth))
        else:
            residual = family in ("residual", "dpcnn", "dilated", "idcnn", "tcn", "depthwise", "grouped", "large_kernel", "byte_hierarchy")
            schedule_depth = len(dilations) if family == "idcnn" else depth
            for i in range(schedule_depth):
                kw = dict(kernel=kernels[i % len(kernels)], dilation=dilations[i % len(dilations)],
                          norm=norm, dropout=dropout, residual=residual,
                          gated=family == "gated" or config.get("gated", False),
                          causal=family == "tcn" and config.get("direction", "causal") != "centered",
                          groups=config.get("groups", 4) if family == "grouped" else 1,
                          separable=family == "depthwise")
                self.blocks.append(ConvBlock(width, **kw))
                if family == "tcn" and config.get("direction") == "bidirectional":
                    self.backward_blocks.append(ConvBlock(width, **kw))
            if family == "idcnn":
                self.shared_repeats = int(config.get("repeats", 3))
                if self.shared_repeats < 1:
                    raise ValueError("IDCNN repeats must be positive")
            if family in ("dpcnn", "byte_hierarchy"):
                # Each level has two residual convolutions; DPCNN additionally
                # uses overlapping centered max-3 before pair decimation.
                self.downsample_after = set(range(1, depth - 1, 2))
            if self.backward_blocks:
                output_width *= 2
        self.pool = EvidencePool(output_width, config.get("pooling", "max"), config.get("topk", 4))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.pool.output_dim, 2)

    def forward_features(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch,length]")
        mask = input_ids.ne(self.pad_id) if attention_mask is None else attention_mask.bool()
        if mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        if input_ids.shape[1] == 0:
            input_ids = F.pad(input_ids, (0, 1), value=self.pad_id)
            mask = F.pad(mask, (0, 1), value=False)
        mask = mask.unsqueeze(1)
        x = apply_mask(self.embedding(input_ids.masked_fill(~mask.squeeze(1), self.pad_id)).transpose(1, 2), mask)
        if self.projection is not None:
            x, _ = self.projection(x, mask)
        if self.study_config["family"] == "textcnn":
            return torch.cat([apply_mask(F.relu(conv(x, mask)[0]), mask) for conv in self.blocks], dim=1), mask
        original, original_mask = x, mask
        for _ in range(self.shared_repeats):
            for i, block in enumerate(self.blocks):
                x, mask = block(x, mask)
                if i in self.downsample_after:
                    if self.study_config["family"] == "dpcnn":
                        x = F.max_pool1d(x.masked_fill(~mask, float("-inf")), 3, stride=1, padding=1)
                        x = apply_mask(x, mask)
                    x, mask = downsample(x, mask)
        if self.backward_blocks:
            y, reversed_mask = original.flip(-1), original_mask.flip(-1)
            for block in self.backward_blocks:
                y, reversed_mask = block(y, reversed_mask)
            x = torch.cat((x, y.flip(-1)), dim=1)
        return apply_mask(x, mask), mask

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x, mask = self.forward_features(input_ids, attention_mask)
        return SimpleNamespace(logits=self.classifier(self.dropout(self.pool(x, mask))))


def build_model(config, vocab_size, pad_id=0):
    return StudyCNN(config, vocab_size, pad_id)


def load_search_configs(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "configs/cnn_search_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    configs = data["configurations"]
    ids = [c["id"] for c in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("configuration IDs must be unique")
    if any(c["family"] not in FAMILIES for c in configs):
        raise ValueError("unknown family in registry")
    return configs


def model_cost(model, sequence_length=256, batch_size=1):
    """Analytical tensor/MAC counts from an unmeasured dummy shape execution.

    Conv/linear MACs exclude pooling, norms, activation functions and embedding
    lookup. Peak is the largest single observed tensor, not allocator peak or
    simultaneous activation storage. Shared modules are counted each execution.
    No latency, device resources, power or quantized speedup is inferred.
    """
    totals = {"conv_linear_macs": 0, "largest_module_output_elements": 0}
    handles = []
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            totals["largest_module_output_elements"] = max(totals["largest_module_output_elements"], output.numel())
        if isinstance(module, nn.Conv1d):
            totals["conv_linear_macs"] += output.numel() * module.kernel_size[0] * module.in_channels // module.groups
        elif isinstance(module, nn.Linear):
            totals["conv_linear_macs"] += output.numel() * module.in_features
    for module in model.modules():
        handles.append(module.register_forward_hook(hook))
    training = model.training
    try:
        model.eval()
        device = next(model.parameters()).device
        ids = torch.full((batch_size, max(sequence_length, 0)), model.pad_id, dtype=torch.long, device=device)
        with torch.no_grad():
            model(ids, torch.ones_like(ids))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    params = sum(p.numel() for p in model.parameters())
    embedding = model.embedding.weight.numel()
    return {**totals, "parameters": params, "embedding_parameters": embedding,
            "nonembedding_parameters": params - embedding, "parameter_bytes_fp32": 4 * params,
            "parameter_bytes_ideal_int8": params, "sequence_length": sequence_length,
            "batch_size": batch_size, "kind": "analytical tensor shape and operation counts",
            "exclusions": "norm/pool/nonlinear operations; workspace and simultaneous tensors; preprocessing/transfer/document aggregation"}
