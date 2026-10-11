"""Two fixed Goal R1 challengers; no pretrained encoder features."""
from types import SimpleNamespace
import re
import zlib

import torch
from torch import nn

FEATURE_BINS = 262144


def word_features(text):
    """Word 1/2-gram term counts, deterministic feature hashing and L2 norm."""
    words = re.findall(r"(?u)\b\w+\b", text.lower())
    counts = {}
    for n in (1, 2):
        for i in range(len(words) - n + 1):
            key = (str(n) + ":" + " ".join(words[i:i+n])).encode("utf-8")
            index = zlib.crc32(key) % FEATURE_BINS
            counts[index] = counts.get(index, 0) + 1
    if not counts:
        return [0], [0.0]
    indices = sorted(counts)
    norm = sum(v*v for v in counts.values()) ** 0.5
    return indices, [counts[i] / norm for i in indices]


class FeatureLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.EmbeddingBag(FEATURE_BINS, 1, mode="sum")
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.coefficients.weight)

    def forward(self, indices, offsets, values):
        positive = self.coefficients(indices, offsets, per_sample_weights=values).flatten() + self.bias
        return SimpleNamespace(logits=torch.stack((torch.zeros_like(positive), positive), dim=1))


class TokenCNN(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=pad_id)
        self.convs = nn.ModuleList(nn.Conv1d(64, 64, k, padding=k//2) for k in (3, 5, 7))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(192, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        mask = attention_mask.bool().unsqueeze(1)
        # Mask embeddings explicitly; padding positions never enter max pooling.
        x = self.embedding(input_ids).transpose(1, 2) * mask
        pooled = [torch.relu(conv(x)).masked_fill(~mask, float("-inf")).amax(dim=2)
                  for conv in self.convs]
        return SimpleNamespace(logits=self.classifier(self.dropout(torch.cat(pooled, dim=1))))
