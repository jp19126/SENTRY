"""Study-owned CNN construction and training-only pooled representation transfer."""
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / 'scripts'))
from cnn_models import StudyCNN


def build_student(config, vocab_size=30522, pad_id=0):
    return StudyCNN(config, vocab_size, pad_id)


def masked_pool(features, mask):
    """Concatenated mean then max; padding excluded, empty rows become zero."""
    if features.ndim != 3 or mask.shape != (features.shape[0], 1, features.shape[2]):
        raise ValueError('Expected BCL features and B1L mask')
    mask = mask.bool()
    mean = features.masked_fill(~mask, 0).sum(-1) / mask.sum(-1).clamp_min(1)
    maximum = features.masked_fill(~mask, -torch.inf).amax(-1)
    maximum = maximum.masked_fill(~mask.any(-1), 0)
    return torch.cat((mean, maximum), dim=1)


def forward_training(model, input_ids, attention_mask):
    """Same deployed graph and RNG consumption; additional pooling has no RNG."""
    features, mask = model.forward_features(input_ids, attention_mask)
    logits = model.classifier(model.dropout(model.pool(features, mask)))
    return SimpleNamespace(logits=logits, representation=masked_pool(features, mask))


class RepresentationHead(nn.Module):
    """Removed before deployment; cosine loss separately normalizes mean and max."""
    def __init__(self, student_width, teacher_width=256):
        super().__init__()
        self.projection = nn.Linear(2 * student_width, 2 * teacher_width)

    def forward(self, student, teacher):
        predicted = self.projection(student).reshape(len(student), 2, -1)
        target = teacher.to(predicted).reshape(len(student), 2, -1)
        # eps handles zero features without NaN. Unit is a window, not a token.
        return (1 - F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)).mean(-1)


def receptive_field(config):
    if config['family'] not in {'dilated', 'residual', 'depthwise'}:
        raise ValueError('Receptive-field formula only covers sequential nonstrided families')
    kernels, dilations = config['kernel_sizes'], config['dilations']
    return 1 + sum((kernels[i % len(kernels)] - 1) * dilations[i % len(dilations)]
                   for i in range(config['depth']))
