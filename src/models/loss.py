"""Contrastive loss for this project's convention: 1=genuine, 0=impostor."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """Mean y*d^2 + (1-y)*max(margin-d, 0)^2, using Euclidean distance.

    Unit embeddings have distances in [0,2]. Margin=1 is an initial training
    setting, not an authentication threshold. Select settings on validation.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        if not math.isfinite(margin) or not 0 < margin <= 2:
            raise ValueError('margin must be finite and in (0,2] for unit embeddings')
        self.margin = margin

    def forward(self, embedding_a, embedding_b, labels):
        if (embedding_a.ndim != 2 or embedding_a.shape != embedding_b.shape
                or embedding_a.shape[0] == 0):
            raise ValueError('Expected nonempty matching [batch, embedding] tensors')
        if labels.shape != (embedding_a.shape[0],) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError('Expected binary labels [batch]: 1=genuine, 0=impostor')
        distance = torch.linalg.vector_norm(embedding_a - embedding_b, dim=1)
        labels = labels.to(device=distance.device, dtype=distance.dtype)
        return (labels * distance.square()
                + (1 - labels) * F.relu(self.margin - distance).square()).mean()
