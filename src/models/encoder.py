"""Shared timing encoder: masked 1D CNN -> packed LSTM -> unit embedding."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class KeystrokeEncoder(nn.Module):
    """Encode normalized [batch, time, H/UD/DD] sequences into 64 dimensions.

    The three timing features are Conv1d input channels; time is preserved.
    Mask after EVERY convolution so padded activations cannot feed back into
    real positions through the next convolution. Packing keeps the LSTM from
    processing padding. No batch normalization or temporal pooling is used.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.projection = nn.Linear(64, 64)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != 3:
            raise ValueError('features must have shape [batch, time, 3]')
        if not features.is_floating_point():
            raise ValueError('features must be floating point')
        if lengths.ndim != 1 or lengths.shape[0] != features.shape[0]:
            raise ValueError('lengths must have one entry per sample')
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('lengths must be integer tensors')
        if features.shape[0] == 0 or (lengths < 1).any() or (lengths > features.shape[1]).any():
            raise ValueError('lengths must be within the nonempty input sequence')
        positions = torch.arange(features.shape[1], device=features.device)
        mask = positions[None, :] < lengths.to(features.device)[:, None]
        x = features.masked_fill(~mask[..., None], 0).transpose(1, 2)
        channel_mask = mask[:, None, :]
        x = self.dropout(F.relu(self.conv1(x))).masked_fill(~channel_mask, 0)
        x = self.dropout(F.relu(self.conv2(x))).masked_fill(~channel_mask, 0)
        packed = pack_padded_sequence(x.transpose(1, 2), lengths.detach().cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return F.normalize(self.projection(hidden[-1]), p=2, dim=1)


class SiameseNetwork(nn.Module):
    """Both branches call the SAME encoder, so all encoder weights are shared."""

    def __init__(self, encoder: KeystrokeEncoder | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else KeystrokeEncoder()

    def forward(self, features_a, lengths_a, features_b, lengths_b):
        return self.encoder(features_a, lengths_a), self.encoder(features_b, lengths_b)
