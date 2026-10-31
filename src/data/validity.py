"""Shared timing validity rules for parsed data and browser samples."""
import numpy as np

POLICY = 'finite_nonnegative_hold_and_dd_v1'


def valid_sequences(features, mask):
    """Return one eligibility flag per sequence. Negative UD remains valid.

    Padding does not participate. Length/mask structure is checked separately.
    """
    features = np.asarray(features)
    mask = np.asarray(mask, dtype=bool)
    if features.ndim != 3 or features.shape[-1] != 3 or features.shape[:2] != mask.shape:
        raise ValueError('Expected features [N,T,3] and mask [N,T]')
    finite = (np.isfinite(features) | ~mask[..., None]).all(axis=(1, 2))
    hold_ok = ((features[..., 0] >= 0) | ~mask).all(axis=1)
    transitions = mask.copy()
    transitions[:, 0] = False
    dd_ok = ((features[..., 2] >= 0) | ~transitions).all(axis=1)
    return finite & hold_ok & dd_ok


def check_lengths(mask, lengths):
    lengths = np.asarray(lengths)
    if (lengths < 1).any() or (lengths > mask.shape[1]).any():
        raise ValueError('Sequence lengths outside padded sequence bounds')
    expected = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError('Mask does not match sequence lengths')
