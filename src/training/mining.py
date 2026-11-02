"""Balanced pair selection from a shared candidate pool."""
import numpy as np


def select_pairs(labels, distances, *, mode, size, seed, epoch, step):
    if mode not in {'random', 'hard'} or size < 2 or size % 2:
        raise ValueError('Selection requires random/hard mode and an even size >= 2')
    labels = np.asarray(labels)
    distances = np.asarray(distances)
    if labels.ndim != 1 or distances.shape != labels.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid candidate labels or distances')
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite candidate scores')
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n = min(size // 2, len(positives), len(negatives))
    if not n:
        return np.empty(0, dtype=np.int64)
    positive_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, step, 0]))
    negative_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, step, 1]))
    chosen_pos = positive_rng.choice(positives, n, replace=False)
    chosen_neg = (negative_rng.choice(negatives, n, replace=False) if mode == 'random'
                  else negatives[np.argsort(distances[negatives], kind='stable')[:n]])
    return np.concatenate([chosen_pos, chosen_neg])
