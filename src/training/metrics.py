"""Empirical verification metrics: accept when Euclidean distance <= threshold."""
import numpy as np


def verification_metrics(distances, labels):
    d, y = np.asarray(distances, dtype=float), np.asarray(labels)
    if d.ndim != 1 or y.shape != d.shape or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected matching finite distance/label vectors')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Both genuine=1 and impostor=0 labels are required')
    order = np.argsort(d, kind='stable')
    d, y = d[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(d)), len(d) - 1]
    thresholds = np.r_[np.nextafter(d[0], -np.inf), d[ends]]
    far = np.r_[0., np.cumsum(y == 0)[ends] / (y == 0).sum()]
    frr = np.r_[1., 1 - np.cumsum(y == 1)[ends] / (y == 1).sum()]
    best = int(np.argmin(np.abs(far - frr)))
    # Discrete operating point, not an interpolated crossing. Ties stay together.
    return dict(eer=float((far[best] + frr[best]) / 2),
                threshold=float(thresholds[best]), far=float(far[best]), frr=float(frr[best]),
                eer_method='nearest_empirical_far_frr')
