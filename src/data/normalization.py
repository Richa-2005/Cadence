"""
Versioned, training-fitted timing transform shared by training and inference.
Signed log with a fixed 0.1-second reference, then per-channel standardisation.
Fit uses all eligible train keystrokes, excluding padding and first-key UD/DD.
Those structural zeros are restored after transformation. No hard clipping.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .pair_generation import DEFAULT_PROCESSED_DIR, load_pair_index
from .schema import FEATURE_NAMES
from .validity import POLICY, valid_sequences, check_lengths

VERSION = 'signed_log_standardize_v1'


@dataclass(frozen=True)
class TimingNormalizer:
    mean: np.ndarray
    std: np.ndarray
    reference_seconds: float = 0.1
    provenance: list[dict] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, 'mean', np.asarray(self.mean, dtype=np.float64))
        object.__setattr__(self, 'std', np.asarray(self.std, dtype=np.float64))
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or (self.std <= 0).any() or not np.isfinite(self.reference_seconds)
                or self.reference_seconds <= 0):
            raise ValueError('Invalid normalization parameters')

    def transform(self, features, mask):
        """Accept one [T,3] sample or a batch [N,T,3]; return float32."""
        features = np.asarray(features)
        mask = np.asarray(mask, dtype=bool)
        single = features.ndim == 2
        if single:
            features, mask = features[None], mask[None]
        if not valid_sequences(features, mask).all():
            raise ValueError('Invalid timing sequence; reject before normalization')
        check_lengths(mask, mask.sum(axis=1))
        # Padding may contain arbitrary values; never transform it.
        x = np.where(mask[..., None], features, 0).astype(np.float64)
        logged = np.sign(x) * np.log1p(np.abs(x) / self.reference_seconds)
        output = ((logged - self.mean) / self.std).astype(np.float32)
        output[~mask] = 0
        output[:, 0, 1:] = 0
        return output[0] if single else output

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=VERSION, feature_names=FEATURE_NAMES, units='seconds',
                       validity_policy=POLICY, reference_seconds=self.reference_seconds,
                       mean=self.mean.tolist(), std=self.std.tolist(),
                       first_transition='restore_zero', provenance=self.provenance)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if (data['version'] != VERSION or data['feature_names'] != FEATURE_NAMES
                or data['units'] != 'seconds' or data['validity_policy'] != POLICY
                or data['first_transition'] != 'restore_zero'):
            raise ValueError('Unsupported normalization artifact')
        return cls(data['mean'], data['std'], data['reference_seconds'], data['provenance'])


def fit_normalizer(indices, reference_seconds=0.1):
    """Merge shard statistics with float64 parallel variance; no full-data copy.

    Multiple TRAIN indices can be supplied for a future mixed-dataset run.
    Each keystroke has equal weight, so Aalto dominates a combined fit.
    """
    if not np.isfinite(reference_seconds) or reference_seconds <= 0:
        raise ValueError('reference_seconds must be positive and finite')
    indices = list(indices)
    if not indices or any(index.split_name != 'train' or index.validity_policy != POLICY
                          for index in indices):
        raise ValueError('Fit requires validity-filtered training indices only')
    counts = np.zeros(3, dtype=np.int64)
    means, m2 = np.zeros(3), np.zeros(3)
    provenance = []
    for index in indices:
        provenance.append(dict(dataset=index.dataset, split='train', samples=len(index.sample_ids),
                               excluded_samples=index.excluded_samples,
                               sample_ids_sha256=hashlib.sha256(
                                   '\n'.join(map(str, index.sample_ids)).encode()).hexdigest()))
        for shard, path in enumerate(index.shard_paths):
            positions = np.flatnonzero(index.shard_indices == shard)
            if not len(positions):
                continue
            rows = index.row_indices[positions]
            with np.load(path, allow_pickle=True) as data:
                x = data['features'][rows].astype(np.float64)
                mask = data['mask'][rows].astype(bool)
                check_lengths(mask, data['seq_len'][rows])
            if not valid_sequences(x, mask).all():
                raise ValueError('Training index contains invalid sequences')
            x = np.where(mask[..., None], x, 0)
            x = np.sign(x) * np.log1p(np.abs(x) / reference_seconds)
            for channel in range(3):
                valid = mask.copy()
                if channel:
                    valid[:, 0] = False
                values = x[..., channel][valid]
                n = len(values)
                if not n:
                    continue
                mean = values.mean()
                delta = mean - means[channel]
                total = counts[channel] + n
                m2[channel] += ((values - mean) ** 2).sum() + delta**2 * counts[channel] * n / total
                means[channel] += delta * n / total
                counts[channel] = total
    if (counts == 0).any():
        raise ValueError('Insufficient train observations for each channel')
    std = np.sqrt(m2 / counts)
    std = np.maximum(std, 1e-8)
    normalizer = TimingNormalizer(means, std, reference_seconds, provenance)
    return normalizer, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cmu', 'aalto'], required=True)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--split-path', type=Path)
    parser.add_argument('--out-path', type=Path, required=True)
    args = parser.parse_args()
    index = load_pair_index(args.dataset, 'train', args.processed_dir, args.split_path)
    print(f'Eligible train sequences: {len(index.sample_ids)}; excluded: {index.excluded_samples}', flush=True)
    print(f'Subjects without valid samples: {len(index.subjects_without_valid_samples)}', flush=True)
    normalizer, counts = fit_normalizer([index])
    normalizer.save(args.out_path)
    print(f'Observations [H, UD, DD]: {counts.tolist()}')
    print(f'Means after signed log: {normalizer.mean.tolist()}')
    print(f'Std after signed log: {normalizer.std.tolist()}')
    print(f'Saved: {args.out_path}')


if __name__ == '__main__':
    main()
