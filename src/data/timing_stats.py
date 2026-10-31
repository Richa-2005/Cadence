"""
Inspect a seeded, uniform sample of TRAIN sequences before choosing scaling.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .pair_generation import DEFAULT_PROCESSED_DIR, load_pair_index
from .schema import FEATURE_NAMES

PERCENTILES = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]


def summarize(values):
    finite = np.isfinite(values)
    clean = values[finite].astype(np.float64)
    result = {'count': int(len(values)), 'nonfinite': int((~finite).sum())}
    if len(clean):
        result.update({
            'mean_seconds': float(clean.mean()),
            'std_seconds': float(clean.std()),
            'negative_fraction': float((clean < 0).mean()),
            'zero_fraction': float((clean == 0).mean()),
            'abs_over_1s_fraction': float((np.abs(clean) > 1).mean()),
            'abs_over_5s_fraction': float((np.abs(clean) > 5).mean()),
            'percentiles_seconds': {str(p): float(v) for p, v in
                                    zip(PERCENTILES, np.percentile(clean, PERCENTILES))},
        })
    return result


def inspect_training(index, max_samples=50000, seed=42):
    if max_samples < 1 or seed < 0:
        raise ValueError('max_samples must be positive; seed must be non-negative')
    count = len(index.sample_ids)
    if not count:
        raise ValueError('No training samples')
    selected = np.sort(np.random.default_rng(seed).choice(
        count, size=min(max_samples, count), replace=False))
    channels = [[] for _ in FEATURE_NAMES]
    lengths = []
    for shard, path in enumerate(index.shard_paths):
        positions = selected[index.shard_indices[selected] == shard]
        if not len(positions):
            continue
        rows = index.row_indices[positions]
        with np.load(path, allow_pickle=True) as data:
            features = data['features'][rows]
            mask = data['mask'][rows].astype(bool)
            seq_len = data['seq_len'][rows]
        expected = np.arange(mask.shape[1])[None, :] < seq_len[:, None]
        if not np.array_equal(mask, expected):
            raise ValueError(f'Invalid length/mask relationship in {path}')
        lengths.extend(seq_len.tolist())
        for channel in range(len(FEATURE_NAMES)):
            valid = mask.copy()
            if channel > 0:
                valid[:, 0] = False
            channels[channel].append(features[:, :, channel][valid])
    return {
        'scope': 'training only; uniformly sampled sequences, not subject-balanced',
        'seed': seed, 'total_train_sequences': count,
        'sampled_sequences': len(selected),
        'length_percentiles': {str(p): float(v) for p, v in
                               zip(PERCENTILES, np.percentile(lengths, PERCENTILES))},
        'features': {name: summarize(np.concatenate(parts))
                     for name, parts in zip(FEATURE_NAMES, channels)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cmu', 'aalto'], required=True)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--split-path', type=Path)
    parser.add_argument('--max-samples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-path', type=Path, required=True)
    args = parser.parse_args()
    index = load_pair_index(args.dataset, 'train', args.processed_dir, args.split_path)
    report = inspect_training(index, args.max_samples, args.seed)
    report.update(dataset=args.dataset, processed_dir=str(args.processed_dir),
                  split_path=str(args.split_path) if args.split_path else 'default')
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Training sequences sampled: {report['sampled_sequences']} / {len(index.sample_ids)}")
    print('Timings in milliseconds; padding and first-key UD/DD excluded.')
    print(f"{'Feature':<9} {'median':>10} {'p95':>10} {'p99':>10} {'p99.9':>10} {'min':>10} {'max':>10}")
    for name, stats in report['features'].items():
        if 'percentiles_seconds' in stats:
            q = stats['percentiles_seconds']
            numbers = ' '.join(f'{q[p] * 1000:10.2f}' for p in ['50', '95', '99', '99.9', '0', '100'])
            print(f'{name:<9} {numbers}')
        print(f"  {name}: nonfinite={stats['nonfinite']}, "
              f"negative={stats.get('negative_fraction', 0):.2%}, "
              f"|timing|>1s={stats.get('abs_over_1s_fraction', 0):.2%}, "
              f"|timing|>5s={stats.get('abs_over_5s_fraction', 0):.2%}")
    print(f'Wrote report: {args.out_path}')


if __name__ == '__main__':
    main()
