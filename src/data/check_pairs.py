"""
End-to-end data smoke check
Creates only temporary evaluation artifacts; does not overwrite saved pairs.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .pair_dataset import PairDataset
from .pair_generation import DEFAULT_PROCESSED_DIR, load_pair_index, generate_fixed_pairs, write_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cmu', 'aalto'], required=True)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--split-path', type=Path)
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    train = load_pair_index(args.dataset, 'train', args.processed_dir, args.split_path)
    evaluation = load_pair_index(args.dataset, 'test', args.processed_dir, args.split_path)
    if set(train.subjects) & set(evaluation.subjects):
        raise AssertionError('Subject leakage')
    dynamic = PairDataset(train, pairs_per_epoch=args.pairs)
    # Check labels, split membership, and session policy for every dynamic pair.
    lookup = {str(s): i for i, s in enumerate(train.sample_ids)}
    for i in range(len(dynamic)):
        pair = dynamic[i]
        a, b = lookup[pair['sample_id_a']], lookup[pair['sample_id_b']]
        label = bool(pair['label'])
        assert a != b and (train.subject_ids[a] == train.subject_ids[b]) == label
        if label and train.cross_session:
            assert train.sessions[a] != train.sessions[b]
    pairs = generate_fixed_pairs(evaluation, args.pairs)
    assert np.array_equal(pairs, generate_fixed_pairs(evaluation, args.pairs))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'pairs.npz'
        write_pairs(pairs, path)
        fixed = PairDataset(evaluation, fixed_pairs_path=path, split_name='test')
        for name, dataset in [('train', dynamic), ('test', fixed)]:
            batch = next(iter(DataLoader(dataset, batch_size=32, num_workers=args.workers)))
            for side in ('a', 'b'):
                sample = batch[side]
                assert tuple(sample['features'].shape[1:]) == (70, 3)
                assert np.isfinite(sample['features'].numpy()).all()
                assert (sample['mask'].sum(dim=1) == sample['length']).all()
                assert (sample['features'][~sample['mask']] == 0).all()
            print(f"{name}: features={tuple(batch['a']['features'].shape)}, "
                  f"masks={tuple(batch['a']['mask'].shape)}, "
                  f"lengths={tuple(batch['a']['length'].shape)}, labels={tuple(batch['label'].shape)}")
    print(f'PASS: {len(train.subjects)} train / {len(evaluation.subjects)} test subjects; '
          f'{args.pairs} dynamic and {len(pairs)} reproducible fixed pairs; '
          f'cross-session genuine policy={train.cross_session}')


if __name__ == '__main__':
    main()
