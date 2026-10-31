"""Measure indexing, feature setup and steady-state pair throughput separately."""
import argparse
from pathlib import Path
import resource
from time import perf_counter

from torch.utils.data import DataLoader

from .pair_generation import DEFAULT_PROCESSED_DIR, load_pair_index
from .pair_dataset import PairDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cmu', 'aalto'], required=True)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--split-path', type=Path)
    parser.add_argument('--loading', choices=['cached', 'memory'], default='memory')
    parser.add_argument('--batches', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error('batches and batch-size must be positive')
    start = perf_counter()
    index = load_pair_index(args.dataset, 'train', args.processed_dir, args.split_path)
    print(f'Metadata indexing: {perf_counter() - start:.1f} seconds', flush=True)
    start = perf_counter()
    dataset = PairDataset(index, pairs_per_epoch=args.batches * args.batch_size, loading=args.loading)
    print(f'Feature setup ({args.loading}): {perf_counter() - start:.1f} seconds', flush=True)
    if args.loading == 'memory':
        print(f'Resident feature arrays: {dataset.store.nbytes / 1024**3:.2f} GiB', flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    start = previous = perf_counter()
    for number, batch in enumerate(loader, start=1):
        now = perf_counter()
        print(f'Batch {number}/{args.batches}: {now - previous:.4f} seconds', flush=True)
        previous = now
    elapsed = perf_counter() - start
    print(f'Throughput: {len(dataset) / elapsed:.1f} pairs/second')
    # Linux reports KiB; macOS reports bytes.
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == 'darwin' else 1024)
    print(f'Peak process memory: {peak_mib:.0f} MiB')


if __name__ == '__main__':
    main()
