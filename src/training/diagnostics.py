"""Validation-only checkpoint diagnostics. Never loads test pairs or subjects."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.normalization import TimingNormalizer
from src.data.pair_dataset import PairDataset
from src.data.pair_generation import load_pair_index
from src.models.encoder import SiameseNetwork
from .metrics import verification_metrics
from .train import sha256


def operating_points(distances, labels, targets=(.01, .001)):
    """Most permissive empirical threshold with FAR <= target, keeping ties intact."""
    d, y = np.asarray(distances), np.asarray(labels)
    verification_metrics(d, y)  # validate inputs and class coverage
    thresholds = np.r_[np.nextafter(float(d.min()), -np.inf), np.unique(d)]
    genuine, impostor = np.sort(d[y == 1]), np.sort(d[y == 0])
    far = np.searchsorted(impostor, thresholds, side='right') / len(impostor)
    frr = 1 - np.searchsorted(genuine, thresholds, side='right') / len(genuine)
    result = []
    for target in targets:
        if not 0 <= target <= 1:
            raise ValueError('FAR target must be in [0,1]')
        position = np.flatnonzero(far <= target)[-1]
        result.append(dict(target_far=target, threshold=float(thresholds[position]),
                           far=float(far[position]), frr=float(frr[position])))
    return result


def error_summary(labels, rejected):
    labels, rejected = np.asarray(labels), np.asarray(rejected)
    genuine, impostor = labels == 1, labels == 0
    return dict(pairs=len(labels), genuine_pairs=int(genuine.sum()), impostor_pairs=int(impostor.sum()),
                frr=float(rejected[genuine].mean()) if genuine.any() else None,
                far=float((~rejected[impostor]).mean()) if impostor.any() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--processed-dir', type=Path, required=True)
    p.add_argument('--split-path', type=Path, required=True)
    p.add_argument('--normalizer-path', type=Path, required=True)
    p.add_argument('--val-pairs-path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    # Trusted checkpoints from our own runs contain Python and NumPy RNG state.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for name in ('split_path', 'normalizer_path', 'val_pairs_path'):
        if saved['config']['artifact_hashes'][name] != sha256(getattr(args, name)):
            raise ValueError(f'Artifact differs from training: {name}')
    normalizer = TimingNormalizer.load(args.normalizer_path)
    index = load_pair_index('aalto', 'val', args.processed_dir, args.split_path)
    dataset = PairDataset(index, fixed_pairs_path=args.val_pairs_path, split_name='val',
                          loading='memory', normalizer=normalizer)
    device = torch.device(args.device)
    model = SiameseNetwork().to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    distances, labels, shortest, ids_a, ids_b = [], [], [], [], []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=256, num_workers=0):
            a, b = model(batch['a']['features'].to(device), batch['a']['length'],
                         batch['b']['features'].to(device), batch['b']['length'])
            distances.extend(torch.linalg.vector_norm(a-b, dim=1).cpu().tolist())
            labels.extend(batch['label'].tolist())
            shortest.extend(torch.minimum(batch['a']['length'], batch['b']['length']).tolist())
            ids_a.extend(batch['sample_id_a'])
            ids_b.extend(batch['sample_id_b'])
    d, y, lengths = np.asarray(distances), np.asarray(labels, dtype=int), np.asarray(shortest)
    eer = verification_metrics(d, y)
    rejected = d > eer['threshold']
    quantiles = {name: dict(zip(['p1', 'p5', 'p50', 'p95', 'p99'],
                                np.percentile(d[y == label], [1, 5, 50, 95, 99]).tolist()))
                 for name, label in [('genuine', 1), ('impostor', 0)]}
    buckets = {}
    for low, high in [(1, 10), (11, 20), (21, 40), (41, 70)]:
        keep = (lengths >= low) & (lengths <= high)
        buckets[f'{low}-{high}'] = error_summary(y[keep], rejected[keep])
    subjects = dict(zip(index.sample_ids.astype(str), index.subject_ids.astype(str)))
    groups = defaultdict(list)
    for row, (a, b) in enumerate(zip(ids_a, ids_b)):
        for subject in {subjects[a], subjects[b]}:
            groups[subject].append(row)
    per_subject = []
    for subject, rows in groups.items():
        per_subject.append(dict(subject_id=subject, **error_summary(y[rows], rejected[rows])))
    report = dict(scope='validation only; thresholds selected and measured on validation',
                  checkpoint_epoch=saved['epoch'], checkpoint_sha256=sha256(args.checkpoint),
                  eer=eer, operating_points=operating_points(d, y), distance_quantiles=quantiles,
                  length_buckets_at_eer_threshold=buckets,
                  length_bucket_definition='minimum sequence length of pair members',
                  subject_note='Pair appearances per subject; small counts are noisy, not independent trials.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'subjects.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'pairs', 'genuine_pairs', 'impostor_pairs', 'frr', 'far'])
        writer.writeheader()
        writer.writerows(per_subject)
    np.savez_compressed(args.output_dir / 'scores.npz', distance=d, label=y,
                        min_length=lengths, sample_id_a=ids_a, sample_id_b=ids_b)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 2, 81)
    for label, name in [(1, 'Genuine'), (0, 'Impostor')]:
        ax.hist(d[y == label], bins=bins, density=True, alpha=.5, label=name)
    ax.axvline(eer['threshold'], color='black', linestyle='--', label='Validation EER threshold')
    ax.set(xlabel='Embedding Euclidean distance', ylabel='Density', title='Aalto validation distances')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / 'distances.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Saved diagnostics: {args.output_dir}')


if __name__ == '__main__':
    main()
