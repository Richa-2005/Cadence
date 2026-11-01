"""Aalto-only pilot: dynamic train pairs, fixed validation, no test access."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.normalization import TimingNormalizer
from src.data.pair_generation import load_pair_index
from src.data.pair_dataset import PairDataset
from src.models.encoder import SiameseNetwork
from src.models.loss import ContrastiveLoss
from .metrics import verification_metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count, distances, labels = 0., 0, [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            a, b = model(batch['a']['features'].to(device), batch['a']['length'],
                         batch['b']['features'].to(device), batch['b']['length'])
            y = batch['label'].to(device)
            loss = criterion(a, b, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.detach().item() * len(y)
            count += len(y)
            if not training:
                distances.extend(torch.linalg.vector_norm(a - b, dim=1).detach().cpu().tolist())
                labels.extend(y.cpu().tolist())
    if not count:
        raise ValueError('Empty loader')
    result = dict(loss=total_loss / count, pairs=count)
    if not training:
        result.update(verification_metrics(distances, labels))
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-dir', type=Path, required=True)
    p.add_argument('--split-path', type=Path, required=True)
    p.add_argument('--normalizer-path', type=Path, required=True)
    p.add_argument('--val-pairs-path', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--pairs-per-epoch', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--margin', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dataset-version', required=True, help='Kaggle input dataset version or identifying description')
    args = p.parse_args()
    if min(args.epochs, args.pairs_per_epoch, args.batch_size) < 1 or args.seed < 0 or not 0 < args.lr < float('inf'):
        p.error('Invalid epoch, pair, batch, seed or learning-rate setting')
    # A fresh directory prevents accidental overwrite of previous runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; enable a Kaggle GPU or explicitly use --device cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    normalizer = TimingNormalizer.load(args.normalizer_path)
    print('Loading valid training and validation samples...', flush=True)
    train_index = load_pair_index('aalto', 'train', args.processed_dir, args.split_path)
    val_index = load_pair_index('aalto', 'val', args.processed_dir, args.split_path)
    if set(train_index.subjects) & set(val_index.subjects):
        raise ValueError('Subject leakage')
    train = PairDataset(train_index, pairs_per_epoch=args.pairs_per_epoch, seed=args.seed,
                        normalizer=normalizer, loading='memory')
    val = PairDataset(val_index, fixed_pairs_path=args.val_pairs_path, split_name='val',
                      normalizer=normalizer, loading='memory')
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=0)
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(git_commit=git.stdout.strip() if git.returncode == 0 else None,
                  torch_version=str(torch.__version__), protocol='aalto_only_pilot',
                  artifact_hashes={k: sha256(getattr(args, k)) for k in
                                   ('split_path', 'normalizer_path', 'val_pairs_path')})
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    normalizer.save(args.output_dir / 'normalizer.json')
    best, history = float('inf'), []
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        start = perf_counter()
        training = run_epoch(model, train_loader, criterion, device, optimizer)
        validation = run_epoch(model, val_loader, criterion, device)
        row = dict(epoch=epoch + 1, train=training, val=validation, seconds=perf_counter() - start)
        history.append(row)
        improved = validation['eer'] < best
        best = min(best, validation['eer'])
        checkpoint = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          best_val_eer=best, config=config, history=history,
                          normalizer=json.loads((args.output_dir / 'normalizer.json').read_text()),
                          torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                          python_rng=random.getstate(),
                          cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        # Replace only after a complete checkpoint has been written.
        temporary = args.output_dir / 'checkpoint.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(args.output_dir / 'last.pt')
        if improved:
            import shutil
            shutil.copyfile(args.output_dir / 'last.pt', args.output_dir / 'best.pt')
        (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={training['loss']:.4f} "
              f"val_loss={validation['loss']:.4f} val_EER={validation['eer']:.2%} "
              f"FAR={validation['far']:.2%} FRR={validation['frr']:.2%} "
              f"seconds={row['seconds']:.1f}", flush=True)
    print(f'Pilot complete. Best validation EER: {best:.2%}. Outputs: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
