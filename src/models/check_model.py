"""One optimizer-step smoke check on TRAIN pairs; no evaluation or saved model."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.normalization import TimingNormalizer
from src.data.pair_dataset import PairDataset
from src.data.pair_generation import DEFAULT_PROCESSED_DIR, load_pair_index
from .encoder import SiameseNetwork
from .loss import ContrastiveLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cmu', 'aalto'], required=True)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--split-path', type=Path)
    parser.add_argument('--normalizer-path', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.manual_seed(42)
    normalizer = TimingNormalizer.load(args.normalizer_path)
    index = load_pair_index(args.dataset, 'train', args.processed_dir, args.split_path)
    dataset = PairDataset(index, pairs_per_epoch=32, loading='memory', normalizer=normalizer)
    batch = next(iter(DataLoader(dataset, batch_size=32, num_workers=0)))
    device = torch.device(args.device)
    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    a, b = model(batch['a']['features'].to(device), batch['a']['length'],
                 batch['b']['features'].to(device), batch['b']['length'])
    assert a.shape == b.shape == (32, 64)
    for embedding in (a, b):
        torch.testing.assert_close(embedding.norm(dim=1), torch.ones(32, device=device))
    loss = ContrastiveLoss()(a, b, batch['label'].to(device))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    print(f'PASS: {args.dataset} normalized train batch -> two {tuple(a.shape)} embeddings')
    print(f'Parameters: {sum(p.numel() for p in model.parameters())}; device: {device}')
    print(f'Initial contrastive loss: {loss.detach().item():.6f}')
    print('Forward, backward, unit norms, finite gradients, optimizer step: PASS')
    print('This checks execution, not learned accuracy. No model saved.')


if __name__ == '__main__':
    main()
