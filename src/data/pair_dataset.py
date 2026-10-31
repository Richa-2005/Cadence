"""
PyTorch pair datasets shared by local CMU and Kaggle Aalto runs.
Labels: 1 = genuine, 0 = impostor. Features remain in raw seconds.
Each worker keeps its own bounded cache of decompressed NPZ shards.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .pair_generation import PairIndex, sample_genuine_pair, sample_impostor_pair


class FeatureStore:
    """Resolve split-local sample indices to shard rows without loading all shards."""

    def __init__(self, index: PairIndex, cache_size: int = 2):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.index = index
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def __getstate__(self):
        # Spawned DataLoader workers start with an empty cache.
        return {**self.__dict__, "_cache": OrderedDict()}

    def get(self, sample_index: int) -> dict[str, torch.Tensor]:
        shard = int(self.index.shard_indices[sample_index])
        row = int(self.index.row_indices[sample_index])
        if shard not in self._cache:
            # Evict before decompressing so peak cache use stays bounded.
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            with np.load(self.index.shard_paths[shard], allow_pickle=True) as data:
                self._cache[shard] = {key: data[key] for key in ("features", "mask", "seq_len")}
        self._cache.move_to_end(shard)
        arrays = self._cache[shard]
        return {
            "features": torch.from_numpy(arrays["features"][row].copy()).float(),
            "mask": torch.from_numpy(arrays["mask"][row].copy()).bool(),
            "length": torch.tensor(int(arrays["seq_len"][row]), dtype=torch.long),
        }


class PairDataset(Dataset):
    """Dynamic train pairs or fixed evaluation pairs, with one batch structure.

    Training sampling is deterministic for (seed, epoch, item), independent of
    worker scheduling. Call set_epoch(epoch) before each epoch. Use a regular
    DataLoader (persistent_workers=False) so workers receive epoch changes.
    Uniformly sampled subjects avoid favouring subjects with more samples.
    """

    def __init__(self, index: PairIndex, *, pairs_per_epoch: int = 10000,
                 seed: int = 42, fixed_pairs_path: Path | None = None,
                 cache_size: int = 2, split_name: str = "train"):
        if split_name not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if split_name != "train" and fixed_pairs_path is None:
            raise ValueError("Evaluation requires fixed pairs")
        if split_name == "train" and fixed_pairs_path is not None:
            raise ValueError("Training pairs must be sampled dynamically")
        if pairs_per_epoch < 1 or seed < 0:
            raise ValueError("pairs_per_epoch must be positive and seed non-negative")
        self.index = index
        self.store = FeatureStore(index, cache_size)
        self.seed, self.epoch = seed, 0
        self.pairs = None
        self.size = pairs_per_epoch
        if fixed_pairs_path is not None:
            lookup = {str(sample): i for i, sample in enumerate(index.sample_ids)}
            with np.load(fixed_pairs_path, allow_pickle=False) as data:
                a, b, labels = data["sample_id_a"], data["sample_id_b"], data["label"]
                if a.ndim != 1 or b.shape != a.shape or labels.shape != a.shape or not len(a):
                    raise ValueError("Invalid fixed pair array shapes")
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("Labels must be 0 or 1")
                try:
                    ai = np.array([lookup[str(x)] for x in a], dtype=np.int64)
                    bi = np.array([lookup[str(x)] for x in b], dtype=np.int64)
                except KeyError as exc:
                    raise ValueError("Fixed pair references a sample outside this split") from exc
                genuine = labels == 1
                same_subject = index.subject_ids[ai] == index.subject_ids[bi]
                if (ai == bi).any() or not np.array_equal(same_subject, genuine):
                    raise ValueError("Invalid self pair or incorrect pair label")
                if index.cross_session and (index.sessions[ai[genuine]] == index.sessions[bi[genuine]]).any():
                    raise ValueError("Genuine pairs must use different sessions; regenerate this pair file")
                self.pairs = np.column_stack((ai, bi, labels)).astype(np.int64)
                self.size = len(labels)
        elif len(index.subjects) < 2 or not index.genuine_subjects:
            raise ValueError("Training requires two subjects and eligible genuine pairs")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self):
        return self.size

    def __getitem__(self, item: int):
        if not 0 <= item < self.size:
            raise IndexError(item)
        if self.pairs is not None:
            a, b, label = map(int, self.pairs[item])
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, item]))
            label = int(rng.integers(2))
            sampler = sample_genuine_pair if label else sample_impostor_pair
            a, b = sampler(self.index, rng)
        return {
            "a": self.store.get(a), "b": self.store.get(b),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id_a": str(self.index.sample_ids[a]),
            "sample_id_b": str(self.index.sample_ids[b]),
        }
