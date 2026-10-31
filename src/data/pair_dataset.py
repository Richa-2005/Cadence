"""
PyTorch pair datasets shared by local CMU and Kaggle Aalto runs.
Labels: 1 = genuine, 0 = impostor. Optional saved normalizer transforms raw seconds.
Choose a bounded shard cache or preload the selected split into RAM.
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
        return sample_tensors(arrays, row)


def sample_tensors(arrays, row):
    return {
        "features": torch.from_numpy(arrays["features"][row].copy()).float(),
        "mask": torch.from_numpy(arrays["mask"][row].copy()).bool(),
        "length": torch.tensor(int(arrays["seq_len"][row]), dtype=torch.long),
    }


class InMemoryFeatureStore:
    """Decompress each relevant shard once and retain only this split's rows.

    Allocate final arrays first and fill shard by shard to avoid concatenating
    a second full copy. Use num_workers=0 to avoid worker copies of these arrays.
    """

    def __init__(self, index: PairIndex):
        self.arrays = {}
        count = len(index.sample_ids)
        if count == 0:
            raise ValueError("Cannot preload an empty split")
        for shard, path in enumerate(index.shard_paths):
            destinations = np.flatnonzero(index.shard_indices == shard)
            if not len(destinations):
                continue
            rows = index.row_indices[destinations]
            with np.load(path, allow_pickle=True) as data:
                for key in ("features", "mask", "seq_len"):
                    source = data[key]
                    if key not in self.arrays:
                        self.arrays[key] = np.empty((count, *source.shape[1:]), dtype=source.dtype)
                    target = self.arrays[key]
                    if source.shape[1:] != target.shape[1:] or source.dtype != target.dtype:
                        raise ValueError(f"Inconsistent {key} shape or dtype in {path}")
                    target[destinations] = source[rows]

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.arrays.values())

    def get(self, sample_index: int) -> dict[str, torch.Tensor]:
        return sample_tensors(self.arrays, sample_index)


class PairDataset(Dataset):
    """Dynamic train pairs or fixed evaluation pairs, with one batch structure.

    Training sampling is deterministic for (seed, epoch, item), independent of
    worker scheduling. Call set_epoch(epoch) before each epoch. Use a regular
    DataLoader (persistent_workers=False) so workers receive epoch changes.
    Uniformly sampled subjects avoid favouring subjects with more samples.
    """

    def __init__(self, index: PairIndex, *, pairs_per_epoch: int = 10000,
                 seed: int = 42, fixed_pairs_path: Path | None = None,
                 cache_size: int = 2, split_name: str = "train",
                 loading: str = "cached", normalizer=None):
        if split_name not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if split_name != "train" and fixed_pairs_path is None:
            raise ValueError("Evaluation requires fixed pairs")
        if split_name == "train" and fixed_pairs_path is not None:
            raise ValueError("Training pairs must be sampled dynamically")
        if pairs_per_epoch < 1 or seed < 0:
            raise ValueError("pairs_per_epoch must be positive and seed non-negative")
        self.index = index
        self.normalizer = normalizer
        if loading not in {"cached", "memory"}:
            raise ValueError("loading must be 'cached' or 'memory'")
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

        # Validate pair metadata before allocating potentially large arrays.
        self.store = (InMemoryFeatureStore(index) if loading == "memory"
                      else FeatureStore(index, cache_size))

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
        sample_a, sample_b = self.store.get(a), self.store.get(b)
        if self.normalizer is not None:
            for sample in (sample_a, sample_b):
                sample["features"] = torch.from_numpy(self.normalizer.transform(
                    sample["features"].numpy(), sample["mask"].numpy()))
        return {
            "a": sample_a, "b": sample_b,
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id_a": str(self.index.sample_ids[a]),
            "sample_id_b": str(self.index.sample_ids[b]),
        }
