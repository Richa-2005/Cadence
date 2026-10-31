"""
Split-safe pair generation for Siamese keystroke training.
"""
import argparse
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass(frozen=True)
class PairIndex:
    """Lightweight structure used by training code for live pair sampling."""

    sample_ids: np.ndarray
    subject_ids: np.ndarray
    subject_to_indices: dict[str, np.ndarray]
    shard_paths: tuple[Path, ...]
    shard_indices: np.ndarray
    row_indices: np.ndarray
    sessions: np.ndarray
    cross_session: bool = False

    @cached_property
    def subjects(self) -> tuple[str, ...]:
        return tuple(self.subject_to_indices)

    @cached_property
    def session_groups(self) -> dict[str, tuple[np.ndarray, ...]]:
        return {
            subject: tuple(indices[self.sessions[indices] == session]
                           for session in np.unique(self.sessions[indices]))
            for subject, indices in self.subject_to_indices.items()
        }

    @cached_property
    def genuine_subjects(self) -> tuple[str, ...]:
        return tuple(subject for subject, indices in self.subject_to_indices.items()
                     if len(indices) >= 2 and
                     (not self.cross_session or len(self.session_groups[subject]) >= 2))


def dataset_npz_paths(dataset: str, processed_dir: Path) -> list[Path]:
    if dataset == "cmu":
        path = processed_dir / "cmu.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing CMU processed file: {path}")
        return [path]
    if dataset == "aalto":
        paths = sorted(processed_dir.glob("aalto_shard_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No Aalto shards found under {processed_dir}")
        return paths
    raise ValueError(f"Unsupported dataset: {dataset}")


def default_split_path(dataset: str, processed_dir: Path) -> Path:
    if dataset == "cmu":
        return processed_dir / "cmu_subject_split_10_train_rest_test.json"
    if dataset == "aalto":
        return processed_dir / "aalto_subject_split_80_10_10.json"
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_split(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        split = json.load(f)
    for name in ("train", "val", "test"):
        if name not in split:
            raise ValueError(f"Split file {path} is missing '{name}'")
    groups = [split[name] for name in ("train", "val", "test")]
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Duplicate subjects within split")
    if any(set(groups[i]) & set(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Subjects overlap across splits")
    return split


def load_pair_index(
    dataset: str,
    split_name: str,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    split_path: Path | None = None,
    cross_session: bool | None = None,
) -> PairIndex:
    """
    Load only sample_id/subject_id metadata for one split.

    This intentionally does not load features. Training code can keep this map
    in memory and use the returned sample indices to fetch features separately.
    """
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be one of: train, val, test")

    if split_path is None:
        split_path = default_split_path(dataset, processed_dir)
    split_subjects = set(load_split(split_path)[split_name])

    sample_ids: list[str] = []
    subject_ids: list[str] = []

    paths = tuple(dataset_npz_paths(dataset, processed_dir))
    shard_indices, row_indices, sessions = [], [], []
    for shard_idx, npz_path in enumerate(paths):
        with np.load(npz_path, allow_pickle=True) as data:
            shard_subjects = data["subject_id"].astype(str)
            shard_sample_ids = data["sample_id"].astype(str)
            keep = np.flatnonzero(np.isin(shard_subjects, list(split_subjects)))
            sample_ids.extend(shard_sample_ids[keep].tolist())
            subject_ids.extend(shard_subjects[keep].tolist())
            shard_indices.extend([shard_idx] * len(keep))
            row_indices.extend(keep.tolist())
            sessions.extend(data["session"][keep].tolist())
    if set(subject_ids) != split_subjects:
        raise ValueError("Split references subjects missing from the processed data")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs in processed data")

    sample_ids_arr = np.array(sample_ids)
    subject_ids_arr = np.array(subject_ids)
    subject_to_indices: dict[str, list[int]] = {}
    for idx, subject_id in enumerate(subject_ids_arr):
        subject_to_indices.setdefault(str(subject_id), []).append(idx)

    frozen_subject_to_indices = {
        subject_id: np.array(indices, dtype=np.int64)
        for subject_id, indices in subject_to_indices.items()
    }
    return PairIndex(sample_ids_arr, subject_ids_arr, frozen_subject_to_indices,
                     paths, np.asarray(shard_indices, dtype=np.int32),
                     np.asarray(row_indices, dtype=np.int64), np.asarray(sessions),
                     dataset == "cmu" if cross_session is None else cross_session)


def sample_genuine_pair(index: PairIndex, rng: np.random.Generator) -> tuple[int, int]:
    if not index.genuine_subjects:
        raise ValueError("No subject has enough samples/sessions for genuine pairs")
    subject_id = index.genuine_subjects[int(rng.integers(len(index.genuine_subjects)))]
    if index.cross_session:
        groups = index.session_groups[subject_id]
        a, b = rng.choice(len(groups), size=2, replace=False)
        return int(rng.choice(groups[a])), int(rng.choice(groups[b]))
    a, b = rng.choice(index.subject_to_indices[subject_id], size=2, replace=False)
    return int(a), int(b)


def sample_impostor_pair(index: PairIndex, rng: np.random.Generator) -> tuple[int, int]:
    subjects = index.subjects
    if len(subjects) < 2:
        raise ValueError("Need at least two subjects for impostor pairs")

    a, b = rng.choice(len(subjects), size=2, replace=False)
    subject_a, subject_b = subjects[a], subjects[b]
    idx_a = int(rng.choice(index.subject_to_indices[subject_a]))
    idx_b = int(rng.choice(index.subject_to_indices[subject_b]))
    return idx_a, idx_b


def sample_pair(index: PairIndex, rng: np.random.Generator) -> tuple[str, str, int]:
    """Return (sample_id_a, sample_id_b, label), where label 1=genuine, 0=impostor."""
    label = int(rng.integers(0, 2))
    if label == 1:
        idx_a, idx_b = sample_genuine_pair(index, rng)
    else:
        idx_a, idx_b = sample_impostor_pair(index, rng)
    return str(index.sample_ids[idx_a]), str(index.sample_ids[idx_b]), label


def generate_fixed_pairs(
    index: PairIndex,
    n_pairs: int,
    seed: int = 42,
    genuine_ratio: float = 0.5,
) -> np.ndarray:
    """
    Generate fixed validation/test pairs as (sample_id_a, sample_id_b, label).

    Pairs are sampled, not exhaustively enumerated. With genuine_ratio=0.5 this
    produces the usual 1:1 genuine/impostor balance.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if not 0.0 <= genuine_ratio <= 1.0:
        raise ValueError("genuine_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n_genuine = int(round(n_pairs * genuine_ratio))
    n_impostor = n_pairs - n_genuine
    rows: list[tuple[str, str, int]] = []

    for _ in range(n_genuine):
        idx_a, idx_b = sample_genuine_pair(index, rng)
        rows.append((str(index.sample_ids[idx_a]), str(index.sample_ids[idx_b]), 1))
    for _ in range(n_impostor):
        idx_a, idx_b = sample_impostor_pair(index, rng)
        rows.append((str(index.sample_ids[idx_a]), str(index.sample_ids[idx_b]), 0))

    rng.shuffle(rows)
    return np.array(rows, dtype=object)


def write_pairs(pairs: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_id_a=pairs[:, 0].astype(str),
        sample_id_b=pairs[:, 1].astype(str),
        label=pairs[:, 2].astype(np.int8),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cmu", "aalto"], required=True)
    ap.add_argument("--split-name", choices=["val", "test"], required=True)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--split-path", type=Path, default=None)
    ap.add_argument("--out-path", type=Path, default=None)
    ap.add_argument("--pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--genuine-ratio", type=float, default=0.5)
    ap.add_argument("--cross-session", action=argparse.BooleanOptionalAction, default=None,
                    help="Require different sessions for genuine pairs (default: CMU only)")
    args = ap.parse_args()

    index = load_pair_index(args.dataset, args.split_name, args.processed_dir, args.split_path, args.cross_session)
    if not index.subject_to_indices:
        raise SystemExit(f"No samples found for {args.dataset} {args.split_name} split")

    pairs = generate_fixed_pairs(index, args.pairs, args.seed, args.genuine_ratio)
    out_path = args.out_path
    if out_path is None:
        out_path = args.processed_dir / f"{args.dataset}_{args.split_name}_pairs_seed{args.seed}.npz"
    write_pairs(pairs, out_path)

    n_genuine = int((pairs[:, 2].astype(np.int8) == 1).sum())
    n_impostor = len(pairs) - n_genuine
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split_name}")
    print(f"Subjects: {len(index.subject_to_indices)}")
    print(f"Samples: {len(index.sample_ids)}")
    print(f"Genuine pairs: {n_genuine}")
    print(f"Impostor pairs: {n_impostor}")
    print(f"Wrote pairs -> {out_path}")


if __name__ == "__main__":
    main()
