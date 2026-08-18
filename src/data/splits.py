import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

def aalto_split(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    out_path: Path | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    shard_paths = sorted(processed_dir.glob("aalto_shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No Aalto shards found under {processed_dir}")

    subjects = set()
    for shard_path in shard_paths:
        subjects.update(load_subjects_from_npz(shard_path))

    split = split_subjects(subjects, seed=seed)
    if out_path is None:
        out_path = processed_dir / "aalto_subject_split_80_10_10.json"
    write_split(split, out_path)
    print_split_summary("aalto", split, out_path, len(shard_paths))
    return split


def cmu_split(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    out_path: Path | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    cmu_path = processed_dir / "cmu.npz"
    if not cmu_path.exists():
        raise FileNotFoundError(f"No CMU file found at {cmu_path}")

    subjects = load_subjects_from_npz(cmu_path)
    split = split_cmu_subjects(subjects, n_train_subjects=10, seed=seed)
    if out_path is None:
        out_path = processed_dir / "cmu_subject_split_10_train_rest_test.json"
    write_split(split, out_path)
    print_split_summary("cmu", split, out_path, 1)
    return split

def load_subjects_from_npz(path: Path) -> set[str]:
    data = np.load(path, allow_pickle=True)
    return set(data["subject_id"].astype(str).tolist())


def split_subjects(
    subjects: set[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    subjects_arr = np.array(sorted(subjects))
    if len(subjects_arr) == 0:
        raise ValueError("No subjects found to split")

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects_arr)

    n_subjects = len(subjects_arr)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    return {
        "train": subjects_arr[:n_train].tolist(),
        "val": subjects_arr[n_train:n_train + n_val].tolist(),
        "test": subjects_arr[n_train + n_val:].tolist(),
    }


def split_cmu_subjects(
    subjects: set[str],
    n_train_subjects: int = 10,
    seed: int = 42,
) -> dict[str, list[str]]:
    subjects_arr = np.array(sorted(subjects))
    if len(subjects_arr) == 0:
        raise ValueError("No subjects found to split")
    if n_train_subjects >= len(subjects_arr):
        raise ValueError("n_train_subjects must be smaller than total subjects")

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects_arr)

    return {
        "train": subjects_arr[:n_train_subjects].tolist(),
        "val": [],
        "test": subjects_arr[n_train_subjects:].tolist(),
    }


def write_split(split: dict[str, list[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

def print_split_summary(
    dataset: str,
    split: dict[str, list[str]],
    out_path: Path,
    n_files: int,
) -> None:
    total = sum(len(subjects) for subjects in split.values())
    print(f"Dataset: {dataset}")
    print(f"Input files: {n_files}")
    print(f"Total subjects: {total}")
    print(f"Train subjects: {len(split['train'])}")
    print(f"Val subjects: {len(split['val'])}")
    print(f"Test subjects: {len(split['test'])}")
    print(f"Wrote split -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["aalto", "cmu"], required=True)
    ap.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    ap.add_argument("--out-path", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "aalto":
        aalto_split(args.processed_dir, args.out_path, args.seed)
    else:
        cmu_split(args.processed_dir, args.out_path, args.seed)


if __name__ == "__main__":
    main()
