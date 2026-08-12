"""
Parses the Aalto 136M Keystrokes dataset (Dhakal et al., 2018) into the
unified schema defined in schema.py.

Source format: one file per participant, named "<PARTICIPANT_ID>_keystrokes.txt",
tab-separated, columns:
    PARTICIPANT_ID  TEST_SECTION_ID  SENTENCE  USER_INPUT  KEYSTROKE_ID
    PRESS_TIME  RELEASE_TIME  LETTER  KEYCODE

Each (PARTICIPANT_ID, TEST_SECTION_ID) group is one sample (one sentence
attempt). Rows are NOT guaranteed to be press-time ordered (overlapping
shift/letter chords can appear out of KEYSTROKE_ID order), so we explicitly
sort by PRESS_TIME within each group.

H  (Hold)     = RELEASE_TIME - PRESS_TIME, for that key
UD (Up-Down)  = this PRESS_TIME - previous key's RELEASE_TIME  (can be
                negative for overlapping/chorded keys -- kept as signed
                info, not clipped, since overlap itself is a rhythm signal)
DD (Down-Down)= this PRESS_TIME - previous key's PRESS_TIME

BKSP and SHIFT rows are kept as real keystroke events (they carry real
timing/rhythm signal), not filtered out.

Memory strategy: the full dataset (~169K participants, ~2.5M samples) is too
large to hold as one array on a laptop. We process participant files in
batches and write sharded .npz output files (data/processed/aalto_shard_*.npz)
rather than one monolithic file.

Usage:
    python src/data/parse_aalto.py [--input-dir data/raw/aalto] [--shard-size 20000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from schema import MAX_SEQ_LEN, N_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw" / "aalto"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"

COLS = ["PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE", "USER_INPUT",
        "KEYSTROKE_ID", "PRESS_TIME", "RELEASE_TIME", "LETTER", "KEYCODE"]


def parse_one_file(path: Path):
    """Yields (subject_id, sample_id, session, seq_len, feat, mask) per sample in this file."""
    try:
        df = pd.read_csv(path, sep="\t", usecols=COLS, dtype={
            "PARTICIPANT_ID": str, "TEST_SECTION_ID": str,
            "PRESS_TIME": np.int64, "RELEASE_TIME": np.int64,
        }, on_bad_lines="skip")
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return

    if df.empty:
        return

    participant_id = df["PARTICIPANT_ID"].iloc[0]

    for test_section_id, group in df.groupby("TEST_SECTION_ID", sort=False):
        group = group.sort_values("PRESS_TIME", kind="mergesort")  # stable sort
        press = group["PRESS_TIME"].to_numpy()
        release = group["RELEASE_TIME"].to_numpy()
        n = len(press)
        if n < 2:
            continue  # too short to be a meaningful sample

        seq_len = min(n, MAX_SEQ_LEN)
        feat = np.zeros((MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=bool)

        h = (release - press).astype(np.float32) / 1000.0  # ms -> seconds, match CMU units
        ud = np.zeros(n, dtype=np.float32)
        dd = np.zeros(n, dtype=np.float32)
        ud[1:] = (press[1:] - release[:-1]).astype(np.float32) / 1000.0
        dd[1:] = (press[1:] - press[:-1]).astype(np.float32) / 1000.0

        feat[:seq_len, 0] = h[:seq_len]
        feat[:seq_len, 1] = ud[:seq_len]
        feat[:seq_len, 2] = dd[:seq_len]
        mask[:seq_len] = True

        subject_id = f"aalto_{participant_id}"
        sample_id = f"aalto_{participant_id}_{test_section_id}"
        yield subject_id, sample_id, 1, seq_len, feat, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--shard-size", type=int, default=20000,
                     help="samples per output shard file")
    ap.add_argument("--limit-files", type=int, default=None,
                     help="only process first N participant files (for testing)")
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("*_keystrokes.txt"))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"No *_keystrokes.txt files found under {args.input_dir}")
    print(f"Found {len(files)} participant files under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- resume support ---
    # Track which participant files have already been fully written into a
    # completed shard, so a re-run after an interruption skips them instead
    # of reprocessing from scratch. Manifest lives next to the shards.
    manifest_path = args.output_dir / "_processed_files.txt"
    already_done = set()
    if manifest_path.exists():
        already_done = set(manifest_path.read_text().splitlines())
        print(f"Resuming: {len(already_done)} files already processed, skipping them")
        files = [f for f in files if f.name not in already_done]
        print(f"{len(files)} files remaining")

    # Start shard numbering after the highest existing shard, so we never
    # overwrite prior output.
    existing_shards = sorted(args.output_dir.glob("aalto_shard_*.npz"))
    start_shard_idx = 0
    if existing_shards:
        last_num = int(existing_shards[-1].stem.split("_")[-1])
        start_shard_idx = last_num + 1
        print(f"Continuing shard numbering from {start_shard_idx:04d}")

    manifest_file = open(manifest_path, "a")

    buf_subject, buf_sample, buf_session = [], [], []
    buf_seqlen, buf_feat, buf_mask = [], [], []
    buf_source_files = []  # participant files contributing to current shard buffer
    shard_idx = start_shard_idx
    total_samples = 0

    def flush_shard():
        nonlocal shard_idx, buf_subject, buf_sample, buf_session, buf_seqlen, buf_feat, buf_mask, buf_source_files
        if not buf_subject:
            return
        out_path = args.output_dir / f"aalto_shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            out_path,
            subject_id=np.array(buf_subject),
            sample_id=np.array(buf_sample),
            session=np.array(buf_session, dtype=np.int32),
            seq_len=np.array(buf_seqlen, dtype=np.int32),
            features=np.stack(buf_feat),
            mask=np.stack(buf_mask),
        )
        print(f"  wrote {out_path} ({len(buf_subject)} samples)")
        # Mark every source file that contributed to this shard as done,
        # and flush to disk immediately so a crash right after doesn't lose it.
        for fname in buf_source_files:
            manifest_file.write(fname + "\n")
        manifest_file.flush()
        shard_idx += 1
        buf_subject, buf_sample, buf_session = [], [], []
        buf_seqlen, buf_feat, buf_mask = [], [], []
        buf_source_files = []

    for fpath in tqdm(files, desc="Parsing participant files"):
        file_had_samples = False
        for subject_id, sample_id, session, seq_len, feat, mask in parse_one_file(fpath):
            buf_subject.append(subject_id)
            buf_sample.append(sample_id)
            buf_session.append(session)
            buf_seqlen.append(seq_len)
            buf_feat.append(feat)
            buf_mask.append(mask)
            total_samples += 1
            file_had_samples = True
            if len(buf_subject) >= args.shard_size:
                flush_shard()
        # Record this file as done even if it contributed 0 samples (e.g. it
        # was unreadable) so we don't retry known-bad files on resume.
        buf_source_files.append(fpath.name)
        if not file_had_samples:
            manifest_file.write(fpath.name + "\n")
            manifest_file.flush()
            buf_source_files.pop()  # already written directly above

    flush_shard()  # final partial shard
    manifest_file.close()
    print(f"\nDone. Total samples so far: {total_samples}, shards written this run: {shard_idx - start_shard_idx}")


if __name__ == "__main__":
    main()