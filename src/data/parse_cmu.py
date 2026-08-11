"""
Parses data/raw/cmu/DSL-StrongPasswordData.csv (Killourhy & Maxion, 2009)
into the unified schema defined in schema.py.

Source format: one row per password entry, with H/DD/UD flattened into
named columns per character of the fixed password ".tie5Roanl" + Return
(11 keystrokes total). We reshape each row back into a (11, 3) sequence.

Usage:
    python src/data/parse_cmu.py
Output:
    data/processed/cmu.npz  (arrays: subject_id, sample_id, session, seq_len,
                              features, mask)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from schema import MAX_SEQ_LEN, N_FEATURES

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "cmu" / "DSL-StrongPasswordData.csv"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "cmu.npz"


def get_key_order(columns: list[str]) -> list[str]:
    """
    CMU columns look like: H.period, DD.period.t, UD.period.t, H.t, DD.t.i, ...
    Each keystroke's Hold-time column (H.<key>) appears in typed order.
    We recover the ordered list of key labels from the H.* columns.
    """
    h_cols = [c for c in columns if c.startswith("H.")]
    return [c[2:] for c in h_cols]  # strip "H." prefix, preserves column order


def parse_cmu() -> None:
    df = pd.read_csv(RAW_PATH)
    key_order = get_key_order(list(df.columns))
    n_keys = len(key_order)
    assert n_keys == 11, f"expected 11 keystrokes for '.tie5Roanl'+Return, got {n_keys}: {key_order}"

    n_samples = len(df)
    features = np.zeros((n_samples, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
    mask = np.zeros((n_samples, MAX_SEQ_LEN), dtype=bool)

    # DD/UD columns are named by the *transition* (prev_key -> this_key), not
    # by the destination key alone, and CMU's naming has quirks (e.g. periods
    # inside key names like "Shift.r"). Safer to walk columns in original
    # order rather than reconstruct names, since order is preserved.
    ordered_cols = list(df.columns)
    non_meta_cols = [c for c in ordered_cols if c not in ("subject", "sessionIndex", "rep")]

    # non_meta_cols alternates, per keystroke i>=1: H.<i>, DD.<i-1,i>, UD.<i-1,i>
    # for i==0 it's just H.<0> (first three real cols: H.period, then DD/UD to next)
    # Actual CMU layout: H.period, DD.period.t, UD.period.t, H.t, DD.t.i, UD.t.i, ...
    # i.e. grouped as (H_k, DD_k->k+1, UD_k->k+1) except the last key has only H.
    col_iter = iter(non_meta_cols)
    key_idx = 0
    col = next(col_iter, None)
    while col is not None:
        assert col.startswith("H."), f"expected H.* column, got {col}"
        features[:, key_idx, 0] = df[col].to_numpy()
        col = next(col_iter, None)
        if col is not None and col.startswith("DD."):
            dd_col = col
            ud_col = next(col_iter)
            assert ud_col.startswith("UD."), f"expected UD.* after {dd_col}, got {ud_col}"
            # DD/UD describe the transition INTO key_idx+1, store on that keystroke
            features[:, key_idx + 1, 2] = df[dd_col].to_numpy()  # DD
            features[:, key_idx + 1, 1] = df[ud_col].to_numpy()  # UD
            col = next(col_iter, None)
        key_idx += 1

    mask[:, :n_keys] = True

    subject_id = ("cmu_" + df["subject"].astype(str)).to_numpy()
    session = df["sessionIndex"].to_numpy()
    sample_id = (
        "cmu_" + df["subject"].astype(str) + "_s" + df["sessionIndex"].astype(str)
        + "_r" + df["rep"].astype(str)
    ).to_numpy()
    seq_len = np.full(n_samples, n_keys, dtype=np.int32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        subject_id=subject_id,
        sample_id=sample_id,
        session=session,
        seq_len=seq_len,
        features=features,
        mask=mask,
    )
    print(f"Wrote {n_samples} samples, {len(set(subject_id))} subjects -> {OUT_PATH}")
    print(f"features shape: {features.shape}, mask shape: {mask.shape}")


if __name__ == "__main__":
    parse_cmu()