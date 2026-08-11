"""
Unified keystroke schema used across all datasets (CMU, Aalto, future ones).

Every parser (parse_cmu.py, parse_aalto.py, ...) must emit data in this exact
shape so the Siamese training pipeline never needs to know which dataset a
sample came from.

A single SAMPLE is one instance of a subject typing something (one password
entry for CMU, one sentence for Aalto). A sample is a *sequence* of keystroke
events, each with three timing features:

    H  (Hold Time)     — key press to key release, for a single key
    UD (Up-Down Time)  — previous key's release to this key's press
    DD (Down-Down Time)— previous key's press to this key's press

Output format (per dataset, saved as a single .parquet or .npz):

    subject_id   : str  — unique per-subject identifier (namespaced per
                   dataset, e.g. "cmu_s002", "aalto_100234", so subject IDs
                   never collide across datasets)
    sample_id    : str  — unique per-sample identifier
    dataset      : str  — "cmu" | "aalto" | ...
    session      : int  — session index within subject, for datasets that
                   have repeated sessions (CMU: 1-8. Aalto: always 1, since
                   Aalto is single-session per subject)
    seq_len      : int  — actual number of keystrokes before padding
    features     : float32 array, shape (MAX_SEQ_LEN, 3) — [H, UD, DD] per
                   keystroke, zero-padded/truncated to MAX_SEQ_LEN
    mask         : bool array, shape (MAX_SEQ_LEN,) — True for real
                   keystrokes, False for padding (used by
                   pack_padded_sequence / attention masking)

MAX_SEQ_LEN is fixed globally so CMU (11 keys) and Aalto (up to ~70+ chars
with corrections) share one tensor shape.
"""

MAX_SEQ_LEN = 70
FEATURE_NAMES = ["H", "UD", "DD"]
N_FEATURES = len(FEATURE_NAMES)