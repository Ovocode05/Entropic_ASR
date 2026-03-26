"""
ingest_synthetic.py
--------------------
Merges the 300-row synthetic_benchmark.csv into the unified manifest.

What this script does:
  1. Reads the existing manifest.csv (MUCS rows, old 7-column schema)
  2. Migrates it to the RICH unified schema (adds 6 new empty columns for MUCS rows)
  3. Reads synthetic_benchmark.csv and maps its columns to the unified schema
  4. Auto-assigns train/val/test splits to synthetic rows (80/10/10)
  5. Writes the unified manifest back to manifest.csv

Unified manifest columns:
  audio_path      - path to WAV file (empty for synthetic rows without audio)
  transcript      - spoken Hinglish text  [MUCS + synthetic]
  normalized      - ITN-normalized text   [synthetic only]
  source          - "mucs" | "synthetic"
  split           - "train" | "val" | "test"
  utt_id          - utterance ID (e.g. synth_0001 for synthetic)
  recording_id    - parent recording (empty for synthetic)
  duration_sec    - clip duration       [MUCS only; empty for synthetic until audio exists]
  intent          - intent label        [synthetic only]
  amount_inr      - numeric amount in ₹ [synthetic only]
  is_disambiguation - True/False        [synthetic only]
  register        - casual/formal/urgent [synthetic only]
  disambiguation_note - annotation about ambiguous "do" [synthetic only]

Usage:
  python scripts/data/ingest_synthetic.py
  python scripts/data/ingest_synthetic.py --synthetic path/to/synthetic_benchmark.csv
  python scripts/data/ingest_synthetic.py --no-backup   (skip creating manifest.bak.csv)
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MANIFEST   = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\raw\manifest.csv"
DEFAULT_SYNTHETIC  = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\scripts\data\raw\synthetic_benchmark.csv"

# Unified column order —— everything for both models
UNIFIED_COLUMNS = [
    "audio_path",           # WAV file path (empty for synthetic w/o audio)
    "transcript",           # spoken Hinglish input text
    "normalized",           # ITN-normalized English output (DistilBERT target)
    "source",               # "mucs" | "synthetic"
    "split",                # train / val / test
    "utt_id",               # utterance ID
    "recording_id",         # parent recording (MUCS only)
    "duration_sec",         # clip length in seconds (MUCS only)
    "intent",               # SEND_MONEY / CHECK_BALANCE / etc. (synthetic only)
    "amount_inr",           # numeric ₹ amount (synthetic only)
    "is_disambiguation",    # True if utterance is ambiguous "do" case
    "register",             # casual / formal / urgent (synthetic only)
    "disambiguation_note",  # notes like "do=VERB:gave" (synthetic only)
]

# Split ratios for synthetic data
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO = 0.10 (remainder)

RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def migrate_mucs(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 6 new synthetic-only columns (filled with NaN) to old MUCS rows."""
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[UNIFIED_COLUMNS]


def load_synthetic(path: Path) -> pd.DataFrame:
    """
    Load synthetic_benchmark.csv and map to unified schema.

    Synthetic CSV columns:
      audio_path, transcript, normalized, intent, amount,
      is_disambiguation, register, note
    """
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)

    # Rename to unified names
    df = df.rename(columns={
        "amount": "amount_inr",
        "note"  : "disambiguation_note",
    })

    # source column
    df["source"] = "synthetic"

    # Generate utt_ids: synth_0001 … synth_0300
    df["utt_id"] = [f"synth_{i+1:04d}" for i in range(len(df))]

    # Empty MUCS-only columns
    df["recording_id"] = np.nan
    df["duration_sec"] = np.nan

    # Fix audio_path: empty strings → NaN
    df["audio_path"] = df["audio_path"].replace("", np.nan)

    # is_disambiguation: coerce to bool-string for consistency
    df["is_disambiguation"] = df["is_disambiguation"].astype(str).str.strip()

    # Assign splits (stratified by intent to keep class balance)
    df = assign_splits(df)

    return df[UNIFIED_COLUMNS]


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test split to synthetic data.
    Stratified by intent label to preserve class proportions.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    splits = np.empty(len(df), dtype=object)

    if "intent" in df.columns and df["intent"].notna().any():
        for intent_val in df["intent"].unique():
            idx = df.index[df["intent"] == intent_val].tolist()
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(n * TRAIN_RATIO)
            n_val   = max(1, int(n * VAL_RATIO))
            for i, j in enumerate(idx):
                if i < n_train:
                    splits[df.index.get_loc(j)] = "train"
                elif i < n_train + n_val:
                    splits[df.index.get_loc(j)] = "val"
                else:
                    splits[df.index.get_loc(j)] = "test"
    else:
        # Fallback: simple sequential split
        n = len(df)
        for i in range(n):
            if i < int(n * TRAIN_RATIO):
                splits[i] = "train"
            elif i < int(n * (TRAIN_RATIO + VAL_RATIO)):
                splits[i] = "val"
            else:
                splits[i] = "test"

    df["split"] = splits
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge synthetic_benchmark.csv into the unified manifest."
    )
    parser.add_argument("--manifest",   default=DEFAULT_MANIFEST)
    parser.add_argument("--synthetic",  default=DEFAULT_SYNTHETIC)
    parser.add_argument("--no-backup",  action="store_true",
                        help="Skip creating manifest.bak.csv before overwriting")
    args = parser.parse_args()

    manifest_path  = Path(args.manifest)
    synthetic_path = Path(args.synthetic)

    # ── Load existing manifest ─────────────────────────────────────────────
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("        Run preprocess_mucs.py first.")
        return

    print(f"Loading manifest: {manifest_path}")
    df_mucs = pd.read_csv(manifest_path, encoding="utf-8", low_memory=False)
    print(f"  Existing rows : {len(df_mucs)}")
    print(f"  Existing cols : {list(df_mucs.columns)}")

    # Check if already migrated (has 'normalized' column)
    already_unified = "normalized" in df_mucs.columns

    if already_unified:
        # Drop any previously ingested synthetic rows to avoid duplicates
        df_mucs_only = df_mucs[df_mucs["source"] == "mucs"].copy()
        synth_existing = (df_mucs["source"] == "synthetic").sum()
        if synth_existing > 0:
            print(f"  Found {synth_existing} existing synthetic rows — will replace them")
        df_mucs = df_mucs_only

    # ── Migrate MUCS rows to unified schema ───────────────────────────────
    print("\nMigrating MUCS rows to unified schema...")
    df_mucs = migrate_mucs(df_mucs)
    print(f"  MUCS rows after migration : {len(df_mucs)}")

    # ── Load synthetic data ────────────────────────────────────────────────
    if not synthetic_path.exists():
        print(f"[ERROR] Synthetic CSV not found: {synthetic_path}")
        return

    print(f"\nLoading synthetic data: {synthetic_path}")
    df_synth = load_synthetic(synthetic_path)
    print(f"  Synthetic rows : {len(df_synth)}")
    print(f"  Split distribution:\n{df_synth['split'].value_counts().to_string()}")
    print(f"  Intent distribution:\n{df_synth['intent'].value_counts().to_string()}")

    # ── Combine ────────────────────────────────────────────────────────────
    df_unified = pd.concat([df_mucs, df_synth], ignore_index=True)
    df_unified = df_unified[UNIFIED_COLUMNS]

    print(f"\nUnified manifest: {len(df_unified)} total rows")
    print(f"  Source distribution:\n{df_unified['source'].value_counts().to_string()}")

    # ── Backup + write ─────────────────────────────────────────────────────
    if not args.no_backup and manifest_path.exists():
        backup = manifest_path.with_suffix(".bak.csv")
        shutil.copy2(manifest_path, backup)
        print(f"\nBackup saved: {backup}")

    df_unified.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"✓ Unified manifest written: {manifest_path}")
    print(f"  Columns: {list(df_unified.columns)}")

    # ── Quick stats ────────────────────────────────────────────────────────
    print("\n── Final unified manifest stats ──────────────────────────────")
    for source in ["mucs", "synthetic"]:
        sub = df_unified[df_unified["source"] == source]
        print(f"  [{source}] {len(sub)} rows | splits: "
              f"{dict(sub['split'].value_counts())}")
    print("─" * 60)


if __name__ == "__main__":
    main()
