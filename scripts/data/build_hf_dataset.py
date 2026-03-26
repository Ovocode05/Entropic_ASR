"""
build_hf_dataset.py
--------------------
Reads data/raw/manifest.csv (unified 13-col schema) and builds two HuggingFace DatasetDicts:

NOTE: Audio is stored as plain file paths (strings), NOT as HF Audio objects.
      This avoids the torchcodec/FFmpeg dependency on Windows.
      During training, load audio with: arr, sr = soundfile.read(row['audio_path'])

  1. data/processed/hinglish_asr/
     → MUCS rows: audio, transcript, utt_id
     → Used for Whisper LoRA fine-tuning

  2. data/processed/financial_benchmark/
     → Synthetic rows: transcript, normalized, intent, amount_inr,
       is_disambiguation, register, disambiguation_note
     → Used for DistilBERT ITN and Intent Classifier training

Prerequisites:
  Run preprocess_mucs.py + ingest_synthetic.py before this script.

Usage:
  python scripts/data/build_hf_dataset.py
  python scripts/data/build_hf_dataset.py --manifest data/raw/manifest.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MANIFEST   = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\raw\manifest.csv"
DEFAULT_OUTPUT_ASR = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\processed\hinglish_asr"
DEFAULT_OUTPUT_FIN = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\processed\financial_benchmark"
TARGET_SR = 16_000


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ──────────────────────────────────────────────────────────────────────────────

def build_asr_dataset(df_mucs: pd.DataFrame, output_dir: Path) -> DatasetDict:
    """
    Build Whisper-ready ASR DatasetDict from MUCS rows.
    Columns: audio, transcript, utt_id
    """
    print(f"\n── Building ASR dataset (MUCS) ──")
    print(f"   Total rows : {len(df_mucs)}")

    # Only keep rows where audio file exists on disk
    valid_mask = df_mucs["audio_path"].apply(
        lambda p: Path(str(p)).exists() if pd.notna(p) else False
    )
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"   [WARN] {n_missing} audio files not found — dropping those rows")
    df_mucs = df_mucs[valid_mask].copy()

    splits = {}
    for split_name in sorted(df_mucs["split"].unique()):
        df_s = df_mucs[df_mucs["split"] == split_name]
        print(f"   split='{split_name}': {len(df_s)} utterances")

        ds = Dataset.from_dict({
            "audio_path" : df_s["audio_path"].tolist(),   # plain path string
            "transcript" : df_s["transcript"].tolist(),
            "utt_id"     : df_s["utt_id"].fillna("").tolist(),
        })
        # NOTE: audio stored as path string — load with soundfile during training:
        #   import soundfile as sf
        #   arr, sr = sf.read(row["audio_path"])
        #   inputs = processor(arr, sampling_rate=16000, return_tensors="pt")
        splits[split_name] = ds

    dataset_dict = DatasetDict(splits)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"   ✓ Saved to: {output_dir}")
    print(f"   {dataset_dict}")
    return dataset_dict


def build_financial_dataset(df_synth: pd.DataFrame, output_dir: Path) -> DatasetDict:
    """
    Build DistilBERT-ready DatasetDict from synthetic rows.

    Columns:
      transcript          → model input (spoken Hinglish)
      normalized          → ITN target (English with ₹ amounts)
      intent              → intent classification label
      amount_inr          → numeric ₹ amount (string, empty if not applicable)
      is_disambiguation   → True/False string flag
      register            → casual / formal / urgent
      disambiguation_note → annotation for ambiguous 'do' cases
      audio               → audio file (if present)
    """
    print(f"\n── Building Financial Benchmark dataset (synthetic) ──")
    print(f"   Total rows : {len(df_synth)}")

    def _str(series: pd.Series) -> list:
        """Fill NaN with empty string and return as list."""
        return series.fillna("").astype(str).tolist()

    splits = {}
    for split_name in sorted(df_synth["split"].unique()):
        df_s = df_synth[df_synth["split"] == split_name]
        print(f"   split='{split_name}': {len(df_s)} utterances")

        row_dict = {
            "transcript"          : _str(df_s["transcript"]),
            "normalized"          : _str(df_s["normalized"]),
            "intent"              : _str(df_s["intent"]),
            "amount_inr"          : _str(df_s["amount_inr"]),
            "is_disambiguation"   : _str(df_s["is_disambiguation"]),
            "register"            : _str(df_s["register"]),
            "disambiguation_note" : _str(df_s["disambiguation_note"]),
        }

        # Add audio_path only where files exist
        has_audio = df_s["audio_path"].apply(
            lambda p: Path(str(p)).exists() if pd.notna(p) and str(p) not in ("", "nan") else False
        )
        if has_audio.any():
            row_dict["audio_path"] = df_s["audio_path"].tolist()  # plain path string

        ds = Dataset.from_dict(row_dict)
        splits[split_name] = ds

    dataset_dict = DatasetDict(splits)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"   ✓ Saved to: {output_dir}")
    print(f"   {dataset_dict}")
    return dataset_dict


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace DatasetDicts from unified manifest CSV."
    )
    parser.add_argument("--manifest",    default=DEFAULT_MANIFEST)
    parser.add_argument("--output-asr",  default=DEFAULT_OUTPUT_ASR)
    parser.add_argument("--output-fin",  default=DEFAULT_OUTPUT_FIN)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run preprocess_mucs.py + ingest_synthetic.py first."
        )

    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, encoding="utf-8", low_memory=False)
    print(f"Total rows : {len(df)}")
    print(f"Columns    : {list(df.columns)}")

    if "source" not in df.columns:
        raise ValueError(
            "Manifest is in the OLD 7-column schema.\n"
            "Run ingest_synthetic.py to migrate to the unified 13-column schema."
        )

    print(f"\nSource distribution:\n{df['source'].value_counts().to_string()}")

    df_mucs  = df[df["source"] == "mucs"].copy()
    df_synth = df[df["source"] == "synthetic"].copy()

    if len(df_mucs) > 0:
        Path(args.output_asr).mkdir(parents=True, exist_ok=True)
        build_asr_dataset(df_mucs, Path(args.output_asr))
    else:
        print("\n[INFO] No MUCS rows in manifest — skipping ASR dataset.")

    if len(df_synth) > 0:
        Path(args.output_fin).mkdir(parents=True, exist_ok=True)
        build_financial_dataset(df_synth, Path(args.output_fin))
    else:
        print("\n[INFO] No synthetic rows — skipping financial benchmark.")
        print("       Run ingest_synthetic.py first.")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
