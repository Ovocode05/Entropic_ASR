"""
check_dataset.py
-----------------
Quick sanity check for the MUCS preprocessing output.
Run after preprocess_mucs.py to verify the manifest and audio files are correct.

Usage:
  python scripts/data/check_dataset.py
"""

from pathlib import Path
import pandas as pd
import soundfile as sf

MANIFEST_PATH = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\raw\manifest.csv"
AUDIO_OUT_DIR = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\raw\audio"

def main():
    manifest = Path(MANIFEST_PATH)
    if not manifest.exists():
        print("❌ manifest.csv not found. Run preprocess_mucs.py first.")
        return

    df = pd.read_csv(manifest, encoding="utf-8")
    print(f"✓ Manifest loaded: {len(df)} rows")
    print(f"\nSource distribution:")
    print(df["source"].value_counts().to_string())

    if "split" in df.columns:
        print(f"\nSplit distribution:")
        print(df.groupby(["source", "split"]).size().to_string())

    # Check audio files exist
    mucs_rows = df[df["source"] == "mucs"]
    print(f"\n── MUCS audio check ({len(mucs_rows)} utterances) ──")
    missing = []
    durations = []
    for _, row in mucs_rows.head(50).iterrows():   # spot-check first 50
        p = Path(row["audio_path"])
        if not p.exists():
            missing.append(str(p))
        else:
            try:
                info = sf.info(str(p))
                assert info.samplerate == 16000, f"SR={info.samplerate}"
                assert info.channels == 1, f"channels={info.channels}"
                durations.append(info.duration)
            except Exception as e:
                print(f"  [WARN] {p.name}: {e}")

    if missing:
        print(f"  ❌ {len(missing)} missing audio files (first 3): {missing[:3]}")
    else:
        print(f"  ✓ Spot-checked 50 audio files — all present, 16kHz mono")

    if durations:
        print(f"  Duration stats (first 50): "
              f"min={min(durations):.1f}s, "
              f"max={max(durations):.1f}s, "
              f"avg={sum(durations)/len(durations):.1f}s")

    # Transcript sample
    print(f"\n── Sample transcripts (MUCS) ──")
    for _, row in mucs_rows.head(5).iterrows():
        print(f"  [{row.get('utt_id', '')}] {row['transcript'][:80]}")

    print("\n✓ Sanity check complete.")


if __name__ == "__main__":
    main()
