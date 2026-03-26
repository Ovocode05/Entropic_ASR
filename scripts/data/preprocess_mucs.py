"""
preprocess_mucs.py
------------------
Preprocesses the MUCS 2021 Hindi-English dataset into sentence-level
16kHz mono WAV clips + a unified CSV manifest.

Dataset layout expected (for each split — train / test):
  Hindi-English/
    <split>/
      *.wav                       ← long/full recording WAV files
      transcripts/
        text                      ← "<utt_id> <transcript>" per line
        wav.scp                   ← "<recording_id> <wav_filename>"
        segments                  ← "<utt_id> <recording_id> <start_sec> <end_sec>"

Output:
  data/raw/audio/mucs_<utt_id>.wav   (16kHz mono, float32)
  data/raw/manifest.csv              (appended, columns below)

Manifest columns:
  audio_path, transcript, source, split, utt_id, recording_id, duration_sec
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TARGET_SR = 16_000          # Whisper expects 16 kHz
MIN_DURATION_SEC = 1.0      # discard clips shorter than this
MAX_DURATION_SEC = 30.0     # discard clips longer than this (Whisper max = 30s)

MANIFEST_COLUMNS = [
    "audio_path", "transcript", "source", "split",
    "utt_id", "recording_id", "duration_sec"
]

# ──────────────────────────────────────────────────────────────────────────────
# Kaldi file parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_text(path: Path) -> dict[str, str]:
    """Return {utt_id: transcript} from a Kaldi 'text' file."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
            else:
                mapping[parts[0]] = ""   # empty transcript (edge case)
    return mapping


def parse_wav_scp(path: Path, wav_dir: Path) -> dict[str, Path]:
    """Return {recording_id: absolute_wav_path} from a Kaldi 'wav.scp' file.
    
    Handles both:
      - Relative filenames: 'FV8hkymGWcy53lrf FV8hkymGWcy53lrf.wav'
      - Absolute paths:     'FV8hkymGWcy53lrf /abs/path/to/file.wav'
    """
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            rec_id, wav_ref = parts
            wav_path = Path(wav_ref)
            if not wav_path.is_absolute():
                wav_path = wav_dir / wav_path
            mapping[rec_id] = wav_path
    return mapping


def parse_segments(path: Path) -> dict[str, tuple[str, float, float]]:
    """Return {utt_id: (recording_id, start_sec, end_sec)}."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            utt_id, rec_id, start, end = parts
            mapping[utt_id] = (rec_id, float(start), float(end))
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Audio processing
# ──────────────────────────────────────────────────────────────────────────────

def slice_and_resample(
    wav_path: Path,
    start_sec: float,
    end_sec: float,
    target_sr: int = TARGET_SR,
) -> np.ndarray | None:
    """Load a slice of a WAV file, convert to mono, resample to target_sr.
    
    Returns numpy float32 array, or None on failure.
    """
    try:
        duration = end_sec - start_sec
        if duration <= 0:
            return None
        # librosa loads as float32, handles mono conversion and resampling
        audio, _ = librosa.load(
            str(wav_path),
            sr=target_sr,
            mono=True,
            offset=start_sec,
            duration=duration,
        )
        return audio
    except Exception as exc:
        print(f"  [WARN] Failed to load {wav_path} @ {start_sec:.1f}s–{end_sec:.1f}s: {exc}",
              file=sys.stderr)
        return None


def is_silent(audio: np.ndarray, silence_threshold_db: float = -60.0) -> bool:
    """Return True if the clip is effectively silent."""
    if len(audio) == 0:
        return True
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return True
    db = 20 * np.log10(rms)
    return db < silence_threshold_db


# ──────────────────────────────────────────────────────────────────────────────
# Main processing function
# ──────────────────────────────────────────────────────────────────────────────

def process_split(
    split_dir: Path,
    split_name: str,          # "train" or "test"
    audio_out_dir: Path,
    manifest_rows: list[dict],
    skip_existing: bool = True,
) -> tuple[int, int]:
    """
    Process one split (train or test) of the MUCS dataset.
    
    Returns (n_processed, n_skipped).
    """
    transcripts_dir = split_dir / "transcripts"

    # Validate required files
    for fname in ("text", "wav.scp", "segments"):
        fpath = transcripts_dir / fname
        if not fpath.exists():
            print(f"[ERROR] Missing {fpath}", file=sys.stderr)
            return 0, 0

    print(f"\n── Processing split: {split_name} ──")
    print(f"   Transcript dir : {transcripts_dir}")
    print(f"   Audio input dir: {split_dir}")
    print(f"   Audio output   : {audio_out_dir}")

    # Parse Kaldi files
    text_map     = parse_text(transcripts_dir / "text")
    wav_scp_map  = parse_wav_scp(transcripts_dir / "wav.scp", wav_dir=split_dir)
    segments_map = parse_segments(transcripts_dir / "segments")

    print(f"   Utterances in text    : {len(text_map)}")
    print(f"   Recordings in wav.scp : {len(wav_scp_map)}")
    print(f"   Segments              : {len(segments_map)}")

    # Cache for loaded full recordings (avoid re-reading same file many times)
    # We process per-recording to be efficient
    utterances_by_recording: dict[str, list[str]] = {}
    for utt_id, (rec_id, _, _) in segments_map.items():
        utterances_by_recording.setdefault(rec_id, []).append(utt_id)

    n_processed = 0
    n_skipped   = 0

    for rec_id, utt_ids in tqdm(utterances_by_recording.items(),
                                 desc=f"  {split_name} recordings"):
        wav_path = wav_scp_map.get(rec_id)
        if wav_path is None:
            print(f"  [WARN] Recording '{rec_id}' not in wav.scp — skipping",
                  file=sys.stderr)
            n_skipped += len(utt_ids)
            continue

        if not wav_path.exists():
            print(f"  [WARN] WAV not found: {wav_path} — skipping {len(utt_ids)} utts",
                  file=sys.stderr)
            n_skipped += len(utt_ids)
            continue

        # Sort utterances by start time for sequential access (faster I/O)
        utt_ids_sorted = sorted(
            utt_ids,
            key=lambda u: segments_map[u][1]
        )

        for utt_id in utt_ids_sorted:
            out_wav = audio_out_dir / f"mucs_{utt_id}.wav"

            # Skip already processed
            if skip_existing and out_wav.exists():
                # Still need to read duration for manifest
                try:
                    info = sf.info(str(out_wav))
                    duration = info.duration
                    transcript = text_map.get(utt_id, "")
                    manifest_rows.append({
                        "audio_path" : str(out_wav),
                        "transcript" : transcript,
                        "source"     : "mucs",
                        "split"      : split_name,
                        "utt_id"     : utt_id,
                        "recording_id": rec_id,
                        "duration_sec": round(duration, 3),
                    })
                    n_processed += 1
                except Exception:
                    pass
                continue

            rec_id_seg, start_sec, end_sec = segments_map[utt_id]
            duration = end_sec - start_sec

            # Duration filter
            if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
                n_skipped += 1
                continue

            # Missing transcript
            transcript = text_map.get(utt_id)
            if transcript is None:
                print(f"  [WARN] No transcript for {utt_id}", file=sys.stderr)
                n_skipped += 1
                continue

            # Load and process audio
            audio = slice_and_resample(wav_path, start_sec, end_sec)
            if audio is None:
                n_skipped += 1
                continue

            # Silence check
            if is_silent(audio):
                n_skipped += 1
                continue

            # Write output WAV
            sf.write(str(out_wav), audio, TARGET_SR, subtype="PCM_16")

            manifest_rows.append({
                "audio_path"  : str(out_wav),   # caller will make relative if needed
                "transcript"  : transcript,
                "source"      : "mucs",
                "split"       : split_name,
                "utt_id"      : utt_id,
                "recording_id": rec_id,
                "duration_sec": round(len(audio) / TARGET_SR, 3),
            })
            n_processed += 1

    return n_processed, n_skipped


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

# Repo root is 3 levels up: scripts/data/preprocess_mucs.py → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MUCS 2021 Hindi-English dataset into 16kHz WAV clips + manifest."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(_REPO_ROOT / "Hindi-English"),
        help="Root directory containing 'train/' and 'test/' subdirs.",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=str(_REPO_ROOT / "data" / "raw" / "audio"),
        help="Directory to write sentence-level WAV clips into.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(_REPO_ROOT / "data" / "raw" / "manifest.csv"),
        help="Path to the output manifest CSV (will be created/appended).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which splits to process (default: train test).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process even if output WAV already exists.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        default=True,
        help="Store relative (repo-root-relative) audio_path in manifest (default: True).",
    )
    args = parser.parse_args()

    dataset_root     = Path(args.dataset_root)
    audio_out_dir    = Path(args.output_audio_dir)
    manifest_path    = Path(args.manifest_path)
    skip_existing    = not args.no_skip_existing

    # Create output dirs
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if manifest already exists (for header)
    write_header = not manifest_path.exists()

    all_rows: list[dict] = []
    total_processed = 0
    total_skipped   = 0

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] Split directory not found: {split_dir} — skipping",
                  file=sys.stderr)
            continue

        n_proc, n_skip = process_split(
            split_dir    = split_dir,
            split_name   = split,
            audio_out_dir= audio_out_dir,
            manifest_rows= all_rows,
            skip_existing= skip_existing,
        )
        total_processed += n_proc
        total_skipped   += n_skip

    # Write / append manifest
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print("\n" + "="*60)
    print(f"  MUCS preprocessing complete")
    print(f"  Utterances processed : {total_processed}")
    print(f"  Utterances skipped   : {total_skipped}")
    print(f"  Manifest written to  : {manifest_path}")
    print(f"  Audio clips saved to : {audio_out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
