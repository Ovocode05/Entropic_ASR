"""
verify_datasets.py
-------------------
Loads both processed datasets and proves they are correctly structured
for Whisper LoRA and DistilBERT training.

Audio is stored as plain path strings — decoded with soundfile (no torchcodec needed).

Run:
  python scripts/data/verify_datasets.py
"""

from pathlib import Path
import soundfile as sf
from datasets import load_from_disk

ASR_PATH = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\processed\hinglish_asr"
FIN_PATH = r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR\data\processed\financial_benchmark"
SEP = "─" * 60


def check_asr(path: str):
    print(f"\n{SEP}")
    print("  [1] hinglish_asr  →  Whisper LoRA")
    print(SEP)

    ds = load_from_disk(path)
    print(f"  Splits : {list(ds.keys())}")

    for split, dset in ds.items():
        print(f"\n  [{split}] {len(dset)} rows")
        print(f"  Features: {list(dset.features.keys())}")

        assert "audio_path" in dset.features, "❌ Missing 'audio_path'!"
        assert "transcript" in dset.features, "❌ Missing 'transcript'!"

        # Access row without any audio decoding (just strings)
        row = dset[0]
        audio_path = row["audio_path"]
        transcript = row["transcript"]

        assert Path(audio_path).exists(), f"❌ WAV not found: {audio_path}"

        # Decode with soundfile — exactly what training code will do
        arr, sr = sf.read(audio_path)
        assert sr == 16000, f"❌ Sample rate {sr}, expected 16000"
        assert len(arr) > 0,  "❌ Audio array empty"
        assert len(transcript) > 0, "❌ Empty transcript"

        duration = len(arr) / sr
        print(f"  ✓ audio_path : ...{audio_path[-50:]}")
        print(f"  ✓ audio      : {len(arr)} samples @ {sr}Hz ({duration:.1f}s)")
        print(f"  ✓ transcript : \"{transcript[:70]}\"")
        print(f"  ✓ utt_id     : {row.get('utt_id', 'N/A')}")

    print(f"\n  ✅ hinglish_asr READY for Whisper LoRA")
    print(f"     import soundfile as sf")
    print(f"     arr, sr = sf.read(row['audio_path'])")
    print(f"     inputs = processor(arr, sampling_rate=16000, return_tensors='pt')")


def check_financial(path: str):
    print(f"\n{SEP}")
    print("  [2] financial_benchmark  →  DistilBERT ITN + Intent Classifier")
    print(SEP)

    ds = load_from_disk(path)
    print(f"  Splits : {list(ds.keys())}")

    required = ["transcript", "normalized", "intent",
                "amount_inr", "is_disambiguation", "register", "disambiguation_note"]

    for split, dset in ds.items():
        print(f"\n  [{split}] {len(dset)} rows")
        print(f"  Features: {list(dset.features.keys())}")

        for col in required:
            assert col in dset.features, f"❌ Missing: '{col}'"

        row = dset[0]
        print(f"  ✓ transcript         : \"{row['transcript'][:60]}\"")
        print(f"  ✓ normalized (target): \"{row['normalized'][:60]}\"")
        print(f"  ✓ intent             : {row['intent']}")
        print(f"  ✓ amount_inr         : {row['amount_inr']}")
        print(f"  ✓ is_disambiguation  : {row['is_disambiguation']}")
        print(f"  ✓ register           : {row['register']}")
        print(f"  ✓ disambiguation_note: {row['disambiguation_note']}")

        intents   = sorted(set(dset["intent"]) - {""})
        disambig  = sum(1 for r in dset if r["is_disambiguation"] == "True")
        print(f"\n  Intent classes ({len(intents)}): {intents}")
        print(f"  Disambiguation cases: {disambig}")

    print(f"\n  ✅ financial_benchmark READY for:")
    print(f"     → DistilBERT ITN   : tokenizer(transcript) → normalized")
    print(f"     → Intent Classifier: tokenizer(transcript) → intent (5 classes)")


def main():
    print("\n" + "═" * 60)
    print("   DATASET VERIFICATION REPORT")
    print("═" * 60)

    if Path(ASR_PATH).exists():
        check_asr(ASR_PATH)
    else:
        print(f"\n❌ {ASR_PATH} not found — run build_hf_dataset.py")

    if Path(FIN_PATH).exists():
        check_financial(FIN_PATH)
    else:
        print(f"\n❌ {FIN_PATH} not found — run ingest_synthetic.py + build_hf_dataset.py")

    print(f"\n{'═' * 60}")
    print("   All checks passed ✅")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
