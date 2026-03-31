"""
augment_intent_data.py — Generate Whisper-style augmented training data
========================================================================
Run on DGX:
  cd /workspace/Krrish/Entropic_ASR
  python augment_intent_data.py

What this does:
  1. Loads the existing 237-sample financial_benchmark dataset
  2. For every training sentence, generates a "Whisper-style" variant:
       - Proper casing  (sentence case)
       - Adds punctuation
       - English synonyms for common Hinglish words
  3. Also generates short/ambiguous variants to teach robustness
  4. Adds ~100 OTHER/UNKNOWN samples (greetings, random speech)
  5. Saves a new extended dataset to data-love/processed/financial_benchmark_v2

The augmented dataset has ~600 samples (vs 237 original).
Retrain with:  python scripts/train/train_intent.py --data financial_benchmark_v2

WHY THIS MATTERS:
  Training-style "das hazaar bhejo"       → conf=0.77 ✓
  Whisper-style  "Send 10,000 rupees."    → conf=0.25 ✗
  After augmentation, Whisper-style should reach conf > 0.60.
"""

import re
import json
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
FIN_DATA = BASE_DIR / "data/processed/financial_benchmark"        # actual DGX path
OUT_DIR  = BASE_DIR / "data/processed/financial_benchmark_v2"

# ── Hinglish → English synonym map ───────────────────────────────────────────
HINGLISH_TO_ENGLISH = {
    "bhejo": "send", "bhej do": "send", "bheja": "sent",
    "rupay": "rupees", "rupaye": "rupees",
    "hazaar": "thousand", "lakh": "hundred thousand",
    "kar do": "please do", "karo": "do it",
    "check karo": "check", "dekho": "see",
    "bill": "bill", "pay karo": "pay",
    "mangao": "request", "chahiye": "need",
    "kharcha": "expense", "kharcha hua": "I spent",
    "likh do": "note down", "record karo": "record",
}

OTHER_SAMPLES = [
    "Hello, my name is Rahul.",
    "Namaste, main yahan kaam karta hoon.",
    "I am a student researcher.",
    "Good morning, how are you?",
    "Theek hai, chalte hain.",
    "Thank you for your help.",
    "Can you repeat that?",
    "Okay, fine.",
    "Mujhe neend aa rahi hai.",
    "Aaj mausam bahut accha hai.",
    "What time is it?",
    "Please help me understand.",
    "I don't know what to say.",
    "Sorry, I was distracted.",
    "Ek second ruko.",
    "Test test test.",
    "Can you hear me?",
    "Acha, theek hai.",
    "Bhai, sun.",
    "Yaar kya kar rahe ho?",
    "My internet is slow today.",
    "This is a test recording.",
    "Hello hello, is this working?",
    "Main busy hoon abhi.",
    "Zaraa ruko, main aa raha hoon.",
    "Oh, I forgot what I wanted to say.",
    "Nothing important.",
    "Just testing the microphone.",
    "Kal milenge.",
    "Bahut time lag gaya.",
]

def to_sentence_case(text: str) -> str:
    """Convert to proper sentence case like Whisper."""
    if not text:
        return text
    text = text.strip()
    return text[0].upper() + text[1:] + ("." if not text.endswith((".","?","!")) else "")

def apply_english_substitutions(text: str) -> str:
    """Replace common Hinglish words with English equivalents."""
    t = text.lower()
    for hi, en in HINGLISH_TO_ENGLISH.items():
        t = t.replace(hi, en)
    return t

def augment_sentence(base: str, intent: str) -> list[dict]:
    """Generate 2-3 variants of a training sentence."""
    variants = [{"transcript": base, "intent": intent}]  # original

    # Variant 1: Sentence case + punctuation (Whisper-style casing)
    v1 = to_sentence_case(base)
    if v1 != base:
        variants.append({"transcript": v1, "intent": intent})

    # Variant 2: English substitutions + sentence case
    v2 = apply_english_substitutions(base)
    v2 = to_sentence_case(v2)
    if v2 not in (base, v1):
        variants.append({"transcript": v2, "intent": intent})

    # Variant 3: Add simple filler prefix (Whisper often prepends context)
    PREFIXES = ["I want to", "Please", "Can you", "Mujhe", "Kya aap"]
    prefix = random.choice(PREFIXES)
    v3 = f"{prefix} {base.lower()}."
    variants.append({"transcript": v3, "intent": intent})

    return variants


def main():
    from datasets import load_from_disk, DatasetDict, Dataset

    print(f"Loading base dataset: {FIN_DATA}")
    try:
        ds = load_from_disk(str(FIN_DATA))
    except Exception as e:
        print(f"ERROR: {e}")
        print("Run this on the DGX server where the dataset exists.")
        return

    print(ds)

    # ── Augment training split ────────────────────────────────────────────────
    aug_rows = []
    for row in ds["train"]:
        variants = augment_sentence(row["transcript"], row["intent"])
        aug_rows.extend(variants)

    # ── Add OTHER class ───────────────────────────────────────────────────────
    for sample in OTHER_SAMPLES:
        aug_rows.append({"transcript": sample,             "intent": "UNKNOWN"})
        aug_rows.append({"transcript": sample.lower(),    "intent": "UNKNOWN"})
        aug_rows.append({"transcript": to_sentence_case(sample.lower()), "intent": "UNKNOWN"})

    # Shuffle
    random.shuffle(aug_rows)

    # ── Val/test splits (keep originals, add OTHER) ───────────────────────────
    val_rows  = list(ds["val"])
    test_rows = list(ds["test"])
    other_val  = [{"transcript": s, "intent": "UNKNOWN"} for s in OTHER_SAMPLES[:8]]
    other_test = [{"transcript": s, "intent": "UNKNOWN"} for s in OTHER_SAMPLES[8:16]]
    val_rows.extend(other_val)
    test_rows.extend(other_test)

    # ── Build HuggingFace DatasetDict ─────────────────────────────────────────
    new_ds = DatasetDict({
        "train": Dataset.from_list(aug_rows),
        "val":   Dataset.from_list(val_rows),
        "test":  Dataset.from_list(test_rows),
    })
    print(f"\nAugmented dataset:")
    print(new_ds)

    from collections import Counter
    print("\nAugmented train intent distribution:")
    c = Counter(r["intent"] for r in aug_rows)
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")

    print(f"\nSample augmented rows:")
    for r in aug_rows[:8]:
        print(f"  [{r['intent']}] {repr(r['transcript'])}")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_ds.save_to_disk(str(OUT_DIR))
    print(f"\n✓ Saved to {OUT_DIR}")
    print(f"\nNext step: retrain intent classifier on augmented data:")
    print(f"  # 1. Edit INTENT_LABELS in train_intent.py to add 'UNKNOWN':")
    print(f"  #    INTENT_LABELS = ['BILL_PAYMENT','CHECK_BALANCE','EXPENSE_LOG','RECEIVE_MONEY','SEND_MONEY','UNKNOWN']")
    print(f"  # 2. Point to v2 dataset:")
    print(f"  #    FIN_DATA = BASE_DIR / 'data/processed/financial_benchmark_v2'")
    print(f"  # 3. Run training:")
    print(f"  python scripts/train/train_intent.py --epochs 25 --lr 2e-5")
    print(f"\nExpected: Whisper-style confidence 0.28 → 0.65+, OOD sentences → UNKNOWN (direct)")


if __name__ == "__main__":
    main()
