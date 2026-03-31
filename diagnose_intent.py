"""
diagnose_intent.py — Intent Model Confidence Diagnostic
========================================================
Run on DGX:
  cd /workspace/Krrish/Entropic_ASR
  python diagnose_intent.py

This script answers exactly ONE question:
  WHY is confidence low for certain inputs?

It probes:
  1. Training data distribution + format vs. real Whisper output format
  2. Label-level confidence for known-good and known-bad inputs
  3. Tokenization of training-style vs Whisper-style text
  4. OOD (out-of-distribution) detection — the model has no "other/reject" class
  5. Softmax entropy as a calibration metric
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

BASE_DIR    = Path(__file__).resolve().parent
INTENT_DIR  = BASE_DIR / "models/adapters/distilbert_intent"
FIN_DATA    = BASE_DIR / "data-love/processed/financial_benchmark"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ── 1. Load config ────────────────────────────────────────────────────────────
cfg = json.loads((INTENT_DIR / "intent_config.json").read_text())
id2label = {int(k): v for k, v in cfg["id2label"].items()}
label2id = cfg["label2id"]
print(f"Labels: {list(id2label.values())}")
print(f"Trained accuracy: {cfg.get('test_accuracy','?')}  F1: {cfg.get('test_f1_macro','?')}\n")

# ── 2. Load model ─────────────────────────────────────────────────────────────
tok   = AutoTokenizer.from_pretrained(str(INTENT_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(INTENT_DIR))
model.eval().to(device)

def predict(text: str) -> dict:
    enc = tok(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))  # higher = more uncertain
    return {
        "text":      text,
        "intent":    id2label[pred_id],
        "conf":      round(float(probs[pred_id]), 4),
        "entropy":   round(entropy, 4),
        "all_probs": {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }

def pp(r):
    bar = "█" * int(r["conf"] * 30)
    print(f"  [{r['intent']:<16}] conf={r['conf']:.4f} entropy={r['entropy']:.3f}  |{bar}")
    for l, p in sorted(r["all_probs"].items(), key=lambda x: -x[1]):
        print(f"       {l:<18} {p:.4f}  {'▓'*int(p*30)}")
    print()

# ── 3. Training-style sentences (how training data was written) ───────────────
print("="*65)
print("A. TRAINING-STYLE INPUTS (synthetic, usually lowercase, short)")
print("="*65)
training_style = [
    "das hazaar bhejo",
    "balance check karo",
    "do sau ka bill pay karo",
    "paanch hazaar receive karna hai",
    "kharcha likh do 300",
    "1000 rupay bhejo rahul ko",
    "kitna balance hai",
]
for t in training_style:
    pp(predict(t))

# ── 4. Whisper-style sentences (what Whisper actually outputs) ────────────────
print("="*65)
print("B. WHISPER-STYLE INPUTS (proper case, punctuation, longer)")
print("="*65)
whisper_style = [
    "Send 1000 rupees to Rahul.",
    "1000 rupees Rahul ko send kar do.",
    "Check my balance please.",
    "I need to check my account balance.",
    "Pay the electricity bill of 500 rupees.",
    "Hello, my name is Krishpunjh and I am working here as a student research intern.",
    "I need two more.",
    "Mujhe 300 rupay aur chahye 2000 ke saath.",
]
for t in whisper_style:
    pp(predict(t))

# ── 5. OOD analysis ───────────────────────────────────────────────────────────
print("="*65)
print("C. OOD INPUTS (no valid financial intent — model MUST pick one)")
print("="*65)
ood = [
    "Hello, my name is Krishpunjh.",
    "What is the weather today?",
    "I am a student researcher.",
    "Please help me.",
    "Okay.",
    "yes",
    "I need two more.",
    "Acha.",
]
for t in ood:
    pp(predict(t))

# ── 6. Tokenization format comparison ────────────────────────────────────────
print("="*65)
print("D. TOKENIZATION COMPARISON — training-style vs whisper-style")
print("="*65)
pairs = [
    ("das hazaar bhejo",                   "Send ten thousand rupees."),
    ("balance check karo",                 "Check my balance."),
    ("1000 rupay bhejo rahul ko",          "1000 rupees Rahul ko send kar do."),
]
for train_txt, whisper_txt in pairs:
    train_toks  = tok.tokenize(train_txt)
    whisper_toks = tok.tokenize(whisper_txt)
    print(f"  Train  : {repr(train_txt)}")
    print(f"    tokens ({len(train_toks)}): {train_toks}")
    print(f"  Whisper: {repr(whisper_txt)}")
    print(f"    tokens ({len(whisper_toks)}): {whisper_toks}")
    print()

# ── 7. Training data format inspection ───────────────────────────────────────
print("="*65)
print("E. TRAINING DATA FORMAT (first 3 per class)")
print("="*65)
try:
    ds = load_from_disk(str(FIN_DATA))
    by_label = {l: [] for l in label2id}
    for r in ds["train"]:
        lbl = r["intent"].strip()
        if lbl in by_label and len(by_label[lbl]) < 3:
            by_label[lbl].append(r["transcript"])
    for lbl, samples in by_label.items():
        print(f"  {lbl}:")
        for s in samples:
            print(f"    {repr(s)}")
    # Count uppercase, punctuation
    has_upper = sum(1 for r in ds["train"] if any(c.isupper() for c in r["transcript"]))
    has_punct = sum(1 for r in ds["train"] if "." in r["transcript"] or "," in r["transcript"])
    total     = len(ds["train"])
    print(f"\n  Training set: {total} rows")
    print(f"  Has uppercase chars:  {has_upper}/{total}  ({has_upper/total:.1%})")
    print(f"  Has . or , (punct):   {has_punct}/{total}  ({has_punct/total:.1%})")
    from statistics import mean, median
    lens = [len(r["transcript"].split()) for r in ds["train"]]
    print(f"  Length (words) — min:{min(lens)} max:{max(lens)} mean:{mean(lens):.1f} median:{median(lens):.1f}")
except Exception as e:
    print(f"  [WARN] Could not load dataset: {e}")

# ── 8. Summary diagnosis ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("F. DIAGNOSIS SUMMARY")
print("="*65)
print("""
The model's low confidence comes from ONE OR MORE of these root causes:

1. DISTRIBUTION SHIFT (most likely):
   The training data is SHORT, LOWERCASE, Hinglish financial phrases.
   Whisper outputs PROPER CASE, PUNCTUATED, LONGER English sentences.
   DistilBERT sees a completely different token distribution at inference.
   → Fix: Normalize pipeline input to match training format before classification.

2. NO REJECT CLASS (guaranteed for OOD inputs):
   The model has 5 labels. Any OOD input (greetings, random speech)
   is FORCED into one of these 5 — confidence will always be low.
   → Fix: Add an "OTHER/UNKNOWN" class to training data + retrain.
         OR use entropy threshold: if entropy > 1.4, treat as OOD.

3. SMALL DATASET (237 rows for 5 classes = ~47/class):
   Very limited generalization. High test accuracy on held-out synthetic data
   does NOT mean high confidence on real Whisper transcriptions.
   → Fix: Augment with Whisper-transcribed versions of training sentences.

4. SOFTMAX OVERCONFIDENCE ON TRAINING DISTRIBUTION:
   The model is well-calibrated for training-style inputs (short Hinglish).
   It degrades gracefully (low confidence) for Whisper-style inputs.
   This is actually CORRECT behavior — low confidence IS the right signal.
""")
