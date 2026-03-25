"""
train_intent.py
----------------
Phase 4: Train DistilBERT 5-class Intent Classifier.

Task:
  Input  : spoken Hinglish transcript
  Output : intent label  (SEND_MONEY | CHECK_BALANCE | RECEIVE_MONEY | EXPENSE_LOG | BILL_PAYMENT)

Additionally implements the confidence gate (USP 3):
  If max softmax probability < CONFIDENCE_THRESHOLD → return "reprompt" at inference time

Model:
  distilbert-base-multilingual-cased  + sequence classification head

Training data: data/processed/financial_benchmark/  (237 train rows)
Output:        models/adapters/distilbert_intent/

Usage:
  python scripts/train/train_intent.py
  python scripts/train/train_intent.py --epochs 20 --lr 2e-5
  python scripts/train/train_intent.py --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR")
FIN_DATA   = BASE_DIR / "data" / "processed" / "financial_benchmark"
MODEL_OUT  = BASE_DIR / "models" / "adapters" / "distilbert_intent"
BASE_MODEL = "distilbert-base-multilingual-cased"

INTENT_LABELS = ["BILL_PAYMENT", "CHECK_BALANCE", "EXPENSE_LOG",
                 "RECEIVE_MONEY", "SEND_MONEY"]
LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTENT_LABELS)}

CONFIDENCE_THRESHOLD = 0.60   # USP 3: below this → reprompt


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class IntentDataset(TorchDataset):
    def __init__(self, hf_split, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.items     = []

        for row in hf_split:
            intent = row["intent"].strip()
            if intent not in LABEL2ID:
                continue   # skip unknown labels

            enc = tokenizer(
                row["transcript"],
                truncation    = True,
                max_length    = max_len,
                padding       = "max_length",
                return_tensors= "pt",
            )
            self.items.append({
                "input_ids"     : enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels"        : torch.tensor(LABEL2ID[intent], dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading financial_benchmark: {FIN_DATA}")
    dataset = load_from_disk(str(FIN_DATA))
    print(dataset)

    if args.dry_run:
        dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
        dataset["val"]   = dataset["val"].select(range(min(20, len(dataset["val"]))))

    # ── Tokenizer + dataset ───────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = IntentDataset(dataset["train"], tokenizer)
    val_ds   = IntentDataset(dataset["val"],   tokenizer)
    test_ds  = IntentDataset(dataset["test"],  tokenizer)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Intent distribution
    from collections import Counter
    label_counts = Counter(row["intent"] for row in dataset["train"])
    print(f"\nTrain intent distribution: {dict(sorted(label_counts.items()))}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels = len(INTENT_LABELS),
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
    )
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ── Training ──────────────────────────────────────────────────────────
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir              = str(MODEL_OUT),
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        learning_rate           = args.lr,
        weight_decay            = 0.01,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "accuracy",
        greater_is_better       = True,
        logging_steps           = 10,
        fp16                    = (device == "cuda"),
        report_to               = "none",
        save_total_limit        = 2,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
    )

    print(f"\n{'='*60}")
    print("Starting DistilBERT Intent Classifier training...")
    print(f"  Labels    : {INTENT_LABELS}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  Confidence gate threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Output    : {MODEL_OUT}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_result = trainer.evaluate(test_ds)
    print(f"  Test Accuracy : {test_result.get('eval_accuracy', '?'):.4f}")
    print(f"  Test F1 macro : {test_result.get('eval_f1_macro', '?'):.4f}")

    # Confusion matrix
    preds_output = trainer.predict(test_ds)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    cm     = confusion_matrix(labels, preds)
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {INTENT_LABELS}")
    print(f"  {cm}")

    # ── Save ─────────────────────────────────────────────────────────────
    model.save_pretrained(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    # Save label map + confidence threshold for inference pipeline
    config_out = {
        "label2id"             : LABEL2ID,
        "id2label"             : ID2LABEL,
        "confidence_threshold" : CONFIDENCE_THRESHOLD,
        "base_model"           : BASE_MODEL,
        "task"                 : "intent_classification",
        "train_samples"        : len(train_ds),
        "test_accuracy"        : round(test_result.get("eval_accuracy", 0), 4),
        "test_f1_macro"        : round(test_result.get("eval_f1_macro", 0), 4),
        "elapsed_min"          : round(elapsed / 60, 1),
    }
    (MODEL_OUT / "intent_config.json").write_text(json.dumps(config_out, indent=2))
    print(f"\n✓ Model saved: {MODEL_OUT}")
    print(f"✓ Config saved: {MODEL_OUT / 'intent_config.json'}")

    print(f"\n✅ Done.")
    print(f"   Test Accuracy: {test_result.get('eval_accuracy', 0):.1%}")
    print(f"   Target: >90%")
    print(f"   Training took: {elapsed/60:.1f} minutes")

    # ── Confidence gate demo ───────────────────────────────────────────────
    print(f"\n── Confidence Gate Demo (threshold={CONFIDENCE_THRESHOLD}) ──")
    model.eval()
    demo_sentences = [
        "das hazaar bhejo",        # SEND_MONEY — high confidence
        "do",                       # ambiguous — expect low confidence
        "balance check karo",       # CHECK_BALANCE — high confidence
    ]
    for text in demo_sentences:
        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        max_prob, pred_id = probs.max(dim=0)
        intent   = ID2LABEL[pred_id.item()]
        conf     = max_prob.item()
        decision = intent if conf >= CONFIDENCE_THRESHOLD else "REPROMPT"
        print(f"  \"{text}\" → {decision} (conf={conf:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
