"""
train_itn.py
-------------
Phase 3: Train DistilBERT Neural ITN (Inverse Text Normalisation).

Task:
  Input  : spoken Hinglish  e.g. "ek hazaar do sau ka transaction karo"
  Output : normalized text  e.g. "Do transaction of ₹1,200"

Model:
  distilbert-base-multilingual-cased used as a sequence classifier / encoder,
  with a linear head that produces token-level predictions.
  
  ITN formulation: token classification with 3 labels
    O   → keep the token as-is
    NUM → this token is part of a number expression (replace with digit)
    SEP → sentence boundary / special token (ignore in prediction)

  For generation of the normalized string we use greedy number-word mapping
  at inference time. This is a practical, lightweight approach that works
  well for the structured financial domain vs. full seq2seq.

Training data: data/processed/financial_benchmark/  (237 train rows)
Output:        models/adapters/distilbert_itn/

GPU: optional (4GB is plenty), but CPU training works at this data size.

Usage:
  python scripts/train/train_itn.py
  python scripts/train/train_itn.py --epochs 10 --lr 3e-5
  python scripts/train/train_itn.py --dry-run
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR")
FIN_DATA   = BASE_DIR / "data" / "processed" / "financial_benchmark"
MODEL_OUT  = BASE_DIR / "models" / "adapters" / "distilbert_itn"
BASE_MODEL = "distilbert-base-multilingual-cased"

LABEL2ID = {"O": 0, "NUM": 1, "SEP": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Hindi/Hinglish number words → digits mapping
NUMBER_WORDS = {
    "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5,
    "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "gyarah": 11, "barah": 12, "tera": 13, "chaudah": 14, "pandrah": 15,
    "solah": 16, "satrah": 17, "atharah": 18, "unnees": 19, "bees": 20,
    "pachas": 50, "saath": 70, "assi": 80, "nabbe": 90,
    "sau": 100, "hazaar": 1000, "lakh": 100000, "crore": 10000000,
    # Roman equivalents also seen in corpus
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "hundred": 100, "thousand": 1000,
}


# ──────────────────────────────────────────────────────────────────────────────
# Token labelling: align transcript tokens to NUM/O labels
# ──────────────────────────────────────────────────────────────────────────────

def label_tokens(transcript: str, normalized: str) -> Tuple[List[str], List[str]]:
    """
    Heuristically assign NUM/O labels to transcript tokens.
    Tokens that are number-words → NUM; rest → O.
    """
    tokens = transcript.lower().split()
    labels = []
    for tok in tokens:
        # Strip punctuation
        clean = re.sub(r"[^\w]", "", tok)
        if clean in NUMBER_WORDS or clean.isdigit():
            labels.append("NUM")
        else:
            labels.append("O")
    return tokens, labels


class ITNDataset(TorchDataset):
    def __init__(self, hf_split, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.items     = []

        for row in hf_split:
            tokens, labels = label_tokens(row["transcript"], row["normalized"])
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            # Align labels to subword tokens
            word_ids    = enc.word_ids()
            label_ids   = []
            prev_word_id = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)   # special tokens: ignore in loss
                elif wid != prev_word_id:
                    label_ids.append(LABEL2ID[labels[wid]])
                else:
                    label_ids.append(-100)   # subword continuation: ignore
                prev_word_id = wid

            self.items.append({
                "input_ids"      : enc["input_ids"].squeeze(),
                "attention_mask" : enc["attention_mask"].squeeze(),
                "labels"         : torch.tensor(label_ids, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


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

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = ITNDataset(dataset["train"], tokenizer)
    val_ds   = ITNDataset(dataset["val"],   tokenizer)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = DistilBertForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels  = NUM_LABELS,
        id2label    = ID2LABEL,
        label2id    = LABEL2ID,
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
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        logging_steps           = 10,
        fp16                    = (device == "cuda"),
        report_to               = "none",
        save_total_limit        = 2,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
    )

    print(f"\n{'='*60}")
    print("Starting DistilBERT ITN training...")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  Output    : {MODEL_OUT}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ── Save ─────────────────────────────────────────────────────────────
    model.save_pretrained(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"\n✓ ITN model saved: {MODEL_OUT}")

    eval_result = trainer.evaluate()
    results = {
        "base_model"   : BASE_MODEL,
        "task"         : "itn_token_classification",
        "label_scheme" : list(LABEL2ID.keys()),
        "train_samples": len(train_ds),
        "val_samples"  : len(val_ds),
        "epochs"       : args.epochs,
        "elapsed_min"  : round(elapsed / 60, 1),
        **eval_result,
    }
    (MODEL_OUT / "itn_results.json").write_text(json.dumps(results, indent=2))
    print(f"✓ Results: {MODEL_OUT / 'itn_results.json'}")
    print(f"\n✅ Done. eval_loss = {eval_result.get('eval_loss', '?'):.4f}")
    print(f"   Training took: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=15)
    parser.add_argument("--lr",      type=float, default=3e-5)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
