"""
train_whisper_lora.py
----------------------
Phase 2: Fine-tune Whisper-small on MUCS Hinglish ASR corpus using LoRA (PEFT).

Architecture:
  Base model : openai/whisper-small (244M params)
  LoRA targets: q_proj, v_proj (encoder + decoder attention)
  LoRA config : r=8, alpha=16, dropout=0.1
  Task        : Seq2Seq speech-to-text (Hinglish)

Training data: data/processed/hinglish_asr/
  - 52,714 train utterances (MUCS 21, real Hinglish speech at 16kHz)
  - 3,128  test  utterances

Outputs:
  models/adapters/whisper_lora/   ← LoRA adapter weights
  models/adapters/whisper_lora/wer_results.json

GPU required: ≥6GB VRAM (Whisper-small). Tested on RTX 3060 12GB.

Usage:
  python scripts/train/train_whisper_lora.py
  python scripts/train/train_whisper_lora.py --epochs 5 --batch-size 8
  python scripts/train/train_whisper_lora.py --dry-run   # 50 steps, quick sanity check
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Defaults
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\HP\GitMakesMeHappy\Entropic_ASR")
ASR_DATA     = BASE_DIR / "data" / "processed" / "hinglish_asr"
MODEL_OUT    = BASE_DIR / "models" / "adapters" / "whisper_lora"
BASE_MODEL   = "openai/whisper-small"
TARGET_SR    = 16_000
MAX_DURATION = 30.0   # Whisper's hard limit


# ──────────────────────────────────────────────────────────────────────────────
# LoRA Config
# ──────────────────────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    task_type    = TaskType.SEQ_2_SEQ_LM,
    r            = 8,          # rank  — 8 is efficient; try 16 for more capacity
    lora_alpha   = 16,         # scaling = alpha/r = 2
    lora_dropout = 0.1,
    target_modules = ["q_proj", "v_proj"],  # attention Q and V matrices only
    bias         = "none",
)


# ──────────────────────────────────────────────────────────────────────────────
# Data collator — loads audio on-the-fly, pads batch
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WhisperDataCollator:
    processor      : WhisperProcessor
    max_label_len  : int = 448    # Whisper tokenizer max

    def _load_audio(self, path: str) -> np.ndarray:
        """Load WAV file with soundfile (no torchcodec needed)."""
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1:          # stereo → mono
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        # Clip to max 30s
        max_samples = int(MAX_DURATION * TARGET_SR)
        arr = arr[:max_samples]
        return arr

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract log-mel features from audio files
        arrays = [self._load_audio(f["audio_path"]) for f in features]
        input_features = self.processor(
            arrays,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        ).input_features   # (B, 80, 3000)

        # Tokenize transcripts
        labels_batch = self.processor.tokenizer(
            [f["transcript"] for f in features],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_label_len,
        )
        labels = labels_batch["input_ids"]
        # Replace padding token id with -100 so loss ignores it
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# WER compute callback for Trainer
# ──────────────────────────────────────────────────────────────────────────────
def make_compute_metrics(processor: WhisperProcessor):
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 padding back to pad_token_id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str   = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str  = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("[WARN] Running on CPU — training will be very slow.")
        print("       Enable GPU / use Kaggle or Colab with T4 for real training.")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {ASR_DATA}")
    dataset = load_from_disk(str(ASR_DATA))
    print(dataset)

    if args.dry_run:
        print("\n[DRY RUN] Limiting to 200 train + 50 test examples...")
        dataset["train"] = dataset["train"].select(range(min(200, len(dataset["train"]))))
        dataset["test"]  = dataset["test"].select(range(min(50,  len(dataset["test"]))))

    # ── Load processor (feature extractor + tokenizer) ────────────────────
    print(f"\nLoading Whisper processor: {BASE_MODEL}")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language="hi",       # Hindi-English code-switched → use "hi" as seed language
        task="transcribe",
    )
    # Force multilingual decoding (do NOT force English)
    processor.tokenizer.set_prefix_tokens(language="hi", task="transcribe")

    # ── Load model + apply LoRA ───────────────────────────────────────────
    print(f"\nLoading base model: {BASE_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = get_peft_model(model, LORA_CONFIG)

    # Configure generation via generation_config (model.config is deprecated for this)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.generation_config.language           = "hi"
    model.generation_config.task               = "transcribe"

    model.print_trainable_parameters()
    # Expected: ~2M trainable / 244M total ≈ 0.9% — LoRA efficiency

    # ── Training args ─────────────────────────────────────────────────────
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir               = str(MODEL_OUT),
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = 4,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate            = 1e-4,
        warmup_steps             = 500,
        max_steps                = args.max_steps if args.dry_run else -1,
        num_train_epochs         = args.epochs,
        eval_strategy            = "epoch",
        save_strategy            = "epoch",
        load_best_model_at_end   = True,
        metric_for_best_model    = "wer",
        greater_is_better        = False,    # lower WER = better
        logging_steps            = 25,
        predict_with_generate    = True,
        generation_max_length    = 448,
        fp16                     = (device == "cuda"),
        dataloader_num_workers   = 0,        # Windows: keep 0 to avoid multiprocessing issues
        remove_unused_columns    = False,    # keep audio_path+transcript for data collator
        report_to                = "none",   # disable W&B by default
        save_total_limit         = 2,
    )

    # ── Data collator ─────────────────────────────────────────────────────
    data_collator = WhisperDataCollator(processor=processor)

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = dataset["train"],
        eval_dataset      = dataset["test"],
        data_collator     = data_collator,
        compute_metrics   = make_compute_metrics(processor),
        processing_class  = processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Starting Whisper LoRA training...")
    print(f"  Train examples : {len(dataset['train'])}")
    print(f"  Eval  examples : {len(dataset['test'])}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Batch size     : {args.batch_size} × grad_accum {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Output dir     : {MODEL_OUT}")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ── Save LoRA adapter ─────────────────────────────────────────────────
    model.save_pretrained(str(MODEL_OUT))
    processor.save_pretrained(str(MODEL_OUT))
    print(f"\n✓ LoRA adapter saved: {MODEL_OUT}")

    # ── Final eval ────────────────────────────────────────────────────────
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final WER: {eval_results.get('eval_wer', 'N/A'):.4f}")
    print(f"Training took: {elapsed/60:.1f} minutes")

    results = {
        "base_model"    : BASE_MODEL,
        "lora_r"        : LORA_CONFIG.r,
        "lora_alpha"    : LORA_CONFIG.lora_alpha,
        "train_samples" : len(dataset["train"]),
        "test_samples"  : len(dataset["test"]),
        "epochs"        : args.epochs,
        "elapsed_min"   : round(elapsed / 60, 1),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in eval_results.items()},
    }
    results_path = MODEL_OUT / "wer_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"✓ Results saved: {results_path}")

    print(f"\n✅ Done. WER = {eval_results.get('eval_wer', '?')}")
    print(f"   WER target: <25% on MUCS test set")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int,   default=4,
                        help="Per-device train batch size (default: 4; lower for <8GB VRAM)")
    parser.add_argument("--grad-accum", type=int,   default=8,
                        help="Gradient accumulation steps. Effective batch = batch*accum")
    parser.add_argument("--max-steps",  type=int,   default=50,
                        help="Max steps for --dry-run (default: 50)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run 50 steps on 200 examples to verify the pipeline")
    args = parser.parse_args()
    main(args)
