#!/usr/bin/env python3
"""
ByT5 Fine-tuning for E-Commerce Typo Correction
=================================================

Supports: google/byt5-small, google/byt5-base, google/byt5-large
Hardware target: RTX 5070 Ti (16 GB VRAM)

Improvements over previous version:
    - Dynamic padding (DataCollator) instead of fixed max_length → faster training
    - Label smoothing (0.1) for better generalisation
    - Character Error Rate (CER) metric alongside accuracy
    - Cosine scheduler with restarts option
    - BF16 auto-detection for Ampere+ GPUs
    - Gradient clipping (max_grad_norm=1.0)

Usage examples:
    # Quick dry-run
    python train.py --model_name google/byt5-small --epochs 1 --batch_size 4

    # Production training (byt5-small, current)
    python train.py --model_name google/byt5-small --epochs 10 --fp16

    # byt5-base (recommended upgrade)
    python train.py --model_name google/byt5-base --epochs 10 --fp16

    # Continue from existing checkpoint
    python train.py --model_name ../byt5-typo-final --epochs 5 --fp16

    # ByT5-large (aggressive settings for 16 GB VRAM)
    python train.py --model_name google/byt5-large --epochs 10 --fp16 --gradient_checkpointing
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

# Allow importing app.metrics from parent directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ======================================================================
# Dataset
# ======================================================================

class TypoCorrectionDataset(Dataset):
    """Loads JSONL with {"input_text": "correct: ...", "target_text": "..."} entries.

    Returns raw tokenised tensors WITHOUT padding — padding is handled
    dynamically by DataCollatorForSeq2Seq for optimal batch efficiency.
    """

    def __init__(self, filepath: Path, tokenizer, max_source_len: int = 128, max_target_len: int = 128):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.examples = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    self.examples.append(obj)

        print(f"  Loaded {len(self.examples)} examples from {filepath.name}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        source = ex["input_text"]
        target = ex["target_text"]

        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_len,
            truncation=True,
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_target_len,
            truncation=True,
        )

        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


# ======================================================================
# Metrics
# ======================================================================

def build_compute_metrics(tokenizer):
    """Return a compute_metrics function compatible with Seq2SeqTrainer.

    Metrics:
        - sentence_accuracy: Exact-match at sentence level
        - token_accuracy: Word-level accuracy
        - cer: Character Error Rate (lower is better)
    """

    def _cer(pred: str, ref: str) -> float:
        """Character Error Rate via simple edit distance."""
        n = len(ref) or 1
        if pred == ref:
            return 0.0
        # Dynamic programming edit distance
        d = list(range(len(ref) + 1))
        for i, pc in enumerate(pred):
            nd = [i + 1] + [0] * len(ref)
            for j, rc in enumerate(ref):
                cost = 0 if pc == rc else 1
                nd[j + 1] = min(nd[j] + 1, d[j + 1] + 1, d[j] + cost)
            d = nd
        return d[-1] / n

    def compute_metrics(eval_preds):
        predictions, label_ids = eval_preds

        # If predictions are logits (3D), take argmax to get token IDs
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Replace -100 and out-of-range IDs with pad token
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        predictions = np.where(
            (predictions >= 0) & (predictions < tokenizer.vocab_size),
            predictions,
            tokenizer.pad_token_id,
        )

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Sentence-level exact-match accuracy
        correct = sum(1 for p, t in zip(decoded_preds, decoded_labels) if p == t)
        accuracy = correct / len(decoded_preds) if decoded_preds else 0.0

        # Token-level accuracy
        total_tokens = 0
        correct_tokens = 0
        for p, t in zip(decoded_preds, decoded_labels):
            p_tok = p.split()
            t_tok = t.split()
            min_len = min(len(p_tok), len(t_tok))
            for i in range(min_len):
                if p_tok[i] == t_tok[i]:
                    correct_tokens += 1
            total_tokens += max(len(p_tok), len(t_tok))
        token_acc = correct_tokens / total_tokens if total_tokens else 0.0

        # Character Error Rate
        cer_scores = [_cer(p, t) for p, t in zip(decoded_preds, decoded_labels)]
        avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0

        return {
            "sentence_accuracy": round(accuracy, 4),
            "token_accuracy": round(token_acc, 4),
            "cer": round(avg_cer, 4),
        }

    return compute_metrics


# ======================================================================
# Recommended configs per model variant
# ======================================================================

CONFIGS = {
    "byt5-small": {
        "batch_size": 16,
        "lr": 3e-4,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 1,
    },
    "byt5-base": {
        "batch_size": 12,
        "lr": 5e-5,
        "gradient_checkpointing": False,
        "gradient_accumulation_steps": 2,
        # effective batch = 12 * 2 = 24 → daha stabil gradients
    },
    "byt5-large": {
        "batch_size": 4,
        "lr": 3e-5,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
    },
}


def get_recommended_config(model_name: str) -> dict:
    """Return sensible defaults based on model variant."""
    model_lower = model_name.lower()
    for key, cfg in CONFIGS.items():
        if key in model_lower:
            return cfg
    # Fallback to small config
    return CONFIGS["byt5-small"]


# ======================================================================
# Training
# ======================================================================

def train(args):
    print("=" * 60)
    print("  ByT5 Fine-tuning for Typo Correction")
    print("=" * 60)

    # Resolve data paths
    data_dir = Path(args.data_dir)
    train_file = data_dir / "train_t5.jsonl"
    eval_file = data_dir / "eval_t5.jsonl"

    if not train_file.exists():
        print(f"\nERROR: {train_file} not found.")
        print("Run:  python prepare_data.py --augment --samples 8000 --multi-word 4000 --sentences 4000 --spacing-variants --symbol-variants")
        sys.exit(1)

    # Get recommended config
    rec = get_recommended_config(args.model_name)
    batch_size = args.batch_size or rec["batch_size"]
    lr = args.lr or rec["lr"]
    grad_accum = args.gradient_accumulation or rec["gradient_accumulation_steps"]
    grad_ckpt = args.gradient_checkpointing or rec["gradient_checkpointing"]

    # Auto-detect precision
    use_fp16 = args.fp16
    use_bf16 = False
    if torch.cuda.is_available() and not args.no_fp16:
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere+ (RTX 30xx, 40xx, 50xx)
            use_bf16 = True
            use_fp16 = False
            print(f"  Auto-detected BF16 support (compute capability {capability[0]}.{capability[1]})")

    print(f"\n  Model:           {args.model_name}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {lr}")
    print(f"  FP16:            {use_fp16}")
    print(f"  BF16:            {use_bf16}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Grad checkpoint: {grad_ckpt}")
    print(f"  Grad accumulate: {grad_accum}")
    print(f"  Output dir:      {args.output_dir}")

    # Load tokenizer and model
    print(f"\n  Loading model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    if grad_ckpt:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} total, {trainable:,} trainable")

    # Load datasets
    print("\n  Loading datasets ...")
    train_ds = TypoCorrectionDataset(train_file, tokenizer, args.max_length, args.max_length)
    eval_ds_full = TypoCorrectionDataset(eval_file, tokenizer, args.max_length, args.max_length)

    # Subsample eval set for faster evaluation during training
    # (ByT5 generate is slow; full eval on 2900 examples takes too long)
    eval_size = min(args.eval_subset, len(eval_ds_full))
    if eval_size < len(eval_ds_full):
        eval_ds = torch.utils.data.Subset(eval_ds_full, list(range(eval_size)))
        print(f"  Using {eval_size}/{len(eval_ds_full)} eval examples for speed")
    else:
        eval_ds = eval_ds_full

    # Data collator with dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate warmup steps (10% of total)
    steps_per_epoch = len(train_ds) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)

    import os
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "logs")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        label_smoothing_factor=args.label_smoothing,
        max_grad_norm=1.0,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="sentence_accuracy",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=1,
        report_to=["tensorboard"],
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # Train
    print("\n" + "=" * 60)
    print("  Starting training ...")
    print("=" * 60 + "\n")
    trainer.train()

    # Save best model
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\n  Best model saved to: {best_dir}")

    # Final evaluation
    print("\n  Running final evaluation ...")
    metrics = trainer.evaluate()
    print(f"\n  Final metrics: {json.dumps(metrics, indent=2)}")

    # Save metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ByT5 for typo correction")

    # Model
    p.add_argument("--model_name", type=str, default="google/byt5-base",
                    help="HuggingFace model ID or local path (google/byt5-small, google/byt5-base, google/byt5-large)")
    p.add_argument("--max_length", type=int, default=128,
                    help="Max sequence length for source and target")

    # Data
    p.add_argument("--data_dir", type=str, default=str(Path(__file__).parent / "data"),
                    help="Directory with train_t5.jsonl and eval_t5.jsonl")

    # Training
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=None,
                    help="Override batch size (auto-selected per model if not set)")
    p.add_argument("--lr", type=float, default=None,
                    help="Override learning rate (auto-selected per model if not set)")
    p.add_argument("--fp16", action="store_true", default=True,
                    help="Use FP16 mixed precision (default: True)")
    p.add_argument("--no_fp16", action="store_true",
                    help="Disable FP16")
    p.add_argument("--gradient_checkpointing", action="store_true",
                    help="Enable gradient checkpointing (saves VRAM)")
    p.add_argument("--gradient_accumulation", type=int, default=None,
                    help="Gradient accumulation steps")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing factor (0.0 = off, 0.1 = recommended)")

    # Evaluation
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--eval_subset", type=int, default=500,
                    help="Max eval examples per evaluation (speeds up ByT5 generate)")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (number of eval steps without improvement)")

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "outputs" / "byt5-typo"),
                    help="Output directory for checkpoints")

    args = p.parse_args()

    if args.no_fp16:
        args.fp16 = False

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
