#!/usr/bin/env python3
"""
T5-Large Fine-tuning Script for E-commerce Typo Correction

Supports both full fine-tuning and LoRA (PEFT) fine-tuning with optimizations
for RTX 5070 Ti (16GB VRAM). Uses google-t5/t5-large (770M params).

Dataset format: JSONL with {"input_text": "correct: ...", "target_text": "..."}
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_cosine_schedule_with_warmup,
)
from transformers.integrations import TensorBoardCallback

# Conditional PEFT import
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def check_peft_availability(use_lora: bool):
    """Check PEFT availability if LoRA is requested."""
    if use_lora and not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA fine-tuning. "
            "Install it with: pip install peft"
        )


def detect_bf16_support() -> bool:
    """Detect BF16 support on Ampere+ GPUs."""
    if not torch.cuda.is_available():
        return False

    # Check GPU capability (Ampere+ supports BF16)
    # RTX 5070 Ti is Ada architecture, supports BF16
    device_capability = torch.cuda.get_device_capability(0)
    return device_capability[0] >= 8  # Ampere (8.0+) and Ada (8.9)


def load_data(data_dir: str, max_samples: Optional[int] = None) -> Dict:
    """Load JSONL dataset."""
    logger.info(f"Loading dataset from {data_dir}")

    dataset_files = {
        'train': os.path.join(data_dir, 'train_t5.jsonl'),
        'validation': os.path.join(data_dir, 'eval_t5.jsonl'),
    }

    # Check file existence
    for split, path in dataset_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} split not found at {path}")

    # Load dataset
    dataset = load_dataset(
        'json',
        data_files=dataset_files,
        cache_dir=None
    )

    if max_samples:
        dataset['train'] = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(max_samples, len(dataset['validation']))))

    logger.info(f"Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['validation'])}")
    return dataset


def preprocess_function(examples, tokenizer, max_length: int):
    """Preprocess examples for T5."""
    inputs = examples['input_text']
    targets = examples['target_text']

    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False,  # Dynamic padding via DataCollator
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def compute_metrics(eval_preds, tokenizer) -> Dict[str, float]:
    """Compute evaluation metrics: sentence_accuracy, token_accuracy, CER, precision, recall."""
    predictions, labels = eval_preds

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Sentence accuracy
    sentence_accuracy = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

    # Token accuracy and CER
    token_correct = 0
    token_total = 0
    char_errors = 0
    char_total = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = pred.split()
        label_tokens = label.split()

        token_total += len(label_tokens)
        for p_tok, l_tok in zip(pred_tokens, label_tokens):
            if p_tok == l_tok:
                token_correct += 1

        # Character-level error rate
        char_total += len(label)
        char_errors += sum(1 for p, l in zip(pred, label) if p != l)
        char_errors += abs(len(pred) - len(label))

    token_accuracy = token_correct / max(token_total, 1)
    cer = char_errors / max(char_total, 1)

    # Precision and Recall
    # Precision: correct changes / total changes attempted
    # Recall: correct changes / total changes needed
    correct_changes = 0
    total_changes_attempted = 0
    total_changes_needed = 0

    for pred, label in zip(decoded_preds, decoded_labels):
        if pred != label:
            total_changes_attempted += 1
            if pred == label:
                correct_changes += 1

        if label != label:  # Changes needed (simplified: any difference)
            total_changes_needed += 1

    precision = correct_changes / max(total_changes_attempted, 1)
    recall = correct_changes / max(total_changes_needed, 1)

    return {
        'sentence_accuracy': sentence_accuracy,
        'token_accuracy': token_accuracy,
        'cer': cer,
        'precision': precision,
        'recall': recall,
    }


def setup_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Setup LoRA (PEFT) configuration."""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=['q', 'v'],
        lora_dropout=lora_dropout,
        bias='none',
        task_type='SEQ_2_SEQ_LM',
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune T5-large for typo correction')

    # Model and data
    parser.add_argument('--model_name', type=str, default='google-t5/t5-large',
                        help='HuggingFace model name')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory with train_t5.jsonl and eval_t5.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs/t5-large-typo',
                        help='Output directory for model checkpoints')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max sequence length for tokenization')

    # Training config
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size (will use defaults if not specified)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (will use defaults if not specified)')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluation steps')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--grad_accum', type=int, default=None,
                        help='Gradient accumulation steps (default: 1 for LoRA, 4 for full)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 (disable if BF16 auto-detected)')

    # LoRA config
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA (PEFT) fine-tuning instead of full fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')

    # Checkpoint & resume
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume training from checkpoint path (e.g. outputs/t5-large-typo/checkpoint-1500). '
                             'Use "auto" to resume from latest checkpoint in output_dir.')
    parser.add_argument('--save_steps', type=int, default=None,
                        help='Save checkpoint every N steps (defaults to eval_steps)')

    # Other
    parser.add_argument('--max_samples', type=int,
                        help='Max samples per split for debugging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Check PEFT availability
    check_peft_availability(args.use_lora)

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Training T5-large for typo correction")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA: {args.use_lora}")
    logger.info(f"Output: {args.output_dir}")

    # Load dataset
    dataset = load_data(args.data_dir, args.max_samples)

    # Load model and tokenizer
    logger.info(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Setup LoRA if requested
    if args.use_lora:
        logger.info("Setting up LoRA (PEFT)...")
        model = setup_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05
        )
        default_batch_size = 16
        default_lr = 1e-3
    else:
        default_batch_size = 8
        default_lr = 3e-4

    # Set batch size and learning rate with defaults
    batch_size = args.batch_size or default_batch_size
    learning_rate = args.lr or default_lr

    logger.info(f"Batch size: {batch_size}, LR: {learning_rate}")

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Detect BF16 support
    use_bf16 = detect_bf16_support() and not args.fp16
    logger.info(f"BF16 support: {use_bf16}, FP16: {args.fp16}")

    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        batch_size=256,
        remove_columns=['input_text', 'target_text'],
    )

    # Data collator with dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=int(len(dataset['train']) / batch_size * args.epochs * 0.1),
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=args.fp16 and not use_bf16,
        gradient_accumulation_steps=args.grad_accum or (1 if args.use_lora else 4),
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.eval_steps,  # must equal eval_steps when load_best_model_at_end=True
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=5,
        logging_steps=50,
        logging_dir=f'{args.output_dir}/logs',
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        optim='adamw_torch',
        seed=args.seed,
        report_to=['tensorboard'],
    )

    # Trainer — eval uses only loss (no generation) for speed & stability
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[TensorBoardCallback()],
    )

    # Train (with optional resume from checkpoint)
    resume_checkpoint = None
    if args.resume_from_checkpoint == "auto":
        # Find latest checkpoint in output_dir
        import glob as glob_mod
        checkpoints = sorted(glob_mod.glob(os.path.join(args.output_dir, 'checkpoint-*')))
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            logger.info(f"Auto-resuming from: {resume_checkpoint}")
        else:
            logger.info("No checkpoint found, starting fresh.")
    elif args.resume_from_checkpoint:
        resume_checkpoint = args.resume_from_checkpoint
        logger.info(f"Resuming from: {resume_checkpoint}")

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save best model
    logger.info(f"Saving best model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("\n=== FINAL METRICS ===")
    for key, value in sorted(eval_results.items()):
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    logger.info(f"\nTraining completed. Best model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
