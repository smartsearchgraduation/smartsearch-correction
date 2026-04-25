#!/usr/bin/env python3
"""
T5-Large v3 Fine-tuning — Google Colab Script
================================================
Copy each section (delimited by # ═══ CELL N ═══) into a separate Colab cell.
Or upload this file and run: %run t5_large_finetune_v3.py

CRITICAL FIXES OVER v2.1:
  1. LR 3e-5        (v2.1 used 5e-4 → 16x too high, caused noisy gradients)
  2. Adafactor       (designed for T5, adapts per-parameter LR, saves memory)
  3. Cosine schedule (v2.1 used linear → premature LR decay)
  4. Warmup 10%      (v2.1 used 6% → unstable first steps)
  5. num_beams=4     (v2.1 eval used greedy → underestimated real accuracy)
  6. EarlyStopping   (v2.1 had none → risk of overfitting on 5 epochs)
  7. No-harm metric  (overcorrection is worse than missing a fix)
  8. 400K examples   (v2.1 had 205K with 40% identity padding)

Expected: 75-82% exact match, 97-99% no-harm rate
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 1: SETUP & INSTALLATION
# ═══════════════════════════════════════════════════════════════════════

import subprocess, sys, os

print("Installing packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.44.0",
    "datasets>=2.20.0",
    "accelerate>=0.33.0",
    "tensorboard>=2.14.0",
    "sentencepiece>=0.2.0",
    "protobuf>=3.20.0",
])
print("Done.\n")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("Google Drive mounted.")

from pathlib import Path
import torch

# ── Paths ──
# Upload 'BYT5-T5 Large v3' folder to: Drive > Grad/Correction/fine_tune/
DRIVE_BASE = Path('/content/drive/MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3')
DATA_DIR   = DRIVE_BASE / 'data'
OUTPUT_DIR = DRIVE_BASE / 'output' / 'T5-Large-v3-final'
LOG_DIR    = OUTPUT_DIR / 'logs'

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Verify data files ──
train_file = DATA_DIR / 'train_v3.jsonl'
eval_file  = DATA_DIR / 'eval_v3.jsonl'

assert train_file.exists(), f"Train data not found: {train_file}\nRun build_training_data_v3.py first, then upload data/ to Drive."
assert eval_file.exists(),  f"Eval data not found: {eval_file}"

train_count = sum(1 for _ in open(train_file, encoding='utf-8'))
eval_count  = sum(1 for _ in open(eval_file, encoding='utf-8'))

print(f"\nData verified:")
print(f"  Train: {train_file.name} ({train_count:,} examples)")
print(f"  Eval:  {eval_file.name}  ({eval_count:,} examples)")
print(f"  Output: {OUTPUT_DIR}")

# Stats file if available
stats_file = DATA_DIR / 'training_stats_v3.json'
if stats_file.exists():
    import json
    with open(stats_file) as f:
        stats = json.load(f)
    print(f"\n  Identity ratio: {stats.get('identity_ratio', 'N/A')}")
    print(f"  Brand count:    {stats.get('brand_count', 'N/A')}")
    print(f"  Categories:     {len(stats.get('categories', {}))}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 2: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

import json

MODEL_NAME = "google-t5/t5-large"

# ── GPU detection & auto-tune ──
gpu_name = "CPU"
gpu_mem_gb = 0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

if gpu_mem_gb >= 35:       # A100-40GB / A100-80GB
    BATCH_SIZE = 16
    GRAD_ACCUM = 4
    FP16 = False
    BF16 = True
    EVAL_BATCH = 32
elif gpu_mem_gb >= 14:     # T4-16GB / L4-24GB / V100-16GB
    BATCH_SIZE = 8
    GRAD_ACCUM = 8
    FP16 = True
    BF16 = False
    EVAL_BATCH = 16
else:                      # CPU or very low GPU
    BATCH_SIZE = 4
    GRAD_ACCUM = 16
    FP16 = False
    BF16 = False
    EVAL_BATCH = 8

# ═══════════════════════════════════════════════════════
# WHY THESE HYPERPARAMETERS (v2.1 mistakes annotated)
# ═══════════════════════════════════════════════════════
#
# learning_rate: 3e-5
#   v2.1 used 5e-4 (16x higher!). For T5-Large (770M params),
#   high LR causes gradient noise, the model oscillates instead of
#   converging. Google's T5 paper recommends 1e-4 for pre-training
#   but 1e-5 to 5e-5 for fine-tuning on downstream tasks.
#   3e-5 is the sweet spot: fast enough to converge in 3 epochs,
#   low enough to avoid overshooting.
#
# optimizer: Adafactor
#   T5 was pre-trained with Adafactor. It adapts LR per-parameter
#   using second-moment estimates (like Adam) but factorizes the
#   accumulator, saving ~30% optimizer memory vs AdamW.
#   scale_parameter=False + relative_step=False = external LR control.
#
# lr_scheduler: cosine
#   Linear decay (v2.1 default) drops LR too aggressively mid-training.
#   Cosine keeps a higher effective LR through the middle epochs,
#   then gently tapers — better for seq2seq fine-tuning.
#
# warmup_ratio: 0.10
#   v2.1 used 0.06. With a lower LR we can afford 10% warmup.
#   Prevents early gradient spikes from corrupting pre-trained weights.
#
# label_smoothing: 0.1
#   v2.1 used 0.05. Slightly higher smoothing helps seq2seq models
#   avoid overconfident token predictions — critical for
#   spelling correction (the model must generalize to unseen typos).
#
# num_epochs: 3
#   v2.1 used 5 with no early stopping, likely overfit on later epochs.
#   With 400K examples and LR 3e-5, 3 epochs is ~18K steps.
#   EarlyStopping with patience=3 guards against overfitting.
#
# generation_num_beams: 4
#   v2.1 evaluated with greedy (beams=1), then tested with beams=4.
#   This mismatch means training optimized for the wrong strategy.
#   We eval with beams=4 so the best checkpoint is truly the best.
#
CONFIG = {
    "model_name":       MODEL_NAME,
    "max_input_length":  64,
    "max_target_length": 64,

    # Training
    "num_epochs":        3,
    "batch_size":        BATCH_SIZE,
    "eval_batch_size":   EVAL_BATCH,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate":     3e-5,
    "warmup_ratio":      0.10,
    "weight_decay":      0.01,
    "label_smoothing":   0.1,
    "max_grad_norm":     1.0,

    # Evaluation & saving
    "eval_steps":        500,
    "save_steps":        500,
    "logging_steps":     50,
    "save_total_limit":  3,
    "early_stopping_patience": 3,

    # Generation
    "generation_num_beams": 4,

    # Precision
    "fp16": FP16,
    "bf16": BF16,
    "gradient_checkpointing": True,
    "seed": 42,
}

eff_batch = CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']
est_steps_per_epoch = train_count // eff_batch
est_total = est_steps_per_epoch * CONFIG['num_epochs']

print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
print(f"Batch: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {eff_batch} effective")
print(f"Precision: {'bf16' if BF16 else 'fp16' if FP16 else 'fp32'}")
print(f"LR: {CONFIG['learning_rate']} (v2.1 was 5e-4 -- 16x higher!)")
print(f"Scheduler: cosine with {CONFIG['warmup_ratio']:.0%} warmup")
print(f"Optimizer: Adafactor (T5-native)")
print(f"Epochs: {CONFIG['num_epochs']} + EarlyStopping(patience={CONFIG['early_stopping_patience']})")
print(f"Eval beams: {CONFIG['generation_num_beams']}")
print(f"Est. steps/epoch: ~{est_steps_per_epoch:,}, total: ~{est_total:,}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 3: LOAD MODEL & TOKENIZER
# ═══════════════════════════════════════════════════════════════════════

import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("[1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_enc = tokenizer("correct: corsiar keybord wireless", return_tensors="pt")
print(f"  Vocab size: {tokenizer.vocab_size:,}")
print(f"  'correct: corsiar keybord wireless' -> {test_enc['input_ids'].shape[1]} tokens")

print(f"\n[2/3] Loading model: {MODEL_NAME} (~2.95 GB)...")
t0 = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
elapsed = time.time() - t0
print(f"  Loaded in {elapsed:.0f}s")

if CONFIG['gradient_checkpointing']:
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing: ON (saves ~40% memory)")

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {total_params:,} ({total_params/1e6:.0f}M)")
print(f"  Trainable:  {trainable:,} (100% -- full fine-tune)")

if torch.cuda.is_available():
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Sanity check
print(f"\n[3/3] Sanity check...")
dummy_in = tokenizer("correct: test query", return_tensors="pt")
dummy_tgt = tokenizer("test query", return_tensors="pt")

with torch.no_grad():
    out = model(input_ids=dummy_in["input_ids"], labels=dummy_tgt["input_ids"])
print(f"  Forward OK (loss={out.loss.item():.4f})")

with torch.no_grad():
    gen = model.generate(dummy_in["input_ids"], max_new_tokens=20, num_beams=4)
decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
print(f"  Generate: 'correct: test query' -> '{decoded}'")
print("\nModel ready.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 4: LOAD & TOKENIZE DATA
# ═══════════════════════════════════════════════════════════════════════

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from collections import Counter

print("Loading dataset from JSONL...")
dataset = load_dataset('json', data_files={
    'train': str(train_file),
    'validation': str(eval_file),
})
print(f"  Train:      {len(dataset['train']):,}")
print(f"  Validation: {len(dataset['validation']):,}")

# Category distribution
cats = Counter(ex['category'] for ex in dataset['train'])
print(f"\nCategories ({len(cats)}):")
for cat, cnt in cats.most_common(15):
    pct = cnt / len(dataset['train']) * 100
    bar = ">" * max(1, int(pct / 2))
    print(f"  {cat:<30s} {cnt:>7,} ({pct:5.1f}%) {bar}")
if len(cats) > 15:
    print(f"  ... and {len(cats) - 15} more")

# Identity ratio check
id_count = sum(1 for ex in dataset['train']
               if ex['input_text'].replace('correct: ', '', 1) == ex['target_text'])
print(f"\nIdentity pairs: {id_count:,} / {len(dataset['train']):,} "
      f"({100 * id_count / len(dataset['train']):.1f}%)")

# Samples
print("\nSample corrections:")
shown_cats = set()
for ex in dataset['train']:
    inp = ex['input_text'].replace('correct: ', '')
    tgt = ex['target_text']
    cat = ex['category']
    if cat not in shown_cats and inp != tgt:
        shown_cats.add(cat)
        print(f"  [{cat:<25s}] '{inp}' -> '{tgt}'")
    if len(shown_cats) >= 8:
        break

# ── Tokenize ──
def tokenize_fn(examples):
    inputs = tokenizer(
        examples['input_text'],
        max_length=CONFIG['max_input_length'],
        truncation=True,
        padding=False,
    )
    targets = tokenizer(
        text_target=examples['target_text'],
        max_length=CONFIG['max_target_length'],
        truncation=True,
        padding=False,
    )
    inputs['labels'] = targets['input_ids']
    return inputs

print("\nTokenizing...")
tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=2000,
    remove_columns=['input_text', 'target_text', 'category'],
    desc="Tokenizing",
    num_proc=2,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# Stats
sample = tokenized['train'][0]
trunc_count = sum(1 for ex in tokenized['train']
                  if len(ex['input_ids']) >= CONFIG['max_input_length'])
print(f"  Sample input:  {len(sample['input_ids'])} tokens")
print(f"  Sample labels: {len(sample['labels'])} tokens")
print(f"  Truncated: {trunc_count:,} / {len(tokenized['train']):,} "
      f"({100 * trunc_count / len(tokenized['train']):.2f}%)")
print("Tokenization complete.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 5: METRICS -- EXACT MATCH + NO-HARM + TOKEN ACCURACY + CER
# ═══════════════════════════════════════════════════════════════════════

import numpy as np

def compute_metrics(eval_preds):
    """
    Metrics:
      - sentence_accuracy: exact match (pred == target)
      - token_accuracy: word-level accuracy
      - cer: character error rate
    """
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 with pad token for decoding
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    n = len(decoded_labels)
    if n == 0:
        return {"sentence_accuracy": 0, "token_accuracy": 0, "cer": 1}

    # ── Exact match ──
    exact = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    sentence_accuracy = exact / n

    # ── Token (word-level) accuracy ──
    tok_correct = tok_total = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        p_words = pred.split()
        l_words = label.split()
        tok_total += len(l_words)
        for pw, lw in zip(p_words, l_words):
            if pw == lw:
                tok_correct += 1
    token_accuracy = tok_correct / max(tok_total, 1)

    # ── Character Error Rate ──
    char_errors = char_total = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        char_total += max(len(label), 1)
        common_len = min(len(pred), len(label))
        char_errors += sum(1 for a, b in zip(pred[:common_len], label[:common_len]) if a != b)
        char_errors += abs(len(pred) - len(label))
    cer = char_errors / max(char_total, 1)

    return {
        "sentence_accuracy": round(sentence_accuracy, 4),
        "token_accuracy":    round(token_accuracy, 4),
        "cer":               round(cer, 4),
    }

print("Metrics defined: sentence_accuracy, token_accuracy, cer")
print("Primary metric for best model: sentence_accuracy")


# ═══════════════════════════════════════════════════════════════════════
# CELL 6: TRAINING
# ═══════════════════════════════════════════════════════════════════════

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback
import time as _time
import gc

# Set True to resume from latest checkpoint
RESUME = False

# ── Custom callback for monitoring ──
class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self._train_start = None
        self._best_acc = 0.0
        self._best_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = _time.time()
        print("\n" + "=" * 70)
        print("  TRAINING STARTED -- T5-Large v3")
        print(f"  Epochs: {args.num_train_epochs} | Steps: {state.max_steps}")
        print(f"  Eval every {args.eval_steps} steps | Beams: {args.generation_num_beams}")
        print(f"  LR: {args.learning_rate} | Scheduler: {args.lr_scheduler_type}")
        print(f"  Early stopping: patience={CONFIG['early_stopping_patience']}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.memory_allocated()/1e9:.1f} GB used)")
        print("=" * 70)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            acc = metrics.get('eval_sentence_accuracy', 0)
            loss = metrics.get('eval_loss', 0)
            cer = metrics.get('eval_cer', 1)
            tok = metrics.get('eval_token_accuracy', 0)

            improved = ""
            if acc > self._best_acc:
                self._best_acc = acc
                self._best_step = state.global_step
                improved = " << NEW BEST"

            elapsed = (_time.time() - self._train_start) / 60 if self._train_start else 0
            print(f"\n  [Step {state.global_step:>5d}] "
                  f"loss={loss:.4f} | acc={acc:.4f} | tok={tok:.4f} | cer={cer:.4f}"
                  f"{improved} ({elapsed:.0f}min)")

    def on_save(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_end(self, args, state, control, **kwargs):
        total_min = (_time.time() - self._train_start) / 60 if self._train_start else 0
        print(f"\n{'=' * 70}")
        print(f"  TRAINING COMPLETE")
        print(f"  Duration: {total_min:.0f} min ({total_min/60:.1f} hours)")
        print(f"  Best accuracy: {self._best_acc:.4f} at step {self._best_step}")
        print(f"{'=' * 70}")

# ── Compute steps ──
eff_batch = CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']
steps_per_epoch = len(tokenized['train']) // eff_batch
total_steps = steps_per_epoch * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

print(f"Steps/epoch: {steps_per_epoch:,}")
print(f"Total steps: {total_steps:,}")
print(f"Warmup: {warmup_steps:,} ({CONFIG['warmup_ratio']:.0%})")

# ── Training arguments ──
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Epochs & batching
    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['eval_batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],

    # Optimizer & LR -- THE CRITICAL FIXES
    learning_rate=CONFIG['learning_rate'],        # 3e-5, NOT 5e-4
    lr_scheduler_type="cosine",                   # cosine, NOT linear
    warmup_steps=warmup_steps,                    # 10%, NOT 6%
    weight_decay=CONFIG['weight_decay'],
    label_smoothing_factor=CONFIG['label_smoothing'],   # 0.1, NOT 0.05
    max_grad_norm=CONFIG['max_grad_norm'],
    optim="adafactor",                            # T5-native, NOT adamw

    # Precision
    fp16=CONFIG['fp16'] and torch.cuda.is_available(),
    bf16=CONFIG['bf16'] and torch.cuda.is_available(),

    # Evaluation
    eval_strategy="steps",
    eval_steps=CONFIG['eval_steps'],
    save_strategy="steps",
    save_steps=CONFIG['save_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="sentence_accuracy",
    greater_is_better=True,
    save_total_limit=CONFIG['save_total_limit'],

    # Generation during eval -- USE BEAM SEARCH
    predict_with_generate=True,
    generation_max_length=CONFIG['max_target_length'],
    generation_num_beams=CONFIG['generation_num_beams'],   # 4, NOT 1

    # Logging
    logging_dir=str(LOG_DIR),
    logging_steps=CONFIG['logging_steps'],
    report_to=['tensorboard'],

    # Performance
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    gradient_checkpointing=CONFIG['gradient_checkpointing'],
    seed=CONFIG['seed'],
    torch_compile=False,
)

# ── Trainer ──
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        TrainingMonitor(),
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG['early_stopping_patience'],
            early_stopping_threshold=0.001,
        ),
    ],
)

# ── Resume logic ──
resume_ckpt = None
if RESUME:
    import glob as _g
    ckpts = sorted(_g.glob(str(OUTPUT_DIR / 'checkpoint-*')))
    if ckpts:
        resume_ckpt = ckpts[-1]
        print(f"Resuming from: {resume_ckpt}")
    else:
        print("No checkpoint found, starting fresh.")

# ── Train ──
print(f"\n{'=' * 70}")
print(f"  STARTING T5-LARGE v3 FINE-TUNING")
print(f"  LR={CONFIG['learning_rate']} | Optimizer=Adafactor | Schedule=cosine")
print(f"  Beams={CONFIG['generation_num_beams']} | Smoothing={CONFIG['label_smoothing']}")
print(f"{'=' * 70}")

result = trainer.train(resume_from_checkpoint=resume_ckpt)

# ── Save best model ──
best_dir = OUTPUT_DIR / 'best_model'
trainer.save_model(str(best_dir))
tokenizer.save_pretrained(str(best_dir))

print(f"\nBest model saved: {best_dir}")
print(f"Final training loss: {result.training_loss:.4f}")

# Save config alongside model
with open(best_dir / 'v3_training_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
print(f"Config saved: {best_dir / 'v3_training_config.json'}")


# ═══════════════════════════════════════════════════════════════════════
# CELL 7: TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════

import json, glob
import matplotlib.pyplot as plt

# Find trainer state
state_files = sorted(glob.glob(str(OUTPUT_DIR / 'checkpoint-*' / 'trainer_state.json')))
state_file = OUTPUT_DIR / 'trainer_state.json'
if state_file.exists():
    state_files.append(str(state_file))

if state_files:
    with open(state_files[-1]) as f:
        state = json.load(f)
    logs = state.get('log_history', [])

    train_steps = [(l['step'], l['loss']) for l in logs if 'loss' in l and 'eval_loss' not in l]
    eval_losses = [(l['step'], l['eval_loss']) for l in logs if 'eval_loss' in l]
    eval_accs   = [(l['step'], l['eval_sentence_accuracy']) for l in logs if 'eval_sentence_accuracy' in l]
    eval_toks   = [(l['step'], l['eval_token_accuracy']) for l in logs if 'eval_token_accuracy' in l]
    eval_cers   = [(l['step'], l['eval_cer']) for l in logs if 'eval_cer' in l]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    if train_steps:
        ax.plot(*zip(*train_steps), label='Train', alpha=0.6, linewidth=0.8)
    if eval_losses:
        ax.plot(*zip(*eval_losses), label='Eval', linewidth=2, color='red')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # Sentence Accuracy
    ax = axes[0, 1]
    if eval_accs:
        steps, accs = zip(*eval_accs)
        ax.plot(steps, accs, color='green', linewidth=2, marker='o', markersize=4)
        best_idx = int(np.argmax(accs))
        ax.axhline(y=accs[best_idx], color='green', linestyle='--', alpha=0.3)
        ax.annotate(f'Best: {accs[best_idx]:.4f}', xy=(steps[best_idx], accs[best_idx]),
                    fontsize=10, color='green', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Exact Match')
    ax.set_title('Sentence Accuracy'); ax.grid(True, alpha=0.3)

    # Token Accuracy
    ax = axes[1, 0]
    if eval_toks:
        ax.plot(*zip(*eval_toks), color='blue', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Step'); ax.set_ylabel('Token Accuracy')
    ax.set_title('Word-Level Accuracy'); ax.grid(True, alpha=0.3)

    # CER
    ax = axes[1, 1]
    if eval_cers:
        ax.plot(*zip(*eval_cers), color='orange', linewidth=2, marker='^', markersize=3)
    ax.set_xlabel('Step'); ax.set_ylabel('CER')
    ax.set_title('Character Error Rate (lower=better)'); ax.grid(True, alpha=0.3)

    plt.suptitle('T5-Large v3 Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'training_curves_v3.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {OUTPUT_DIR / 'training_curves_v3.png'}")
else:
    print("No trainer state found yet.")


# ═══════════════════════════════════════════════════════════════════════
# CELL 8: COMPREHENSIVE EVALUATION -- THE REAL TEST
# ═══════════════════════════════════════════════════════════════════════

import time

# ── Load best model ──
best_path = OUTPUT_DIR / 'best_model'
if not best_path.exists():
    ckpts = sorted(glob.glob(str(OUTPUT_DIR / 'checkpoint-*')))
    best_path = Path(ckpts[-1]) if ckpts else OUTPUT_DIR
    print(f"No best_model dir, using: {best_path}")

print(f"Loading model from: {best_path}")
eval_tokenizer = AutoTokenizer.from_pretrained(str(best_path))
eval_model = AutoModelForSeq2SeqLM.from_pretrained(str(best_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_model = eval_model.to(device).eval()
print(f"Model on {device}")

def correct(query: str, num_beams: int = 4) -> str:
    """Run inference with beam search."""
    enc = eval_tokenizer(
        f"correct: {query}",
        return_tensors='pt',
        max_length=64,
        truncation=True,
    )
    with torch.no_grad():
        out = eval_model.generate(
            enc.input_ids.to(device),
            attention_mask=enc.attention_mask.to(device),
            max_length=64,
            num_beams=num_beams,
            length_penalty=1.0,
            early_stopping=True,
        )
    return eval_tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════
# TEST SUITE -- organized by category for systematic evaluation
# Format: (input, expected_output, category)
# ═══════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── IDENTITY: Must NOT change these (overcorrection = bad) ──
    ("wireless mouse",          "wireless mouse",          "identity"),
    ("samsung galaxy",          "samsung galaxy",          "identity"),
    ("iphone 15 pro max",       "iphone 15 pro max",       "identity"),
    ("running shoes",           "running shoes",           "identity"),
    ("laptop bag",              "laptop bag",              "identity"),
    ("keyboard",                "keyboard",                "identity"),
    ("128gb ssd",               "128gb ssd",               "identity"),
    ("headphones",              "headphones",              "identity"),
    ("asus rog laptop",         "asus rog laptop",         "identity"),
    ("corsair k70 rgb",         "corsair k70 rgb",         "identity"),
    ("logitech mx master",      "logitech mx master",      "identity"),
    ("nvidia rtx 4090",         "nvidia rtx 4090",         "identity"),
    ("razer blade 16",          "razer blade 16",          "identity"),
    ("sport bra",               "sport bra",               "identity"),
    ("protein powder",          "protein powder",          "identity"),
    ("yoga mat",                "yoga mat",                "identity"),
    ("usb c cable",             "usb c cable",             "identity"),
    ("gaming chair",            "gaming chair",            "identity"),
    ("air purifier",            "air purifier",            "identity"),
    ("coffee maker",            "coffee maker",            "identity"),

    # ── BRAND TYPOS: Core capability ──
    ("samsng galaxy",           "samsung galaxy",          "brand_typo"),
    ("iphnoe 15",               "iphone 15",               "brand_typo"),
    ("logitec mouse",           "logitech mouse",          "brand_typo"),
    ("corsiar k70",             "corsair k70",             "brand_typo"),
    ("nvidai rtx 4080",         "nvidia rtx 4080",         "brand_typo"),
    ("assu rog laptop",         "asus rog laptop",         "brand_typo"),
    ("lenvo thinkpad",          "lenovo thinkpad",         "brand_typo"),
    ("steelsires headset",      "steelseries headset",     "brand_typo"),
    ("razr blade",              "razer blade",             "brand_typo"),
    ("mackbook pro",            "macbook pro",             "brand_typo"),

    # ── PRODUCT TYPOS: Generic terms ──
    ("wireles mouse",           "wireless mouse",          "product_typo"),
    ("keybord",                 "keyboard",                "product_typo"),
    ("heaphones",               "headphones",              "product_typo"),
    ("lapto computer",          "laptop computer",         "product_typo"),
    ("runnng shoes",            "running shoes",           "product_typo"),
    ("moniter 27 inch",         "monitor 27 inch",         "product_typo"),
    ("wireles charger",         "wireless charger",        "product_typo"),
    ("smartphon case",          "smartphone case",         "product_typo"),
    ("bluethooth speaker",      "bluetooth speaker",       "product_typo"),
    ("gameing mouse",           "gaming mouse",            "product_typo"),

    # ── COMPOUND TYPOS: Multiple errors per query ──
    ("samsng galxy s24",        "samsung galaxy s24",      "compound"),
    ("wireles keybord",         "wireless keyboard",       "compound"),
    ("corsiar heaphones",       "corsair headphones",      "compound"),
    ("nvidai geforc rtx",       "nvidia geforce rtx",      "compound"),
    ("logitec wireles mouse",   "logitech wireless mouse", "compound"),

    # ── SPACE ERRORS: Merged/split words ──
    ("air pods pro",            "airpods pro",             "space_error"),
    ("head phones wireless",    "headphones wireless",     "space_error"),
    ("lap top gaming",          "laptop gaming",           "space_error"),
    ("key board mechanical",    "keyboard mechanical",     "space_error"),
    ("smart watch",             "smartwatch",              "space_error"),

    # ── PHONETIC ERRORS: Sound-alike substitutions ──
    ("grafics card",            "graphics card",           "phonetic"),
    ("processer intel",         "processor intel",         "phonetic"),
    ("baterry pack",            "battery pack",            "phonetic"),
    ("dispaly 4k",              "display 4k",              "phonetic"),
    ("performace laptop",       "performance laptop",      "phonetic"),
]

# ── Run tests ──
print(f"\n{'_' * 100}")
print(f"{'INPUT':<35} {'EXPECTED':<35} {'PREDICTED':<35} {'RESULT'}")
print(f"{'_' * 100}")

results_by_cat = {}
all_ok = all_total = 0

for inp, expected, cat in TEST_CASES:
    pred = correct(inp)
    ok = pred.lower().strip() == expected.lower().strip()

    if cat not in results_by_cat:
        results_by_cat[cat] = {"ok": 0, "total": 0, "fails": []}
    results_by_cat[cat]["total"] += 1
    results_by_cat[cat]["ok"] += int(ok)
    if not ok:
        results_by_cat[cat]["fails"].append((inp, expected, pred))

    all_ok += int(ok)
    all_total += 1

    status = "  OK  " if ok else " FAIL "
    print(f"{inp:<35} {expected:<35} {pred:<35} [{status}]")

# ── Summary ──
print(f"\n{'=' * 100}")
print(f"  TEST RESULTS SUMMARY")
print(f"{'=' * 100}")

for cat in ["identity", "brand_typo", "product_typo", "compound", "space_error", "phonetic"]:
    if cat in results_by_cat:
        r = results_by_cat[cat]
        pct = 100 * r["ok"] / r["total"]
        status = "PASS" if pct >= 80 else "WARN" if pct >= 60 else "FAIL"
        print(f"  {cat:<20s} {r['ok']:>2d}/{r['total']:<2d} ({pct:5.1f}%)  [{status}]")

print(f"  {'_' * 45}")
print(f"  {'OVERALL':<20s} {all_ok:>2d}/{all_total:<2d} ({100 * all_ok / all_total:5.1f}%)")

# No-harm rate: for identity tests, how many did the model NOT break?
id_results = results_by_cat.get("identity", {"ok": 0, "total": 0})
if id_results["total"] > 0:
    no_harm = 100 * id_results["ok"] / id_results["total"]
    print(f"\n  NO-HARM RATE (identity preservation): {id_results['ok']}/{id_results['total']} ({no_harm:.1f}%)")
    if no_harm < 95:
        print(f"  WARNING: No-harm rate below 95% -- model may overcorrect!")

# Show failures
if any(r["fails"] for r in results_by_cat.values()):
    print(f"\n  FAILURES:")
    for cat, r in results_by_cat.items():
        for inp, exp, pred in r["fails"]:
            print(f"    [{cat}] '{inp}' -> expected '{exp}', got '{pred}'")

# ── Latency test ──
print(f"\n  LATENCY (beam=4, {device}):")
latencies = []
test_queries = ["wireles mouse", "samsng galaxy", "keybord", "nvidai rtx 4080", "logitec mouse"]
for q in test_queries:
    times = []
    for _ in range(3):  # 3 runs, take median
        t0 = time.time()
        r = correct(q, 4)
        times.append((time.time() - t0) * 1000)
    med = sorted(times)[1]
    latencies.append(med)
    print(f"    '{q}' -> '{r}' ({med:.0f}ms)")
print(f"    Median latency: {sorted(latencies)[len(latencies)//2]:.0f}ms")


# ═══════════════════════════════════════════════════════════════════════
# CELL 9: EXPORT FOR DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════

import shutil
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
export_dir = OUTPUT_DIR / f't5_large_v3_final_{ts}'
export_dir.mkdir(parents=True, exist_ok=True)

# Save model
eval_model.save_pretrained(str(export_dir))
eval_tokenizer.save_pretrained(str(export_dir))

# Save training config
with open(export_dir / 'training_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)

# Save metadata
metadata = {
    "model_name": CONFIG['model_name'],
    "version": "v3",
    "date": datetime.now().isoformat(),
    "train_examples": train_count,
    "eval_examples": eval_count,
    "hyperparameters": {
        "learning_rate": CONFIG['learning_rate'],
        "optimizer": "adafactor",
        "scheduler": "cosine",
        "warmup_ratio": CONFIG['warmup_ratio'],
        "label_smoothing": CONFIG['label_smoothing'],
        "num_epochs": CONFIG['num_epochs'],
        "effective_batch_size": eff_batch,
        "generation_num_beams": CONFIG['generation_num_beams'],
    },
    "test_results": {
        "overall_accuracy": f"{all_ok}/{all_total} ({100*all_ok/all_total:.1f}%)",
        "identity_preservation": f"{id_results['ok']}/{id_results['total']}",
        "categories": {cat: f"{r['ok']}/{r['total']}" for cat, r in results_by_cat.items()},
    },
    "gpu": gpu_name,
    "training_data_version": "v3",
    "improvements_over_v2_1": [
        "LR 3e-5 (was 5e-4)",
        "Adafactor optimizer (was AdamW)",
        "Cosine schedule (was linear)",
        "10% warmup (was 6%)",
        "beam=4 eval (was greedy)",
        "400K examples with 22% identity (was 205K/40%)",
        "Phonetic + compound + space error typos (was QWERTY-only)",
        "Broad e-commerce coverage (was electronics-only)",
        "EarlyStopping patience=3 (was none)",
    ],
}
with open(export_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Create zip for easy download
zip_base = str(export_dir)
shutil.make_archive(zip_base, 'zip', export_dir)
zip_size_mb = os.path.getsize(f"{zip_base}.zip") / (1024**2)
model_size_mb = sum(f.stat().st_size for f in export_dir.rglob('*') if f.is_file()) / (1024**2)

print(f"Exported to: {export_dir}")
print(f"  Model size: {model_size_mb:.0f} MB")
print(f"  Zip: {zip_base}.zip ({zip_size_mb:.0f} MB)")
print(f"\nTo deploy:")
print(f"  1. Download the zip or the best_model/ folder")
print(f"  2. Place in your server's model directory")
print(f"  3. Load with: AutoModelForSeq2SeqLM.from_pretrained(path)")
print(f"  4. Use num_beams=4 for best quality")
print(f"\nDone! T5-Large v3 is ready for deployment.")
