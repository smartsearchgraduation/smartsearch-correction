#!/usr/bin/env python3
"""
ByT5-Large v3 Fine-tuning -- Google Colab Script
==================================================
Copy each section (delimited by # === CELL N ===) into a separate Colab cell.

ByT5-Large (1.2B params) is a byte-level seq2seq model -- no tokenizer vocab,
every UTF-8 byte is a token. This means:
  - "keyboard" = 8 tokens (vs ~2 subword tokens in T5)
  - Sequences are ~3-5x longer than T5 for the same text
  - Can handle ANY character (emojis, accents, symbols)
  - Better at character-level edits (typos!) but slower inference

PRE-REQUISITE:
  Run download_byt5_to_drive.ipynb FIRST to cache the model on Drive.
  This notebook loads from Drive cache (no internet needed for model).

DIFFERENCES FROM T5-Large v3:
  - max_length: 128 (vs 64) -- byte-level needs more tokens
  - Batch size: smaller (model is 1.2B vs 770M)
  - LR: 1e-5 (lower than T5's 3e-5 -- larger model needs gentler updates)
  - Training time: ~2-3x longer than T5-Large
"""

# =====================================================================
# CELL 1: SETUP & VERIFY MODEL CACHE
# =====================================================================

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

from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("Google Drive mounted.")

from pathlib import Path
import torch

# -- Paths --
DRIVE_BASE  = Path('/content/drive/MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3')
DATA_DIR    = DRIVE_BASE / 'data'
OUTPUT_DIR  = DRIVE_BASE / 'output' / 'BYT5-Large-v3-final'
LOG_DIR     = OUTPUT_DIR / 'logs'
MODEL_CACHE = DRIVE_BASE / 'hf_model_cache'

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Verify data --
train_file = DATA_DIR / 'train_v3.jsonl'
eval_file  = DATA_DIR / 'eval_v3.jsonl'

assert train_file.exists(), f"Train data not found: {train_file}"
assert eval_file.exists(),  f"Eval data not found: {eval_file}"

train_count = sum(1 for _ in open(train_file, encoding='utf-8'))
eval_count  = sum(1 for _ in open(eval_file, encoding='utf-8'))

print(f"\nData verified:")
print(f"  Train: {train_file.name} ({train_count:,} examples)")
print(f"  Eval:  {eval_file.name}  ({eval_count:,} examples)")

# -- Verify model cache (must run download_byt5_to_drive.ipynb first) --
MODEL_NAME = "google/byt5-large"
LOCAL_MODEL = MODEL_CACHE / "byt5-large"
weight_file = LOCAL_MODEL / "pytorch_model.bin"

assert LOCAL_MODEL.exists() and (LOCAL_MODEL / "config.json").exists(), (
    f"Model cache not found at {LOCAL_MODEL}\n"
    "Run download_byt5_to_drive.ipynb first to download the model to Drive."
)
assert weight_file.exists(), (
    f"pytorch_model.bin not found in {LOCAL_MODEL}\n"
    "Run download_byt5_to_drive.ipynb first."
)
model_size = weight_file.stat().st_size / 1e9
assert model_size > 4.0, (
    f"pytorch_model.bin is only {model_size:.1f} GB (expected ~4.7 GB) -- truncated.\n"
    "Delete the folder and re-run download_byt5_to_drive.ipynb."
)
model_path = str(LOCAL_MODEL)
print(f"\nModel cache verified:")
print(f"  Path: {LOCAL_MODEL}")
print(f"  Weight: pytorch_model.bin ({model_size:.1f} GB)")

stats_file = DATA_DIR / 'training_stats_v3.json'
if stats_file.exists():
    import json
    with open(stats_file) as f:
        stats = json.load(f)
    print(f"\n  Identity ratio: {stats.get('identity_ratio', 'N/A')}")
    print(f"  Brand count:    {stats.get('brand_count', 'N/A')}")
    print(f"  Categories:     {len(stats.get('categories', {}))}")


# =====================================================================
# CELL 2: CONFIGURATION
# =====================================================================

import json

# -- GPU detection & auto-tune --
gpu_name = "CPU"
gpu_mem_gb = 0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

# ByT5-Large is 1.2B params -- needs careful memory management
if gpu_mem_gb >= 35:       # A100-40GB / A100-80GB
    BATCH_SIZE = 8
    GRAD_ACCUM = 8
    FP16 = False
    BF16 = True
    EVAL_BATCH = 16
elif gpu_mem_gb >= 14:     # T4-16GB / L4-24GB
    BATCH_SIZE = 4
    GRAD_ACCUM = 16
    FP16 = True
    BF16 = False
    EVAL_BATCH = 8
else:                      # CPU or very low GPU
    BATCH_SIZE = 2
    GRAD_ACCUM = 32
    FP16 = False
    BF16 = False
    EVAL_BATCH = 4

# ---------------------------------------------------------------
# HYPERPARAMETER RATIONALE FOR ByT5-Large
# ---------------------------------------------------------------
#
# learning_rate: 1e-5
#   Lower than T5-Large's 3e-5 because ByT5-Large has 1.2B params
#   (vs 770M). Larger models need gentler LR to avoid catastrophic
#   forgetting. Google's ByT5 paper used 1e-3 for pre-training,
#   but fine-tuning should be 10-100x lower.
#
# max_input_length: 128 (vs T5's 64)
#   ByT5 tokenizes at byte level: "wireless keyboard" = 17 bytes
#   vs ~4 subword tokens in T5. P99 query length = 36 chars.
#   With prefix: "correct: " (9) + query (~36) = ~45 bytes.
#   128 is plenty of headroom while saving memory vs 256.
#
# max_target_length: 128
#   Same reasoning -- output is also byte-level.
#
# num_epochs: 3
#   Same as T5-Large. With 400K examples and lower LR,
#   3 epochs + early stopping is sufficient.
#
# generation_num_beams: 4
#   Beam search is more important for byte-level models because
#   they make decisions at character granularity. Greedy can
#   miss the correct character sequence.
#
CONFIG = {
    "model_name":       MODEL_NAME,
    "model_path":       str(LOCAL_MODEL),
    "max_input_length":  128,       # P99 query=36 chars + "correct: "(9) = ~45 bytes; 128 is plenty
    "max_target_length": 128,       # same reasoning -- output also byte-level

    # Training
    "num_epochs":        3,
    "batch_size":        BATCH_SIZE,
    "eval_batch_size":   EVAL_BATCH,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate":     1e-5,      # lower than T5's 3e-5 -- larger model
    "warmup_ratio":      0.10,
    "weight_decay":      0.0,       # Adafactor has its own decay; external WD conflicts
    "label_smoothing":   0.1,
    "max_grad_norm":     1.0,

    # Eval & saving
    "eval_steps":        2000,      # was 500 -> 84 evals too many for 902K data on Colab
    "save_steps":        2000,      # sync with eval_steps
    "logging_steps":     100,
    "save_total_limit":  5,         # was 2! Must be >= patience+1 so best ckpt survives
    "early_stopping_patience": 3,
    "eval_subset_size":  5000,      # subsample eval during training (full eval in Cell 8)

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
print(f"Model: ByT5-Large (1.2B params, byte-level)")
print(f"Batch: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {eff_batch} effective")
print(f"Precision: {'bf16' if BF16 else 'fp16' if FP16 else 'fp32'}")
print(f"LR: {CONFIG['learning_rate']} (T5-Large used 3e-5)")
print(f"Max length: {CONFIG['max_input_length']} bytes (T5 used 64 subwords)")
print(f"Scheduler: cosine with {CONFIG['warmup_ratio']:.0%} warmup")
print(f"Optimizer: Adafactor (no external weight_decay -- Adafactor handles its own)")
print(f"Epochs: {CONFIG['num_epochs']} + EarlyStopping(patience={CONFIG['early_stopping_patience']})")
print(f"Eval: beam={CONFIG['generation_num_beams']}, every {CONFIG['eval_steps']} steps, subset={CONFIG.get('eval_subset_size', 'full')}")
print(f"Est. steps/epoch: ~{est_steps_per_epoch:,}, total: ~{est_total:,}")


# =====================================================================
# CELL 3: LOAD MODEL & TOKENIZER (from Drive cache)
# =====================================================================

import time
from transformers import AutoTokenizer, T5ForConditionalGeneration

load_path = CONFIG['model_path']

print(f"[1/3] Loading tokenizer from {load_path}...")
tokenizer = AutoTokenizer.from_pretrained(load_path)

# ByT5 tokenizer demo -- byte-level, no vocabulary
test_text = "correct: corsiar keybord"
test_enc = tokenizer(test_text, return_tensors="pt")
print(f"  Tokenizer: {type(tokenizer).__name__}")
print(f"  '{test_text}' -> {test_enc['input_ids'].shape[1]} byte-tokens")
print(f"  (T5-Large would use ~6 subword tokens for this)")

print(f"\n[2/3] Loading ByT5-Large model (~4.7 GB from Drive)...")
t0 = time.time()

# Use T5ForConditionalGeneration (ByT5 uses T5 architecture)
model = T5ForConditionalGeneration.from_pretrained(load_path)

elapsed = time.time() - t0
print(f"  Loaded in {elapsed:.0f}s (from Drive cache -- no internet needed)")

if CONFIG['gradient_checkpointing']:
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing: ON")

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
    gen = model.generate(dummy_in["input_ids"], max_new_tokens=50, num_beams=4)
decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
print(f"  Generate: 'correct: test query' -> '{decoded}'")
print("\nByT5-Large ready.")


# =====================================================================
# CELL 4: LOAD & TOKENIZE DATA
# =====================================================================

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
for cat, cnt in cats.most_common(10):
    pct = cnt / len(dataset['train']) * 100
    bar = ">" * max(1, int(pct / 2))
    print(f"  {cat:<30s} {cnt:>7,} ({pct:5.1f}%) {bar}")

# Identity ratio
id_count = sum(1 for ex in dataset['train']
               if ex['input_text'].replace('correct: ', '', 1) == ex['target_text'])
print(f"\nIdentity pairs: {id_count:,} / {len(dataset['train']):,} "
      f"({100 * id_count / len(dataset['train']):.1f}%)")

# Samples
print("\nSample corrections:")
shown = set()
for ex in dataset['train']:
    inp = ex['input_text'].replace('correct: ', '')
    tgt = ex['target_text']
    cat = ex['category']
    if cat not in shown and inp != tgt:
        shown.add(cat)
        # Show byte-token count for ByT5
        n_bytes = len(tokenizer(ex['input_text'])['input_ids'])
        print(f"  [{cat:<22s}] '{inp}' -> '{tgt}' ({n_bytes} byte-tokens)")
    if len(shown) >= 6:
        break

# -- Tokenize --
# ByT5 byte-level: max_length=256 bytes covers queries up to ~240 chars
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

print("\nTokenizing (byte-level, sequences are longer than T5)...")
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
avg_len = sum(len(ex['input_ids']) for ex in tokenized['train']) / len(tokenized['train'])
print(f"  Sample input:  {len(sample['input_ids'])} byte-tokens")
print(f"  Sample labels: {len(sample['labels'])} byte-tokens")
print(f"  Avg input length: {avg_len:.0f} byte-tokens")
print(f"  Truncated: {trunc_count:,} / {len(tokenized['train']):,} "
      f"({100 * trunc_count / len(tokenized['train']):.2f}%)")

# -- Eval subset for training-time evaluation (faster, avoids Colab timeout) --
eval_subset_size = CONFIG.get('eval_subset_size', 5000)
if len(tokenized['validation']) > eval_subset_size:
    eval_subset = tokenized['validation'].shuffle(seed=CONFIG['seed']).select(range(eval_subset_size))
    print(f"  Eval subset: {eval_subset_size:,} / {len(tokenized['validation']):,} (for training-time eval)")
    print(f"  Full eval set ({len(tokenized['validation']):,}) used in Cell 8 for final evaluation")
else:
    eval_subset = tokenized['validation']
    print(f"  Eval set small enough, using full set for training-time eval")

print("Tokenization complete.")


# =====================================================================
# CELL 5: METRICS
# =====================================================================

import numpy as np
from collections import Counter

# ---------------------------------------------------------------
# Levenshtein distance (edit distance) -- used for real CER
# ---------------------------------------------------------------
def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[len(s2)]


def _word_f1(pred_words: list, label_words: list) -> float:
    """Word-level F1: handles insertions, deletions, reorderings."""
    pred_set = Counter(pred_words)
    label_set = Counter(label_words)
    common = sum((pred_set & label_set).values())
    if common == 0:
        return 0.0
    precision = common / max(len(pred_words), 1)
    recall = common / max(len(label_words), 1)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

    n = len(decoded_labels)
    if n == 0:
        return {"sentence_accuracy": 0, "word_f1": 0, "cer": 1}

    # 1. Exact match (sentence accuracy)
    exact = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    sentence_accuracy = exact / n

    # 2. Word-level F1 (replaces broken zip-based token_accuracy)
    word_f1_sum = sum(
        _word_f1(p.split(), l.split())
        for p, l in zip(decoded_preds, decoded_labels)
    )
    word_f1 = word_f1_sum / n

    # 3. Levenshtein CER (replaces broken position-based CER)
    char_errors = char_total = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        char_total += max(len(label), 1)
        char_errors += _levenshtein(pred, label)
    cer = char_errors / max(char_total, 1)

    return {
        "sentence_accuracy": round(sentence_accuracy, 4),
        "word_f1":           round(word_f1, 4),
        "cer":               round(cer, 4),
    }

print("Metrics: sentence_accuracy (exact match), word_f1 (F1-based), cer (Levenshtein-based)")


# =====================================================================
# CELL 6: TRAINING
# =====================================================================

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback
import time as _time
import gc

RESUME = False

class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self._train_start = None
        self._best_acc = 0.0
        self._best_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = _time.time()
        print("\n" + "=" * 70)
        print("  TRAINING STARTED -- ByT5-Large v3 (byte-level)")
        print(f"  Epochs: {args.num_train_epochs} | Steps: {state.max_steps}")
        print(f"  Eval every {args.eval_steps} steps | Beams: {args.generation_num_beams}")
        print(f"  LR: {args.learning_rate} | Scheduler: {args.lr_scheduler_type}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.memory_allocated()/1e9:.1f} GB used)")
        print("=" * 70)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            acc = metrics.get('eval_sentence_accuracy', 0)
            loss = metrics.get('eval_loss', 0)
            cer = metrics.get('eval_cer', 1)

            improved = ""
            if acc > self._best_acc:
                self._best_acc = acc
                self._best_step = state.global_step
                improved = " << NEW BEST"

            elapsed = (_time.time() - self._train_start) / 60 if self._train_start else 0
            print(f"\n  [Step {state.global_step:>5d}] "
                  f"loss={loss:.4f} | acc={acc:.4f} | cer={cer:.4f}"
                  f"{improved} ({elapsed:.0f}min)")

    def on_save(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_end(self, args, state, control, **kwargs):
        total_min = (_time.time() - self._train_start) / 60 if self._train_start else 0
        print(f"\n{'=' * 70}")
        print(f"  TRAINING COMPLETE -- ByT5-Large v3")
        print(f"  Duration: {total_min:.0f} min ({total_min/60:.1f} hours)")
        print(f"  Best accuracy: {self._best_acc:.4f} at step {self._best_step}")
        print(f"{'=' * 70}")

# Steps
eff_batch = CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']
steps_per_epoch = len(tokenized['train']) // eff_batch
total_steps = steps_per_epoch * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

print(f"Steps/epoch: {steps_per_epoch:,}")
print(f"Total steps: {total_steps:,}")
print(f"Warmup: {warmup_steps:,}")

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['eval_batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],

    learning_rate=CONFIG['learning_rate'],
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    weight_decay=CONFIG['weight_decay'],
    label_smoothing_factor=CONFIG['label_smoothing'],
    max_grad_norm=CONFIG['max_grad_norm'],
    optim="adafactor",

    fp16=CONFIG['fp16'] and torch.cuda.is_available(),
    bf16=CONFIG['bf16'] and torch.cuda.is_available(),

    eval_strategy="steps",
    eval_steps=CONFIG['eval_steps'],
    save_strategy="steps",
    save_steps=CONFIG['save_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="sentence_accuracy",
    greater_is_better=True,
    save_total_limit=CONFIG['save_total_limit'],

    predict_with_generate=True,
    generation_max_length=CONFIG['max_target_length'],
    generation_num_beams=CONFIG['generation_num_beams'],

    logging_dir=str(LOG_DIR),
    logging_steps=CONFIG['logging_steps'],
    report_to=['tensorboard'],

    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    gradient_checkpointing=CONFIG['gradient_checkpointing'],
    seed=CONFIG['seed'],
    torch_compile=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=eval_subset,          # subset for fast training-time eval
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

resume_ckpt = None
if RESUME:
    import glob as _g
    ckpts = sorted(_g.glob(str(OUTPUT_DIR / 'checkpoint-*')))
    if ckpts:
        resume_ckpt = ckpts[-1]
        print(f"Resuming from: {resume_ckpt}")

print(f"\n{'=' * 70}")
print(f"  STARTING ByT5-LARGE v3 FINE-TUNING (byte-level)")
print(f"  LR={CONFIG['learning_rate']} | Optimizer=Adafactor | Schedule=cosine")
print(f"  Max length: {CONFIG['max_input_length']} bytes")
print(f"{'=' * 70}")

result = trainer.train(resume_from_checkpoint=resume_ckpt)

best_dir = OUTPUT_DIR / 'best_model'
trainer.save_model(str(best_dir))
tokenizer.save_pretrained(str(best_dir))

print(f"\nBest model saved: {best_dir}")
print(f"Final training loss: {result.training_loss:.4f}")

with open(best_dir / 'v3_training_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2, default=str)


# =====================================================================
# CELL 7: TRAINING CURVES
# =====================================================================

import json, glob
import matplotlib.pyplot as plt

state_files = sorted(glob.glob(str(OUTPUT_DIR / 'checkpoint-*' / 'trainer_state.json')))
sf = OUTPUT_DIR / 'trainer_state.json'
if sf.exists():
    state_files.append(str(sf))

if state_files:
    with open(state_files[-1]) as f:
        state = json.load(f)
    logs = state.get('log_history', [])

    train_steps = [(l['step'], l['loss']) for l in logs if 'loss' in l and 'eval_loss' not in l]
    eval_losses = [(l['step'], l['eval_loss']) for l in logs if 'eval_loss' in l]
    eval_accs   = [(l['step'], l['eval_sentence_accuracy']) for l in logs if 'eval_sentence_accuracy' in l]
    eval_cers   = [(l['step'], l['eval_cer']) for l in logs if 'eval_cer' in l]

    eval_f1s = [(l['step'], l['eval_word_f1']) for l in logs if 'eval_word_f1' in l]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    ax = axes[0]
    if train_steps: ax.plot(*zip(*train_steps), label='Train', alpha=0.6)
    if eval_losses: ax.plot(*zip(*eval_losses), label='Eval', linewidth=2, color='red')
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if eval_accs:
        steps, accs = zip(*eval_accs)
        ax.plot(steps, accs, color='green', linewidth=2, marker='o', markersize=4)
        best_idx = int(np.argmax(accs))
        ax.annotate(f'Best: {accs[best_idx]:.4f}', xy=(steps[best_idx], accs[best_idx]),
                    fontsize=10, color='green', fontweight='bold')
    ax.set_title('Sentence Accuracy'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if eval_f1s:
        s, f = zip(*eval_f1s)
        ax.plot(s, f, color='blue', linewidth=2, marker='s', markersize=4)
    ax.set_title('Word F1'); ax.grid(True, alpha=0.3)

    ax = axes[3]
    if eval_cers: ax.plot(*zip(*eval_cers), color='orange', linewidth=2)
    ax.set_title('CER / Levenshtein (lower=better)'); ax.grid(True, alpha=0.3)

    plt.suptitle('ByT5-Large v3 Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'training_curves_byt5_v3.png'), dpi=150)
    plt.show()
else:
    print("No trainer state found yet.")


# =====================================================================
# CELL 8: COMPREHENSIVE EVALUATION
# =====================================================================

import time

best_path = OUTPUT_DIR / 'best_model'
if not best_path.exists():
    ckpts = sorted(glob.glob(str(OUTPUT_DIR / 'checkpoint-*')))
    best_path = Path(ckpts[-1]) if ckpts else OUTPUT_DIR

print(f"Loading from: {best_path}")
eval_tokenizer = AutoTokenizer.from_pretrained(str(best_path))
eval_model = T5ForConditionalGeneration.from_pretrained(str(best_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_model = eval_model.to(device).eval()
print(f"Model on {device}")

def correct(query: str, num_beams: int = 4) -> str:
    enc = eval_tokenizer(
        f"correct: {query}",
        return_tensors='pt',
        max_length=128,
        truncation=True,
    )
    with torch.no_grad():
        out = eval_model.generate(
            enc.input_ids.to(device),
            attention_mask=enc.attention_mask.to(device),
            max_length=128,
            num_beams=num_beams,
            length_penalty=1.0,
        )
    return eval_tokenizer.decode(out[0], skip_special_tokens=True).strip()


TEST_CASES = [
    # Identity
    ("wireless mouse","wireless mouse","identity"),
    ("samsung galaxy","samsung galaxy","identity"),
    ("iphone 15 pro max","iphone 15 pro max","identity"),
    ("running shoes","running shoes","identity"),
    ("keyboard","keyboard","identity"),
    ("128gb ssd","128gb ssd","identity"),
    ("headphones","headphones","identity"),
    ("corsair k70 rgb","corsair k70 rgb","identity"),
    ("nvidia rtx 4090","nvidia rtx 4090","identity"),
    ("razer blade 16","razer blade 16","identity"),
    ("sport bra","sport bra","identity"),
    ("yoga mat","yoga mat","identity"),
    ("usb c cable","usb c cable","identity"),
    ("gaming chair","gaming chair","identity"),
    ("coffee maker","coffee maker","identity"),
    # Brand typos
    ("samsng galaxy","samsung galaxy","brand_typo"),
    ("iphnoe 15","iphone 15","brand_typo"),
    ("logitec mouse","logitech mouse","brand_typo"),
    ("corsiar k70","corsair k70","brand_typo"),
    ("nvidai rtx 4080","nvidia rtx 4080","brand_typo"),
    ("assu rog laptop","asus rog laptop","brand_typo"),
    ("lenvo thinkpad","lenovo thinkpad","brand_typo"),
    ("steelsires headset","steelseries headset","brand_typo"),
    ("razr blade","razer blade","brand_typo"),
    ("mackbook pro","macbook pro","brand_typo"),
    # Product typos
    ("wireles mouse","wireless mouse","product_typo"),
    ("keybord","keyboard","product_typo"),
    ("heaphones","headphones","product_typo"),
    ("lapto computer","laptop computer","product_typo"),
    ("runnng shoes","running shoes","product_typo"),
    ("moniter 27 inch","monitor 27 inch","product_typo"),
    ("bluethooth speaker","bluetooth speaker","product_typo"),
    ("gameing mouse","gaming mouse","product_typo"),
    # Compound
    ("samsng galxy s24","samsung galaxy s24","compound"),
    ("wireles keybord","wireless keyboard","compound"),
    ("nvidai geforc rtx","nvidia geforce rtx","compound"),
    ("logitec wireles mouse","logitech wireless mouse","compound"),
    # Space errors
    ("air pods pro","airpods pro","space_error"),
    ("head phones wireless","headphones wireless","space_error"),
    ("lap top gaming","laptop gaming","space_error"),
    ("smart watch","smartwatch","space_error"),
    # Phonetic
    ("grafics card","graphics card","phonetic"),
    ("processer intel","processor intel","phonetic"),
    ("baterry pack","battery pack","phonetic"),
    ("dispaly 4k","display 4k","phonetic"),
]

print(f"\n{'INPUT':<35} {'EXPECTED':<35} {'PREDICTED':<35} RESULT")
print("-" * 110)

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

print(f"\n{'='*80}")
print(f"  ByT5-Large v3 TEST RESULTS")
print(f"{'='*80}")
for cat in ["identity", "brand_typo", "product_typo", "compound", "space_error", "phonetic"]:
    if cat in results_by_cat:
        r = results_by_cat[cat]
        pct = 100 * r["ok"] / r["total"]
        status = "PASS" if pct >= 80 else "WARN" if pct >= 60 else "FAIL"
        print(f"  {cat:<20s} {r['ok']:>2d}/{r['total']:<2d} ({pct:5.1f}%)  [{status}]")

print(f"  {'-'*45}")
print(f"  {'OVERALL':<20s} {all_ok:>2d}/{all_total:<2d} ({100 * all_ok / all_total:5.1f}%)")

id_r = results_by_cat.get("identity", {"ok": 0, "total": 0})
if id_r["total"] > 0:
    nh = 100 * id_r["ok"] / id_r["total"]
    print(f"\n  NO-HARM RATE: {id_r['ok']}/{id_r['total']} ({nh:.1f}%)")

if any(r["fails"] for r in results_by_cat.values()):
    print(f"\n  FAILURES:")
    for cat, r in results_by_cat.items():
        for inp, exp, pred in r["fails"]:
            print(f"    [{cat}] '{inp}' -> expected '{exp}', got '{pred}'")

# ---------------------------------------------------------------
# FULL EVAL SET: No-Harm Rate + Accuracy (on ENTIRE eval split)
# ---------------------------------------------------------------
print(f"\n{'='*80}")
print(f"  FULL EVAL SET EVALUATION ({eval_count:,} examples)")
print(f"{'='*80}")

full_eval_results = trainer.predict(tokenized['validation'])
full_metrics = full_eval_results.metrics
print(f"  Sentence Accuracy: {full_metrics.get('test_sentence_accuracy', 'N/A')}")
print(f"  Word F1:           {full_metrics.get('test_word_f1', 'N/A')}")
print(f"  CER (Levenshtein): {full_metrics.get('test_cer', 'N/A')}")

# No-harm rate on identity pairs from eval set
print(f"\n  Computing NO-HARM rate on eval identity pairs...")
eval_ds = load_dataset('json', data_files={'eval': str(eval_file)})['eval']
identity_pairs = [(ex['input_text'].replace('correct: ', '', 1), ex['target_text'])
                  for ex in eval_ds
                  if ex['input_text'].replace('correct: ', '', 1) == ex['target_text']]

if identity_pairs:
    import random as _rnd
    _rnd.seed(42)
    id_sample = _rnd.sample(identity_pairs, min(2000, len(identity_pairs)))
    id_ok = 0
    id_fails = []
    for inp, expected in id_sample:
        pred = correct(inp, num_beams=4)
        if pred.lower().strip() == expected.lower().strip():
            id_ok += 1
        elif len(id_fails) < 20:
            id_fails.append((inp, pred))
    nh_rate = 100 * id_ok / len(id_sample)
    print(f"  NO-HARM RATE: {id_ok}/{len(id_sample)} ({nh_rate:.1f}%) "
          f"[sampled from {len(identity_pairs):,} total identity pairs]")
    if id_fails:
        print(f"  Sample failures (overcorrections):")
        for inp, pred in id_fails[:10]:
            print(f"    '{inp}' -> '{pred}' (should be unchanged)")
else:
    print(f"  No identity pairs found in eval set.")

# Latency
print(f"\n  LATENCY (beam=4, {device}):")
latencies = []
for q in ["wireles mouse", "samsng galaxy", "keybord", "nvidai rtx 4080"]:
    times = []
    for _ in range(3):
        t0 = time.time()
        r = correct(q, 4)
        times.append((time.time() - t0) * 1000)
    med = sorted(times)[1]
    latencies.append(med)
    print(f"    '{q}' -> '{r}' ({med:.0f}ms)")
print(f"    Median: {sorted(latencies)[len(latencies)//2]:.0f}ms")
print(f"    (ByT5-Large is typically 2-3x slower than T5-Large)")


# =====================================================================
# CELL 9: EXPORT
# =====================================================================

import shutil
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
export_dir = OUTPUT_DIR / f'byt5_large_v3_final_{ts}'
export_dir.mkdir(parents=True, exist_ok=True)

eval_model.save_pretrained(str(export_dir))
eval_tokenizer.save_pretrained(str(export_dir))

with open(export_dir / 'training_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2, default=str)

metadata = {
    "model_name": "google/byt5-large",
    "version": "v3",
    "architecture": "byte-level seq2seq (1.2B params)",
    "date": datetime.now().isoformat(),
    "train_examples": train_count,
    "eval_examples": eval_count,
    "hyperparameters": {
        "learning_rate": CONFIG['learning_rate'],
        "optimizer": "adafactor",
        "scheduler": "cosine",
        "warmup_ratio": CONFIG['warmup_ratio'],
        "label_smoothing": CONFIG['label_smoothing'],
        "max_input_length": CONFIG['max_input_length'],
        "generation_num_beams": CONFIG['generation_num_beams'],
    },
    "test_results": {
        "overall": f"{all_ok}/{all_total}",
        "no_harm_rate": f"{id_r['ok']}/{id_r['total']}",
        "categories": {c: f"{r['ok']}/{r['total']}" for c, r in results_by_cat.items()},
    },
    "gpu": gpu_name,
    "no_harm_rate": f"{id_ok}/{len(id_sample)} ({nh_rate:.1f}%)" if identity_pairs else "N/A",
    "full_eval_metrics": {k: v for k, v in full_metrics.items() if k.startswith('test_')},
    "notes": [
        "Byte-level model -- no subword tokenizer, processes raw UTF-8 bytes",
        "Higher latency than T5-Large (~300-500ms vs ~150ms) but better character understanding",
        "Download fix: hf_transfer + Drive caching solves Colab timeout",
        "max_length reduced from 256 to 128 (P99 query=36 chars, 128 bytes is plenty)",
        "Levenshtein CER (was position-based)", "Word F1 (was zip-based token acc)",
        "Eval subset 5K during training (full 47K eval in Cell 8)",
        "Full eval-set no-harm rate (was 15 hardcoded only)",
        "save_total_limit=5 (was 2, risked losing best checkpoint)",
    ],
}
with open(export_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

zip_base = str(export_dir)
shutil.make_archive(zip_base, 'zip', export_dir)
zip_size_mb = os.path.getsize(f"{zip_base}.zip") / (1024**2)

print(f"Exported: {export_dir}")
print(f"  Zip: {zip_base}.zip ({zip_size_mb:.0f} MB)")
print(f"\nTo deploy, use num_beams=4 and max_length=256.")
print("Done!")
