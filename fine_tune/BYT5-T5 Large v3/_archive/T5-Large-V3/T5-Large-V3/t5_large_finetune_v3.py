#!/usr/bin/env python3
"""
T5-Large v3 Fine-tuning -- Google Colab Script
================================================
Copy each section (delimited by # === CELL N ===) into a separate Colab cell.

CRITICAL FIXES OVER v2.1:
  1. LR 3e-5        (v2.1 used 5e-4 -> 16x too high, caused noisy gradients)
  2. Adafactor       (designed for T5, adapts per-parameter LR, saves memory)
  3. Cosine schedule (v2.1 used linear -> premature LR decay)
  4. Warmup 10%      (v2.1 used 6% -> unstable first steps)
  5. num_beams=4     (v2.1 eval used greedy -> underestimated real accuracy)
  6. EarlyStopping   (v2.1 had none -> risk of overfitting on 5 epochs)
  7. No-harm metric  (overcorrection is worse than missing a fix)
  8. 400K examples   (v2.1 had 205K with 40% identity padding)

Expected: 75-82% exact match, 97-99% no-harm rate
"""

# =====================================================================
# CELL 1: SETUP & INSTALLATION
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
# Upload 'BYT5-T5 Large v3' folder to: Drive > Grad/Correction/fine_tune/
DRIVE_BASE = Path('/content/drive/MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3')
DATA_DIR   = DRIVE_BASE / 'data'
OUTPUT_DIR = DRIVE_BASE / 'output' / 'T5-Large-v3-final'
LOG_DIR    = OUTPUT_DIR / 'logs'

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

MODEL_NAME = "google-t5/t5-large"

gpu_name = "CPU"
gpu_mem_gb = 0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

if gpu_mem_gb >= 35:       # A100-40GB / A100-80GB
    BATCH_SIZE = 16; GRAD_ACCUM = 4; FP16 = False; BF16 = True; EVAL_BATCH = 32
elif gpu_mem_gb >= 14:     # T4-16GB / L4-24GB
    BATCH_SIZE = 8; GRAD_ACCUM = 8; FP16 = True; BF16 = False; EVAL_BATCH = 16
else:
    BATCH_SIZE = 4; GRAD_ACCUM = 16; FP16 = False; BF16 = False; EVAL_BATCH = 8

# ---------------------------------------------------------------
# WHY THESE HYPERPARAMETERS (v2.1 mistakes annotated)
# ---------------------------------------------------------------
# learning_rate: 3e-5
#   v2.1 used 5e-4 (16x higher!). T5 paper recommends 1e-5 to 5e-5
#   for fine-tuning. 3e-5 balances convergence speed and stability.
#
# optimizer: Adafactor
#   T5 was pre-trained with Adafactor. Saves ~30% optimizer memory.
#
# lr_scheduler: cosine
#   Linear (v2.1 default) drops LR too aggressively mid-training.
#   Cosine keeps higher LR through middle epochs.
#
# warmup_ratio: 0.10 (v2.1 used 0.06)
# label_smoothing: 0.1 (v2.1 used 0.05)
# generation_num_beams: 4 (v2.1 eval used 1 = greedy)
# num_epochs: 3 + EarlyStopping (v2.1 used 5 with no stopping)
# ---------------------------------------------------------------

CONFIG = {
    "model_name":       MODEL_NAME,
    "max_input_length":  64,
    "max_target_length": 64,
    "num_epochs":        3,
    "batch_size":        BATCH_SIZE,
    "eval_batch_size":   EVAL_BATCH,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "learning_rate":     3e-5,
    "warmup_ratio":      0.10,
    "weight_decay":      0.0,        # Adafactor has its own decay; external WD conflicts
    "label_smoothing":   0.1,
    "max_grad_norm":     1.0,
    "eval_steps":        2000,       # was 500 -> 84 evals was WAY too many for 902K data
    "save_steps":        2000,       # sync with eval_steps
    "logging_steps":     100,
    "save_total_limit":  5,          # >= patience+1 so best checkpoint survives
    "early_stopping_patience": 3,
    "eval_subset_size":  5000,       # subsample eval during training (full eval in Cell 8)
    "generation_num_beams": 4,
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
print(f"Optimizer: Adafactor (no external weight_decay -- Adafactor handles its own)")
print(f"Epochs: {CONFIG['num_epochs']} + EarlyStopping(patience={CONFIG['early_stopping_patience']})")
print(f"Eval: beam={CONFIG['generation_num_beams']}, every {CONFIG['eval_steps']} steps, subset={CONFIG.get('eval_subset_size', 'full')}")
print(f"Est. steps/epoch: ~{est_steps_per_epoch:,}, total: ~{est_total:,}")


# =====================================================================
# CELL 3: LOAD MODEL & TOKENIZER
# =====================================================================

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
print(f"  Loaded in {time.time() - t0:.0f}s")

if CONFIG['gradient_checkpointing']:
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing: ON (saves ~40% memory)")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,} ({total_params/1e6:.0f}M)")
if torch.cuda.is_available():
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print(f"\n[3/3] Sanity check...")
dummy_in = tokenizer("correct: test query", return_tensors="pt")
dummy_tgt = tokenizer("test query", return_tensors="pt")
with torch.no_grad():
    out = model(input_ids=dummy_in["input_ids"], labels=dummy_tgt["input_ids"])
print(f"  Forward OK (loss={out.loss.item():.4f})")
with torch.no_grad():
    gen = model.generate(dummy_in["input_ids"], max_new_tokens=20, num_beams=4)
print(f"  Generate: 'correct: test query' -> '{tokenizer.decode(gen[0], skip_special_tokens=True)}'")
print("\nModel ready.")


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

cats = Counter(ex['category'] for ex in dataset['train'])
print(f"\nCategories ({len(cats)}):")
for cat, cnt in cats.most_common(15):
    pct = cnt / len(dataset['train']) * 100
    print(f"  {cat:<30s} {cnt:>7,} ({pct:5.1f}%)")

id_count = sum(1 for ex in dataset['train']
               if ex['input_text'].replace('correct: ', '', 1) == ex['target_text'])
print(f"\nIdentity pairs: {id_count:,} / {len(dataset['train']):,} "
      f"({100 * id_count / len(dataset['train']):.1f}%)")

def tokenize_fn(examples):
    inputs = tokenizer(examples['input_text'], max_length=CONFIG['max_input_length'],
                       truncation=True, padding=False)
    targets = tokenizer(text_target=examples['target_text'], max_length=CONFIG['max_target_length'],
                        truncation=True, padding=False)
    inputs['labels'] = targets['input_ids']
    return inputs

print("\nTokenizing...")
tokenized = dataset.map(tokenize_fn, batched=True, batch_size=2000,
    remove_columns=['input_text', 'target_text', 'category'], desc="Tokenizing", num_proc=2)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
    label_pad_token_id=-100, pad_to_multiple_of=8)

trunc_count = sum(1 for ex in tokenized['train'] if len(ex['input_ids']) >= CONFIG['max_input_length'])
print(f"  Truncated: {trunc_count:,} / {len(tokenized['train']):,} ({100*trunc_count/len(tokenized['train']):.2f}%)")

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

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
import time as _time, gc

RESUME = False

class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self._t0 = None; self._best_acc = 0; self._best_step = 0
    def on_train_begin(self, args, state, control, **kw):
        self._t0 = _time.time()
        print(f"\n{'='*70}\n  TRAINING STARTED -- T5-Large v3")
        print(f"  Steps: {state.max_steps} | Eval every {args.eval_steps} | Beams: {args.generation_num_beams}")
        print(f"  LR: {args.learning_rate} | Scheduler: {args.lr_scheduler_type}")
        print(f"{'='*70}")
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if not metrics: return
        acc = metrics.get('eval_sentence_accuracy', 0)
        loss = metrics.get('eval_loss', 0)
        cer = metrics.get('eval_cer', 1)
        improved = ""
        if acc > self._best_acc: self._best_acc = acc; self._best_step = state.global_step; improved = " << NEW BEST"
        elapsed = (_time.time() - self._t0) / 60 if self._t0 else 0
        print(f"\n  [Step {state.global_step:>5d}] loss={loss:.4f} | acc={acc:.4f} | cer={cer:.4f}{improved} ({elapsed:.0f}min)")
    def on_save(self, *a, **kw): gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()
    def on_train_end(self, args, state, control, **kw):
        t = (_time.time() - self._t0) / 60 if self._t0 else 0
        print(f"\n{'='*70}\n  TRAINING COMPLETE: {t:.0f}min ({t/60:.1f}h) | Best: {self._best_acc:.4f} @ step {self._best_step}\n{'='*70}")

eff_batch = CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']
steps_per_epoch = len(tokenized['train']) // eff_batch
total_steps = steps_per_epoch * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
print(f"Steps/epoch: {steps_per_epoch:,} | Total: {total_steps:,} | Warmup: {warmup_steps:,}")

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
    eval_strategy="steps", eval_steps=CONFIG['eval_steps'],
    save_strategy="steps", save_steps=CONFIG['save_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="sentence_accuracy", greater_is_better=True,
    save_total_limit=CONFIG['save_total_limit'],
    predict_with_generate=True,
    generation_max_length=CONFIG['max_target_length'],
    generation_num_beams=CONFIG['generation_num_beams'],
    logging_dir=str(LOG_DIR), logging_steps=CONFIG['logging_steps'],
    report_to=['tensorboard'],
    dataloader_pin_memory=True, dataloader_num_workers=2,
    gradient_checkpointing=CONFIG['gradient_checkpointing'],
    seed=CONFIG['seed'], torch_compile=False,
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=eval_subset,          # subset for fast training-time eval
    data_collator=data_collator, processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[TrainingMonitor(), EarlyStoppingCallback(
        early_stopping_patience=CONFIG['early_stopping_patience'],
        early_stopping_threshold=0.001)],
)

resume_ckpt = None
if RESUME:
    import glob as _g
    ckpts = sorted(_g.glob(str(OUTPUT_DIR / 'checkpoint-*')))
    if ckpts: resume_ckpt = ckpts[-1]; print(f"Resuming from: {resume_ckpt}")

print(f"\n{'='*70}\n  STARTING T5-LARGE v3 FINE-TUNING\n  LR={CONFIG['learning_rate']} | Adafactor | cosine | beams={CONFIG['generation_num_beams']}\n{'='*70}")
result = trainer.train(resume_from_checkpoint=resume_ckpt)

best_dir = OUTPUT_DIR / 'best_model'
trainer.save_model(str(best_dir))
tokenizer.save_pretrained(str(best_dir))
with open(best_dir / 'v3_training_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
print(f"\nBest model saved: {best_dir}\nFinal loss: {result.training_loss:.4f}")


# =====================================================================
# CELL 7: TRAINING CURVES
# =====================================================================

import json, glob
import matplotlib.pyplot as plt

state_files = sorted(glob.glob(str(OUTPUT_DIR / 'checkpoint-*' / 'trainer_state.json')))
sf = OUTPUT_DIR / 'trainer_state.json'
if sf.exists(): state_files.append(str(sf))

if state_files:
    with open(state_files[-1]) as f: state = json.load(f)
    logs = state.get('log_history', [])
    train_s = [(l['step'], l['loss']) for l in logs if 'loss' in l and 'eval_loss' not in l]
    eval_l = [(l['step'], l['eval_loss']) for l in logs if 'eval_loss' in l]
    eval_a = [(l['step'], l['eval_sentence_accuracy']) for l in logs if 'eval_sentence_accuracy' in l]
    eval_c = [(l['step'], l['eval_cer']) for l in logs if 'eval_cer' in l]

    eval_f1 = [(l['step'], l['eval_word_f1']) for l in logs if 'eval_word_f1' in l]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    if train_s: axes[0].plot(*zip(*train_s), label='Train', alpha=0.6)
    if eval_l: axes[0].plot(*zip(*eval_l), label='Eval', linewidth=2, color='red')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if eval_a:
        s, a = zip(*eval_a); axes[1].plot(s, a, color='green', linewidth=2, marker='o', markersize=4)
        bi = int(np.argmax(a)); axes[1].annotate(f'Best: {a[bi]:.4f}', xy=(s[bi], a[bi]), fontsize=10, color='green')
    axes[1].set_title('Sentence Accuracy'); axes[1].grid(True, alpha=0.3)
    if eval_f1:
        s, f = zip(*eval_f1); axes[2].plot(s, f, color='blue', linewidth=2, marker='s', markersize=4)
    axes[2].set_title('Word F1'); axes[2].grid(True, alpha=0.3)
    if eval_c: axes[3].plot(*zip(*eval_c), color='orange', linewidth=2)
    axes[3].set_title('CER / Levenshtein (lower=better)'); axes[3].grid(True, alpha=0.3)
    plt.suptitle('T5-Large v3 Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(str(OUTPUT_DIR / 'training_curves_v3.png'), dpi=150); plt.show()
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
eval_model = AutoModelForSeq2SeqLM.from_pretrained(str(best_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_model = eval_model.to(device).eval()

def correct(query, num_beams=4):
    enc = eval_tokenizer(f"correct: {query}", return_tensors='pt', max_length=64, truncation=True)
    with torch.no_grad():
        out = eval_model.generate(enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device),
                                  max_length=64, num_beams=num_beams, length_penalty=1.0)
    return eval_tokenizer.decode(out[0], skip_special_tokens=True).strip()

TEST_CASES = [
    # Identity (must NOT change)
    ("wireless mouse","wireless mouse","identity"),("samsung galaxy","samsung galaxy","identity"),
    ("iphone 15 pro max","iphone 15 pro max","identity"),("running shoes","running shoes","identity"),
    ("keyboard","keyboard","identity"),("128gb ssd","128gb ssd","identity"),
    ("headphones","headphones","identity"),("corsair k70 rgb","corsair k70 rgb","identity"),
    ("nvidia rtx 4090","nvidia rtx 4090","identity"),("razer blade 16","razer blade 16","identity"),
    ("sport bra","sport bra","identity"),("yoga mat","yoga mat","identity"),
    ("usb c cable","usb c cable","identity"),("gaming chair","gaming chair","identity"),
    ("coffee maker","coffee maker","identity"),
    # Brand typos
    ("samsng galaxy","samsung galaxy","brand_typo"),("iphnoe 15","iphone 15","brand_typo"),
    ("logitec mouse","logitech mouse","brand_typo"),("corsiar k70","corsair k70","brand_typo"),
    ("nvidai rtx 4080","nvidia rtx 4080","brand_typo"),("assu rog laptop","asus rog laptop","brand_typo"),
    ("lenvo thinkpad","lenovo thinkpad","brand_typo"),("steelsires headset","steelseries headset","brand_typo"),
    ("razr blade","razer blade","brand_typo"),("mackbook pro","macbook pro","brand_typo"),
    # Product typos
    ("wireles mouse","wireless mouse","product_typo"),("keybord","keyboard","product_typo"),
    ("heaphones","headphones","product_typo"),("lapto computer","laptop computer","product_typo"),
    ("runnng shoes","running shoes","product_typo"),("moniter 27 inch","monitor 27 inch","product_typo"),
    ("bluethooth speaker","bluetooth speaker","product_typo"),("gameing mouse","gaming mouse","product_typo"),
    # Compound
    ("samsng galxy s24","samsung galaxy s24","compound"),("wireles keybord","wireless keyboard","compound"),
    ("nvidai geforc rtx","nvidia geforce rtx","compound"),("logitec wireles mouse","logitech wireless mouse","compound"),
    # Space errors
    ("air pods pro","airpods pro","space_error"),("head phones wireless","headphones wireless","space_error"),
    ("lap top gaming","laptop gaming","space_error"),("smart watch","smartwatch","space_error"),
    # Phonetic
    ("grafics card","graphics card","phonetic"),("processer intel","processor intel","phonetic"),
    ("baterry pack","battery pack","phonetic"),("dispaly 4k","display 4k","phonetic"),
]

print(f"\n{'INPUT':<35} {'EXPECTED':<35} {'PREDICTED':<35} RESULT")
print("-" * 110)

results_by_cat = {}
all_ok = all_total = 0
for inp, expected, cat in TEST_CASES:
    pred = correct(inp)
    ok = pred.lower().strip() == expected.lower().strip()
    if cat not in results_by_cat: results_by_cat[cat] = {"ok": 0, "total": 0, "fails": []}
    results_by_cat[cat]["total"] += 1; results_by_cat[cat]["ok"] += int(ok)
    if not ok: results_by_cat[cat]["fails"].append((inp, expected, pred))
    all_ok += int(ok); all_total += 1
    print(f"{inp:<35} {expected:<35} {pred:<35} {'OK' if ok else 'FAIL'}")

print(f"\n{'='*80}\n  TEST RESULTS SUMMARY\n{'='*80}")
for cat in ["identity","brand_typo","product_typo","compound","space_error","phonetic"]:
    if cat in results_by_cat:
        r = results_by_cat[cat]; pct = 100*r["ok"]/r["total"]
        print(f"  {cat:<20s} {r['ok']:>2d}/{r['total']:<2d} ({pct:5.1f}%)  [{'PASS' if pct>=80 else 'WARN' if pct>=60 else 'FAIL'}]")
print(f"  {'-'*45}\n  {'OVERALL':<20s} {all_ok:>2d}/{all_total:<2d} ({100*all_ok/all_total:5.1f}%)")

id_r = results_by_cat.get("identity", {"ok":0,"total":0})
if id_r["total"] > 0:
    print(f"\n  NO-HARM RATE: {id_r['ok']}/{id_r['total']} ({100*id_r['ok']/id_r['total']:.1f}%)")

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

from torch.utils.data import DataLoader

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
    # Sample up to 2000 for tractable beam-search evaluation
    import random as _rnd
    _rnd.seed(42)
    id_sample = _rnd.sample(identity_pairs, min(2000, len(identity_pairs)))
    id_ok = 0
    id_fails = []
    for inp, expected in id_sample:
        pred = correct(inp, num_beams=4)
        if pred.lower().strip() == expected.lower().strip():
            id_ok += 1
        elif len(id_fails) < 20:  # keep first 20 failures for inspection
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
for q in ["wireles mouse","samsng galaxy","keybord","nvidai rtx 4080","logitec mouse"]:
    times = [0]*3
    for i in range(3): t0=time.time(); correct(q,4); times[i]=(time.time()-t0)*1000
    print(f"    '{q}' -> '{correct(q)}' ({sorted(times)[1]:.0f}ms)")


# =====================================================================
# CELL 9: EXPORT
# =====================================================================

import shutil
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
export_dir = OUTPUT_DIR / f't5_large_v3_final_{ts}'
export_dir.mkdir(parents=True, exist_ok=True)
eval_model.save_pretrained(str(export_dir))
eval_tokenizer.save_pretrained(str(export_dir))

with open(export_dir / 'training_config.json', 'w') as f: json.dump(CONFIG, f, indent=2)
metadata = {
    "model": "google-t5/t5-large", "version": "v3", "date": datetime.now().isoformat(),
    "train_examples": train_count, "eval_examples": eval_count,
    "hyperparameters": {"lr": CONFIG['learning_rate'], "optimizer": "adafactor",
        "scheduler": "cosine", "warmup": CONFIG['warmup_ratio'],
        "smoothing": CONFIG['label_smoothing'], "beams": CONFIG['generation_num_beams']},
    "test_results": {"overall": f"{all_ok}/{all_total}",
        "no_harm": f"{id_r['ok']}/{id_r['total']}",
        "categories": {c: f"{r['ok']}/{r['total']}" for c,r in results_by_cat.items()}},
    "gpu": gpu_name,
    "no_harm_rate": f"{id_ok}/{len(id_sample)} ({nh_rate:.1f}%)" if identity_pairs else "N/A",
    "full_eval_metrics": {k: v for k, v in full_metrics.items() if k.startswith('test_')},
    "fixes_over_v2_1": ["LR 3e-5 (was 5e-4)","Adafactor (was AdamW)","cosine (was linear)",
        "10% warmup (was 6%)","beam=4 eval (was greedy)","902K data/18% identity (was 205K/40%)",
        "phonetic+compound+space typos (was QWERTY-only)","broad e-commerce (was electronics-only)",
        "EarlyStopping patience=3 (was none)","Levenshtein CER (was position-based)",
        "Word F1 (was zip-based token acc)","Eval subset 5K (was full 47K every 500 steps)",
        "Full eval-set no-harm rate (was 15 hardcoded only)"],
}
with open(export_dir / 'metadata.json', 'w') as f: json.dump(metadata, f, indent=2)
shutil.make_archive(str(export_dir), 'zip', export_dir)
print(f"Exported: {export_dir}\nUse num_beams=4 for best quality.\nDone!")
