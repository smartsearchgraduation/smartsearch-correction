#!/usr/bin/env python3
"""
ByT5-Large v3.1 Fine-tuning -- Google Colab Script
====================================================
Copy each section (delimited by # === CELL N ===) into a separate Colab cell.

ByT5-Large (1.2B params) is a byte-level seq2seq model -- no tokenizer vocab,
every UTF-8 byte is a token.

v3.1 CHANGES over v3.0:
  - FIX: `total_memory` (not `total_mem`) - previous version crashed on GPU detect.
  - FIX: Conditional torch_compile (+20-30% throughput when CUDA is available).
  - NEW: group_by_length=True for better pad efficiency.
  - NEW: Overfitting guard - tracks (train - eval) loss gap; stops early if it
         widens for N consecutive evals.
  - NEW: Per-category accuracy callback during eval (uses a small fixed probe set).
  - NEW: More detailed Colab console logging (per-step ETA, sample predictions,
         running best metrics, GPU mem/util snapshots).
  - NEW: Optional Cell 7B - 3D trajectory visualisation of train/eval loss
         vs steps (plotly).
  - TUNE: eval_steps 2000 -> 4000, eval_subset_size 5000 -> 2000 for faster
          iteration without losing signal.
  - TUNE: save_total_limit 5 -> 3 (saves disk / Drive quota).
  - TUNE: Increased num_beams for eval from 4 -> 5 (small quality bump).

PRE-REQUISITE:
  Run download_byt5_to_drive.ipynb FIRST to cache the model on Drive.
"""

# =====================================================================
# CELL 1: SETUP & VERIFY MODEL CACHE
# =====================================================================
# !pip install -q transformers==4.44.2 datasets==2.21.0 accelerate==0.33.0 \
#               sentencepiece==0.2.0 evaluate==0.4.3 python-Levenshtein==0.25.1 \
#               plotly==5.24.1

import os, sys, time, json, gc, math, random
from pathlib import Path

import torch

# Mount Google Drive (Colab only)
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive")
except Exception:
    DRIVE_ROOT = Path(os.environ.get("DRIVE_ROOT", "/tmp/drive"))

# Paths
PROJECT_ROOT   = DRIVE_ROOT / "Grad" / "Correction" / "fine_tune" / "BYT5-T5 Large v3"
DATA_DIR       = PROJECT_ROOT / "data"
COLLAB_DIR     = PROJECT_ROOT / "collab" / "BYT5-Large-V3"
OUTPUT_DIR     = PROJECT_ROOT / "output" / "byt5-large-v3"
MODEL_CACHE    = PROJECT_ROOT / "model_cache" / "byt5-large"

for p in [DATA_DIR, COLLAB_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("─" * 70)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR:     {DATA_DIR}")
print(f"OUTPUT_DIR:   {OUTPUT_DIR}")
print(f"MODEL_CACHE:  {MODEL_CACHE}")
print("─" * 70)

# Verify data files exist
TRAIN_FILE = DATA_DIR / "train_v3.jsonl"
EVAL_FILE  = DATA_DIR / "eval_v3.jsonl"
assert TRAIN_FILE.exists(), f"Missing: {TRAIN_FILE}"
assert EVAL_FILE.exists(),  f"Missing: {EVAL_FILE}"
print(f"✓ train_v3.jsonl:  {TRAIN_FILE.stat().st_size / 1e6:.1f} MB")
print(f"✓ eval_v3.jsonl:   {EVAL_FILE.stat().st_size / 1e6:.1f} MB")

# Verify model cache
if MODEL_CACHE.exists():
    files = list(MODEL_CACHE.iterdir())
    cache_size_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
    print(f"✓ Model cache:  {len(files)} files, {cache_size_gb:.2f} GB")
else:
    print("⚠ Model cache NOT FOUND — run download_byt5_to_drive.ipynb first!")


# =====================================================================
# CELL 2: CONFIGURATION
# =====================================================================
import torch

# v3.1 FIX: correct property is `total_memory`, not `total_mem`
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} | VRAM: {gpu_mem_gb:.1f} GB")
else:
    gpu_name = "cpu"
    gpu_mem_gb = 0
    print("⚠ No CUDA GPU available — training will be very slow.")

# GPU-adaptive batch size
if "A100" in gpu_name or gpu_mem_gb >= 35:
    BS, GRAD_ACCUM = 8, 8
    MIXED_PRECISION = "bf16"
    TORCH_COMPILE = True
    DEVICE_TIER = "A100"
elif gpu_mem_gb >= 22:                       # L4 / A10 / A40
    BS, GRAD_ACCUM = 6, 10
    MIXED_PRECISION = "bf16"
    TORCH_COMPILE = True
    DEVICE_TIER = "L4/A10"
elif gpu_mem_gb >= 14:                       # T4 16G / V100 16G / RTX 5070Ti 16G
    BS, GRAD_ACCUM = 4, 16
    MIXED_PRECISION = "fp16"
    TORCH_COMPILE = False                     # fp16 + compile is flaky on T4
    DEVICE_TIER = "T4/V100/5070Ti"
else:
    BS, GRAD_ACCUM = 2, 32
    MIXED_PRECISION = "fp32"
    TORCH_COMPILE = False
    DEVICE_TIER = "small/cpu"

CONFIG = {
    # Model
    "model_name_or_path":    str(MODEL_CACHE),
    "max_input_length":      128,
    "max_target_length":     128,

    # Training
    "num_train_epochs":      3,
    "per_device_train_batch_size":  BS,
    "per_device_eval_batch_size":   max(2, BS // 2),
    "gradient_accumulation_steps":  GRAD_ACCUM,
    "learning_rate":         1e-5,
    "warmup_ratio":          0.10,
    "weight_decay":          0.01,       # v3.1: mild weight decay for regularisation
    "label_smoothing_factor": 0.1,
    "lr_scheduler_type":     "cosine",
    "optim":                 "adafactor",
    "mixed_precision":       MIXED_PRECISION,
    "torch_compile":         TORCH_COMPILE,   # v3.1 NEW
    "group_by_length":       True,            # v3.1 NEW
    "gradient_checkpointing": True,

    # Eval / save cadence (v3.1 tuned for shorter total runtime)
    "eval_steps":            4000,            # was 2000
    "save_steps":            4000,            # was 2000
    "save_total_limit":      3,               # was 5
    "logging_steps":         100,
    "eval_subset_size":      2000,            # was 5000 — fast enough, still signal
    "generation_num_beams":  5,               # was 4

    # Early stopping
    "early_stopping_patience":    3,
    "early_stopping_threshold":   1e-4,

    # Overfit guard (v3.1 NEW)
    "overfit_guard_patience":     3,          # consecutive evals with widening gap
    "overfit_guard_threshold":    0.05,       # min train-eval loss gap in absolute terms

    # Output
    "output_dir":            str(OUTPUT_DIR),
    "report_to":             ["none"],
    "seed":                  42,
}

print(f"\nDevice tier:       {DEVICE_TIER}")
print(f"Batch size:        {BS} (effective: {BS * GRAD_ACCUM})")
print(f"Mixed precision:   {MIXED_PRECISION}")
print(f"torch.compile:     {TORCH_COMPILE}")
print(f"Epochs:            {CONFIG['num_train_epochs']}")
print(f"LR:                {CONFIG['learning_rate']} (warmup {CONFIG['warmup_ratio']})")
print(f"Eval cadence:      every {CONFIG['eval_steps']} steps, subset {CONFIG['eval_subset_size']}")
print(f"Beam search:       num_beams={CONFIG['generation_num_beams']}")
print(f"Early stop:        patience={CONFIG['early_stopping_patience']}")
print(f"Overfit guard:     patience={CONFIG['overfit_guard_patience']}  gap>={CONFIG['overfit_guard_threshold']}")


# =====================================================================
# CELL 3: LOAD MODEL & TOKENIZER
# =====================================================================
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

print(f"Loading ByT5-Large from {CONFIG['model_name_or_path']}...")
t0 = time.time()
tokenizer = ByT5Tokenizer.from_pretrained(CONFIG["model_name_or_path"])
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name_or_path"])
print(f"Loaded in {time.time() - t0:.1f}s")
print(f"Params:    {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

if CONFIG["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")

if torch.cuda.is_available():
    model = model.cuda()

# Quick inference sanity check (pre-train)
model.eval()
with torch.no_grad():
    dummy_in = tokenizer("correct: corsiar keybord", return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        dummy_in = {k: v.cuda() for k, v in dummy_in.items()}
    gen = model.generate(dummy_in["input_ids"], max_new_tokens=50, num_beams=4)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"Pre-train sample: 'correct: corsiar keybord' -> '{out}'")


# =====================================================================
# CELL 4: LOAD & TOKENIZE DATA
# =====================================================================
from datasets import load_dataset

t0 = time.time()
raw = load_dataset("json", data_files={
    "train": str(TRAIN_FILE),
    "validation": str(EVAL_FILE),
})
print(f"Raw data loaded in {time.time() - t0:.1f}s")
print(f"  train:      {len(raw['train']):,}")
print(f"  validation: {len(raw['validation']):,}")

# Category distribution snapshot (helps sanity-check rebuild)
from collections import Counter
cat_train = Counter(raw["train"]["category"])
print("\nTop categories in train split:")
for c, n in cat_train.most_common(10):
    print(f"  {c:<28s} {n:>7,}")

def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=CONFIG["max_input_length"],
        padding=False,
        truncation=True,
    )
    # transformers >= 4.22: as_target_tokenizer() deprecated; use text_target=
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=CONFIG["max_target_length"],
        padding=False,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("\nTokenizing train split...")
t0 = time.time()
tokenized_train = raw["train"].map(
    tokenize_fn, batched=True,
    remove_columns=raw["train"].column_names,
    desc="tokenize-train",
    num_proc=2,
)
print(f"  train tokenized in {time.time() - t0:.1f}s")

print("Tokenizing eval split...")
t0 = time.time()
tokenized_eval = raw["validation"].map(
    tokenize_fn, batched=True,
    remove_columns=raw["validation"].column_names,
    desc="tokenize-eval",
    num_proc=2,
)
print(f"  eval  tokenized in {time.time() - t0:.1f}s")

# Eval subset (v3.1: smaller subset, faster evals during training)
if CONFIG["eval_subset_size"] and len(tokenized_eval) > CONFIG["eval_subset_size"]:
    tokenized_eval_sub = tokenized_eval.shuffle(seed=CONFIG["seed"]).select(
        range(CONFIG["eval_subset_size"])
    )
    print(f"  Using eval subset of {len(tokenized_eval_sub):,} for training-loop evals")
else:
    tokenized_eval_sub = tokenized_eval


# =====================================================================
# CELL 5: METRICS
# =====================================================================
try:
    import Levenshtein as _Lev
    def _levenshtein(a: str, b: str) -> int:
        return _Lev.distance(a, b)
except Exception:
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2): s1, s2 = s2, s1
        if not s2: return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, 1):
            cur = [i]
            for j, c2 in enumerate(s2, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c1 != c2)))
            prev = cur
        return prev[-1]

def _word_f1(pred_words: list, label_words: list) -> float:
    from collections import Counter as _C
    if not label_words or not pred_words:
        return 0.0
    pc, lc = _C(pred_words), _C(label_words)
    common = sum((pc & lc).values())
    if common == 0:
        return 0.0
    p = common / len(pred_words)
    r = common / len(label_words)
    return 2 * p * r / (p + r)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    import numpy as np
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.ndim == 3:
        preds = preds.argmax(-1)
    preds = preds.tolist()
    labels = [[t for t in l if t != -100] for l in labels.tolist()]
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact, f1_sum, cer_sum = 0, 0.0, 0.0
    for p, l in zip(decoded_preds, decoded_labels):
        p, l = p.strip(), l.strip()
        if p == l:
            exact += 1
        f1_sum  += _word_f1(p.split(), l.split())
        cer_sum += _levenshtein(p, l) / max(1, len(l))
    n = max(1, len(decoded_preds))
    return {
        "sentence_accuracy": round(exact / n, 4),
        "word_f1":            round(f1_sum / n, 4),
        "cer":                round(cer_sum / n, 4),
    }


# =====================================================================
# CELL 6: TRAINING
# =====================================================================
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback,
)

# Probe set for per-category accuracy (small but representative)
PROBE_CASES = [
    # identity_brand
    ("razer",                       "razer",                        "identity_brand"),
    ("asus",                        "asus",                         "identity_brand"),
    ("logitech",                    "logitech",                     "identity_brand"),
    ("corsair",                     "corsair",                      "identity_brand"),
    # ecom_query_typo
    ("corsiar keybord",             "corsair keyboard",             "ecom_query_typo"),
    ("razr mouse",                  "razer mouse",                  "ecom_query_typo"),
    ("logiteck mx keys",            "logitech mx keys",             "ecom_query_typo"),
    ("nvidai rtx 4090",             "nvidia rtx 4090",              "ecom_query_typo"),
    # phonetic
    ("blutooth headphones",         "bluetooth headphones",         "phonetic_typo"),
    ("wereless mouse",              "wireless mouse",               "phonetic_typo"),
    # space errors
    ("air pods pro",                "airpods pro",                  "space_error"),
    ("iphonecase",                  "iphone case",                  "space_error"),
    # price
    ("iphone 15 under 500 dolars",  "iphone 15 under 500 dollars",  "price_typo"),
    ("samsung tv under $1000",      "samsung tv under $1000",       "price_identity"),
    # units
    ("ssd 1tb",                     "ssd 1TB",                      "unit_typo"),
    ("monitor 144 hz",              "monitor 144 Hz",               "unit_typo"),
    # brand-category mismatch
    ("nvidia tuf chair",            "asus tuf gaming chair",        "brand_category_mismatch"),
    # everyday english
    ("teh best laptop",             "the best laptop",              "everyday_english_typo"),
    # identity_tech
    ("rtx 4090",                    "rtx 4090",                     "identity_tech"),
    ("ddr5",                        "ddr5",                         "identity_tech"),
]


def run_probe(model_ref, tokenizer_ref, cases=PROBE_CASES, num_beams=4) -> dict:
    """Run a tiny decoding probe to surface per-category accuracy."""
    from collections import defaultdict
    buckets = defaultdict(lambda: [0, 0])
    model_ref.eval()
    with torch.no_grad():
        for inp, tgt, cat in cases:
            enc = tokenizer_ref(f"correct: {inp}", return_tensors="pt", truncation=True, max_length=128)
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            out = model_ref.generate(enc["input_ids"], max_new_tokens=64,
                                     num_beams=num_beams, early_stopping=True)
            pred = tokenizer_ref.decode(out[0], skip_special_tokens=True).strip()
            ok = int(pred == tgt)
            buckets[cat][0] += ok
            buckets[cat][1] += 1
    return {c: (n, d, n / d if d else 0.0) for c, (n, d) in buckets.items()}


class TrainingMonitor(TrainerCallback):
    """
    v3.1 rich logger:
      - per-step ETA, LR, loss
      - running best sentence_accuracy
      - probe-set per-category accuracy at each eval
      - overfitting guard: stops if eval-train gap widens for N evals
    """
    def __init__(self):
        self.best_acc = 0.0
        self.best_step = 0
        self.widen_streak = 0
        self.last_gap = None
        self.train_loss_ema = None
        self.t_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_start = time.time()
        print("=" * 70)
        print("  TRAINING BEGIN")
        print(f"  Total steps:       {state.max_steps}")
        print(f"  Eval every:        {args.eval_steps}")
        print(f"  Num beams (eval):  {args.generation_num_beams}")
        print("=" * 70)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is not None:
            self.train_loss_ema = loss if self.train_loss_ema is None \
                else 0.9 * self.train_loss_ema + 0.1 * loss
        lr = logs.get("learning_rate")
        elapsed = time.time() - self.t_start if self.t_start else 0
        progress = state.global_step / max(1, state.max_steps)
        eta = elapsed * (1 - progress) / max(progress, 1e-9)
        if state.global_step % max(1, args.logging_steps) == 0 and loss is not None:
            print(f"  step {state.global_step:>6d}/{state.max_steps}  "
                  f"loss={loss:.4f}  ema={self.train_loss_ema:.4f}  "
                  f"lr={lr:.2e}  "
                  f"elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        acc = metrics.get("eval_sentence_accuracy", 0.0)
        f1  = metrics.get("eval_word_f1", 0.0)
        cer = metrics.get("eval_cer", 0.0)
        eval_loss  = metrics.get("eval_loss")
        train_loss = self.train_loss_ema
        gap = (eval_loss - train_loss) if (eval_loss is not None and train_loss is not None) else None
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_step = state.global_step
        print("-" * 70)
        print(f"  EVAL @ step {state.global_step}")
        if gap is not None:
            print(f"    loss={eval_loss:.4f}  train_ema={train_loss:.4f}  gap={gap:.4f}")
        else:
            print(f"    eval_loss={eval_loss}")
        print(f"    sentence_acc={acc:.4f}  (best={self.best_acc:.4f} @ step {self.best_step})")
        print(f"    word_f1={f1:.4f}  cer={cer:.4f}")
        # Overfit guard
        if gap is not None:
            if self.last_gap is not None and gap - self.last_gap > 0 and gap > CONFIG["overfit_guard_threshold"]:
                self.widen_streak += 1
            else:
                self.widen_streak = 0
            self.last_gap = gap
            if self.widen_streak >= CONFIG["overfit_guard_patience"]:
                print(f"    ! Overfit guard: eval-train loss gap widened {self.widen_streak}x in a row -> stopping.")
                control.should_training_stop = True
        # Probe (lightweight -- doesn't touch the main eval dataset)
        try:
            probe = run_probe(kwargs.get("model"), tokenizer)
            print(f"    probe per-category accuracy:")
            for c, (n, d, r) in sorted(probe.items()):
                print(f"      {c:<26s} {n}/{d}  {100*r:5.1f}%")
        except Exception as e:
            print(f"    probe failed: {e}")
        # GPU snapshot
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_resvd = torch.cuda.memory_reserved() / 1e9
            print(f"    GPU mem: alloc={mem_alloc:.1f}G  reserved={mem_resvd:.1f}G")
        print("-" * 70)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.t_start if self.t_start else 0
        print("=" * 70)
        print(f"  TRAINING END  (elapsed {elapsed/60:.1f} min)")
        print(f"  Best sentence_acc: {self.best_acc:.4f} @ step {self.best_step}")
        print("=" * 70)


data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding="longest", pad_to_multiple_of=8,
)

# Compute total steps for cosine schedule
n_train = len(tokenized_train)
effective_bs = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
steps_per_epoch = math.ceil(n_train / effective_bs)
total_steps = steps_per_epoch * CONFIG["num_train_epochs"]
print(f"Steps per epoch: {steps_per_epoch:,}  |  Total steps: {total_steps:,}")

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_train_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    label_smoothing_factor=CONFIG["label_smoothing_factor"],
    optim=CONFIG["optim"],
    fp16=CONFIG["mixed_precision"] == "fp16",
    bf16=CONFIG["mixed_precision"] == "bf16",
    gradient_checkpointing=CONFIG["gradient_checkpointing"],
    # v3.1 NEW throughput options
    torch_compile=CONFIG["torch_compile"],
    group_by_length=CONFIG["group_by_length"],
    # Eval / save
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="sentence_accuracy",
    greater_is_better=True,
    # Logging
    logging_steps=CONFIG["logging_steps"],
    logging_dir=str(OUTPUT_DIR / "logs"),
    report_to=CONFIG["report_to"],
    # Generation
    predict_with_generate=True,
    generation_max_length=CONFIG["max_target_length"],
    generation_num_beams=CONFIG["generation_num_beams"],
    # Misc
    seed=CONFIG["seed"],
    dataloader_num_workers=2,
    remove_unused_columns=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval_sub,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG["early_stopping_patience"],
            early_stopping_threshold=CONFIG["early_stopping_threshold"],
        ),
        TrainingMonitor(),
    ],
)

print("\n" + "=" * 70)
print(" STARTING TRAINING")
print("=" * 70)
train_result = trainer.train()

# Save final model + tokenizer
trainer.save_model(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
print(f"\n✓ Final model saved to: {OUTPUT_DIR / 'final'}")


# =====================================================================
# CELL 7: TRAINING CURVES (2D)
# =====================================================================
import matplotlib.pyplot as plt
from pathlib import Path as _P

log_history = trainer.state.log_history
train_steps, train_losses = [], []
eval_steps, eval_losses, eval_accs = [], [], []
for row in log_history:
    if "loss" in row and "eval_loss" not in row:
        train_steps.append(row.get("step", 0)); train_losses.append(row["loss"])
    if "eval_loss" in row:
        eval_steps.append(row.get("step", 0))
        eval_losses.append(row["eval_loss"])
        eval_accs.append(row.get("eval_sentence_accuracy", 0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_steps, train_losses, label="train", alpha=0.6)
axes[0].plot(eval_steps, eval_losses, label="eval", marker="o")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(eval_steps, eval_accs, marker="o", color="green")
axes[1].set_xlabel("step"); axes[1].set_ylabel("sentence accuracy")
axes[1].set_title("Eval sentence accuracy"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
png_out = _P(CONFIG["output_dir"]) / "training_curves.png"
plt.savefig(png_out, dpi=120, bbox_inches="tight")
plt.show()
print(f"✓ Saved 2D curves: {png_out}")


# =====================================================================
# CELL 7B (OPTIONAL): 3D TRAINING TRAJECTORY (plotly)
# =====================================================================
# If plotly is available, renders a 3D path over (step, loss, sentence_acc).
try:
    import plotly.graph_objects as go
    if eval_steps:
        fig3d = go.Figure(data=[go.Scatter3d(
            x=eval_steps, y=eval_losses, z=eval_accs,
            mode="lines+markers",
            marker=dict(size=5, color=eval_accs, colorscale="Viridis", showscale=True,
                        colorbar=dict(title="sent_acc")),
            line=dict(width=4, color="darkblue"),
        )])
        fig3d.update_layout(
            title="Training trajectory (eval)",
            scene=dict(
                xaxis_title="step",
                yaxis_title="eval_loss",
                zaxis_title="sentence_accuracy",
            ),
            height=600,
        )
        html_out = _P(CONFIG["output_dir"]) / "training_trajectory_3d.html"
        fig3d.write_html(str(html_out))
        print(f"✓ Saved 3D trajectory: {html_out}")
        fig3d.show()
except ImportError:
    print("plotly not installed -- skip 3D trajectory.")


# =====================================================================
# CELL 8: COMPREHENSIVE EVALUATION
# =====================================================================
from collections import defaultdict

TEST_CASES = [
    # identity
    ("razer",                           "razer",                            "identity"),
    ("corsair",                         "corsair",                          "identity"),
    ("logitech",                        "logitech",                         "identity"),
    ("asus tuf gaming",                 "asus tuf gaming",                  "identity"),
    ("iphone 15 pro max",               "iphone 15 pro max",                "identity"),
    ("rtx 4090",                        "rtx 4090",                         "identity"),
    ("ddr5 ram 32gb",                   "ddr5 ram 32gb",                    "identity"),
    # brand typo
    ("corsiar keybord",                 "corsair keyboard",                 "brand_typo"),
    ("razr mouse",                      "razer mouse",                      "brand_typo"),
    ("logiteck mx master",              "logitech mx master",               "brand_typo"),
    ("nvidai rtx 4090",                 "nvidia rtx 4090",                  "brand_typo"),
    ("samsng galaxy s24",               "samsung galaxy s24",               "brand_typo"),
    # product typo
    ("best wreless headphnoes",         "best wireless headphones",         "product_typo"),
    ("cheap keybord under 50",          "cheap keyboard under 50",          "product_typo"),
    ("blutooth speker",                 "bluetooth speaker",                "product_typo"),
    ("mechnical gaming keybord",        "mechanical gaming keyboard",       "product_typo"),
    # compound typo
    ("corsiar blutooth mehanical keybord", "corsair bluetooth mechanical keyboard", "compound"),
    ("razr wreless mechnical mouse",       "razer wireless mechanical mouse",        "compound"),
    # space error
    ("air pods pro",                    "airpods pro",                      "space_error"),
    ("iphonecase",                      "iphone case",                      "space_error"),
    ("keyboardmouse combo",             "keyboard mouse combo",             "space_error"),
    # phonetic
    ("blutooth mowse",                  "bluetooth mouse",                  "phonetic"),
    ("wereless hedphones",              "wireless headphones",              "phonetic"),
    ("munitor 144 hz",                  "monitor 144 Hz",                   "phonetic"),
    # price
    ("iphone 15 under 500 dolars",      "iphone 15 under 500 dollars",      "price_typo"),
    ("samsung tv under $1000",          "samsung tv under $1000",           "price_identity"),
    ("gaming laptop between $800 and $1500", "gaming laptop between $800 and $1500", "price_identity"),
    ("coffe maker for $50",             "coffee maker for $50",             "price_typo"),
    # units
    ("ssd 1tb",                         "ssd 1TB",                          "unit_typo"),
    ("monitor 144hz",                   "monitor 144 Hz",                   "unit_typo"),
    ("power bank 20000mah",             "power bank 20000 mAh",             "unit_typo"),
    ("1080p webcam",                    "1080p webcam",                     "unit_identity"),
    ("4k monitor",                      "4k monitor",                       "unit_identity"),
    # brand-category mismatch
    ("nvidia tuf chair",                "asus tuf gaming chair",            "brand_category_mismatch"),
    ("intel ryzen 7 laptop",            "amd ryzen 7 laptop",               "brand_category_mismatch"),
    ("samsung airpods case",            "apple airpods case",               "brand_category_mismatch"),
    # everyday english
    ("teh best laptop",                 "the best laptop",                  "everyday_english"),
    ("recieve shipping",                "receive shipping",                 "everyday_english"),
    ("definately buy",                  "definitely buy",                   "everyday_english"),
    # tricky identity (should NOT be corrected)
    ("razer mouse",                     "razer mouse",                      "tricky_identity"),
    ("asus rog strix",                  "asus rog strix",                   "tricky_identity"),
    ("birkenstock sandals",             "birkenstock sandals",              "tricky_identity"),
]


def correct(query: str, num_beams: int = 5) -> str:
    model.eval()
    enc = tokenizer(f"correct: {query}", return_tensors="pt", truncation=True, max_length=128)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_new_tokens=128,
            num_beams=num_beams,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


results = defaultdict(lambda: {"correct": 0, "total": 0, "cases": []})
print("\n" + "=" * 70)
print("  COMPREHENSIVE EVAL (num_beams=5)")
print("=" * 70)
for inp, tgt, cat in TEST_CASES:
    pred = correct(inp, num_beams=5)
    ok = pred.lower().strip() == tgt.lower().strip()
    results[cat]["correct"] += int(ok)
    results[cat]["total"] += 1
    results[cat]["cases"].append((inp, tgt, pred, ok))
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] [{cat:<26s}] '{inp}' -> '{pred}' (expected '{tgt}')")

print("\n" + "-" * 70)
print("  SUMMARY BY CATEGORY")
print("-" * 70)
all_correct = all_total = 0
for cat, r in results.items():
    pct = 100 * r["correct"] / r["total"] if r["total"] else 0
    print(f"  {cat:<28s}  {r['correct']:>3}/{r['total']:<3}  ({pct:5.1f}%)")
    all_correct += r["correct"]; all_total += r["total"]
print("-" * 70)
pct_all = 100 * all_correct / all_total if all_total else 0
print(f"  OVERALL: {all_correct}/{all_total} ({pct_all:.1f}%)")


# =====================================================================
# CELL 9: EXPORT
# =====================================================================
final_dir = _P(CONFIG["output_dir"]) / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

meta = {
    "version": "v3.1",
    "model": "ByT5-Large",
    "train_examples": len(tokenized_train),
    "eval_examples": len(tokenized_eval),
    "epochs": CONFIG["num_train_epochs"],
    "effective_batch_size": CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"],
    "learning_rate": CONFIG["learning_rate"],
    "generation_num_beams": CONFIG["generation_num_beams"],
    "best_sentence_acc": getattr(trainer.state, "best_metric", None),
    "device_tier": DEVICE_TIER,
    "mixed_precision": CONFIG["mixed_precision"],
    "torch_compile": CONFIG["torch_compile"],
    "group_by_length": CONFIG["group_by_length"],
}
with open(final_dir / "training_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✓ Model + meta saved to: {final_dir}")
print("\nTo deploy, use num_beams=5 and max_length=128.")
