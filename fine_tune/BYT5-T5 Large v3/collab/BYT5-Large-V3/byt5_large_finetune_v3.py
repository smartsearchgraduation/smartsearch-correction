#!/usr/bin/env python3
"""
ByT5-Large v3.3 Fine-tuning -- Google Colab Script (A100 40GB tuned)
=====================================================================
This is the canonical source for `byt5_large_finetune_v3.ipynb`. Each
# === CELL N === block corresponds to one notebook cell.

ByT5-Large (1.2B params) is a byte-level seq2seq model -- no tokenizer vocab,
every UTF-8 byte is a token.

v3.3 CHANGES over v3.2 (A100 40GB optimisations):
  - Epochs default 3 -> 2 (fits inside one Colab Pro A100 session)
  - dataloader_num_workers 4 -> 8, prefetch_factor 4 -> 8
  - tokenize_num_proc 4 -> 8 (A100 VMs have plenty of CPU)
  - New TRAIN_SUBSET_SIZE knob (None = use all, or cap to e.g. 1_000_000)
  - Log cadence tightened: eval/save every 4000 -> 3000 steps
  - Drops prepare_external_datasets / build_training_data runner deps --
    those now live in data/builder_tools/ (local only, NOT run in Colab)

WHAT YOU RUN IN COLAB (only these 2 notebooks, in order):
  1) download_byt5_to_drive.ipynb   -- caches google/byt5-large on Drive
  2) byt5_large_finetune_v3.ipynb   -- trains; resumable from checkpoints

PRE-REQUISITE ON DRIVE:
  /MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3/
    data_v3_7/train_v3_7.jsonl         (~347 MB, 2.66M pairs, v3.7)
    data_v3_7/eval_v3_7.jsonl          (~15 MB,  122k pairs)
    data_v3_7/training_stats_v3_7.json (metadata)
    model_cache/byt5-large/...         (filled by download notebook)

v3.7 CHANGES over v3.6 (gap-fill patch):
  - +52k targeted gap-fill pairs addressing hard-test failures:
      * short_identity_v37       -- prevents over-correction of short words
      * niche_brand_identity_v37 -- Oura / DJI / ZV-1F / WH-1000XM5 preservation
      * split_to_compound_v37    -- boosts "mac book -> macbook" from 15 examples
      * number_preservation_v37  -- keeps "1000 watts" intact across contexts
      * external_*_v37           -- Birkbeck / Wikipedia / Aspell / Holbrook corpora
  - Oversampled critical categories (5x short_identity + niche_brand, 3x compound, 2x numbers)
  - New metric: identity_accuracy -- fraction of correct-input cases returned unchanged
"""

# =====================================================================
# CELL 1: SETUP & VERIFY MODEL CACHE
# =====================================================================
# !pip install -q -U "transformers>=4.47,<4.50" "datasets>=3.0,<3.3" \
#                   "accelerate>=1.0" sentencepiece evaluate \
#                   python-Levenshtein plotly safetensors
# Notes:
#   * Use modern versions that are numpy 2.x-compatible. Pinning the
#     old versions (transformers==4.44.2, datasets==2.21.0) forces
#     numpy<2, which breaks Colab's preinstalled C-extensions compiled
#     against numpy 2.x and produces the error:
#        "ValueError: numpy.dtype size changed, may indicate binary
#         incompatibility. Expected 96 from C header, got 88 from PyObject"
#   * If you already ran the old pip command in this session, go to
#     Runtime -> Restart session after reinstalling, then re-run from Cell 1.

import os, sys, time, json, gc, math, random, shutil, subprocess
from pathlib import Path

import torch

# ----- Mount Google Drive (Colab only) -----
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive")
    IN_COLAB = True
except Exception:
    DRIVE_ROOT = Path(os.environ.get("DRIVE_ROOT", "/tmp/drive"))
    IN_COLAB = False

# ----- Paths -----
PROJECT_ROOT   = DRIVE_ROOT / "Grad" / "Correction" / "fine_tune" / "BYT5-T5 Large v3"
DATA_DIR       = PROJECT_ROOT / "data_v3_7"           # v3.7: point at new dataset
COLLAB_DIR     = PROJECT_ROOT / "collab" / "BYT5-Large-V3"
DRIVE_OUTPUT   = PROJECT_ROOT / "output" / "byt5-large-v3"
MODEL_CACHE    = PROJECT_ROOT / "model_cache" / "byt5-large"

# v3.2 NEW: local scratch dir avoids Drive I/O tax during training
SCRATCH_ROOT   = Path("/content/scratch") if IN_COLAB else Path("/tmp/scratch")
LOCAL_OUTPUT   = SCRATCH_ROOT / "byt5-large-v3"

for p in [DATA_DIR, COLLAB_DIR, DRIVE_OUTPUT, LOCAL_OUTPUT]:
    p.mkdir(parents=True, exist_ok=True)

print("-" * 70)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR:     {DATA_DIR}")
print(f"DRIVE_OUTPUT: {DRIVE_OUTPUT}")
print(f"LOCAL_OUTPUT: {LOCAL_OUTPUT}  (scratch; synced to Drive periodically)")
print(f"MODEL_CACHE:  {MODEL_CACHE}")
print("-" * 70)

# Verify data files exist
TRAIN_FILE = DATA_DIR / "train_v3_7.jsonl"
EVAL_FILE  = DATA_DIR / "eval_v3_7.jsonl"
assert TRAIN_FILE.exists(), f"Missing: {TRAIN_FILE}"
assert EVAL_FILE.exists(),  f"Missing: {EVAL_FILE}"
print(f"OK train_v3_7.jsonl: {TRAIN_FILE.stat().st_size / 1e6:.1f} MB")
print(f"OK eval_v3_7.jsonl:  {EVAL_FILE.stat().st_size / 1e6:.1f} MB")

# Verify model cache
if MODEL_CACHE.exists():
    files = list(MODEL_CACHE.iterdir())
    cache_size_gb = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
    print(f"OK Model cache:  {len(files)} files, {cache_size_gb:.2f} GB")
else:
    print("!! Model cache NOT FOUND -- run download_byt5_to_drive.ipynb first!")


# =====================================================================
# CELL 2: CONFIGURATION
# =====================================================================
import torch

# v3.2: enable TF32 matmul (Ampere/Hopper Tensor Cores)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

FORCE_DETERMINISTIC = False

if torch.cuda.is_available():
    gpu_name   = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name} | VRAM: {gpu_mem_gb:.1f} GB | CC: {cc_major}.{cc_minor}")
else:
    gpu_name, gpu_mem_gb = "cpu", 0
    cc_major = cc_minor = 0
    print("!! No CUDA GPU available -- training will be very slow.")

IS_AMPERE_OR_NEWER = cc_major >= 8

# ---------- GPU-adaptive batch / compile / precision ----------
if "A100" in gpu_name or gpu_mem_gb >= 70:                 # A100 80GB
    BS, GRAD_ACCUM      = 24, 3       # eff 72
    MIXED_PRECISION     = "bf16"
    TORCH_COMPILE       = True
    GRAD_CHECKPOINT     = False
    DEVICE_TIER         = "A100-80G"
elif "A100" in gpu_name or gpu_mem_gb >= 35:               # A100 40GB (Colab Pro)
    BS, GRAD_ACCUM      = 16, 4       # eff 64
    MIXED_PRECISION     = "bf16"
    TORCH_COMPILE       = True
    GRAD_CHECKPOINT     = False       # critical: disabling gives +30-40% throughput
    DEVICE_TIER         = "A100-40G"
elif gpu_mem_gb >= 22:                                     # L4 / A10 / A40 24G
    BS, GRAD_ACCUM      = 8, 8
    MIXED_PRECISION     = "bf16"
    TORCH_COMPILE       = IS_AMPERE_OR_NEWER
    GRAD_CHECKPOINT     = False
    DEVICE_TIER         = "L4/A10/A40"
elif gpu_mem_gb >= 14:                                     # T4 16G / V100 16G / RTX 5070Ti
    BS, GRAD_ACCUM      = 4, 16
    MIXED_PRECISION     = "fp16" if cc_major < 8 else "bf16"
    TORCH_COMPILE       = False
    GRAD_CHECKPOINT     = True
    DEVICE_TIER         = "T4/V100/5070Ti"
else:
    BS, GRAD_ACCUM      = 2, 32
    MIXED_PRECISION     = "fp32"
    TORCH_COMPILE       = False
    GRAD_CHECKPOINT     = True
    DEVICE_TIER         = "small/cpu"

# v3.3: optional cap on training set size -- leave None to use all 2.88M pairs.
# Rule of thumb on A100 40G (bf16 + torch.compile):
#   None (2.88M) x 2 epochs  -> ~18-22h total  (needs Colab Pro+ or 2 sessions)
#   1_000_000     x 2 epochs -> ~8-10h         (fits one Colab Pro A100 session)
#   500_000       x 2 epochs -> ~4-5h          (great for quick iterations)
TRAIN_SUBSET_SIZE = None   # or: 1_000_000, 500_000

CONFIG = {
    "model_name_or_path":        str(MODEL_CACHE),
    "max_input_length":          128,
    "max_target_length":         128,

    "num_train_epochs":          2,   # v3.3: reduced from 3; 2 is enough with 2.88M pairs + label smoothing
    "per_device_train_batch_size":  BS,
    "per_device_eval_batch_size":   max(2, BS // 2),
    "gradient_accumulation_steps":  GRAD_ACCUM,
    "learning_rate":             3e-4,
    "warmup_ratio":              0.06,
    "weight_decay":              0.01,
    # v3.7: dropped label_smoothing 0.1 -> 0.05 to reduce the soft-distribution
    # pressure that can encourage over-correction (model guesses a common typo
    # fix even when input is already correct). Paired with oversampled
    # identity data in v3.7, this nudges toward "when in doubt, don't edit".
    "label_smoothing_factor":    0.05,
    "lr_scheduler_type":         "cosine",
    "optim":                     "adamw_torch_fused" if IS_AMPERE_OR_NEWER else "adamw_torch",
    "mixed_precision":           MIXED_PRECISION,
    "torch_compile":             TORCH_COMPILE,
    "group_by_length":           True,
    "gradient_checkpointing":    GRAD_CHECKPOINT,

    "eval_steps":                3000,   # v3.3: tighter cadence for better early-stop responsiveness
    "save_steps":                3000,
    "save_total_limit":          3,
    "logging_steps":             50,
    "eval_subset_size":          2000,
    "generation_num_beams":      5,

    "dataloader_num_workers":    8,      # v3.3: A100 VMs expose 12+ vCPU -> 8 workers
    "dataloader_pin_memory":     True,
    "dataloader_prefetch_factor": 8,     # v3.3: hides slow JSONL read from training step
    "tokenize_num_proc":         8,      # v3.3: parallel byte-level tokenisation

    "train_subset_size":         TRAIN_SUBSET_SIZE,

    "early_stopping_patience":   5,   # v3.3: 5 evals with no improvement -> stop
    "early_stopping_threshold":  1e-4,

    "overfit_guard_patience":    3,
    "overfit_guard_threshold":   0.05,

    # v3.3: strict "accuracy dropped N times in a row" guard (user request).
    # Distinct from early_stopping_patience (which triggers on no-improvement);
    # this one fires only on actual consecutive decreases of eval_sentence_accuracy.
    "consecutive_drops_patience": 5,

    "local_output_dir":          str(LOCAL_OUTPUT),
    "drive_output_dir":          str(DRIVE_OUTPUT),
    "report_to":                 ["none"],
    "seed":                      42,

    "resume_from_checkpoint":    True,
}

random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])
if FORCE_DETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False

print(f"\nDevice tier:        {DEVICE_TIER}")
print(f"Batch size:         {BS}  (effective: {BS * GRAD_ACCUM})")
print(f"Mixed precision:    {MIXED_PRECISION}  (TF32: on)")
print(f"torch.compile:      {TORCH_COMPILE}")
print(f"grad checkpointing: {GRAD_CHECKPOINT}")
print(f"Optimizer:          {CONFIG['optim']}")
print(f"LR / warmup:        {CONFIG['learning_rate']}  /  {CONFIG['warmup_ratio']}")
print(f"Eval every:         {CONFIG['eval_steps']} steps  (subset {CONFIG['eval_subset_size']})")
print(f"Beam search:        num_beams={CONFIG['generation_num_beams']}")
print(f"Dataloader:         {CONFIG['dataloader_num_workers']} workers, pin_memory={CONFIG['dataloader_pin_memory']}")
print(f"Resume checkpoint:  {CONFIG['resume_from_checkpoint']}")


# =====================================================================
# CELL 3: LOAD MODEL & TOKENIZER
# =====================================================================
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

print(f"Loading ByT5-Large from {CONFIG['model_name_or_path']} ...")
t0 = time.time()
tokenizer = ByT5Tokenizer.from_pretrained(CONFIG["model_name_or_path"])
model     = T5ForConditionalGeneration.from_pretrained(
    CONFIG["model_name_or_path"],
    torch_dtype=torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float32,
)
print(f"Loaded in {time.time() - t0:.1f}s  |  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

if CONFIG["gradient_checkpointing"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    print("OK Gradient checkpointing enabled (non-reentrant)")

if torch.cuda.is_available():
    model = model.cuda()

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
    "train":      str(TRAIN_FILE),
    "validation": str(EVAL_FILE),
})
print(f"Raw data loaded in {time.time() - t0:.1f}s")
print(f"  train:      {len(raw['train']):,}")
print(f"  validation: {len(raw['validation']):,}")

from collections import Counter
if "category" in raw["train"].column_names:
    cat_train = Counter(raw["train"]["category"])
    print("\nTop categories in train split:")
    for c, n in cat_train.most_common(10):
        print(f"  {c:<28s} {n:>7,}")

# v3.7: capture identity-case target strings so compute_metrics can report
# identity_accuracy separately from correction_accuracy.
IDENTITY_EVAL_TARGETS = set()
for rec in raw["validation"]:
    src = rec["input_text"].removeprefix("correct: ").strip()
    tgt = rec["target_text"].strip()
    if src == tgt:
        IDENTITY_EVAL_TARGETS.add(tgt)
print(f"\nIdentity-case eval targets captured: {len(IDENTITY_EVAL_TARGETS):,}")

def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=CONFIG["max_input_length"],
        padding=False,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=CONFIG["max_target_length"],
        padding=False,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["length"] = [len(x) for x in model_inputs["input_ids"]]
    return model_inputs

print("\nTokenizing train split ...")
t0 = time.time()
tokenized_train = raw["train"].map(
    tokenize_fn, batched=True,
    remove_columns=raw["train"].column_names,
    desc="tokenize-train",
    num_proc=CONFIG["tokenize_num_proc"],
    load_from_cache_file=True,
)
print(f"  train tokenized in {time.time() - t0:.1f}s")

# v3.3: optional train-set cap (preserves category distribution via shuffle-then-head).
if CONFIG["train_subset_size"] and len(tokenized_train) > CONFIG["train_subset_size"]:
    orig_n = len(tokenized_train)
    tokenized_train = tokenized_train.shuffle(seed=CONFIG["seed"]).select(
        range(CONFIG["train_subset_size"])
    )
    print(f"  train subset: {orig_n:,} -> {len(tokenized_train):,} (cap = {CONFIG['train_subset_size']:,})")

print("Tokenizing eval split ...")
t0 = time.time()
tokenized_eval = raw["validation"].map(
    tokenize_fn, batched=True,
    remove_columns=raw["validation"].column_names,
    desc="tokenize-eval",
    num_proc=CONFIG["tokenize_num_proc"],
    load_from_cache_file=True,
)
print(f"  eval  tokenized in {time.time() - t0:.1f}s")

if CONFIG["eval_subset_size"] and len(tokenized_eval) > CONFIG["eval_subset_size"]:
    tokenized_eval_sub = tokenized_eval.shuffle(seed=CONFIG["seed"]).select(
        range(CONFIG["eval_subset_size"])
    )
    print(f"  Using eval subset of {len(tokenized_eval_sub):,} for in-training evals")
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
    """
    v3.4: hardened against -100 in preds and out-of-vocab IDs.
    v3.7: added identity_accuracy -- fraction of "input already correct"
          eval samples where the model correctly returns the input unchanged.
          This is the critical anti-hallucination metric.

    ByT5Tokenizer._convert_id_to_token does chr(index - 3). If preds contain
    -100 (HF Seq2SeqTrainer occasionally leaks label ignore_index into
    predictions during beam-search + max_length padding across batch
    boundaries), chr(-103) raises ValueError. We also clamp to vocab range
    to defend against any stray out-of-range IDs from generate().
    """
    import numpy as np
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.ndim == 3:
        preds = preds.argmax(-1)

    pad_id     = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    # 1) Replace -100 (label ignore_index) with pad_token_id in BOTH preds and labels.
    preds  = np.where(preds  != -100, preds,  pad_id)
    labels = np.where(labels != -100, labels, pad_id)

    # 2) Clamp to valid vocab range to defend against stray values.
    preds  = np.clip(preds,  0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)

    decoded_preds  = tokenizer.batch_decode(preds.tolist(),  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

    # v3.7: identity_accuracy needs the original inputs so we can check whether
    # the eval row was an "already correct" case (input == target). We reuse
    # the globally-bound `tokenized_eval_sub` to pull `input_ids` for this batch,
    # but the Trainer doesn't pass inputs here -- so we detect identity via
    # the labels themselves: any sample whose label equals the prediction is
    # "exact match"; but the row was an *identity case* iff its decoded_label
    # already looks like natural text with no typo. We approximate identity
    # cases by checking whether the decoded_label is in a rolling set of
    # "input_text == target_text" samples captured at tokenize time. See the
    # EvalInputsMixin callback that populates IDENTITY_EVAL_TARGETS.
    identity_targets = globals().get("IDENTITY_EVAL_TARGETS", set())

    exact, f1_sum, cer_sum = 0, 0.0, 0.0
    id_total, id_correct = 0, 0
    nonid_total, nonid_correct = 0, 0
    for p, l in zip(decoded_preds, decoded_labels):
        p, l = p.strip(), l.strip()
        is_match = (p == l)
        if is_match:
            exact += 1
        f1_sum  += _word_f1(p.split(), l.split())
        cer_sum += _levenshtein(p, l) / max(1, len(l))
        # Identity bucket: label text was seen as "input == target" at tokenize time.
        if l in identity_targets:
            id_total += 1
            if is_match:
                id_correct += 1
        else:
            nonid_total += 1
            if is_match:
                nonid_correct += 1
    n = max(1, len(decoded_preds))
    out = {
        "sentence_accuracy": round(exact / n, 4),
        "word_f1":           round(f1_sum / n, 4),
        "cer":               round(cer_sum / n, 4),
    }
    if id_total > 0:
        out["identity_accuracy"]     = round(id_correct / id_total, 4)
        out["identity_eval_count"]   = id_total
    if nonid_total > 0:
        out["correction_accuracy"]   = round(nonid_correct / nonid_total, 4)
        out["correction_eval_count"] = nonid_total
    return out


# =====================================================================
# CELL 6: TRAINING
# =====================================================================
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

PROBE_CASES = [
    ("razer",                       "razer",                        "identity_brand"),
    ("asus",                        "asus",                         "identity_brand"),
    ("logitech",                    "logitech",                     "identity_brand"),
    ("corsair",                     "corsair",                      "identity_brand"),
    ("corsiar keybord",             "corsair keyboard",             "ecom_query_typo"),
    ("razr mouse",                  "razer mouse",                  "ecom_query_typo"),
    ("logiteck mx keys",            "logitech mx keys",             "ecom_query_typo"),
    ("nvidai rtx 4090",             "nvidia rtx 4090",              "ecom_query_typo"),
    ("blutooth headphones",         "bluetooth headphones",         "phonetic_typo"),
    ("wereless mouse",              "wireless mouse",               "phonetic_typo"),
    ("air pods pro",                "airpods pro",                  "space_error"),
    ("iphonecase",                  "iphone case",                  "space_error"),
    ("iphone 15 under 500 dolars",  "iphone 15 under 500 dollars",  "price_typo"),
    ("samsung tv under $1000",      "samsung tv under $1000",       "price_identity"),
    ("ssd 1tb",                     "ssd 1TB",                      "unit_typo"),
    ("monitor 144 hz",              "monitor 144 Hz",               "unit_typo"),
    ("nvidia tuf chair",            "asus tuf gaming chair",        "brand_category_mismatch"),
    ("teh best laptop",             "the best laptop",              "everyday_english_typo"),
    ("rtx 4090",                    "rtx 4090",                     "identity_tech"),
    ("ddr5",                        "ddr5",                         "identity_tech"),
]


def run_probe(model_ref, tokenizer_ref, cases=PROBE_CASES, num_beams=1) -> dict:
    from collections import defaultdict
    buckets = defaultdict(lambda: [0, 0])
    model_ref.eval()
    with torch.no_grad():
        for inp, tgt, cat in cases:
            enc = tokenizer_ref(f"correct: {inp}", return_tensors="pt",
                                truncation=True, max_length=128)
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            out = model_ref.generate(
                enc["input_ids"],
                max_new_tokens=64,
                num_beams=num_beams,
                do_sample=False,
            )
            pred = tokenizer_ref.decode(out[0], skip_special_tokens=True).strip()
            ok = int(pred == tgt)
            buckets[cat][0] += ok
            buckets[cat][1] += 1
    return {c: (n, d, n / d if d else 0.0) for c, (n, d) in buckets.items()}


class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self.best_acc        = 0.0
        self.best_step       = 0
        self.widen_streak    = 0
        self.last_gap        = None
        self.train_loss_ema  = None
        self.t_start         = None
        # v3.3: strict consecutive-accuracy-drop tracker.
        self.drop_streak     = 0
        self.last_acc        = None

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
        elapsed  = time.time() - self.t_start if self.t_start else 0
        progress = state.global_step / max(1, state.max_steps)
        eta      = elapsed * (1 - progress) / max(progress, 1e-9)
        if state.global_step % max(1, args.logging_steps) == 0 and loss is not None:
            lr_str = f"{lr:.2e}" if lr is not None else "n/a"
            print(f"  step {state.global_step:>6d}/{state.max_steps}  "
                  f"loss={loss:.4f}  ema={self.train_loss_ema:.4f}  "
                  f"lr={lr_str}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        acc        = metrics.get("eval_sentence_accuracy", 0.0)
        f1         = metrics.get("eval_word_f1", 0.0)
        cer        = metrics.get("eval_cer", 0.0)
        eval_loss  = metrics.get("eval_loss")
        train_loss = self.train_loss_ema
        gap = (eval_loss - train_loss) if (eval_loss is not None and train_loss is not None) else None

        if acc > self.best_acc:
            self.best_acc  = acc
            self.best_step = state.global_step

        print("-" * 70)
        print(f"  EVAL @ step {state.global_step}")
        if gap is not None:
            print(f"    loss={eval_loss:.4f}  train_ema={train_loss:.4f}  gap={gap:.4f}")
        else:
            print(f"    eval_loss={eval_loss}")
        print(f"    sentence_acc={acc:.4f}  (best={self.best_acc:.4f} @ step {self.best_step})")
        print(f"    word_f1={f1:.4f}  cer={cer:.4f}")

        if gap is not None:
            if self.last_gap is not None and gap - self.last_gap > 0 and gap > CONFIG["overfit_guard_threshold"]:
                self.widen_streak += 1
            else:
                self.widen_streak = 0
            self.last_gap = gap
            if self.widen_streak >= CONFIG["overfit_guard_patience"]:
                print(f"    ! Overfit guard: eval-train gap widened {self.widen_streak}x -> stopping.")
                control.should_training_stop = True

        # v3.3: strict consecutive-drop guard on sentence_accuracy.
        if self.last_acc is not None:
            if acc < self.last_acc:
                self.drop_streak += 1
            else:
                self.drop_streak = 0
        self.last_acc = acc
        drops_limit = CONFIG.get("consecutive_drops_patience", 5)
        if self.drop_streak >= drops_limit:
            print(f"    ! Drop guard: eval sentence_acc fell {self.drop_streak}x in a row -> stopping.")
            control.should_training_stop = True
        elif self.drop_streak > 0:
            print(f"    drop_streak: {self.drop_streak}/{drops_limit}")

        try:
            probe = run_probe(kwargs.get("model"), tokenizer, num_beams=1)
            print(f"    probe per-category accuracy (greedy):")
            for c, (n, d, r) in sorted(probe.items()):
                print(f"      {c:<26s} {n}/{d}  {100*r:5.1f}%")
        except Exception as e:
            print(f"    probe failed: {e}")

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


class DriveSyncCallback(TrainerCallback):
    """rsync local checkpoints to Drive after each save (fast local, durable drive)."""
    def __init__(self, local_dir: str, drive_dir: str):
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir)

    def on_save(self, args, state, control, **kwargs):
        if not self.local_dir.exists():
            return
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            subprocess.run(
                ["rsync", "-a", "--delete", f"{self.local_dir}/", f"{self.drive_dir}/"],
                check=True, capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            if self.drive_dir.exists():
                shutil.rmtree(self.drive_dir)
            shutil.copytree(self.local_dir, self.drive_dir)
        print(f"    OK synced local -> drive in {time.time()-t0:.1f}s")


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="longest",
    pad_to_multiple_of=64 if IS_AMPERE_OR_NEWER else 8,
    return_tensors="pt",
)

n_train         = len(tokenized_train)
effective_bs    = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
steps_per_epoch = math.ceil(n_train / effective_bs)
total_steps     = steps_per_epoch * CONFIG["num_train_epochs"]
print(f"Steps per epoch: {steps_per_epoch:,}  |  Total steps: {total_steps:,}")

def _find_resume_path():
    candidates = []
    for root in [LOCAL_OUTPUT, DRIVE_OUTPUT]:
        if root.exists():
            cp = get_last_checkpoint(str(root))
            if cp:
                candidates.append(Path(cp))
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)

resume_path = _find_resume_path() if CONFIG["resume_from_checkpoint"] else None
if resume_path:
    print(f"OK Resuming from: {resume_path}")
    if str(DRIVE_OUTPUT) in resume_path and not (LOCAL_OUTPUT / Path(resume_path).name).exists():
        dst = LOCAL_OUTPUT / Path(resume_path).name
        print(f"  copying to local scratch: {dst}")
        shutil.copytree(resume_path, dst)
        resume_path = str(dst)
else:
    print("  (no checkpoint found -- starting from scratch)")

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["local_output_dir"],
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
    tf32=IS_AMPERE_OR_NEWER,
    gradient_checkpointing=CONFIG["gradient_checkpointing"],
    torch_compile=CONFIG["torch_compile"],
    # v3.4: "default" instead of "reduce-overhead".
    # reduce-overhead activates CUDA graphs, which crash with predict_with_generate=True:
    #   RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten ...
    # "default" still JIT-compiles but does NOT capture CUDA graphs -> safe for generate().
    torch_compile_mode="default" if CONFIG["torch_compile"] else None,
    group_by_length=CONFIG["group_by_length"],
    length_column_name="length",
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    save_safetensors=True,
    load_best_model_at_end=True,
    metric_for_best_model="sentence_accuracy",
    greater_is_better=True,
    logging_steps=CONFIG["logging_steps"],
    logging_dir=str(LOCAL_OUTPUT / "logs"),
    report_to=CONFIG["report_to"],
    predict_with_generate=True,
    generation_max_length=CONFIG["max_target_length"],
    generation_num_beams=CONFIG["generation_num_beams"],
    dataloader_num_workers=CONFIG["dataloader_num_workers"],
    dataloader_pin_memory=CONFIG["dataloader_pin_memory"],
    dataloader_prefetch_factor=CONFIG["dataloader_prefetch_factor"],
    dataloader_persistent_workers=True,
    seed=CONFIG["seed"],
    remove_unused_columns=True,
    push_to_hub=False,
    disable_tqdm=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval_sub,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG["early_stopping_patience"],
            early_stopping_threshold=CONFIG["early_stopping_threshold"],
        ),
        TrainingMonitor(),
        DriveSyncCallback(CONFIG["local_output_dir"], CONFIG["drive_output_dir"]),
    ],
)

print("\n" + "=" * 70)
print(" STARTING TRAINING")
print("=" * 70)
train_result = trainer.train(resume_from_checkpoint=resume_path)

final_local = Path(CONFIG["local_output_dir"]) / "final"
final_drive = Path(CONFIG["drive_output_dir"]) / "final"
trainer.save_model(str(final_local))
tokenizer.save_pretrained(str(final_local))
final_drive.parent.mkdir(parents=True, exist_ok=True)
if final_drive.exists():
    shutil.rmtree(final_drive)
shutil.copytree(final_local, final_drive)
print(f"\nOK Final model saved to: {final_local}")
print(f"OK Mirrored to Drive:    {final_drive}")


# =====================================================================
# CELL 7: TRAINING CURVES (2D)
# =====================================================================
import matplotlib.pyplot as plt
from pathlib import Path as _P

log_history = trainer.state.log_history
train_steps, train_losses = [], []
eval_steps,  eval_losses, eval_accs = [], [], []
for row in log_history:
    if "loss" in row and "eval_loss" not in row:
        train_steps.append(row.get("step", 0)); train_losses.append(row["loss"])
    if "eval_loss" in row:
        eval_steps.append(row.get("step", 0))
        eval_losses.append(row["eval_loss"])
        eval_accs.append(row.get("eval_sentence_accuracy", 0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_steps, train_losses, label="train", alpha=0.6)
axes[0].plot(eval_steps,  eval_losses,  label="eval",  marker="o")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(eval_steps, eval_accs, marker="o", color="green")
axes[1].set_xlabel("step"); axes[1].set_ylabel("sentence accuracy")
axes[1].set_title("Eval sentence accuracy"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
png_out = _P(CONFIG["local_output_dir"]) / "training_curves.png"
plt.savefig(png_out, dpi=120, bbox_inches="tight")
plt.show()
shutil.copy2(png_out, _P(CONFIG["drive_output_dir"]) / png_out.name)
print(f"OK Saved 2D curves: {png_out}")


# =====================================================================
# CELL 7B (OPTIONAL): 3D TRAINING TRAJECTORY (plotly)
# =====================================================================
try:
    import plotly.graph_objects as go
    if eval_steps:
        fig3d = go.Figure(data=[go.Scatter3d(
            x=eval_steps, y=eval_losses, z=eval_accs,
            mode="lines+markers",
            marker=dict(size=5, color=eval_accs, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="sent_acc")),
            line=dict(width=4, color="darkblue"),
        )])
        fig3d.update_layout(
            title="Training trajectory (eval)",
            scene=dict(xaxis_title="step", yaxis_title="eval_loss", zaxis_title="sentence_accuracy"),
            height=600,
        )
        html_out = _P(CONFIG["local_output_dir"]) / "training_trajectory_3d.html"
        fig3d.write_html(str(html_out))
        shutil.copy2(html_out, _P(CONFIG["drive_output_dir"]) / html_out.name)
        print(f"OK Saved 3D trajectory: {html_out}")
        fig3d.show()
except ImportError:
    print("plotly not installed -- skip 3D trajectory.")


# =====================================================================
# CELL 8: COMPREHENSIVE EVALUATION
# =====================================================================
from collections import defaultdict

TEST_CASES = [
    ("razer",                           "razer",                            "identity"),
    ("corsair",                         "corsair",                          "identity"),
    ("logitech",                        "logitech",                         "identity"),
    ("asus tuf gaming",                 "asus tuf gaming",                  "identity"),
    ("iphone 15 pro max",               "iphone 15 pro max",                "identity"),
    ("rtx 4090",                        "rtx 4090",                         "identity"),
    ("ddr5 ram 32gb",                   "ddr5 ram 32gb",                    "identity"),
    ("corsiar keybord",                 "corsair keyboard",                 "brand_typo"),
    ("razr mouse",                      "razer mouse",                      "brand_typo"),
    ("logiteck mx master",              "logitech mx master",               "brand_typo"),
    ("nvidai rtx 4090",                 "nvidia rtx 4090",                  "brand_typo"),
    ("samsng galaxy s24",               "samsung galaxy s24",               "brand_typo"),
    ("best wreless headphnoes",         "best wireless headphones",         "product_typo"),
    ("cheap keybord under 50",          "cheap keyboard under 50",          "product_typo"),
    ("blutooth speker",                 "bluetooth speaker",                "product_typo"),
    ("mechnical gaming keybord",        "mechanical gaming keyboard",       "product_typo"),
    ("corsiar blutooth mehanical keybord", "corsair bluetooth mechanical keyboard", "compound"),
    ("razr wreless mechnical mouse",       "razer wireless mechanical mouse",        "compound"),
    ("air pods pro",                    "airpods pro",                      "space_error"),
    ("iphonecase",                      "iphone case",                      "space_error"),
    ("keyboardmouse combo",             "keyboard mouse combo",             "space_error"),
    ("blutooth mowse",                  "bluetooth mouse",                  "phonetic"),
    ("wereless hedphones",              "wireless headphones",              "phonetic"),
    ("munitor 144 hz",                  "monitor 144 Hz",                   "phonetic"),
    ("iphone 15 under 500 dolars",      "iphone 15 under 500 dollars",      "price_typo"),
    ("samsung tv under $1000",          "samsung tv under $1000",           "price_identity"),
    ("gaming laptop between $800 and $1500", "gaming laptop between $800 and $1500", "price_identity"),
    ("coffe maker for $50",             "coffee maker for $50",             "price_typo"),
    ("ssd 1tb",                         "ssd 1TB",                          "unit_typo"),
    ("monitor 144hz",                   "monitor 144 Hz",                   "unit_typo"),
    ("power bank 20000mah",             "power bank 20000 mAh",             "unit_typo"),
    ("1080p webcam",                    "1080p webcam",                     "unit_identity"),
    ("4k monitor",                      "4k monitor",                       "unit_identity"),
    ("nvidia tuf chair",                "asus tuf gaming chair",            "brand_category_mismatch"),
    ("intel ryzen 7 laptop",            "amd ryzen 7 laptop",               "brand_category_mismatch"),
    ("samsung airpods case",            "apple airpods case",               "brand_category_mismatch"),
    ("teh best laptop",                 "the best laptop",                  "everyday_english"),
    ("recieve shipping",                "receive shipping",                 "everyday_english"),
    ("definately buy",                  "definitely buy",                   "everyday_english"),
    ("razer mouse",                     "razer mouse",                      "tricky_identity"),
    ("asus rog strix",                  "asus rog strix",                   "tricky_identity"),
    ("birkenstock sandals",             "birkenstock sandals",              "tricky_identity"),
]


def correct(query: str, num_beams: int = 5) -> str:
    model.eval()
    enc = tokenizer(f"correct: {query}", return_tensors="pt",
                    truncation=True, max_length=128)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_new_tokens=128,
            num_beams=num_beams,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


results = defaultdict(lambda: {"correct": 0, "total": 0, "cases": []})
print("\n" + "=" * 70)
print("  COMPREHENSIVE EVAL (num_beams=5)")
print("=" * 70)
for inp, tgt, cat in TEST_CASES:
    pred = correct(inp, num_beams=5)
    ok   = pred.lower().strip() == tgt.lower().strip()
    results[cat]["correct"] += int(ok)
    results[cat]["total"]   += 1
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
final_dir = _P(CONFIG["drive_output_dir"]) / "final"
meta = {
    "version":                  "v3.3",
    "model":                    "ByT5-Large",
    "train_examples":           len(tokenized_train),
    "eval_examples":            len(tokenized_eval),
    "train_subset_size":        CONFIG["train_subset_size"],
    "epochs":                   CONFIG["num_train_epochs"],
    "effective_batch_size":     CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"],
    "per_device_batch_size":    CONFIG["per_device_train_batch_size"],
    "grad_accum":               CONFIG["gradient_accumulation_steps"],
    "optim":                    CONFIG["optim"],
    "learning_rate":            CONFIG["learning_rate"],
    "generation_num_beams":     CONFIG["generation_num_beams"],
    "best_sentence_acc":        getattr(trainer.state, "best_metric", None),
    "device_tier":              DEVICE_TIER,
    "mixed_precision":          CONFIG["mixed_precision"],
    "torch_compile":            CONFIG["torch_compile"],
    "gradient_checkpointing":   CONFIG["gradient_checkpointing"],
    "group_by_length":          CONFIG["group_by_length"],
    "tf32":                     IS_AMPERE_OR_NEWER,
    "dataloader_num_workers":   CONFIG["dataloader_num_workers"],
    "pad_to_multiple_of":       64 if IS_AMPERE_OR_NEWER else 8,
}
with open(final_dir / "training_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nOK Model + meta saved to: {final_dir}")
print("\nTo deploy, use num_beams=5 and max_length=128.")
