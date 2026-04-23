#!/usr/bin/env python3
"""
ByT5-Large Model Downloader -- Google Colab (v3.3)
===================================================
Caches google/byt5-large (~4.7 GB) into your Google Drive so the training
notebook can load it offline every session.

Run this ONCE before the first training run. Safe to re-run: it skips files
that are already present and valid, and resumes big files on failure.

Target (on Drive):
  /MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3/model_cache/byt5-large/

After this notebook completes, open byt5_large_finetune_v3.ipynb.
"""

# =====================================================================
# CELL 1: SETUP
# =====================================================================

import subprocess, sys, os

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub>=0.23.0",
])

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

from pathlib import Path

MODEL_CACHE = Path('/content/drive/MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3/model_cache')
LOCAL_MODEL = MODEL_CACHE / "byt5-large"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# -- HF Token for faster downloads --
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF Token: set (authenticated, faster downloads)")
except Exception:
    print("HF Token: not set (anonymous, slower but still works)")

print(f"Download target: {LOCAL_MODEL}")


# =====================================================================
# CELL 2: DOWNLOAD
# =====================================================================

from huggingface_hub import hf_hub_download
import time, os, shutil

MODEL_NAME = "google/byt5-large"

# google/byt5-large repo contains these files:
#   config.json, generation_config.json, pytorch_model.bin (~4.7 GB),
#   special_tokens_map.json, tokenizer_config.json,
#   flax_model.msgpack (skip), tf_model.h5 (skip)
#
# We only need the PyTorch files:
FILES_TO_DOWNLOAD = [
    ("config.json",              "tiny"),
    ("generation_config.json",   "tiny"),
    ("special_tokens_map.json",  "tiny"),
    ("tokenizer_config.json",    "tiny"),
    ("pytorch_model.bin",        "4.7 GB"),   # the big one
]

LOCAL_MODEL.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_NAME} to Google Drive...")
print(f"Target: {LOCAL_MODEL}\n")

for fname, size_hint in FILES_TO_DOWNLOAD:
    dest = LOCAL_MODEL / fname

    # Skip if already downloaded and non-empty
    if dest.exists() and dest.stat().st_size > 0:
        # Extra check for the big file: must be >4GB
        if fname == "pytorch_model.bin" and dest.stat().st_size < 4e9:
            print(f"  {fname}: truncated ({dest.stat().st_size/1e9:.1f} GB), re-downloading...")
        else:
            size_mb = dest.stat().st_size / 1e6
            print(f"  {fname}: already exists ({size_mb:.1f} MB), skipping")
            continue

    # Download with retry
    for attempt in range(1, 6):  # 5 attempts for the big file
        try:
            print(f"  {fname} ({size_hint}): downloading (attempt {attempt})...", end=" ", flush=True)
            t0 = time.time()

            # hf_hub_download returns the actual saved path
            # In some huggingface_hub versions, local_dir puts files
            # in a subdirectory or cache structure instead of directly
            saved_path = hf_hub_download(
                repo_id=MODEL_NAME,
                filename=fname,
                local_dir=str(LOCAL_MODEL),
            )

            # If the file ended up somewhere other than dest, copy it
            saved_path = Path(saved_path)
            if saved_path.resolve() != dest.resolve():
                print(f"(relocated) ", end="", flush=True)
                shutil.copy2(str(saved_path), str(dest))

            # Verify the file is at the expected destination
            if not dest.exists() or dest.stat().st_size == 0:
                raise RuntimeError(
                    f"File not found at {dest} after download. "
                    f"hf_hub_download returned: {saved_path}"
                )

            elapsed = time.time() - t0
            actual_size = dest.stat().st_size / 1e6
            print(f"OK ({actual_size:.0f} MB, {elapsed:.0f}s)")
            break

        except Exception as e:
            print(f"FAILED")
            print(f"    Error: {e}")
            if attempt < 5:
                wait = 10 * attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  FATAL: Could not download {fname} after 5 attempts.")
                print(f"  Manual download: https://huggingface.co/google/byt5-large/resolve/main/{fname}")
                print(f"  Upload to: {LOCAL_MODEL}/")
                raise

# Clean up any .cache or huggingface subdirectories that hf_hub_download created
for subdir in [LOCAL_MODEL / ".cache", LOCAL_MODEL / "huggingface"]:
    if subdir.exists() and subdir.is_dir():
        shutil.rmtree(str(subdir))
        print(f"  Cleaned up temp dir: {subdir.name}/")


# =====================================================================
# CELL 3: VERIFY
# =====================================================================

print(f"\n{'='*60}")
print(f"  VERIFICATION")
print(f"{'='*60}")

all_ok = True

for fname, _ in FILES_TO_DOWNLOAD:
    fpath = LOCAL_MODEL / fname
    if not fpath.exists():
        print(f"  MISSING: {fname}")
        all_ok = False
    else:
        size_mb = fpath.stat().st_size / 1e6
        print(f"  OK: {fname} ({size_mb:.1f} MB)")

# Size check on the big file
weight_file = LOCAL_MODEL / "pytorch_model.bin"
if weight_file.exists():
    size_gb = weight_file.stat().st_size / 1e9
    if size_gb < 4.0:
        print(f"\n  WARNING: pytorch_model.bin is {size_gb:.1f} GB (expected ~4.7 GB)")
        print(f"  Download may be truncated. Delete it and re-run this notebook.")
        all_ok = False
    else:
        print(f"\n  pytorch_model.bin: {size_gb:.2f} GB (OK)")

total_size = sum(f.stat().st_size for f in LOCAL_MODEL.rglob("*") if f.is_file()) / 1e9

if all_ok:
    print(f"\n  ALL FILES OK ({total_size:.1f} GB total)")
    print(f"  Location: {LOCAL_MODEL}")
    print(f"\n  You can now run the ByT5-Large training notebook.")
    print(f"  It will load the model from Drive cache (no internet needed).")
else:
    print(f"\n  SOME FILES MISSING OR CORRUPT")
    print(f"  Re-run this notebook to retry, or download manually.")
