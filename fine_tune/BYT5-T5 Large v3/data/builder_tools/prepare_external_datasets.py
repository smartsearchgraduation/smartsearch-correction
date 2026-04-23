#!/usr/bin/env python3
"""
ByT5-Large v3.2 — External Datasets Preparation (Colab)
========================================================

Downloads and processes two public Amazon-sourced datasets:
  1. Amazon ESCI Shopping Queries Dataset (amazon-science, 1.3M query-product pairs)
  2. Amazon Reviews 2023 product metadata (McAuley-Lab, 8 categories matching v3.1 builder)

Outputs (all written to Drive's `data/external/` so v3.1 builder auto-discovers them):
  - esci_queries.tsv          (wrong<TAB>right pairs from ESCI queries)
  - esci_titles.tsv           (ESCI product titles with typo mutations)
  - amazon_meta_{category}.tsv (per-category Amazon product title pairs)
  - amazon_brands.tsv         (brand+product identity pairs from Amazon metadata)

Total expected output: ~500k-700k new (wrong, right) pairs ready for the v3.1 builder.
"""

# =====================================================================
# CELL 1: SETUP — Mount Drive, install deps
# =====================================================================
# !pip install -q datasets==2.21.0 pandas==2.2.2 pyarrow==17.0.0 tqdm

import os, sys, json, gc, random, re
from pathlib import Path

random.seed(42)

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive")
except Exception:
    DRIVE_ROOT = Path(os.environ.get("DRIVE_ROOT", "/tmp/drive"))

PROJECT_ROOT = DRIVE_ROOT / "Grad" / "Correction" / "fine_tune" / "BYT5-T5 Large v3"
DATA_DIR     = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

print("─" * 70)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"EXTERNAL_DIR: {EXTERNAL_DIR}  (outputs go here)")
print("─" * 70)


# =====================================================================
# CELL 2: CONFIGURATION
# =====================================================================

# Per-category sample budget for Amazon Reviews 2023 metadata.
# Streaming mode — we stop after N items per category, never download full set.
CATEGORY_SAMPLE_BUDGET = {
    "All_Beauty":          20000,
    "Amazon_Fashion":      25000,
    "Appliances":          15000,
    "Arts_Crafts_and_Sewing": 10000,
    "Automotive":          20000,
    "Beauty_and_Personal_Care": 25000,
    "Cell_Phones_and_Accessories": 25000,
    "Clothing_Shoes_and_Jewelry": 30000,
    "Electronics":         30000,
    "Grocery_and_Gourmet_Food": 20000,
    "Health_and_Household":     20000,
    "Home_and_Kitchen":         30000,
    "Industrial_and_Scientific": 10000,
    "Office_Products":          15000,
    "Pet_Supplies":             15000,
    "Sports_and_Outdoors":      25000,
    "Tools_and_Home_Improvement": 20000,
    "Toys_and_Games":           20000,
    "Video_Games":              10000,
}

# ESCI config
ESCI_MAX_QUERIES   = 60000   # unique queries to process
ESCI_MAX_PRODUCTS  = 80000   # unique product titles to process

# Typo mutation count per clean title
TYPO_MUTATIONS_PER_TITLE = 2

# Max title length to keep (longer → truncate to first N words)
MAX_TITLE_WORDS = 7

print(f"Total category budget: {sum(CATEGORY_SAMPLE_BUDGET.values()):,} products")
print(f"ESCI:    {ESCI_MAX_QUERIES:,} queries + {ESCI_MAX_PRODUCTS:,} titles")
print(f"Mutations per title: {TYPO_MUTATIONS_PER_TITLE}")


# =====================================================================
# CELL 3: IMPORT TYPO GENERATORS FROM v3.1 BUILDER
# =====================================================================
# We reuse the exact generators from build_training_data_v3.py so mutations
# are consistent with the main training set.

BUILDER_PATH = DATA_DIR / "build_training_data_v3.py"
assert BUILDER_PATH.exists(), f"Missing v3.1 builder at {BUILDER_PATH}"

import importlib.util
spec = importlib.util.spec_from_file_location("btd", str(BUILDER_PATH))
btd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(btd)

generate_single_typo = btd.generate_single_typo
generate_compound_typo = btd.generate_compound_typo
make_query_typo = btd.make_query_typo
domain_plausible = btd.domain_plausible
levenshtein = btd.levenshtein

print("✓ Imported v3.1 typo generators")


def mutate_title(title: str, n_errors: int = 1) -> str:
    """Apply single or compound typo to a product title. Returns None on failure."""
    return make_query_typo(title, n_errors=n_errors)


def clean_title(raw: str, max_words: int = MAX_TITLE_WORDS) -> str:
    """Normalize Amazon title to a realistic search query form."""
    if not raw:
        return ""
    # Strip packaging text in parens/brackets
    t = re.sub(r"\([^)]*\)", " ", raw)
    t = re.sub(r"\[[^\]]*\]", " ", t)
    # Drop non-ASCII
    try:
        t.encode("ascii")
    except UnicodeEncodeError:
        return ""
    # Lowercase, collapse whitespace
    t = re.sub(r"\s+", " ", t).strip().lower()
    # Drop common boilerplate words at end
    t = re.sub(r"\b(pack of \d+|set of \d+|\d+ count|\d+-pack|bundle)\b.*$", "", t).strip()
    words = t.split()
    if len(words) < 2 or len(words) > 15:
        return ""
    # Keep first N words as the "search query" form
    words = words[:max_words]
    t = " ".join(words)
    # Total length sanity
    if len(t) < 5 or len(t) > 80:
        return ""
    return t


# =====================================================================
# CELL 4: DOWNLOAD AND PROCESS ESCI
# =====================================================================
# amazon-science/esci-data — English shopping queries with product mapping.
# We use the US split (en-US).

from datasets import load_dataset

print("Loading ESCI dataset (tasksource/esci)...")
# tasksource/esci mirrors the Amazon repo on HuggingFace
esci = load_dataset("tasksource/esci", split="train", streaming=False)
print(f"ESCI total rows: {len(esci):,}")

# Filter to English only
esci_en = esci.filter(lambda r: r.get("product_locale", "") == "us" or r.get("query_locale", "") == "us")
print(f"ESCI US/English rows: {len(esci_en):,}")

# Extract unique queries + their associated product titles (Exact matches only)
print("Extracting queries and product titles...")
queries = set()
titles = set()
count = 0
for row in esci_en:
    q = (row.get("query") or "").strip().lower()
    t = (row.get("product_title") or "").strip()
    label = row.get("esci_label", "")  # E=Exact, S=Substitute, C=Complement, I=Irrelevant
    if q and 2 <= len(q.split()) <= 8:
        queries.add(q)
    if t and label == "E":  # Only Exact matches give us reliable query→title
        ct = clean_title(t)
        if ct:
            titles.add(ct)
    count += 1
    if len(queries) >= ESCI_MAX_QUERIES and len(titles) >= ESCI_MAX_PRODUCTS:
        break

queries = list(queries)[:ESCI_MAX_QUERIES]
titles = list(titles)[:ESCI_MAX_PRODUCTS]
print(f"  ESCI queries kept: {len(queries):,}")
print(f"  ESCI titles kept:  {len(titles):,}")

# Generate typo pairs from queries
print("Mutating ESCI queries to generate typo pairs...")
pairs_q = []
for q in queries:
    for _ in range(TYPO_MUTATIONS_PER_TITLE):
        typo = mutate_title(q, n_errors=random.choice([1, 1, 2]))
        if typo and typo != q and len(typo) <= 80:
            pairs_q.append((typo, q))

# Generate typo pairs from titles
print("Mutating ESCI titles to generate typo pairs...")
pairs_t = []
for t in titles:
    for _ in range(TYPO_MUTATIONS_PER_TITLE):
        typo = mutate_title(t, n_errors=random.choice([1, 1, 2]))
        if typo and typo != t and len(typo) <= 80:
            pairs_t.append((typo, t))

# Dedup + write
def write_tsv(pairs, path):
    seen = set()
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for wrong, right in pairs:
            key = (wrong, right)
            if key in seen:
                continue
            if not wrong or not right or wrong == right:
                continue
            seen.add(key)
            f.write(f"{wrong}\t{right}\n")
            n += 1
    return n

out1 = EXTERNAL_DIR / "esci_queries.tsv"
out2 = EXTERNAL_DIR / "esci_titles.tsv"
n1 = write_tsv(pairs_q, out1)
n2 = write_tsv(pairs_t, out2)
print(f"✓ Wrote {n1:,} ESCI query pairs   → {out1.name}")
print(f"✓ Wrote {n2:,} ESCI title pairs   → {out2.name}")
del esci, esci_en, queries, titles, pairs_q, pairs_t
gc.collect()


# =====================================================================
# CELL 5: DOWNLOAD AND PROCESS AMAZON REVIEWS 2023 PRODUCT METADATA
# =====================================================================
# McAuley-Lab/Amazon-Reviews-2023 — per-category product metadata (raw_meta_*).
# We stream each category and take N samples, then write per-category TSV.

from datasets import load_dataset
from tqdm.auto import tqdm

print("Processing Amazon Reviews 2023 product metadata (streaming)...")
all_meta_pairs = []       # (wrong, right) — typo mutations
all_identity = []         # (right, right) — brand+product identity
all_brand_pairs = set()   # unique brand+product phrases for identity

for category, budget in CATEGORY_SAMPLE_BUDGET.items():
    print(f"\n─── {category} (budget {budget:,}) ───")
    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
            streaming=True,
        )
    except Exception as e:
        print(f"  (skipping: {type(e).__name__}: {str(e)[:80]})")
        continue

    collected = 0
    typo_pairs = []
    identity_pairs = []
    for item in ds:
        if collected >= budget:
            break
        title_raw = item.get("title", "") or ""
        brand = (item.get("store", "") or "").strip().lower()
        clean = clean_title(title_raw)
        if not clean:
            continue
        # Only include titles where at least one token is domain_plausible
        toks = clean.split()
        if not any(domain_plausible(t) for t in toks if t.isalpha()):
            continue
        # Identity pair (teach correct form)
        identity_pairs.append((clean, clean))
        all_brand_pairs.add(clean)
        # Typo mutation pairs
        for _ in range(TYPO_MUTATIONS_PER_TITLE):
            typo = mutate_title(clean, n_errors=random.choice([1, 1, 2]))
            if typo and typo != clean and len(typo) <= 80:
                typo_pairs.append((typo, clean))
        collected += 1

    # Write per-category TSV
    cat_out = EXTERNAL_DIR / f"amazon_meta_{category.lower()}.tsv"
    n = write_tsv(typo_pairs, cat_out)
    print(f"  collected {collected:,} products → {n:,} typo pairs → {cat_out.name}")
    all_meta_pairs.extend(typo_pairs)
    all_identity.extend(identity_pairs)
    gc.collect()

# Write combined brand/identity file
brands_out = EXTERNAL_DIR / "amazon_identity.tsv"
n_id = write_tsv(all_identity, brands_out)
print(f"\n✓ Wrote {n_id:,} identity pairs → {brands_out.name}")
print(f"\n── Total Amazon Reviews 2023 pairs: {len(all_meta_pairs):,} typo + {len(all_identity):,} identity ──")


# =====================================================================
# CELL 6: SUMMARY + VERIFICATION
# =====================================================================

print("\n" + "=" * 70)
print("EXTERNAL DATASETS PREP — SUMMARY")
print("=" * 70)
total_lines = 0
for p in sorted(EXTERNAL_DIR.glob("*.tsv")):
    with open(p) as f:
        n = sum(1 for _ in f)
    total_lines += n
    print(f"  {p.name:45s}  {n:>10,} lines")
print("-" * 70)
print(f"  {'TOTAL':45s}  {total_lines:>10,} pairs")
print("=" * 70)
print(f"\nAll files in: {EXTERNAL_DIR}")
print("\nNext step: re-run build_training_data_v3.py — it will auto-discover")
print("these files via its `_discover_external_corpora()` scan of data/external/.")
print("Expected impact: +500k-700k real-world pairs integrated into v3.2 dataset.")
