# SmartSearch — Correction Microservice

> Real-time spelling correction for e-commerce search queries. Part of the **SmartSearch** graduation project (4 microservices: Frontend, Backend, **Correction**, Retrieval).

A FastAPI service on **port 5001** that takes a (possibly typo'd) search query and returns a cleaned-up version. It supports multiple correction models — from a 30 ms ByT5-small to a 4-stage T5+FastText pipeline — and wraps every model with a **brand masking pipeline** that protects identifiers like `iPhone`, `Samsung Galaxy S24 Ultra`, `RTX 4090`, or `MacBook Pro` from byte-level corruption.

---

## Table of Contents

- [Architecture](#architecture)
- [Brand Masking Pipeline](#brand-masking-pipeline)
- [Available Models](#available-models)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Code Examples](#code-examples)
- [Backend Integration](#backend-integration)
- [Design Decisions](#design-decisions)
- [Tests & Benchmarks](#tests--benchmarks)
- [Tech Stack](#tech-stack)

---

## Architecture

```text
                       ┌───────────────────────────────┐
                       │  Frontend (React)             │
                       │  Dropdown selects model       │
                       └───────────────┬───────────────┘
                                       │ raw_text
                                       ▼
                       ┌───────────────────────────────┐
                       │  Backend (Flask, port 5000)   │
                       │  TextCorrectorService         │
                       └───────────────┬───────────────┘
                                       │ POST /correct
                                       │ { query, model }
                                       ▼
                       ┌───────────────────────────────┐
                       │  Correction (FastAPI, 5001)   │
                       │  api.py → TypoCorrector       │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  ★ Brand Masking (in)         │   ← model-agnostic
                       │  Aho-Corasick + SymSpell       │     (skipped for V2.1)
                       │  "iphone" → <<M0>>             │
                       │  "samsng glaxy" → <<M1>>       │
                       └───────────────┬───────────────┘
                                       │
        ┌────────┬─────────────────────┼─────────────────┬───────────┬──────────┐
        ▼        ▼                     ▼                 ▼           ▼          ▼
  ┌─────────┐ ┌─────────┐    ┌────────────────┐  ┌──────────┐ ┌────────┐ ┌────────┐
  │ ByT5    │ │ ByT5    │    │ ByT5-Large-V3  │  │ T5-Large │ │ T5-V2.1│ │ Qwen   │
  │ small   │ │ base    │    │ * default      │  │ V2       │ │ pipe.  │ │ 3.5 2B │
  │ ~30 ms  │ │ ~50 ms  │    │ ~100 ms        │  │ ~200 ms  │ │ ~205ms │ │ ~800ms │
  └─────────┘ └─────────┘    └────────────────┘  └──────────┘ └────┬───┘ └────────┘
                                                                    │
                                                  ┌─────────────────┴────────────────┐
                                                  │  T5-Large-V2.1 sub-pipeline       │
                                                  │  Stage 0: brand whitelist         │
                                                  │  Stage 1: T5-Large                │
                                                  │  Stage 2: FastText + FAISS        │
                                                  │  Stage 3: postprocess + restore   │
                                                  └────────────────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  ★ Brand Masking (out)        │
                       │  3-tier failsafe restore      │
                       │  <<M0>> → "iPhone"            │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                                  to Backend
                                  { corrected_query, changed,
                                    model_used, latency_ms, ... }
```

---

## Brand Masking Pipeline

A **persistent, model-agnostic infrastructure layer** that wraps every correction model. Brand identifiers in the user's query are temporarily replaced with placeholder tokens (`<<M0>>`, `<<M1>>`, …) before the model runs, then restored to their canonical form afterward. This guarantees that even byte-level models like ByT5 cannot corrupt brand names — and the layer survives any future model swap.

### How it works

```text
INPUT:   "samsng glaxy s24 ultra is the best, beats iphone"

▼ Stage 1 — Mask
  ExactMatcher (Aho-Corasick, O(n+m))
    "iphone" → <<M0>> = "iPhone"
  FuzzyMatcher (SymSpell, adaptive ED)
    "samsng glaxy s24 ultra" → <<M1>> = "Samsung Galaxy S24 Ultra"
  Masked text:  "<<M1>> is the best, beats <<M0>>"
  Mask map:     {<<M0>>: "iPhone", <<M1>>: "Samsung Galaxy S24 Ultra"}

▼ Stage 2 — Model (any model from the registry)
  Sees masked text. Mask tokens look like uncommon byte sequences and
  pass through unchanged. Other words get corrected normally.

▼ Stage 3 — Unmask (3-tier failsafe)
  Tier 1 — Strict regex:    <<M\d+>>     → exact match by index
  Tier 2 — Lenient regex:   <<m0>>, <<MO>>, <<M 0>>, <M0>  →  recover
  Tier 3 — Positional fallback: count remaining mask shapes,
            restore in original generation order

OUTPUT:  "Samsung Galaxy S24 Ultra is the best, beats iPhone"
```

### False-positive guards (in priority order)

False positives — masking a real English word as a brand — corrupt the user's query and are **not** recoverable by the downstream model. So the matcher is conservative by design:

| Guard | Where applied | What it does |
|-------|---------------|--------------|
| **English-wordlist filter** | Input token | Skips fuzzy match if the input is itself a real English word (`apply`, `keyboard`, `mouse`) |
| **Deny list** | Both sides | Brands too ambiguous (`apple`, `shell`, `tide`, `gap`, `blade`, `max`, `pro`) are excluded entirely |
| **Adaptive edit distance** | Input length | ED 0 for `len < 4`; ED 1 for `len ≥ 4`. ED 2 deferred to V2 |
| **Length ratio sanity** | Input vs canonical | Match rejected if ratio ∉ [0.6, 1.6] (catches `go → GoPro`) |
| **`min_match_length` per entry** | Per dataset entry | Short canonicals (e.g. `LG`) require exact match; fuzzy floor is 4 chars |
| **Word boundary check** | Exact match | Match must start/end at whitespace, punctuation, or string boundary (prevents `samsungs` → `samsung`) |

### Mask survival strategy

ByT5 byte-level models occasionally mangle special tokens. Three-tier restore handles every observed corruption mode:

| Corruption | Tier triggered | Result |
|------------|---------------|--------|
| `<<M0>>` (clean) | Tier 1 strict | Restored |
| `<<m0>>` (lowercased M) | Tier 2 lenient | Restored |
| `<<MO>>` (O instead of 0) | Tier 2 lenient | Restored |
| `<< M0 >>` (whitespace) | Tier 2 lenient | Restored |
| `<M0>` (single brackets) | Tier 2 lenient | Restored |
| `<<X0>>` (wrong letter) | Tier 3 positional | Restored by index |
| Token entirely deleted | Best-effort, logged | `mask_map` retained |

### Dataset

| File | Purpose | Approx size |
|------|---------|-------------|
| `data/brands_v2.jsonl` | Canonical brand catalog with aliases, category, fuzzy-eligibility flags | ~1,600 entries |
| `data/english_words.txt` | SCOWL Webster's 2nd + modern tech vocab — false-positive guard | ~214,000 words |
| `data/deny_list.txt` | Brands too ambiguous to mask | ~50 entries |

Each line in `brands_v2.jsonl` is a JSON object:

```json
{
  "canonical": "iPhone 15 Pro Max",
  "brand": "iPhone",
  "category": "smartphone",
  "exact_aliases": ["iphone 15 pro max", "iPhone 15 Pro Max", "IPHONE 15 PRO MAX"],
  "symspell_eligible": true,
  "min_match_length": 14
}
```

### Rebuilding the dataset

```bash
python scripts/download_scowl.py        # ~234k English words → data/english_words.txt
python scripts/build_brand_dataset.py   # ~1,600 brands → data/brands_v2.jsonl
```

`build_brand_dataset.py` applies multiple cleanups automatically:

- Drops pure-numeric (`13`, `14`) and spec-shaped tokens (`128gb`, `1080p`, `120hz`, `1920x1080`, `2.4ghz`)
- Drops common English words from seeds (unless explicitly in `CANONICAL_FORMS`)
- Drops generic non-brand words (`laptop`, `mouse`, `gaming`, `keyboard` alone)
- Drops ED-1 typo variants automatically detected against the wordlist (`samsng`, `glaxy`, `loptop`, `keybord`…)
- Re-cases multi-word entries via `CANONICAL_FORMS` + `ALL_CAPS_WORDS` (so `RTX 4090` not `Rtx 4090`, `MacBook Pro` not `Macbook Pro`)
- Deduplicates by lowercased canonical, preferring the entry with more aliases

The script reads two seed files:

- `data/brand_products.txt` — 754 brand/product lines
- `data/electronics_vocab.txt` — 928 electronics terms

…and merges them with two embedded curated datasets:

- `CANONICAL_FORMS` — ~860 lowercased term → canonical-case mapping (Apple, iPhone, MacBook, Galaxy, …)
- `CURATED_ADDITIONAL_BRANDS` — ~770 high-volume brands across electronics, fashion, beauty, food, automotive, sports, smart home

---

## Available Models

Default model: **`BYT5-Large-V3`**.

| Name | Type | Latency | Path | Brand Masking | Notes |
| ---- | ---- | ------- | ---- | :-----------: | ----- |
| `byt5-small` | Byte-level T5 | ~30 ms | `models/byt5-typo-final/` | ✓ | Lightweight |
| `byt5-base` | Byte-level T5 | ~50 ms | `models/byt5-typo-best/` | ✓ | General-purpose |
| `byt5-large` | Byte-level T5 | ~100 ms | `models/byt5-large/best/` | ✓ | More accurate |
| **`BYT5-Large-V3`** | Byte-level T5 | ~100 ms | `models/byt5-large-v3/` | ✓ | **Default — primary fine-tuned model** |
| `T5-Large-V2` | T5-Large + LoRA | ~200 ms | `models/t5-large-typo/v2/t5_correction_v2-1/` | ✓ | Standalone v2.1 |
| `T5-Large-V2.1` | T5-Large + Pipeline | ~205 ms | same as above | (own brand layer) | Full pipeline: T5 + FastText/FAISS fallback |
| `qwen-3.5-2b` | Qwen LLM | ~800 ms | `models/qwen3.5-2b/` | ✓ | Instruction-tuned |

> **Why V2.1 skips the masking layer:** it has its own `brand_lookup`-based protection in stages 0 and 3 of the internal pipeline. Wrapping it again would double-mask.

### T5-Large-V2.1 internal pipeline

```text
Input: "corsiar k70 keybord"
  │
  ├─ Stage 0: Pre-processing
  │            Normalize text, protect known brands via whitelist
  │
  ├─ Stage 1: T5-Large v2.1 primary correction (~200 ms)
  │            Context-aware, handles multi-word queries
  │
  ├─ Stage 2: FastText + FAISS fallback (~5 ms)
  │            Triggered when T5 has low confidence
  │
  └─ Stage 3: Post-processing
               Restore protected entities, validate output
               │
               ▼
Output: "corsair k70 keyboard"
        { correction_source: "t5+fasttext", suggestions: [...] }
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` covers FastAPI, PyTorch, transformers, plus the masking pipeline:

| Package | Purpose |
|---------|---------|
| `fastapi`, `uvicorn[standard]`, `pydantic` | API server |
| `torch`, `transformers`, `accelerate`, `huggingface_hub`, `sentencepiece` | ML models |
| `fasttext-wheel`, `faiss-cpu`, `numpy` | V2.1 fallback |
| `textdistance` | Evaluation |
| `pyahocorasick>=2.0.0` | **Brand masking** — Aho-Corasick exact match |
| `symspellpy>=6.7.7` | **Brand masking** — SymSpell fuzzy match |
| `english-words>=2.0.0` | **Brand masking** — false-positive guard |
| `pytest>=7.0` | Test runner |

### 2. One-time data setup

```bash
python scripts/download_scowl.py        # → data/english_words.txt
python scripts/build_brand_dataset.py   # → data/brands_v2.jsonl
```

### 3. Run the server

```bash
python api.py
# Server starts on http://localhost:5001
# Brand masking pipeline boots automatically (eager init)
```

Or use the convenience script on Windows:

```bat
run_api.bat
```

### 4. Test the masking pipeline

```bash
pytest tests/ -v
# 60 unit + integration tests, ~1 s
```

### 5. Test correction across all models

```bash
python test_all_models.py     # full benchmark — produces benchmark_results.json
```

### 6. Optional — open the demo web UI

Open `web_ui.html` in a browser. It posts directly to `http://localhost:5001/correct` with a model dropdown.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service status JSON |
| `GET` | `/health` | Health check (`{"status": "healthy"}`) |
| `POST` | `/correct` | Correct a search query (with brand masking) |
| `GET` | `/models` | List available models with `loaded` status |

### `POST /correct`

**Request:**
```json
{ "query": "samsng glaxy s24 broekn", "model": "BYT5-Large-V3" }
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `query` | string | ✓ | The user's raw search text |
| `model` | string | ✗ | Model name from the registry. Defaults to `BYT5-Large-V3`. Unknown values fall back to default. |

**Response (200 OK):**
```json
{
  "original_query": "samsng glaxy s24 broekn",
  "corrected_query": "Samsung Galaxy S24 broken",
  "changed": true,
  "model_used": "BYT5-Large-V3",
  "latency_ms": 102.4,
  "correction_source": "",
  "suggestions": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `original_query` | string | The unmodified user input |
| `corrected_query` | string | Final text — masking restored, model corrections applied |
| `changed` | bool | True iff `corrected_query.lower().strip() != original_query.lower().strip()` |
| `model_used` | string | The model that ran (resolved name) |
| `latency_ms` | float | Inference latency in milliseconds |
| `correction_source` | string | Populated by V2.1 pipeline: `"t5"`, `"fasttext"`, `"t5+fasttext"`, `""` |
| `suggestions` | list[string] | "Did you mean …?" candidates from V2.1 fallback |

### `GET /models`

Response shape mirrors the Retrieval service for consistency:

```json
{
  "status": "success",
  "data": {
    "correction_models": [
      {
        "name": "BYT5-Large-V3",
        "description": "ByT5-Large v3 fine-tuned",
        "architecture": "ByT5 (encoder-decoder, byte-level)",
        "type": "seq2seq",
        "loaded": true,
        "default": true
      },
      ...
    ],
    "defaults": { "correction": "BYT5-Large-V3" }
  }
}
```

### Example with curl

```bash
curl -X POST http://localhost:5001/correct \
  -H "Content-Type: application/json" \
  -d '{"query": "iphne 15 pro mxa with samsng glaxy", "model": "BYT5-Large-V3"}'
```

The brand layer recovers `iPhone` and `Samsung Galaxy` (canonical case), while the model fixes `mxa` → `max`.

---

## Project Structure

```text
Correction/
├── api.py                          # FastAPI server (port 5001)
├── run_api.bat                     # Windows convenience launcher
├── test_all_models.py              # Full-model benchmark suite
├── benchmark_results.json          # Most recent benchmark snapshot
├── web_ui.html                     # Standalone browser demo
├── msg_filter.py                   # Tiny logging filter
├── requirements.txt                # All dependencies
├── README.md                       # This file
│
├── app/
│   ├── __init__.py                 # Re-exports TypoCorrector
│   ├── corrector.py                # ★ Main orchestrator (default: BYT5-Large-V3)
│   │                               #   wraps every model with masking layer
│   ├── corrector_v2.py             # Optional brand-lookup-aware orchestrator
│   ├── corrector_v3.py             # Legacy V3 standalone pipeline
│   ├── brand_lookup.py             # O(1) brand/abbreviation/unit protection (V2.1)
│   ├── domain_vocab.py             # Vocabulary loader
│   ├── metrics.py                  # Evaluation metrics
│   │
│   ├── masking/                    # ★ Brand masking pipeline (NEW)
│   │   ├── __init__.py             #   Public API: MaskingPipeline, MaskingConfig
│   │   ├── tokenizer.py            #   Whitespace + punct tokenizer with offsets
│   │   ├── mask_format.py          #   <<M0>> token format + 3-tier restoration
│   │   ├── dataset.py              #   brands_v2.jsonl + deny_list.txt loader
│   │   ├── english_filter.py       #   SCOWL wordlist false-positive guard
│   │   ├── exact_matcher.py        #   Aho-Corasick exact matcher (longest-match-first)
│   │   ├── fuzzy_matcher.py        #   SymSpell fuzzy matcher (adaptive ED, guards)
│   │   └── pipeline.py             #   Orchestrator: tokenize → match → mask/unmask
│   │
│   └── models/
│       ├── __init__.py
│       ├── base.py                 # Abstract BaseCorrector
│       ├── byt5.py                 # ByT5 wrapper (small/base/large/V3)
│       ├── t5_large.py             # T5-Large wrapper (INT8 + LoRA capable)
│       ├── t5_pipeline.py          # T5-Large + FastText/FAISS pipeline (V2.1)
│       ├── fasttext_fallback.py    # FastText + FAISS word-level corrector
│       └── qwen.py                 # Qwen 3.5 2B wrapper
│
├── data/
│   ├── brands_v2.jsonl             # ★ Brand catalog (1,600+ entries) — masking input
│   ├── english_words.txt           # ★ English wordlist (~214k) — FP guard
│   ├── deny_list.txt               # ★ Brands too ambiguous to mask (~50)
│   ├── brand_products.txt          # 754 brand/product seed terms
│   ├── electronics_vocab.txt       # 928 electronics seed terms
│   ├── typo_mappings.txt           # 790 curated typo → correct pairs
│   ├── typo_dataset.csv            # Tabular typo dataset
│   ├── domain_vocab.txt            # Domain vocabulary
│   ├── ecommerce_corpus.txt        # FastText training corpus
│   ├── curated_common_words.txt    # Common-word list
│   ├── expanded_common_words.txt   # Larger common-word list
│   ├── common_words.txt            # Big common-word file
│   └── nltk_words.txt              # NLTK words backup
│
├── fine_tune/                      # Training pipelines
│   ├── BYT5-T5 Large v3/           # ★ V3 training notebooks & data
│   ├── t5_v3/                      # V3 dataset builders
│   ├── t5-large-v2-1/              # V2.1 (204,995 train examples)
│   ├── data/                       # Shared training data
│   ├── real_data/                  # Real-world examples
│   └── build_notebook.py           # Notebook builder
│
├── models/                         # Model weights (gitignored)
│   ├── byt5-typo-final/            # ByT5-small
│   ├── byt5-typo-best/             # ByT5-base
│   ├── byt5-large/best/            # ByT5-large
│   ├── byt5-large-v3/              # ★ ByT5-Large V3 (default model)
│   ├── t5-large-typo/v2/           # T5-Large v2.1
│   ├── qwen3.5-2b/                 # Qwen 3.5 2B
│   ├── fasttext-electronics.bin    # FastText fallback model
│   └── fasttext_corpus.txt         # FastText training corpus
│
├── scripts/
│   ├── build_brand_dataset.py      # ★ Build brands_v2.jsonl with cleanup
│   ├── download_scowl.py           # ★ Generate english_words.txt
│   └── train_fasttext.py           # FastText fallback training
│
└── tests/                          # ★ Unit + integration tests (60 cases, ~1 s)
    ├── conftest.py                 # Fixtures (isolated_tmp, app stub)
    ├── __init__.py
    ├── test_tokenizer.py           # 8 tests
    ├── test_mask_format.py         # 9 tests
    ├── test_english_filter.py      # 4 tests
    ├── test_exact_matcher.py       # 8 tests
    ├── test_fuzzy_matcher.py       # 9 tests
    ├── test_pipeline.py            # 18 tests (mask/unmask roundtrips)
    └── test_corrector_integration.py  # 4 tests (with mocked model)
```

★ marks files added or significantly modified by the brand-masking work.

---

## Code Examples

### Direct use of `TypoCorrector`

```python
from app.corrector import TypoCorrector

corrector = TypoCorrector()
# Brand masking pipeline boots eagerly during __init__()

# Simple — returns just the corrected string
text = corrector.correct_query("iphnoe 15 pro")
# → "iPhone 15 pro"  (canonical case preserved)

# With explicit model selection
text = corrector.correct_query("samsng galaxy s24", model="BYT5-Large-V3")
# → "Samsung Galaxy s24"

# Full result with metadata
result = corrector.correct("corsiar k70 keybord", model="BYT5-Large-V3")
# {
#   "original_query":  "corsiar k70 keybord",
#   "corrected_query": "Corsair K70 keyboard",
#   "changed":         True,
#   "model_used":      "BYT5-Large-V3",
#   "latency_ms":      102.4,
# }

# Batch correction with aggregate stats
batch = corrector.correct_batch(
    ["iphne 15", "samsng s24", "macbok pro"],
    model="BYT5-Large-V3",
)
# {
#   "results": [...],
#   "batch_stats": {
#     "total_queries": 3,
#     "total_latency_ms": 312.5,
#     "avg_latency_ms": 104.2,
#   },
# }
```

### Direct use of `MaskingPipeline`

```python
from app.masking import MaskingPipeline, MaskingConfig

pipe = MaskingPipeline()

# Mask
masked, mask_map = pipe.mask("iphne and samsng galaxy")
# → ("<<M0>> and <<M1>>",
#    {"<<M0>>": "iPhone", "<<M1>>": "Samsung Galaxy"})

# (your model would correct `masked_query` here)

# Unmask — survives realistic corruption modes
restored = pipe.unmask("<<m0>> and << M1 >>", mask_map)
# → "iPhone and Samsung Galaxy"

# Diagnostics
print(pipe.stats())
# {'healthy': True, 'exact_patterns': 1718, 'fuzzy_enabled': True, 'fuzzy_dict': 1025}

# Custom configuration (e.g. point at a different dataset)
from pathlib import Path
custom = MaskingConfig(
    brands_path=Path("/tmp/my_brands.jsonl"),
    english_words_path=Path("/tmp/my_english.txt"),
    deny_list_path=Path("/tmp/my_deny.txt"),
    enable_fuzzy=True,
)
custom_pipe = MaskingPipeline(custom)
```

### Smoke-test the live server

```python
import requests

r = requests.post(
    "http://localhost:5001/correct",
    json={"query": "iphne and samsng galxy s24 ultra", "model": "BYT5-Large-V3"},
    timeout=10,
)
print(r.json()["corrected_query"])
# → "iPhone and Samsung Galaxy S24 Ultra"
```

---

## Backend Integration

This service is consumed by `Backend/services/text_corrector_service.py` (Flask). The contract is **stable** — the brand masking layer is invisible to the backend.

### Contract

| Aspect | Value |
|--------|-------|
| URL | `http://127.0.0.1:5001/correct` (configurable via `.env`) |
| Method | `POST` |
| Request | `{ "query": str, "model": str (optional) }` |
| Response (fields backend reads) | `corrected_query`, `changed`, `latency_ms`, `model_used` |

### Backend's request flow (`TextCorrectorService.correct()`)

```python
# Backend maps frontend engine names to internal model identifiers
engine_map = {
    'symspell': ENGINE_SYMSPELL,
    'symspell_keyboard': ENGINE_SYMSPELL,
    'byt5': ENGINE_BYT5,        # ENGINE_BYT5 = 'byt5-small'
    'byt5-base': ENGINE_BYT5,
    'byt5-small': 'byt5-small',
    'byt5-large': 'byt5-large',
    'qwen-3.5-2b': 'qwen-3.5-2b',
}
correction_engine = engine_map.get(engine, engine)  # passthrough for unknown
```

> Unknown engines (like `BYT5-Large-V3`) pass through unchanged via the `.get(engine, engine)` fallback. No backend code change is required to support new models; they appear in the dropdown when `/models` is called.

### Graceful degradation

If the Correction service is offline, the backend's `requests.post(...)` raises `ConnectionError`, the service logs a warning, and returns `{corrected_text: raw_text, success: False, ...}`. Search still works — just without spell correction.

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| **`BYT5-Large-V3` as default** | Best fine-tune accuracy/latency tradeoff for our domain; brand masking compensates for ByT5's case sensitivity |
| **Brand masking is model-agnostic** | Persistent infrastructure; new models drop in without touching this layer |
| **Aho-Corasick over Levenshtein scan** | O(n+m+z) lookup; scales to 50k+ patterns vs O(k·n) of brute-force fuzzy |
| **SymSpell over BK-trees** | Faster for ED ≤ 2; precomputed delete dictionary, prefix_length=7 |
| **English-word filter on input only** | Old wordlists contain brands like Razer/Bose/Adobe; filtering canonicals would silently drop major brands |
| **Conservative ED (V1: ED ≤ 1)** | False positives corrupt user queries irrecoverably; false negatives are recoverable by the model itself |
| **3-tier mask restoration** | Even byte-level model corruption of `<<M0>>` is recoverable by lenient regex + positional fallback |
| **Skip masking for V2.1** | V2.1 has its own `brand_lookup` protection in stages 0/3 — avoid double-masking |
| **Eager pipeline boot** | Aho-Corasick build (~5 s on 1,600 patterns) runs once at startup, not per-request |
| **Spec-pattern filter on dataset build** | `128gb`/`1080p` etc. are not brands — drop at build time, not at runtime |
| **ED-1 typo auto-detection in build** | Catches dataset pollution like `samsng`, `glaxy`, `keybord` without manual blacklist maintenance |
| **FastText for V2.1 fallback** | Subword n-grams handle brand typos, <5 ms, no GPU |
| **Whitelist protection (legacy)** | Used by V2.1 stages — prevents `razer → razor` |

---

## Tests & Benchmarks

### Unit + integration tests

```bash
pytest tests/ -v
```

60 tests across 7 files — all green:

| File | Tests | What it covers |
|------|-------|----------------|
| `test_tokenizer.py` | 8 | Whitespace splitting, offset tracking, punctuation handling |
| `test_mask_format.py` | 9 | `make_mask`, strict/lenient/positional regex restoration |
| `test_english_filter.py` | 4 | Wordlist load, case-insensitivity, missing-file fallback |
| `test_exact_matcher.py` | 8 | Aho-Corasick longest-match, word boundaries, deny list |
| `test_fuzzy_matcher.py` | 9 | SymSpell guards: typo / English-word / deny / numeric / ratio |
| `test_pipeline.py` | 18 | End-to-end mask/unmask roundtrips, including corrupted-mask survival |
| `test_corrector_integration.py` | 4 | `TypoCorrector` with mocked model — verifies BYT5-Large-V3 goes through masking, T5-V2.1 skips |

### Full-model benchmarks

```bash
python test_all_models.py
```

Loads every model in the registry and runs a curated test set; writes results to `benchmark_results.json`.

---

## Tech Stack

- **Python** 3.10+
- **FastAPI** + **Uvicorn** (HTTP server)
- **PyTorch** 2.0+ + **Transformers** (model inference)
- **pyahocorasick** + **symspellpy** (brand masking matchers)
- **english-words** (FP-guard wordlist; falls back to NLTK)
- **FastText** (`fasttext-wheel`) + **FAISS** (`faiss-cpu`) (V2.1 fallback)
- **textdistance**, **pytest** (eval + tests)
- **GPU**: RTX 5070 Ti (local dev)

---

## Authors

SmartSearch Graduation Team — Correction module.
