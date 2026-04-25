# SmartSearch — Correction Microservice

Real-time spelling correction for e-commerce search queries. Part of the **SmartSearch** graduation project (4 microservices).

Built with **FastAPI** and multiple ML models — from lightweight byte-level transformers to full correction pipelines with FastText/FAISS fallback. Every model call is wrapped by a **brand masking pipeline** that protects brand and product-model identifiers (iPhone, Samsung Galaxy, RTX 4090) from being corrupted by the correction model.

## Architecture

```text
                       ┌────────────────────────────┐
                       │  Frontend (Dropdown)       │
                       │  Select correction model   │
                       └──────────────┬─────────────┘
                                      │
                                      ▼
                       ┌────────────────────────────┐
                       │  POST /correct             │
                       │  { query, model }          │
                       └──────────────┬─────────────┘
                                      │
                                      ▼
                       ┌────────────────────────────┐
                       │  Brand Masking (in)        │  ← model-agnostic layer
                       │  Aho-Corasick + SymSpell   │     (skipped for V2.1)
                       │  iphone → <<M0>>           │
                       └──────────────┬─────────────┘
                                      │
            ┌──────────┬──────────────┼──────────────┬──────────┬──────────┐
            ▼          ▼              ▼              ▼          ▼          ▼
       ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌──────────┐ ┌────────┐ ┌────────┐
       │ ByT5    │ │ ByT5    │ │ ByT5-Large  │ │ T5-Large │ │ T5-V2.1│ │ Qwen   │
       │ small   │ │ base *  │ │ V3          │ │ V2       │ │ pipe.  │ │ 3.5 2B │
       └─────────┘ └─────────┘ └─────────────┘ └──────────┘ └────────┘ └────────┘
                                                                 │
                                                          ┌──────┴──────┐
                                                          │ FastText +  │
                                                          │ FAISS       │
                                                          └─────────────┘
                                      │
                                      ▼
                       ┌────────────────────────────┐
                       │  Brand Masking (out)       │
                       │  <<M0>> → iPhone           │
                       │  3-tier failsafe restore   │
                       └──────────────┬─────────────┘
                                      ▼
                                 to backend

  * default model
```

## Brand Masking Pipeline

A **model-agnostic, persistent layer** that runs before and after every correction model. It guarantees that correctly-spelled (or barely-typo'd) brand names like "iPhone", "Samsung Galaxy S24 Ultra", "Razer Blade", or "RTX 4090" are never corrupted by the underlying byte-level model. The masking layer survives model swaps — when you replace ByT5-Large-V3 with a future model, this protection stays intact.

### How it works

```text
Input:  "samsng glaxy s24 ultra is the best, beats iphone"

Stage 1 — Mask
  ExactMatcher (Aho-Corasick) finds "iphone" → <<M0>>
  FuzzyMatcher (SymSpell) finds "samsng" → Samsung, "glaxy" → Galaxy
  Multi-word match: "samsng glaxy s24 ultra" → <<M1>>
  Masked text: "<<M1>> is the best, beats <<M0>>"
  Mask map: {<<M0>>: "iPhone", <<M1>>: "Samsung Galaxy S24 Ultra"}

Stage 2 — Model (any of ByT5/T5/Qwen)
  Sees masked text, corrects the rest. Mask tokens pass through.

Stage 3 — Unmask
  3-tier failsafe restoration:
    1. Strict regex     (<<M0>> exact)
    2. Lenient regex    (<<m0>>, <<MO>>, << M0 >>, <M0>)
    3. Positional fallback (count remaining shapes, restore by order)

Output: "Samsung Galaxy S24 Ultra is the best, beats iPhone"
```

### False-positive guards

False positives — masking a real English word as a brand — corrupt the user's query and are not recoverable. So the matcher is conservative by design:

- **English-wordlist guard**: input tokens that are real English words (`apply`, `keyboard`) are never flagged as brand typos.
- **Deny list**: brands too ambiguous (`apple`, `shell`, `tide`, `gap`, `blade`, `max`) are excluded entirely.
- **Adaptive edit distance**: ED 0 for tokens shorter than 4 chars, ED 1 otherwise. ED 2 deferred to V2.
- **Length-ratio sanity**: rejects matches where input/canonical length ratio < 0.6 or > 1.6.
- **`min_match_length` per entry**: short canonicals (e.g. "LG") require exact match only.

### Dataset

| File | Purpose | Size |
|------|---------|------|
| `data/brands_v2.jsonl` | Canonical brand catalog with aliases, category, fuzzy-eligibility flags | ~1,600 entries |
| `data/english_words.txt` | SCOWL Webster's 2nd + modern tech vocab — false-positive guard | ~234,500 words |
| `data/deny_list.txt` | Brands too ambiguous to mask | ~50 entries |

Rebuild from seeds:

```bash
python scripts/download_scowl.py        # generates english_words.txt
python scripts/build_brand_dataset.py   # generates brands_v2.jsonl
```

The build script applies multiple cleanups: drops pure-numeric and spec-shaped tokens (`128gb`, `1080p`), drops common English words from seeds (unless explicitly in `CANONICAL_FORMS`), drops generic non-brand words (`laptop`, `mouse`), drops ED-1 typo variants automatically detected against the wordlist, and re-cases multi-word entries (`RTX 4090` not `Rtx 4090`).

## Available Models

| Model | Type | Latency | Brand Masking | Description |
| ----- | ---- | ------- | ------------- | ----------- |
| `byt5-small` | Byte-level T5 | ~30ms | ✓ | Lightweight variant |
| `byt5-base` | Byte-level T5 | ~50ms | ✓ | **Default.** General-purpose correction |
| `byt5-large` | Byte-level T5 | ~100ms | ✓ | Heavier, more accurate |
| `BYT5-Large-V3` | Byte-level T5 | ~100ms | ✓ | **Primary model** — fine-tuned ByT5-Large v3 |
| `T5-Large-V2` | T5-Large | ~200ms | ✓ | Standalone v2.1 fine-tuned |
| `T5-Large-V2.1` | T5-Large + Pipeline | ~205ms | (own brand layer) | Full pipeline: T5 + FastText/FAISS fallback |
| `qwen-3.5-2b` | LLM | ~800ms | ✓ | Qwen 3.5 2B instruction-tuned |

> The masking layer is skipped only for `T5-Large-V2.1` because that pipeline already runs its own `brand_lookup`-based protection in stages 0/3.

### T5-Large-V2.1 Pipeline

Activated when the user selects `T5-Large-V2.1`:

```text
Input: "corsiar k70 keybord"
  │
  ├─ Stage 0: Pre-processing
  │            Normalize text, protect known brands via whitelist
  │
  ├─ Stage 1: T5-Large v2.1 primary correction (~200ms)
  │            Context-aware, handles multi-word queries
  │
  ├─ Stage 2: FastText + FAISS fallback (~5ms)
  │            Only if T5 fails or has low confidence
  │
  └─ Stage 3: Post-processing
               Restore protected entities, validate output
               │
               ▼
Output: "corsair k70 keyboard"
        { correction_source: "t5+fasttext", suggestions: [...] }
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

New brand-masking dependencies (already in `requirements.txt`):

- `pyahocorasick` — Aho-Corasick exact multi-pattern matcher
- `symspellpy` — fuzzy match for typo'd brand names
- `english-words` — wordlist for false-positive guard
- `pytest` — test runner

### One-time data setup

```bash
python scripts/download_scowl.py        # ~234k English words → data/english_words.txt
python scripts/build_brand_dataset.py   # ~1,600 brands → data/brands_v2.jsonl
```

### Run the Server

```bash
python api.py
# Server starts on http://localhost:5001
# Brand masking pipeline boots automatically (eager init)
```

### Run the test suite

```bash
pytest tests/ -v
# 60 unit + integration tests, ~1s
```

### API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check |
| `POST` | `/correct` | Correct a search query (with brand masking) |
| `GET` | `/models` | List available models with loaded status |

### Example Request

```bash
curl -X POST http://localhost:5001/correct \
  -H "Content-Type: application/json" \
  -d '{"query": "samsng glaxy s24 broekn", "model": "BYT5-Large-V3"}'
```

### Example Response

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

Notice that `samsng glaxy` (typo'd brand) is restored to canonical case `Samsung Galaxy`, while `broekn` (typo'd common word) is fixed by the model. The brand layer and the model cooperate without stepping on each other.

## Project Structure

```text
Correction/
├── api.py                          # FastAPI server (port 5001)
├── app/
│   ├── corrector.py                # Main orchestrator + masking integration
│   ├── corrector_v2.py             # V2 orchestrator (brand_lookup support)
│   ├── corrector_v3.py             # V3 standalone pipeline (legacy)
│   ├── brand_lookup.py             # O(1) brand/abbreviation/unit protection (V2.1)
│   ├── domain_vocab.py             # Vocabulary loader
│   ├── metrics.py                  # Evaluation metrics
│   ├── masking/                    # ★ Brand masking pipeline (new)
│   │   ├── __init__.py             #   Public API: MaskingPipeline
│   │   ├── tokenizer.py            #   Whitespace + punct tokenizer
│   │   ├── mask_format.py          #   <<M0>> tokens + 3-tier restoration
│   │   ├── dataset.py              #   brands_v2.jsonl loader
│   │   ├── english_filter.py       #   SCOWL false-positive guard
│   │   ├── exact_matcher.py        #   Aho-Corasick exact matcher
│   │   ├── fuzzy_matcher.py        #   SymSpell fuzzy matcher
│   │   └── pipeline.py             #   Orchestrator: tokenize → match → mask/unmask
│   └── models/
│       ├── base.py                 # Abstract BaseCorrector
│       ├── byt5.py                 # ByT5 corrector (small/base/large/V3)
│       ├── t5_large.py             # T5-Large corrector (INT8 + LoRA)
│       ├── t5_pipeline.py          # T5-Large + FastText/FAISS pipeline
│       ├── fasttext_fallback.py    # FastText + FAISS word-level corrector
│       └── qwen.py                 # Qwen 3.5 2B
├── data/
│   ├── brands_v2.jsonl             # ★ Brand catalog (1,600+ entries)
│   ├── english_words.txt           # ★ English wordlist for FP guard (~234k)
│   ├── deny_list.txt               # ★ Brands too ambiguous to mask
│   ├── typo_mappings.txt           # 790 curated typo → correct pairs
│   ├── brand_products.txt          # 754 brand/product seed terms
│   ├── electronics_vocab.txt       # 928 electronics seed terms
│   ├── domain_vocab.txt            # Domain vocabulary
│   └── curated_common_words.txt    # Common word list
├── fine_tune/                      # Training pipelines & data
│   ├── t5-large-v2-1/              # V2.1 (204,995 train examples)
│   └── BYT5-T5 Large v3/           # V3 training pipelines
├── models/                         # Model weights (gitignored)
│   ├── byt5-typo-best/             # ByT5-base
│   ├── byt5-typo-final/            # ByT5-small
│   ├── byt5-large/best/            # ByT5-large
│   ├── byt5-large-v3/              # ★ ByT5-Large V3 (primary)
│   ├── t5-large-typo/v2/           # T5-Large v2.1
│   └── qwen3.5-2b/                 # Qwen 3.5 2B
├── scripts/
│   ├── build_brand_dataset.py      # ★ Build brands_v2.jsonl from seeds
│   ├── download_scowl.py           # ★ Generate english_words.txt
│   └── train_fasttext.py           # FastText fallback training
└── tests/                          # ★ Unit + integration tests (60 cases)
    ├── conftest.py
    ├── test_tokenizer.py
    ├── test_mask_format.py
    ├── test_dataset.py
    ├── test_english_filter.py
    ├── test_exact_matcher.py
    ├── test_fuzzy_matcher.py
    ├── test_pipeline.py
    └── test_corrector_integration.py
```

## Usage in Code

```python
from app.corrector import TypoCorrector

corrector = TypoCorrector()
# Brand masking pipeline boots eagerly during __init__()

# Simple — returns corrected string
text = corrector.correct_query("iphnoe 15 pro")
# → "iPhone 15 pro"   (brand canonical preserved)

# With model selection
text = corrector.correct_query("samsng galaxy s24", model="BYT5-Large-V3")
# → "Samsung Galaxy s24"

# Full result with metadata
result = corrector.correct("corsiar k70 keybord", model="BYT5-Large-V3")
# → { "original_query":  "corsiar k70 keybord",
#     "corrected_query": "Corsair K70 keyboard",
#     "changed":         True,
#     "model_used":      "BYT5-Large-V3",
#     "latency_ms":      102.4 }

# Direct access to the masking layer (for diagnostics)
masked, mask_map = corrector._mask_pipeline.mask("iphne and samsng")
# → ("<<M0>> and <<M1>>", {"<<M0>>": "iPhone", "<<M1>>": "Samsung"})

restored = corrector._mask_pipeline.unmask(masked, mask_map)
# → "iPhone and Samsung"
```

## Design Decisions

| Decision | Reasoning |
| -------- | --------- |
| **`byt5-base` as default** | Fast, reliable, handles diverse e-commerce queries |
| **`BYT5-Large-V3` as primary** | Higher-quality byte-level corrections; brand masking compensates for its case sensitivity |
| **Brand masking is model-agnostic** | Persistent infrastructure; survives model swaps |
| **Aho-Corasick over Levenshtein** | O(n + m) lookup; scales to 50k+ patterns vs O(k·n) of brute-force fuzzy |
| **SymSpell over BK-trees** | Faster for ED ≤ 2; precomputed delete dictionary |
| **English-word filter on input only** | Old wordlists contain brands like Razer/Bose; filtering canonicals would lose them |
| **Conservative ED (V1: ED ≤ 1)** | False positives corrupt user queries irrecoverably; false negatives are recoverable by the model |
| **3-tier mask restoration** | Even byte-level model corruption of `<<M0>>` is recoverable by lenient regex + positional fallback |
| **Skip masking for V2.1** | V2.1 has its own `brand_lookup` protection in stages 0/3 — avoid double-masking |
| **FastText for fallback** | Subword n-grams handle brand typos, <5ms, no GPU |

## Backend Integration

This service is consumed by `Backend/services/text_corrector_service.py`. The contract:

- **Endpoint**: `POST /correct` on port 5001
- **Request**: `{ "query": str, "model": str (optional) }`
- **Response** (backend reads these fields): `corrected_query`, `changed`, `latency_ms`, `model_used`

The backend uses `requests.get(..., default)` everywhere, so adding new optional response fields never breaks it. If the correction service is offline, the backend silently falls back to the original query — search still works, just without spell correction.

## Tech Stack

- **Python 3.10+**
- **FastAPI** + Uvicorn
- **PyTorch 2.0+** / Transformers
- **pyahocorasick** + **symspellpy** (brand masking)
- **FastText** + **FAISS** (V2.1 fallback pipeline)
- **GPU**: RTX 5070 Ti (local dev)

## Authors

SmartSearch Graduation Team
