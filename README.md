# SmartSearch - Correction Microservice

Real-time spelling correction for e-commerce search queries. Part of the **SmartSearch** graduation project (4 microservices).

Built with **FastAPI** and multiple ML models -- from lightweight byte-level transformers to full correction pipelines with FastText/FAISS fallback.

## Architecture

```text
                         ┌─────────────────────────┐
                         │   Frontend (Dropdown)    │
                         │   Select correction model│
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  POST /correct           │
                         │  { query, model }        │
                         └────────────┬────────────┘
                                      │
              ┌───────────┬───────────┼───────────┬────────────┐
              ▼           ▼           ▼           ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐
         │ ByT5    │ │ ByT5    │ │ ByT5    │ │ T5-Large │ │ Qwen   │
         │ small   │ │ base *  │ │ large   │ │ V2 / V2.1│ │ 3.5 2B │
         └─────────┘ └─────────┘ └─────────┘ └──────────┘ └────────┘
                                                  │
                                          (V2.1 only)
                                                  │
                                    ┌─────────────┴──────────────┐
                                    │  FastText + FAISS Fallback  │
                                    │  Word-level nearest neighbor│
                                    │  ~5ms, no GPU needed        │
                                    └─────────────────────────────┘

  * default model
```

## Available Models

| Model | Type | Latency | Description |
| ----- | ---- | ------- | ----------- |
| `byt5-base` | Byte-level T5 | ~50ms | **Default.** Reliable general-purpose correction |
| `byt5-small` | Byte-level T5 | ~30ms | Lighter, faster variant |
| `byt5-large` | Byte-level T5 | ~100ms | Heavier, more accurate |
| `T5-Large-V2` | T5-Large | ~200ms | Standalone v2.1 fine-tuned correction |
| `T5-Large-V2.1` | T5-Large + Pipeline | ~205ms | Full pipeline: T5 + FastText/FAISS fallback |
| `qwen-3.5-2b` | LLM | ~800ms | Qwen 3.5 2B instruction-tuned |

### T5-Large-V2.1 Pipeline

The most advanced option -- a 4-stage pipeline activated when the user selects `T5-Large-V2.1`:

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

### Run the Server

```bash
python api.py
# Server starts on http://localhost:5001
```

### API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/` | Service status |
| `GET` | `/health` | Health check |
| `POST` | `/correct` | Correct a search query |
| `GET` | `/models` | List available models with loaded status |

### Example Request

```bash
curl -X POST http://localhost:5001/correct \
  -H "Content-Type: application/json" \
  -d '{"query": "corsiar k70 keybord", "model": "byt5-base"}'
```

### Example Response

```json
{
  "original_query": "corsiar k70 keybord",
  "corrected_query": "corsair k70 keyboard",
  "changed": true,
  "model_used": "byt5-base",
  "latency_ms": 48.2,
  "correction_source": "",
  "suggestions": []
}
```

## Project Structure

```text
Correction/
├── api.py                          # FastAPI server (port 5001)
├── app/
│   ├── corrector.py                # Main orchestrator & model registry
│   ├── corrector_v2.py             # V2 orchestrator (brand lookup support)
│   ├── brand_lookup.py             # O(1) brand/abbreviation/unit protection
│   ├── domain_vocab.py             # Vocabulary loader
│   ├── metrics.py                  # Evaluation metrics
│   └── models/
│       ├── base.py                 # Abstract BaseCorrector
│       ├── byt5.py                 # ByT5 corrector (small/base/large)
│       ├── t5_large.py             # T5-Large corrector (INT8 + LoRA)
│       ├── t5_pipeline.py          # T5-Large + FastText/FAISS pipeline
│       ├── fasttext_fallback.py    # FastText + FAISS word-level corrector
│       └── qwen.py                 # Qwen 3.5 2B
├── data/
│   ├── typo_mappings.txt           # 790 curated typo → correct pairs
│   ├── brand_products.txt          # 754 brand/product terms
│   ├── electronics_vocab.txt       # 928 electronics terms
│   ├── domain_vocab.txt            # Domain vocabulary
│   └── curated_common_words.txt    # Common word list
├── fine_tune/                      # Training pipelines & data
│   └── t5-large-v2-1/             # V2.1 (204,995 train examples)
├── models/                         # Model weights (gitignored)
│   ├── byt5-typo-best/            # ByT5-base
│   ├── byt5-typo-final/           # ByT5-small
│   ├── byt5-large/best/           # ByT5-large
│   ├── t5-large-typo/v2/          # T5-Large v2.1
│   └── qwen3.5-2b/               # Qwen 3.5 2B
└── scripts/                        # Utility scripts
```

## Usage in Code

```python
from app.corrector import TypoCorrector

corrector = TypoCorrector()

# Simple -- returns corrected string
corrected = corrector.correct_query("iphnoe 15 pro")
# → "iphone 15 pro"

# With model selection
corrected = corrector.correct_query("samsng galaxy s24", model="T5-Large-V2")
# → "samsung galaxy s24"

# Full result with metadata
result = corrector.correct("corsiar k70 keybord", model="byt5-base")
# → { "original_query": "corsiar k70 keybord",
#     "corrected_query": "corsair k70 keyboard",
#     "changed": True, "model_used": "byt5-base", "latency_ms": 48.2 }
```

## Design Decisions

| Decision | Reasoning |
| -------- | --------- |
| **ByT5-base as default** | Fast, reliable, handles diverse e-commerce queries well |
| **SymSpell rejected** | Poor performance on brand names |
| **Gemma 4 quantized rejected** | Too slow for real-time search |
| **FastText for fallback** | Subword n-grams handle brand typos naturally, <5ms, no GPU |
| **Whitelist protection** | Prevents false corrections (e.g. "razer" is not "razor") |
| **High similarity threshold (0.90)** | "Don't touch if unsure" -- avoids false positives |

## Tech Stack

- **Python 3.11+**
- **FastAPI** + Uvicorn
- **PyTorch 2.0+** / Transformers
- **FastText** + FAISS (fallback pipeline)
- **GPU**: RTX 5070 Ti (local dev)

## Authors

SmartSearch Graduation Team
