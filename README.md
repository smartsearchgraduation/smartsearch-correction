# SmartSearch Typo Correction Module# SmartSearch Typo Correction Module



Production-ready e-commerce search query spell correction with multiple model support including fine-tuned LLM fallback models.Production-ready e-commerce search query spell correction with multiple model support.



## 🏗️ Project Structure## 🚀 Performance (v2.0)



```| Model | Accuracy | Latency | Use Case |

ss-correction-model/|-------|----------|---------|----------|

├── api.py                  # FastAPI endpoint| **SymSpell** | 96.4% | ~0.1ms | Primary - Ultra-fast |

├── requirements.txt        # Main dependencies| **Keyboard** | 89.3% | ~5ms | Adjacent key errors |

│| **E5_ONNX** | 90%+ | ~10ms | Semantic similarity |

├── app/                    # Core correction module| **Mistral** | 100% | ~1000ms | Backup - LLM fallback |

│   ├── typo_corrector.py   # Main TypoCorrector class

│   ├── domain_vocab.py     # Domain-specific vocabulary## 📦 Installation

│   └── metrics.py          # Evaluation metrics

│```bash

├── data/                   # Vocabulary & mappingspip install -r requirements.txt

│   ├── typo_mappings.txt   # Explicit typo→correct mappings```

│   ├── brand_products.txt  # Brand & product names

│   ├── electronics_vocab.txt## 🔧 API Usage

│   └── domain_vocab.txt

│### Simple API (Recommended for Backend)

├── fine_tune/              # LLM Fine-tuning pipeline

│   ├── prepare_data.py     # Training data generator```python

│   ├── ecommerce_vocab.py  # E-commerce vocabulary builderfrom app.typo_corrector import TypoCorrector, CorrectionModel

│   ├── models/             # Fine-tune scripts

│   │   ├── llama_finetune.pycorrector = TypoCorrector()

│   │   ├── qwen_finetune.py

│   │   └── byt5_finetune.py# Just get corrected string

│   └── README.md           # Fine-tuning guidecorrected = corrector.correct_query("iphnoe 15 pro")

│# Returns: "iphone 15 pro"

├── models/                 # Embedding models

│   └── e5-small-v2-onnx/# With model selection (frontend/admin choice)

│corrected = corrector.correct_query("samsng galaxy", model="symspell")

└── tests/                  # Test files```

    ├── test_models.py

    └── test_spell_correction.py### Full API (With Metadata)

```

```python

## 🚀 Performance Summaryresult = corrector.correct("iphnoe 15 pro", model=CorrectionModel.SYMSPELL)

# Returns:

### Primary Models (Fast)# {

#     "original_query": "iphnoe 15 pro",

| Model | Accuracy | Latency | Use Case |#     "normalized_query": "iphone 15 pro",

|-------|----------|---------|----------|#     "changed": True,

| **SymSpell** | 96.4% | ~0.1ms | Primary - Ultra-fast |#     "latency_ms": 0.1

| **Keyboard** | 89.3% | ~5ms | Adjacent key errors |# }

| **E5_ONNX** | 90%+ | ~10ms | Semantic similarity |```



### Fine-tuned Fallback Models (For zero-retrieval cases)### Available Models



| Model | Accuracy | Latency | Size |```python

|-------|----------|---------|------|CorrectionModel.SYMSPELL   # "symspell" - Fast dictionary-based (PRIMARY)

| **Qwen2.5-1.5B** | 80% | ~800ms | 1.5B |CorrectionModel.KEYBOARD   # "keyboard" - Keyboard proximity-aware

| **Llama-3.2-1B** | 75% | ~450ms | 1B |CorrectionModel.E5_ONNX    # "e5_onnx"  - Semantic similarity

| **Phi-3-Mini** | 77% | ~1200ms | 3.8B |CorrectionModel.MISTRAL    # "mistral"  - LLM backup (SLOW)

| **ByT5-small** | 73% | ~350ms | 300M |```



## 📦 Installation## 🔌 Backend Integration



```bash```python

# Clone the repofrom app.typo_corrector import TypoCorrector

git clone https://github.com/KaanTufan/ss-correction-model.git

cd ss-correction-model# Initialize once at startup

corrector = TypoCorrector()

# Install dependencies

pip install -r requirements.txt@app.route('/api/search', methods=['POST'])

```def search():

    query = request.json.get('query')

## 🔧 Quick Start    model = request.json.get('correction_model', 'symspell')

    

### Basic Usage    # Apply typo correction

    corrected = corrector.correct_query(query, model=model)

```python    

from app.typo_corrector import TypoCorrector, CorrectionModel    # Send to retrieval

    results = retrieval.search(corrected)

# Initialize    

corrector = TypoCorrector()    # If empty results, try Mistral backup

    if not results and model != 'mistral':

# Simple correction (returns string)        corrected = corrector.correct_query(query, model='mistral')

corrected = corrector.correct_query("iphnoe 15 pro")        results = retrieval.search(corrected)

# Returns: "iphone 15 pro"    

    return jsonify({"results": results, "corrected_query": corrected})

# With model selection```

corrected = corrector.correct_query("samsng galaxy s24", model="symspell")

# Returns: "samsung galaxy s24"## 📁 Project Structure

```

```

### Full API (with metadata)├── app/

│   ├── typo_corrector.py     # Main correction module (ACTIVE)

```python│   ├── domain_vocab.py       # Vocabulary loader

result = corrector.correct("16 gm gamng laptop", model=CorrectionModel.SYMSPELL)│   └── metrics.py            # Evaluation metrics

# Returns:├── data/

# {│   ├── typo_mappings.txt     # Explicit typo→correct mappings

#     "original_query": "16 gm gamng laptop",│   ├── brand_products.txt    # E-commerce brand vocabulary

#     "normalized_query": "16 gb gaming laptop",│   ├── electronics_vocab.txt # Electronics domain words

#     "changed": True,│   ├── curated_common_words.txt

#     "latency_ms": 0.1│   └── nltk_words.txt        # Large English vocabulary

# }└── requirements.txt

``````



### Available Models## ⚠️ Deprecated Files (DO NOT USE)



```pythonThe following files are from earlier experiments and are **NOT part of production**:

CorrectionModel.SYMSPELL   # "symspell" - Fast dictionary-based (PRIMARY)

CorrectionModel.KEYBOARD   # "keyboard" - Keyboard proximity-aware| File/Folder | Status | Notes |

CorrectionModel.E5_ONNX    # "e5_onnx"  - Semantic similarity|-------------|--------|-------|

CorrectionModel.MISTRAL    # "mistral"  - LLM backup (slow)| `neuspell_env/` | ❌ DEPRECATED | NeuSpell experiment (incompatible) |

| `neuspell_local/` | ❌ DEPRECATED | Local NeuSpell attempt |

# Fine-tuned models (requires trained weights)| `subwordbert-probwordnoise/` | ❌ DEPRECATED | SubwordBERT experiment |

CorrectionModel.LLAMA      # "llama"    - Llama 3.2 1B fine-tuned| `scripts/` | ⚠️ OLD | Old evaluation/demo scripts |

CorrectionModel.QWEN       # "qwen"     - Qwen 2.5 1.5B fine-tuned| `test_script.py` | ❌ DEPRECATED | Old test file |

CorrectionModel.BYT5       # "byt5"     - ByT5-small fine-tuned| `test_corrector.py` | ❌ DEPRECATED | Old test file |

CorrectionModel.PHI3       # "phi3"     - Phi-3 Mini fine-tuned| `data/typo_dataset.csv` | ⚠️ OLD | Old evaluation dataset |

```| `data/common_words.txt` | ⚠️ OLD | Replaced by nltk_words.txt |

| `data/expanded_common_words.txt` | ⚠️ OLD | No longer used |

## 🔌 Backend Integration Example

## 🧪 Testing

```python

from app.typo_corrector import TypoCorrector```python

from app.typo_corrector import TypoCorrector, CorrectionModel

# Initialize once at startup

corrector = TypoCorrector()corrector = TypoCorrector()



@app.route('/api/search', methods=['POST'])# Test queries

def search():tests = [

    query = request.json.get('query')    ("iphnoe 15 pro", "iphone 15 pro"),

    model = request.json.get('correction_model', 'symspell')    ("samsng galaxy s24", "samsung galaxy s24"),

        ("nvidea rtx 4090", "nvidia rtx 4090"),

    # Apply typo correction    ("i ned a chp laptop", "i need a cheap laptop"),

    corrected = corrector.correct_query(query, model=model)]

    

    # Send to retrievalfor query, expected in tests:

    results = retrieval.search(corrected)    result = corrector.correct_query(query)

        status = "✅" if result == expected else "❌"

    # If empty results, try fine-tuned model as backup    print(f"{status} {query} → {result}")

    if not results and model != 'llama':```

        corrected = corrector.correct_query(query, model='llama')

        results = retrieval.search(corrected)## 📊 Model Selection Guide

    

    return {"results": results, "corrected_query": corrected}| Scenario | Recommended Model |

```|----------|-------------------|

| Normal search queries | `symspell` (default) |

## 🎯 Fine-tuning Your Own Models| Keyboard typos (adjacent keys) | `keyboard` |

| Unknown brand names | `e5_onnx` |

See [fine_tune/README.md](fine_tune/README.md) for detailed instructions.| All else fails / empty results | `mistral` (backup) |



### Quick Start## 🔄 Changelog



```bash### v2.0 (December 2024)

cd fine_tune- ✅ 4 correction models (SymSpell, Keyboard, E5_ONNX, Mistral)

- ✅ Simple API: `correct_query()` returns just string

# 1. Generate training data- ✅ 96%+ accuracy on e-commerce queries

python prepare_data.py \- ✅ Sub-millisecond latency (SymSpell)

  --augment \- ✅ 475+ typo mappings for common errors

  --samples 8000 \- ✅ Logging instead of print statements

  --multi-word 4000 \- ✅ Production-ready code

  --sentences 6000 \

  --spacing-variants \### v1.0 (November 2024)

  --symbol-variants- Initial SymSpell implementation

- Basic vocabulary support

# 2. Fine-tune Llama (recommended for speed/accuracy balance)
python models/llama_finetune.py --epochs 3 --batch-size 4

# 3. Test the model
python test_all_finetuned_models.py --model llama
```

### Training Data Features

The `prepare_data.py` script generates:

- **Single-word typos**: `"iphnoe"` → `"iphone"`
- **Multi-word queries**: `"16 gm ram laptop"` → `"16 gb ram laptop"`
- **Full sentences**: `"i need a gamng laptop for my son"` → `"i need a gaming laptop for my son"`
- **Spacing variants**: `"rtx4070"` → `"rtx 4070"`
- **Symbol variants**: `"rtx-4070-ti"` → `"rtx 4070 ti"`
- **Identity examples**: Correct text stays unchanged (prevents over-correction)

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/

# Test specific model
python tests/test_spell_correction.py
```

## 📊 Benchmark

```bash
cd fine_tune
python benchmark_all_models.py
```

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers 4.40+
- symspellpy
- ONNX Runtime (for E5 model)

For fine-tuning:
- trl 0.25+
- peft
- bitsandbytes (optional, for quantization)
- GPU with 8GB+ VRAM recommended

## 📝 License

MIT License

## 👥 Authors

- SmartSearch Graduation Team
