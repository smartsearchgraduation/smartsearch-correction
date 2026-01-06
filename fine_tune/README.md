# Fine-Tuning LLMs for E-Commerce Typo Correction

Bu klasör, e-commerce search query typo correction için küçük LLM'lerin fine-tune edilmesini içerir.

## 🎯 Amaç

SymSpell (dictionary-based) başarısız olduğunda fallback olarak kullanılacak hızlı ve doğru bir model eğitmek.

```
User Query → SymSpell (0.1ms) → Retrieval → Sonuç var? → ✅ Dön
                                    ↓
                              Zero Retrieval?
                                    ↓
                         Fine-tuned Model → Retrieval → Dön
```

## 📁 Klasör Yapısı

```
fine_tune/
├── README.md
├── requirements.txt          # Fine-tune dependencies
├── prepare_data.py           # Data hazırlama scripti
├── models/
│   ├── qwen_finetune.py      # Qwen2.5 0.5B/1.5B fine-tune
│   ├── llama_finetune.py     # Llama 3.2 1B fine-tune
│   └── byt5_finetune.py      # ByT5-small fine-tune
├── test_finetuned_models.py  # Tüm modelleri test et
├── data/                     # Training data (generated)
│   ├── train.jsonl
│   └── eval.jsonl
└── outputs/                  # Fine-tuned model checkpoints
    ├── qwen-typo/
    ├── llama-typo/
    └── byt5-typo/
```

## 🚀 Kullanım

### 1. Kurulum

```bash
pip install -r requirements.txt
```

### 2. Data Hazırlama

```bash
python prepare_data.py
```

### 3. Model Fine-tune

```bash
# Qwen2.5 (önerilen - en küçük)
python models/qwen_finetune.py

# Llama 3.2 1B
python models/llama_finetune.py

# ByT5-small (en hızlı inference)
python models/byt5_finetune.py
```

### 4. Test ve Karşılaştırma

```bash
python test_finetuned_models.py
```

## 📊 Model Karşılaştırması

| Model        | Boyut | Fine-tune GPU | Inference  | E-commerce Uygunluğu |
| ------------ | ----- | ------------- | ---------- | -------------------- |
| Qwen2.5 0.5B | 500MB | 4GB           | ~100-200ms | ⭐⭐⭐⭐             |
| Qwen2.5 1.5B | 1.5GB | 6GB           | ~200-400ms | ⭐⭐⭐⭐⭐           |
| Llama 3.2 1B | 1.3GB | 6GB           | ~200-400ms | ⭐⭐⭐⭐             |
| ByT5-small   | 300MB | 4GB           | ~20-50ms   | ⭐⭐⭐⭐⭐           |

## 🔧 Fine-tune Yaklaşımları

### LoRA/QLoRA (LLM'ler için)

- Daha az VRAM kullanımı
- Hızlı eğitim
- Original weights'i bozmaz

### Full Fine-tune (ByT5 için)

- Küçük model olduğu için full fine-tune yapılabilir
- ONNX export ile production'da çok hızlı

## 📝 Notlar

- Tüm modeller `ss-correction-model/data/` klasöründeki verileri kullanır
- Fine-tune sonrası en iyi model `typo_corrector.py`'a entegre edilecek
- GPU yoksa Colab/Kaggle kullanılabilir
