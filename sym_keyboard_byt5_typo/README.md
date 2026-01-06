# SmartSearch Typo Correction API

E-commerce arama sorguları için yazım hatası düzeltme servisi.

## 🚀 Production Models

| Model | Endpoint | Latency | Accuracy | Kullanım |
|-------|----------|---------|----------|----------|
| `symspell_keyboard` | Default | ~0.1ms | ~98% | Hızlı, gerçek zamanlı |
| `byt5` | `?model=byt5` | ~300-500ms | 100% | Maksimum doğruluk |

## 📦 Kurulum

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Git LFS (Model için gerekli)

Model dosyası 1.1GB olduğu için Git LFS kullanılıyor:

```bash
# Git LFS kurulu değilse
brew install git-lfs  # macOS
# veya
apt-get install git-lfs  # Ubuntu

# LFS'i aktifleştir
git lfs install
git lfs pull
```

### 3. API'yi Başlat

```bash
cd sym_keyboard_byt5_typo
python api.py
```

API `http://localhost:5001` adresinde çalışacak.

## 🔌 API Kullanımı

### Temel Kullanım (Default: symspell_keyboard)

```bash
# GET request
curl "http://localhost:5001/correct?query=samsugn%20galxy%20s24"
```

**Response:**
```json
{
  "original_query": "samsugn galxy s24",
  "corrected_query": "samsung galaxy s24",
  "changed": true,
  "model_used": "symspell_keyboard",
  "latency_ms": 0.12
}
```

### ByT5 Model (Maksimum Doğruluk)

```bash
curl "http://localhost:5001/correct?query=samsugn%20galxy%20s24&model=byt5"
```

### POST Request

```bash
curl -X POST "http://localhost:5001/correct" \
  -H "Content-Type: application/json" \
  -d '{"query": "iphnoe 15 pro max", "model": "symspell_keyboard"}'
```

### Mevcut Modelleri Listele

```bash
curl "http://localhost:5001/models"
```

## 📁 Klasör Yapısı

```
sym_keyboard_byt5_typo/
├── api.py                      # FastAPI uygulaması (Port 5001)
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Bu dosya
├── app/
│   ├── __init__.py
│   ├── typo_corrector.py       # Ana düzeltme mantığı
│   ├── domain_vocab.py         # Vocabulary yükleyici
│   └── metrics.py              # Metrik hesaplamaları
├── data/
│   ├── brand_products.txt      # Marka ve ürün isimleri
│   ├── typo_mappings.txt       # Bilinen typo -> doğru eşlemeleri
│   ├── electronics_vocab.txt   # Elektronik terimleri
│   ├── domain_vocab.txt        # Domain-specific kelimeler
│   └── symspell_dictionary.txt # SymSpell sözlüğü
├── models/
│   └── byt5-typo-final/        # Fine-tuned ByT5 model (1.1GB)
│       ├── config.json
│       ├── model.safetensors   # Model ağırlıkları (LFS)
│       ├── tokenizer_config.json
│       └── ...
└── tests/
    ├── __init__.py
    └── baseline_vs_offline.py  # Baseline SymSpell implementasyonu
```

## 🧠 Model Açıklamaları

### symspell_keyboard (Default)

- **Strateji**: SymSpell + Keyboard Proximity Hybrid
- **Nasıl çalışır**:
  1. SymSpell ile hızlı dictionary lookup
  2. Düşük confidence durumunda keyboard proximity fallback
  3. Komşu tuş hatalarını yakalar (örn: `mbifia` → `nvidia`)
- **Avantaj**: Çok hızlı (~0.1ms), yüksek accuracy (~98%)

### byt5 (Fine-tuned)

- **Model**: Google ByT5-small, Colab'da fine-tune edildi
- **Training**: 117K sample, 10 epoch, T4 GPU
- **Avantaj**: En yüksek accuracy (100% test accuracy)
- **Dezavantaj**: Yavaş (~300-500ms CPU'da)

## 🔧 Backend Entegrasyonu

### Python

```python
import requests

def correct_query(query: str, model: str = "symspell_keyboard") -> str:
    response = requests.get(
        "http://localhost:5001/correct",
        params={"query": query, "model": model}
    )
    data = response.json()
    return data["corrected_query"]

# Kullanım
corrected = correct_query("samsugn galxy s24")
print(corrected)  # "samsung galaxy s24"
```

### JavaScript/Node.js

```javascript
async function correctQuery(query, model = "symspell_keyboard") {
  const response = await fetch(
    `http://localhost:5001/correct?query=${encodeURIComponent(query)}&model=${model}`
  );
  const data = await response.json();
  return data.corrected_query;
}

// Kullanım
const corrected = await correctQuery("iphnoe 15 pro max");
console.log(corrected); // "iphone 15 pro max"
```

## ⚙️ Konfigürasyon

### Environment Variables

| Değişken | Default | Açıklama |
|----------|---------|----------|
| `PORT` | 5001 | API port numarası |

### CORS

API tüm origin'lere açıktır (`allow_origins=["*"]`). Production'da kısıtlamak için `api.py`'daki CORS middleware'ini düzenleyin.

## 📊 Performans

| Model | CPU Latency | GPU Latency | Memory |
|-------|-------------|-------------|--------|
| symspell_keyboard | ~0.1ms | ~0.1ms | ~50MB |
| byt5 | ~300-500ms | ~50ms | ~1.2GB |

## 🧪 Test

```bash
# Basit test
curl "http://localhost:5001/correct?query=logitehc%20mouse%20wireles"

# Expected: {"corrected_query": "logitech mouse wireless", ...}
```

## 📝 Notlar

- `symspell` parametresi gönderilirse otomatik olarak `symspell_keyboard`'a yönlendirilir
- ByT5 ilk çağrıda model yükleme nedeniyle yavaş olabilir (~2-3s), sonraki çağrılar hızlıdır
- GPU varsa ByT5 çok daha hızlı çalışır (~50ms)

## 👥 Geliştirici

SmartSearch Graduation Team
