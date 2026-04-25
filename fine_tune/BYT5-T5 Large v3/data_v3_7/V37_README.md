# ByT5-Large v3.7 — Gap-Fill Retrain Rehberi

## 1. Neden v3.7?

Önceki modelin hard test suite'inde %80 doğrulukta görülen spesifik hata kalıplarını hedefler:

- **Over-correction (hallucination)** — `bay`, `neo`, `ai`, `oura` gibi zaten doğru kısa kelimeleri değiştiriyordu.
- **Compound merging** — `mac book → macbook`, `air pods → airpods` birleşmiyordu. v3.6'da bu kategoride sadece **15 örnek** vardı.
- **Niche brand preservation** — `Oura Ring`, `DJI Mavic 3`, `Sony ZV-1F`, `WH-1000XM5`, `Anker SoundCore` gibi modern teknoloji markaları "düzeltiliyordu".
- **Number dropping** — `"1000 watts"` bazı bağlamlarda sayı kısmını kaybediyordu.
- **Real-world typo coverage** — Akademik standart typo korporalarından (Birkbeck, Wikipedia, Aspell, Holbrook) yararlanılmıyordu.

## 2. Veri seti farkları

| Metrik                  | v3.6       | v3.7       |
|-------------------------|------------|------------|
| Train kayıt sayısı      | 2,580,000  | **2,663,803** |
| Eval kayıt sayısı       | 121,226    | **122,416** |
| Identity ratio (train)  | ~20.0 %    | **20.57 %** |
| split_to_compound örneği| 15         | **~14,500** (3x oversample) |
| short_identity örneği   | —          | **~15,000** (5x oversample) |
| niche_brand örneği      | —          | **~15,000** (5x oversample) |
| number_preservation     | —          | **~5,000** (2x oversample) |
| External typo corpus    | —          | **~37,800 yeni typo çifti** |

### Yeni kategoriler

- `short_identity_v37` — 2-5 karakterli kısa kelimelerin identity koruması.
- `niche_brand_identity_v37` — Modern teknoloji markalarının (Oura/DJI/Anker/ZV-1F/WH-1000XM5) dokunulmadan geçirilmesi.
- `split_to_compound_v37` — `mac book → macbook`, `smart watch → smartwatch` vs. v3.6'nın 15 örnek boşluğunu doldurur.
- `number_preservation_v37` — `"1000 watts"` gibi sayı+birim ifadelerinin prefix/suffix bağlamlarında korunması.
- `external_birkbeck_v37`, `external_wikipedia_v37`, `external_aspell_v37`, `external_holbrook_v37`, `external_wiki_ml_v37` — akademik standart typo korporalarının kalite filtresinden geçirilmiş versiyonları (tek kelime + e-ticaret bağlamı).

## 3. Kod değişiklikleri

### `byt5_large_finetune_v3.py` / `.ipynb`

1. **Veri yolları** → `DATA_DIR = PROJECT_ROOT / "data_v3_7"`, `TRAIN_FILE = train_v3_7.jsonl`, `EVAL_FILE = eval_v3_7.jsonl`.
2. **`IDENTITY_EVAL_TARGETS`** — Eval yüklenirken input == target olan satırların target metinleri bir set içine alınır.
3. **`compute_metrics`** — Artık üç yeni metrik raporluyor:
   - `identity_accuracy` — modelin "düzeltmemesi gereken" eval satırlarında input'u bozmadan geçirme oranı.
   - `correction_accuracy` — gerçek typo düzeltme satırlarındaki doğruluk.
   - `identity_eval_count` / `correction_eval_count` — görünürlük için bucket büyüklükleri.
4. **`label_smoothing_factor`** — 0.10 → **0.05** (over-correction baskısını azaltmak için).

## 4. Colab'da ne yapacaksın?

1. **Drive'a yükle:** `data_v3_7/` klasörünü şuraya koy:
   `MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3/data_v3_7/`

   İçinde şunlar olmalı:
   - `train_v3_7.jsonl` (~347 MB)
   - `eval_v3_7.jsonl`  (~15 MB)
   - `training_stats_v3_7.json`

2. **Model cache'i kontrol et:** `model_cache/byt5-large/` hâlâ yerinde olmalı (v3.6'dan). Yoksa önce `download_byt5_to_drive.ipynb` çalıştır.

3. **Notebook'u yenile:** `byt5_large_finetune_v3.ipynb` güncel kopyasını Drive'a yükle (bu commit'teki hali).

4. **Önceki checkpoint'leri iste/tut:** v3.7 yeni veri üzerinde eğitildiği için eski `output/byt5-large-v3/checkpoint-*` klasörünü silmek veya yeniden adlandırmak gerekir. `CONFIG["resume_from_checkpoint"] = True` açıkta; eski checkpoint'ler varsa HuggingFace onlardan devam edip yeni veri dağılımını tam öğrenemez. **Öneri:** Drive'daki eski output klasörünü `output/byt5-large-v3-v36-archive/` olarak yeniden adlandır, yeni temiz bir `output/byt5-large-v3/` ile başla.

5. **Çalıştır:** Runtime → Run all. A100 40GB'de 2 epoch ≈ **10-12 saat** (efektif batch 64, bf16, torch.compile açık).

## 5. Ne göreceksin

Her 3000 step'te bir eval log'unda şimdi şunları göreceksin:

```
sentence_accuracy:  0.86
word_f1:            0.92
cer:                0.04
identity_accuracy:  0.94   <-- YENİ: "düzeltme"meyi öğrenme metriği
correction_accuracy: 0.79  <-- YENİ: gerçek typo düzeltme oranı
identity_eval_count: 12842
correction_eval_count: 109574
```

**Hedef:** `identity_accuracy ≥ 0.95` ve `correction_accuracy ≥ 0.85` ikisi birden. Eğer identity_accuracy hâlâ düşük kalırsa label_smoothing'i 0.0'a indir.

## 6. Dosyalar

- Data: `/content/drive/MyDrive/Grad/Correction/fine_tune/BYT5-T5 Large v3/data_v3_7/`
- Notebook: `collab/BYT5-Large-V3/byt5_large_finetune_v3.ipynb`
- Script mirror: `collab/BYT5-Large-V3/byt5_large_finetune_v3.py`
- Yerelde external corpus build scriptleri: `v37_external/parse_externals.py`, `v37_external/generate_gap_fills.py`, `v37_external/build_v37.py` (Colab'da çalıştırılmayacak, sadece referans).
