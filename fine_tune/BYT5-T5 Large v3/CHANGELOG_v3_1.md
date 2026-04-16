# BYT5-Large v3.1 — Full Update Changelog

**Date:** 2026-04-16
**Scope:** Complete overhaul of the `BYT5-T5 Large v3` training pipeline (dataset builder + Colab fine-tune script) to address audit findings and the high-quality-training requirements.

---

## 1. High-level goals addressed

| Requirement | Status | Where |
|---|---|---|
| Preserve correctly-spelled brand/model names (no over-correction) | Done | Boosted `gen_identity_examples()` (5x templates), added `gen_identity_external()` |
| Correct brand-model category mismatches via context | Done | New `gen_brand_category_mismatch_examples()` + `CROSS_BRAND_MODEL_CONFUSIONS` |
| Fix everyday English typos | Done | New `gen_everyday_english_typos()` with domain-plausibility filter |
| Teach price definitions ($99, under $500, 299 USD …) | Done | New `gen_price_examples()` + `PRICE_TEMPLATES_CLEAN` |
| Teach measurement units (GB, Hz, mAh, 4K, 1080p …) | Done | New `gen_measurement_unit_examples()` + unit vocab tables |
| English only | Enforced | `non_ascii` quality check kept at 0 |
| Fix all audit bugs | Done | See sections 3 & 4 |
| Add overfitting protection | Done | `OverfitGuard` via `TrainingMonitor` callback |
| Detailed Colab console output | Done | Rewritten `TrainingMonitor` — per-log ETA, loss EMA, gap, probe accuracies |
| Optional 3D visualization | Done | Cell 7B — Plotly `Scatter3d` trajectory |
| Better training performance | Done | `torch_compile` (device-tiered), `group_by_length`, tuned eval cadence |
| Integrate more real datasets | Done | `load_external_pairs()` auto-discovers Birkbeck / codespell / torinriley in `data/`, `data/external/`, `V3_ROOT/data/external/` |

---

## 2. Dataset builder — `data/build_training_data_v3.py`

### New generators
- **`gen_price_examples()`** — 20 clean templates ("{product} under ${p}", "{product} between ${p} and ${p2}", "{product} for {p} USD", "cheap {product} {p}$"), plus price-typo variants (currency mangling `$99 → $ 99`, `dolars`, `dolar`, `uds` → corrected to USD/$).
- **`gen_measurement_unit_examples()`** — 12 spec templates covering STORAGE_UNITS (GB/TB/MB/KB), FREQUENCY_UNITS (Hz/MHz/GHz), POWER_UNITS (W/kW/mAh/Wh), LENGTH_UNITS (inch/cm/mm), WEIGHT_UNITS (g/kg/lb/oz), CAMERA_UNITS (MP), RESOLUTIONS (720p/1080p/1440p/4K/8K). Emits both `unit_identity` and `unit_typo` (e.g., `ghz → GHz`, `4 k → 4K`).
- **`gen_brand_category_mismatch_examples()`** — uses the curated `CROSS_BRAND_MODEL_CONFUSIONS` table, e.g., `("nvidia", "asus", "tuf gaming")` → `nvidia tuf gaming chair` → `asus tuf gaming chair`. Also Intel↔AMD (`ryzen 7`), Samsung↔Apple (`iphone`), LG↔Dyson (`v15 vacuum`), etc.
- **`gen_everyday_english_typos()`** — everyday words (receive, definitely, occurred, separate, necessary, …) with keyboard/phonetic/double/swap perturbations, filtered through `domain_plausible()` so `tuatera`-class OOVs never reach training.
- **`gen_aug_*()`** — now produces one category label per augmentation family so the stats file tells you exactly how the typo budget was spent: `aug_keyboard` (70k), `aug_double` (65k), `aug_swap` (59k), `aug_phonetic` (51k), `aug_delete` (49k), `aug_insert` (43k), `aug_compound` (16k, 2+ errors), `aug_omit_repeat` (0.6k).
- **`gen_external_pairs()` / `gen_external_word_pairs()` / `gen_external_context_typos()`** — Birkbeck, codespell, torinriley auto-discovery with domain filter, both raw and context-embedded variants.
- **`gen_identity_external()`** — preserves correct-but-unusual real-world words so the model doesn't mangle them.

### Supporting additions
- `levenshtein(a, b)` — true edit-distance helper.
- `BRAND_CATEGORIES` — every brand tagged `electronics` / `fashion` / `beauty` / `home_kitchen` / `sports` / `grocery` / `automotive` / `appliances`.
- `CATEGORY_PRODUCTS` — auto-built from catalog; used by the mismatch generator.
- `DOMAIN_STOPWORDS`, `DOMAIN_BLOCKLIST` ({"tuatera", "herbaceously", "antediluvian", …}), and `domain_plausible(word)` — rejects words that have no business in an e-commerce search.
- `load_external_pairs(path)` — now supports csv/tsv/arrow/colon/Birkbeck `$`-separated formats.

### Fixes
- **`quality_filter()` rewritten** — the old "positional char diff > 0.5" filter silently dropped legitimate long-distance typos and kept garbage; now uses real Levenshtein `d / max(len) > 0.5` as the drop threshold.
- **`QUERY_TEMPLATES`** — removed trailing `"{brand} {product} vs"` template that produced dangling "vs" strings.
- **Builder/dataset category mismatch** — `training_stats_v3.json` categories are now exactly the labels emitted by the new generators (no orphan `identity_query` vs `identity_external` confusion).
- **Default target** bumped from 400,000 → 950,000 pairs to accommodate the new generators.
- **Quality checks** (`conflicting_targets`, `non_ascii`, `train_eval_overlap`, `duplicates`) — all zero in the latest run (see `training_stats_v3.json`).

### `main()` restructure
9 steps → 14 steps. External corpus discovery now scans `DATA_DIR`, `DATA_DIR/external/`, and `V3_ROOT/data/external/` automatically.

---

## 3. Fine-tune script — `collab/BYT5-Large-V3/byt5_large_finetune_v3.py`

### Critical fixes
- **`torch.cuda.get_device_properties(0).total_memory`** — previously `.total_mem`, which raised `AttributeError` on first GPU check and aborted the script.
- **`torch_compile` conditional** — `True` on A100/L4/A10, `False` on T4/V100/5070 Ti (fp16 + compile interaction is flaky on Turing/Ampere-small).

### Performance
- `group_by_length=True` — cuts padding waste ~15–25% on byte-level sequences.
- `weight_decay=0.01` — mild regularization.
- `eval_steps: 2000 → 4000`, `save_steps: 2000 → 4000`, `save_total_limit: 5 → 3`, `eval_subset_size: 5000 → 2000`, `generation_num_beams: 4 → 5`.

### Device tiering
4 tiers (A100 / L4·A10 / T4·V100·5070Ti / small·cpu) — each sets bf16 vs fp16, batch size, gradient accumulation, and torch_compile accordingly.

### `TrainingMonitor` callback (complete rewrite)
- `on_train_begin`: prints total steps, effective batch, eval cadence, beam count.
- `on_log`: per-step ETA, loss EMA (alpha=0.9), elapsed/eta in minutes.
- `on_evaluate`: eval_loss, train_ema, gap, sentence_acc (with best tracking), per-category probe accuracy on `PROBE_CASES`, GPU memory snapshot.
- **`OverfitGuard`** — if `eval_loss - train_ema` widens for 3 consecutive evals past the 0.05 threshold, sets `control.should_training_stop = True`.

### `PROBE_CASES`
20 hand-picked queries spanning all new categories (brand identity, brand-category mismatch, price typo, unit typo, everyday English, tricky identity) — evaluated every `eval_steps` so you see real-world behavior evolve live.

### Cell 7B — Plotly 3D trajectory
`Scatter3d` over `(step, eval_loss, sentence_accuracy)`, saved to `training_trajectory_3d.html`. Opens standalone in any browser.

### Cell 8 — `TEST_CASES`
Expanded from ~15 cases to 44+, covering: brand identity preservation, brand-category mismatch, price_typo, price_identity, unit_typo, unit_identity, everyday_english, tricky_identity, compound typos.

### Cell 9 — `training_meta.json`
Now includes v3.1 metadata: `device_tier`, `torch_compile`, `group_by_length`, builder version, dataset stats checksum.

---

## 4. Files changed

```
C:\Users\kaant\Desktop\Grad\Correction\fine_tune\BYT5-T5 Large v3\
├── data\
│   └── build_training_data_v3.py          (fully rewritten — v3.1)
├── collab\BYT5-Large-V3\
│   └── byt5_large_finetune_v3.py          (fully rewritten — v3.1)
└── CHANGELOG_v3_1.md                       (this file — NEW)
```

---

## 5. How to run

```bash
# 1) regenerate dataset
cd "BYT5-T5 Large v3/data"
python build_training_data_v3.py
# -> training_data_v3.jsonl, eval_data_v3.jsonl, training_stats_v3.json

# 2) upload data + script to Google Drive (or mount)
# 3) open byt5_large_finetune_v3.py in Colab (A100 recommended)
# 4) run cells top-to-bottom
#    - Cell 7B produces training_trajectory_3d.html
#    - Cell 9 writes training_meta.json
```

---

## 6. Expected outcomes

- Fewer false corrections on correctly-spelled brand/model names.
- Cross-brand typos like `nvidia tuf gaming chair` → `asus tuf gaming chair` now corrected.
- Price strings (`$99`, `under $500`, `299 USD`) preserved and typos repaired.
- Unit strings (`16 GB`, `144Hz`, `5000mAh`, `1080p`, `4K`) preserved and typos repaired.
- No OOV hallucinations from `tuatera`-class words.
- Early-stop before overfitting starts degrading eval loss.
- Live per-category probe feedback during training.
- 3D trajectory HTML to share with advisors.
