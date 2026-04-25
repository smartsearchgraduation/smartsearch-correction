# SmartSearch Correction — Model Comparison Report

Total queries: **487** | Models tested: **5**

## Headline numbers

| Model | Overall Acc | FP on Clean (L1) | Hit Rate Typos (L2-L5) | p50 ms | p90 ms |
|-------|------------:|-----------------:|-----------------------:|-------:|-------:|
| BYT5-Large-V3 | 57.70% | 8.99% | 50.25% | 466 | 996 |
| T5-Large-V2.1 | 44.97% | 0.00% | 32.66% | 234 | 439 |
| byt5-base | 17.45% | 74.16% | 15.58% | 268 | 556 |
| byt5-small | 10.47% | 94.38% | 11.56% | 190 | 296 |
| qwen-3.5-2b | 31.42% | 37.08% | 24.37% | 540 | 877 |

## Accuracy by difficulty

| Model | L1 clean | L2 easy | L3 medium | L4 hard | L5 v.hard |
|-------|---------:|--------:|----------:|--------:|----------:|
| BYT5-Large-V3 | 91.0% | 77.0% | 66.7% | 44.1% | 12.0% |
| T5-Large-V2.1 | 100.0% | 66.0% | 40.0% | 17.2% | 6.0% |
| byt5-base | 25.8% | 24.0% | 20.0% | 12.9% | 5.0% |
| byt5-small | 5.6% | 19.0% | 17.1% | 7.5% | 2.0% |
| qwen-3.5-2b | 62.9% | 44.0% | 26.7% | 16.1% | 10.0% |

## Accuracy by length

| Model | Short (1-2) | Medium (3-5) | Long (6+) |
|-------|------------:|-------------:|----------:|
| BYT5-Large-V3 | 67.2% | 68.8% | 43.1% |
| T5-Large-V2.1 | 53.6% | 57.5% | 29.7% |
| byt5-base | 50.4% | 13.1% | 0.5% |
| byt5-small | 36.0% | 3.1% | 0.5% |
| qwen-3.5-2b | 56.8% | 33.8% | 13.9% |

## Latency

| Model | p50 ms | p90 ms | p99 ms | mean ms |
|-------|-------:|-------:|-------:|--------:|
| BYT5-Large-V3 | 466 | 996 | 1123 | 570 |
| T5-Large-V2.1 | 234 | 439 | 547 | 248 |
| byt5-base | 268 | 556 | 933 | 319 |
| byt5-small | 190 | 296 | 362 | 187 |
| qwen-3.5-2b | 540 | 877 | 1082 | 560 |

## Per-category accuracy

| Model | adversarial_compound | brand_corrupt_with_typo | clean | dense_compound_typo | dense_phonetic | double_edit_or_transpose | extreme_brand_typo | extreme_compression | single_edit_typo |
|-------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| BYT5-Large-V3 | 2% | 52% | 91% | 33% | 56% | 67% | 30% | 10% | 77% |
| T5-Large-V2.1 | 0% | 48% | 100% | 0% | 16% | 40% | 17% | 5% | 66% |
| byt5-base | 0% | 20% | 26% | 0% | 28% | 20% | 3% | 20% | 24% |
| byt5-small | 0% | 4% | 6% | 0% | 24% | 17% | 3% | 5% | 19% |
| qwen-3.5-2b | 8% | 32% | 63% | 0% | 28% | 27% | 10% | 15% | 44% |
