"""
analyze_results.py
==================
Score the results.json from run_eval.py and produce a comparison report.

Metrics per model:
    - Overall exact-match accuracy
    - Per-difficulty accuracy (L1..L5)
    - Per-length accuracy (short / medium / long)
    - Per-category accuracy
    - FP rate on L1 (correctly-spelled queries the model wrongly changed)
    - Hit rate on L2..L5 (typo'd queries the model fixed)
    - Latency p50 / p90 / p99
    - Error count (HTTP failures, etc.)

Outputs:
    eval/report.md          — readable Markdown summary
    eval/scores.json        — structured scores
    eval/comparison.csv     — side-by-side comparison table
    eval/qualitative.md     — interesting wins/losses across models
"""

import csv
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.json")
DATASET = os.path.join(HERE, "dataset.json")
REPORT_MD = os.path.join(HERE, "report.md")
SCORES_JSON = os.path.join(HERE, "scores.json")
COMP_CSV = os.path.join(HERE, "comparison.csv")
QUAL_MD = os.path.join(HERE, "qualitative.md")


def norm(s):
    return (s or "").strip().lower()


def is_correct(row):
    return norm(row["corrected"]) == norm(row["expected"])


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    with open(RESULTS, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    by_id = {q["id"]: q for q in dataset}

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    models = sorted(by_model.keys())

    scores = {}
    for m in models:
        rows = by_model[m]
        ok_rows = [r for r in rows if not r.get("error")]
        err_count = len(rows) - len(ok_rows)

        total = len(ok_rows)
        correct = sum(1 for r in ok_rows if is_correct(r))

        per_diff = defaultdict(lambda: [0, 0])    # [correct, total]
        per_len = defaultdict(lambda: [0, 0])
        per_cat = defaultdict(lambda: [0, 0])

        # FP rate on L1 (clean): model corrupted a clean input
        l1_total = 0
        l1_corrupted = 0

        latencies = []

        for r in ok_rows:
            d = r["difficulty"]
            ln = r["length"]
            cat = r["category"]
            ok = is_correct(r)

            per_diff[d][1] += 1
            per_diff[d][0] += int(ok)
            per_len[ln][1] += 1
            per_len[ln][0] += int(ok)
            per_cat[cat][1] += 1
            per_cat[cat][0] += int(ok)

            if d == "L1":
                l1_total += 1
                if not ok:
                    l1_corrupted += 1

            latencies.append(r["latency_ms"])

        # Hit rate on typo queries (L2..L5)
        typo_rows = [r for r in ok_rows if r["difficulty"] != "L1"]
        typo_correct = sum(1 for r in typo_rows if is_correct(r))
        hit_rate = typo_correct / len(typo_rows) if typo_rows else 0.0

        scores[m] = {
            "n": total,
            "errors": err_count,
            "overall_accuracy": correct / total if total else 0.0,
            "by_difficulty": {
                k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1] if v[1] else 0.0}
                for k, v in sorted(per_diff.items())
            },
            "by_length": {
                k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1] if v[1] else 0.0}
                for k, v in sorted(per_len.items())
            },
            "by_category": {
                k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1] if v[1] else 0.0}
                for k, v in sorted(per_cat.items(), key=lambda kv: kv[0])
            },
            "fp_rate_on_clean_L1": l1_corrupted / l1_total if l1_total else 0.0,
            "fp_count_on_clean_L1": l1_corrupted,
            "fp_total_clean_L1": l1_total,
            "hit_rate_on_typos_L2_L5": hit_rate,
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p90": percentile(latencies, 90),
            "latency_ms_p99": percentile(latencies, 99),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
        }

    with open(SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # --- comparison.csv ---
    fields = [
        "model", "n", "errors",
        "overall_accuracy", "fp_rate_on_clean_L1", "hit_rate_on_typos_L2_L5",
        "acc_L1", "acc_L2", "acc_L3", "acc_L4", "acc_L5",
        "acc_short", "acc_medium", "acc_long",
        "lat_p50_ms", "lat_p90_ms", "lat_p99_ms",
    ]
    with open(COMP_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for m in models:
            s = scores[m]
            w.writerow([
                m, s["n"], s["errors"],
                f"{s['overall_accuracy']:.4f}",
                f"{s['fp_rate_on_clean_L1']:.4f}",
                f"{s['hit_rate_on_typos_L2_L5']:.4f}",
                f"{s['by_difficulty'].get('L1', {}).get('acc', 0):.4f}",
                f"{s['by_difficulty'].get('L2', {}).get('acc', 0):.4f}",
                f"{s['by_difficulty'].get('L3', {}).get('acc', 0):.4f}",
                f"{s['by_difficulty'].get('L4', {}).get('acc', 0):.4f}",
                f"{s['by_difficulty'].get('L5', {}).get('acc', 0):.4f}",
                f"{s['by_length'].get('short', {}).get('acc', 0):.4f}",
                f"{s['by_length'].get('medium', {}).get('acc', 0):.4f}",
                f"{s['by_length'].get('long', {}).get('acc', 0):.4f}",
                f"{s['latency_ms_p50']:.1f}",
                f"{s['latency_ms_p90']:.1f}",
                f"{s['latency_ms_p99']:.1f}",
            ])

    # --- report.md ---
    lines = []
    lines.append("# SmartSearch Correction — Model Comparison Report")
    lines.append("")
    lines.append(f"Total queries: **{len(dataset)}** | Models tested: **{len(models)}**")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Model | Overall Acc | FP on Clean (L1) | Hit Rate Typos (L2-L5) | p50 ms | p90 ms |")
    lines.append("|-------|------------:|-----------------:|-----------------------:|-------:|-------:|")
    for m in models:
        s = scores[m]
        lines.append(
            f"| {m} | {s['overall_accuracy']*100:.2f}% | "
            f"{s['fp_rate_on_clean_L1']*100:.2f}% | "
            f"{s['hit_rate_on_typos_L2_L5']*100:.2f}% | "
            f"{s['latency_ms_p50']:.0f} | {s['latency_ms_p90']:.0f} |"
        )
    lines.append("")

    lines.append("## Accuracy by difficulty")
    lines.append("")
    lines.append("| Model | L1 clean | L2 easy | L3 medium | L4 hard | L5 v.hard |")
    lines.append("|-------|---------:|--------:|----------:|--------:|----------:|")
    for m in models:
        s = scores[m]
        d = s["by_difficulty"]
        lines.append(
            f"| {m} | "
            f"{d.get('L1', {}).get('acc', 0)*100:.1f}% | "
            f"{d.get('L2', {}).get('acc', 0)*100:.1f}% | "
            f"{d.get('L3', {}).get('acc', 0)*100:.1f}% | "
            f"{d.get('L4', {}).get('acc', 0)*100:.1f}% | "
            f"{d.get('L5', {}).get('acc', 0)*100:.1f}% |"
        )
    lines.append("")

    lines.append("## Accuracy by length")
    lines.append("")
    lines.append("| Model | Short (1-2) | Medium (3-5) | Long (6+) |")
    lines.append("|-------|------------:|-------------:|----------:|")
    for m in models:
        s = scores[m]
        ln = s["by_length"]
        lines.append(
            f"| {m} | "
            f"{ln.get('short', {}).get('acc', 0)*100:.1f}% | "
            f"{ln.get('medium', {}).get('acc', 0)*100:.1f}% | "
            f"{ln.get('long', {}).get('acc', 0)*100:.1f}% |"
        )
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    lines.append("| Model | p50 ms | p90 ms | p99 ms | mean ms |")
    lines.append("|-------|-------:|-------:|-------:|--------:|")
    for m in models:
        s = scores[m]
        lines.append(
            f"| {m} | {s['latency_ms_p50']:.0f} | {s['latency_ms_p90']:.0f} | "
            f"{s['latency_ms_p99']:.0f} | {s['latency_ms_mean']:.0f} |"
        )
    lines.append("")

    lines.append("## Per-category accuracy")
    lines.append("")
    cats = sorted({c for m in models for c in scores[m]["by_category"]})
    header = "| Model | " + " | ".join(cats) + " |"
    sep = "|-------|" + "|".join(["-------:"] * len(cats)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = []
        for c in cats:
            v = scores[m]["by_category"].get(c, {}).get("acc")
            cells.append(f"{v*100:.0f}%" if v is not None else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- qualitative.md: interesting cases ---
    by_model_id = defaultdict(dict)
    for r in results:
        by_model_id[r["model"]][r["id"]] = r

    qlines = []
    qlines.append("# Qualitative — where models disagree")
    qlines.append("")
    qlines.append("Cases where at least one model got it right and at least one got it wrong.")
    qlines.append("Sorted by difficulty descending.")
    qlines.append("")

    interesting = []
    for q in dataset:
        per_m = {}
        for m in models:
            r = by_model_id[m].get(q["id"])
            if not r or r.get("error"):
                continue
            per_m[m] = (is_correct(r), r["corrected"])
        if not per_m:
            continue
        oks = [v[0] for v in per_m.values()]
        if any(oks) and not all(oks):
            interesting.append((q, per_m))

    interesting.sort(key=lambda kv: kv[0]["difficulty"], reverse=True)

    for q, per_m in interesting[:60]:
        qlines.append(f"### `{q['id']}` [{q['difficulty']} / {q['length']} / {q['category']}]")
        qlines.append("")
        qlines.append(f"- **Input:** `{q['query']}`")
        qlines.append(f"- **Gold:**  `{q['expected']}`")
        for m, (ok, corr) in per_m.items():
            mark = "✓" if ok else "✗"
            qlines.append(f"- {mark} **{m}:** `{corr}`")
        qlines.append("")

    with open(QUAL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(qlines))

    # --- console summary ---
    print("\n=== HEADLINE ===")
    for m in models:
        s = scores[m]
        print(f"  {m:18s}  acc={s['overall_accuracy']*100:5.2f}%  "
              f"fp={s['fp_rate_on_clean_L1']*100:5.2f}%  "
              f"hit={s['hit_rate_on_typos_L2_L5']*100:5.2f}%  "
              f"p50={s['latency_ms_p50']:.0f}ms")
    print(f"\nWrote: {REPORT_MD}")
    print(f"Wrote: {SCORES_JSON}")
    print(f"Wrote: {COMP_CSV}")
    print(f"Wrote: {QUAL_MD}")


if __name__ == "__main__":
    main()
