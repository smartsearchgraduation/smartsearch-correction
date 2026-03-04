#!/usr/bin/env python3
"""
Benchmark for ByT5 & T5-spell typo correction models.

60 queries across 4 difficulty levels: easy / medium / hard / extreme.

Usage:
    python benchmark.py --models byt5
    python benchmark.py --models byt5 t5_spell
    python benchmark.py --models byt5 --save
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from test_models import ByT5Tester, T5SpellTester, find_model_path

# ======================================================================
# Benchmark dataset
# ======================================================================

BENCHMARK_QUERIES = {
    "easy": [
        ("iphnoe 15 pro max", "iphone 15 pro max"),
        ("samsng galaxy s24", "samsung galaxy s24"),
        ("macbok air m2", "macbook air m2"),
        ("airpods pro 2", "airpods pro 2"),
        ("nvdia rtx 4090", "nvidia rtx 4090"),
        ("logitec mouse wireless", "logitech mouse wireless"),
        ("sonny headphones", "sony headphones"),
        ("dell monitor 27 inch", "dell monitor 27 inch"),
        ("aple watch ultra", "apple watch ultra"),
        ("razr keyboard gaming", "razer keyboard gaming"),
    ],
    "medium": [
        ("samsng galxy s24 ultra", "samsung galaxy s24 ultra"),
        ("macbok pro m3 max", "macbook pro m3 max"),
        ("iphnoe 15 pro case", "iphone 15 pro case"),
        ("nvdia gefrce rtx 4080", "nvidia geforce rtx 4080"),
        ("logitrch mx master 3", "logitech mx master 3"),
        ("airpds pro 2 case", "airpods pro 2 case"),
        ("samsng buds 2 pro", "samsung buds 2 pro"),
        ("aple macbok charger", "apple macbook charger"),
        ("asus rog strix laptpo", "asus rog strix laptop"),
        ("lenvo thinkpad x1 carbon", "lenovo thinkpad x1 carbon"),
        ("hp spectr x360 laptop", "hp spectre x360 laptop"),
        ("msi gamng monitor 144hz", "msi gaming monitor 144hz"),
        ("corsiar vengeance ram", "corsair vengeance ram"),
        ("kingston fury ddr5 32gb", "kingston fury ddr5 32gb"),
        ("segate barracuda 2tb", "seagate barracuda 2tb"),
        ("wstrn digital ssd 1tb", "western digital ssd 1tb"),
        ("creativ sound blaster", "creative sound blaster"),
        ("steelsries arctis nova", "steelseries arctis nova"),
        ("hpyer x cloud ii", "hyperx cloud ii"),
        ("benq zowie gaming mouse", "benq zowie gaming mouse"),
    ],
    "hard": [
        ("samsng galxy bds 2 pro", "samsung galaxy buds 2 pro"),
        ("aple mcbook pro m3 mxa", "apple macbook pro m3 max"),
        ("nvdia gefroce rtx 4070 ti", "nvidia geforce rtx 4070 ti"),
        ("logitch mx keybord wireless", "logitech mx keyboard wireless"),
        ("razre blackwidow v4 pro", "razer blackwidow v4 pro"),
        ("crosair k100 rgb keybrd", "corsair k100 rgb keyboard"),
        ("asus zenbok pro duo oled", "asus zenbook pro duo oled"),
        ("micorsoft surface pro 9", "microsoft surface pro 9"),
        ("aple ipda pro 12.9 m2", "apple ipad pro 12.9 m2"),
        ("samsng odyysey g9 49 inch", "samsung odyssey g9 49 inch"),
        ("dji mavci 3 pro drone", "dji mavic 3 pro drone"),
        ("gopor hero 12 black", "gopro hero 12 black"),
        ("canno eos r5 mirrorless", "canon eos r5 mirrorless"),
        ("sono beam soundbar gen2", "sonos beam soundbar gen2"),
        ("boes quietcomfort ultra", "bose quietcomfort ultra"),
    ],
    "extreme": [
        ("smasng glxy s24 ultr case", "samsung galaxy s24 ultra case"),
        ("aple mcbok air m2 chrger", "apple macbook air m2 charger"),
        ("nvda gfrce rtx 4090 grphc", "nvidia geforce rtx 4090 graphic"),
        ("lgtch mx mstr 3s mouse", "logitech mx master 3s mouse"),
        ("razr blckwidw v4 pro kybd", "razer blackwidow v4 pro keyboard"),
        ("crsar k100 rgb mechnical", "corsair k100 rgb mechanical"),
        ("iphn 15 pr mx spcgray", "iphone 15 pro max spacegray"),
        ("smsng glaxy buds fe case", "samsung galaxy buds fe case"),
        ("apl wtch ultr 2 titanum", "apple watch ultra 2 titanium"),
        ("nvda shld tv pro 2019", "nvidia shield tv pro 2019"),
        ("lnvo thkpad x1 crbn gen11", "lenovo thinkpad x1 carbon gen11"),
        ("dl xps 15 9530 oled", "dell xps 15 9530 oled"),
        ("micrsft srfc lptp stdo 2", "microsoft surface laptop studio 2"),
        ("assu rog zphyrs g14 2024", "asus rog zephyrus g14 2024"),
        ("sny wh1000xm5 headphnes", "sony wh1000xm5 headphones"),
    ],
}


# ======================================================================
# Utilities
# ======================================================================

def calculate_accuracy(predicted: str, expected: str) -> Tuple[bool, float]:
    pred = predicted.lower().strip()
    exp = expected.lower().strip()
    exact = pred == exp

    pred_words = set(pred.split())
    exp_words = set(exp.split())
    if not exp_words:
        return exact, 1.0 if exact else 0.0
    similarity = len(pred_words & exp_words) / len(exp_words)
    return exact, similarity


def load_tester(model_name: str):
    """Load a tester by name."""
    if model_name == "byt5":
        path = find_model_path("byt5")
        if path:
            return ByT5Tester(str(path))
        print(f"  byt5 checkpoint not found, skipping")
        return None
    elif model_name == "t5_spell":
        path = find_model_path("t5_spell")
        if path:
            return T5SpellTester(str(path))
        # Fall back to HuggingFace model
        print("  Loading T5-large-spell from HuggingFace ...")
        return T5SpellTester("ai-forever/T5-large-spell")
    else:
        print(f"  Unknown model: {model_name}")
        return None


# ======================================================================
# Benchmark runner
# ======================================================================

def test_single_model(model_name: str) -> Dict | None:
    print(f"\n{'=' * 60}")
    print(f"  Testing: {model_name.upper()}")
    print(f"{'=' * 60}")

    tester = load_tester(model_name)
    if tester is None:
        return None

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "by_difficulty": {},
        "total": {"correct": 0, "total": 0, "accuracy": 0.0, "avg_latency_ms": 0.0},
        "details": [],
    }

    all_latencies = []

    for difficulty, queries in BENCHMARK_QUERIES.items():
        print(f"\n  {difficulty.upper()} ({len(queries)} queries)")
        print("  " + "-" * 45)

        diff_correct = 0

        for typo_query, expected in queries:
            result = tester.test(typo_query)

            if result.get("error"):
                print(f"    X {typo_query[:30]:<30} -> ERROR")
                results["details"].append({
                    "difficulty": difficulty, "input": typo_query,
                    "expected": expected, "output": "ERROR",
                    "correct": False, "similarity": 0.0, "latency_ms": 0,
                })
                results["total"]["total"] += 1
                continue

            predicted = result["output"]
            latency = result["latency_ms"]
            exact, similarity = calculate_accuracy(predicted, expected)

            all_latencies.append(latency)
            results["total"]["total"] += 1
            if exact:
                diff_correct += 1
                results["total"]["correct"] += 1

            status = "OK" if exact else ("~" if similarity >= 0.5 else "X")
            pred_short = predicted[:28] + ".." if len(predicted) > 28 else predicted
            print(f"    {status:>2} {typo_query[:25]:<25} -> {pred_short:<30} ({latency:.0f}ms)")

            results["details"].append({
                "difficulty": difficulty, "input": typo_query,
                "expected": expected, "output": predicted,
                "correct": exact, "similarity": similarity, "latency_ms": latency,
            })

        acc = diff_correct / len(queries) * 100
        results["by_difficulty"][difficulty] = {
            "correct": diff_correct, "total": len(queries), "accuracy": acc,
        }
        print(f"  -> {diff_correct}/{len(queries)} ({acc:.1f}%)")

    if all_latencies:
        results["total"]["avg_latency_ms"] = round(sum(all_latencies) / len(all_latencies), 1)
    if results["total"]["total"] > 0:
        results["total"]["accuracy"] = round(
            results["total"]["correct"] / results["total"]["total"] * 100, 1
        )
    return results


def print_summary(all_results: List[Dict]):
    print("\n" + "=" * 75)
    print("  BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"\n  {'Model':<15} {'Easy':<10} {'Medium':<10} {'Hard':<10} {'Extreme':<10} {'TOTAL':<10} {'Latency'}")
    print("  " + "-" * 70)

    for r in all_results:
        if r is None:
            continue
        vals = []
        for d in ["easy", "medium", "hard", "extreme"]:
            vals.append(r["by_difficulty"].get(d, {}).get("accuracy", 0))
        total = r["total"]["accuracy"]
        lat = r["total"]["avg_latency_ms"]
        print(
            f"  {r['model']:<15} "
            + "".join(f"{v:>6.1f}%   " for v in vals)
            + f"{total:>6.1f}%   {lat:>6.0f}ms"
        )

    print("  " + "-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark typo correction models")
    parser.add_argument("--models", nargs="+", default=["byt5"],
                        help="Models to test: byt5, t5_spell")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    total_q = sum(len(q) for q in BENCHMARK_QUERIES.values())
    print(f"\n  Queries: {total_q} total  |  Models: {', '.join(args.models)}")

    all_results = []
    for model_name in args.models:
        try:
            result = test_single_model(model_name)
            all_results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            all_results.append(None)

    print_summary(all_results)

    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path(__file__).parent / f"benchmark_results_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": [r for r in all_results if r is not None],
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    main()
