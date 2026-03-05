#!/usr/bin/env python3
"""
Long-query stress test for ByT5 typo correction models.

Generates an English-only deterministic dataset:
    - 250 medium-noise long queries
    - 250 hard-noise long queries
    - 250 extreme-noise long queries
    - 100 clean long queries (no typo)

Evaluation is STRICT and binary only:
    - correct
    - not_corrected

No partial/similarity scoring is used.

Usage:
    python long_query_stress_test.py
    python long_query_stress_test.py --models byt5-base byt5-small --save-json --save-csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.corrector import TypoCorrector


MEDIUM_COUNT = 250
HARD_COUNT = 250
EXTREME_COUNT = 250
CLEAN_COUNT = 100
TOTAL_COUNT = MEDIUM_COUNT + HARD_COUNT + EXTREME_COUNT + CLEAN_COUNT


@dataclass
class Sample:
    sample_id: int
    split: str
    input_query: str
    expected_query: str


BRANDS = [
    "apple", "samsung", "sony", "dell", "hp", "lenovo", "asus", "acer", "msi", "nvidia",
    "amd", "logitech", "razer", "corsair", "steelseries", "bose", "canon", "nikon", "gopro", "dji",
]

PRODUCTS = [
    "laptop", "gaming laptop", "wireless mouse", "mechanical keyboard", "monitor", "noise cancelling headphones",
    "smart watch", "tablet", "smartphone", "graphics card", "ssd", "router", "webcam", "microphone",
    "drone", "mirrorless camera", "soundbar", "earbuds", "power bank", "usb c hub",
]

ADJECTIVES = [
    "lightweight", "refurbished", "premium", "budget", "durable", "portable", "high performance", "silent",
    "compact", "ultra thin", "long battery", "fast charging", "water resistant", "professional", "entry level",
]

FEATURES = [
    "with 32gb ram", "with 16gb ram", "with oled display", "with 4k display", "with 240hz refresh rate",
    "with usb c charging", "with bluetooth 5.3", "with ai noise reduction", "with rgb lighting",
    "with optical switches", "with low latency mode", "with active cooling", "with dual band wifi",
    "with image stabilization", "with extended warranty",
]

USE_CASES = [
    "for video editing", "for software development", "for competitive gaming", "for remote work", "for university study",
    "for travel", "for streaming setup", "for content creation", "for daily office tasks", "for home entertainment",
]

CONSTRAINTS = [
    "under 1500 dollars", "under 1000 dollars", "under 700 dollars", "available in black", "available in white",
    "available in silver", "with next day shipping", "from verified seller", "with good customer reviews",
    "with official warranty",
]


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def build_clean_long_query(index: int) -> str:
    brand = BRANDS[index % len(BRANDS)]
    product = PRODUCTS[(index * 3 + 1) % len(PRODUCTS)]
    adjective = ADJECTIVES[(index * 5 + 2) % len(ADJECTIVES)]
    feature_1 = FEATURES[(index * 7 + 3) % len(FEATURES)]
    feature_2 = FEATURES[(index * 11 + 4) % len(FEATURES)]
    use_case = USE_CASES[(index * 13 + 5) % len(USE_CASES)]
    constraint = CONSTRAINTS[(index * 17 + 6) % len(CONSTRAINTS)]

    return (
        f"looking for a {adjective} {brand} {product} {feature_1} and {feature_2} "
        f"{use_case} {constraint}"
    )


def mutate_word(word: str, rng: random.Random, intensity: int) -> str:
    if len(word) < 4 or not any(c.isalpha() for c in word):
        return word

    letters = list(word)
    operations = ["swap", "delete", "replace", "insert"]

    for _ in range(max(1, intensity)):
        op = rng.choice(operations)
        alpha_positions = [i for i, ch in enumerate(letters) if ch.isalpha()]
        if not alpha_positions:
            break

        if op == "swap" and len(alpha_positions) >= 2:
            pos = rng.choice(alpha_positions[:-1])
            nxt = pos + 1
            letters[pos], letters[nxt] = letters[nxt], letters[pos]

        elif op == "delete" and len(alpha_positions) >= 3:
            pos = rng.choice(alpha_positions)
            del letters[pos]

        elif op == "replace":
            pos = rng.choice(alpha_positions)
            letters[pos] = rng.choice(string.ascii_lowercase)

        elif op == "insert":
            pos = rng.choice(alpha_positions)
            letters.insert(pos, rng.choice(string.ascii_lowercase))

    return "".join(letters)


def apply_noise(query: str, level: str, rng: random.Random) -> str:
    words = query.split()
    candidate_positions = [i for i, w in enumerate(words) if len(w) >= 4 and any(c.isalpha() for c in w)]

    if not candidate_positions:
        return query

    if level == "medium":
        edits = rng.randint(2, 4)
        intensity = 1
    elif level == "hard":
        edits = rng.randint(4, 7)
        intensity = 2
    else:
        edits = rng.randint(7, 12)
        intensity = 3

    edits = min(edits, len(candidate_positions))
    selected_positions = rng.sample(candidate_positions, edits)

    for pos in selected_positions:
        words[pos] = mutate_word(words[pos], rng, intensity)

    if level in {"hard", "extreme"} and len(words) > 8:
        if rng.random() < 0.35:
            drop_pos = rng.randint(2, len(words) - 3)
            del words[drop_pos]
        if rng.random() < 0.30 and len(words) > 9:
            merge_pos = rng.randint(1, len(words) - 2)
            words[merge_pos] = words[merge_pos] + words[merge_pos + 1]
            del words[merge_pos + 1]

    if level == "extreme":
        if rng.random() < 0.6:
            insert_pos = rng.randint(0, len(words))
            noise_token = "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(2, 4)))
            words.insert(insert_pos, noise_token)
        if rng.random() < 0.35:
            punct = rng.choice(["??", "!!!", "..", "--"])
            words[rng.randint(0, len(words) - 1)] += punct

    return " ".join(words)


def generate_dataset(seed: int) -> List[Sample]:
    rng = random.Random(seed)
    dataset: List[Sample] = []

    for idx in range(MEDIUM_COUNT):
        expected = build_clean_long_query(idx)
        noisy = apply_noise(expected, "medium", random.Random(rng.randint(0, 10**9)))
        dataset.append(Sample(sample_id=len(dataset) + 1, split="medium", input_query=noisy, expected_query=expected))

    for idx in range(MEDIUM_COUNT, MEDIUM_COUNT + HARD_COUNT):
        expected = build_clean_long_query(idx)
        noisy = apply_noise(expected, "hard", random.Random(rng.randint(0, 10**9)))
        dataset.append(Sample(sample_id=len(dataset) + 1, split="hard", input_query=noisy, expected_query=expected))

    for idx in range(MEDIUM_COUNT + HARD_COUNT, MEDIUM_COUNT + HARD_COUNT + EXTREME_COUNT):
        expected = build_clean_long_query(idx)
        noisy = apply_noise(expected, "extreme", random.Random(rng.randint(0, 10**9)))
        dataset.append(Sample(sample_id=len(dataset) + 1, split="extreme", input_query=noisy, expected_query=expected))

    for idx in range(MEDIUM_COUNT + HARD_COUNT + EXTREME_COUNT, TOTAL_COUNT):
        expected = build_clean_long_query(idx)
        dataset.append(Sample(sample_id=len(dataset) + 1, split="clean", input_query=expected, expected_query=expected))

    return dataset


def evaluate_model(
    model_name: str,
    dataset: List[Sample],
    corrector: TypoCorrector,
    model_max_length: int,
    show_each: bool,
) -> Dict:
    print(f"\n{'=' * 72}")
    print(f"Testing model: {model_name}")
    print(f"{'=' * 72}")

    model_obj = corrector._get_model(model_name)
    model_obj.max_length = model_max_length
    print(f"  inference max_length={model_obj.max_length}")

    split_stats = {
        "medium": {"correct": 0, "total": 0},
        "hard": {"correct": 0, "total": 0},
        "extreme": {"correct": 0, "total": 0},
        "clean": {"correct": 0, "total": 0},
    }
    total_correct = 0
    total_latency = 0.0
    rows = []

    for idx, sample in enumerate(dataset, start=1):
        out = corrector.correct(sample.input_query, model=model_name)
        predicted = out.get("corrected_query", sample.input_query)
        latency_ms = float(out.get("latency_ms", 0.0))

        is_correct = normalize(predicted) == normalize(sample.expected_query)
        status = "correct" if is_correct else "not_corrected"

        split_stats[sample.split]["total"] += 1
        if is_correct:
            split_stats[sample.split]["correct"] += 1
            total_correct += 1

        total_latency += latency_ms

        rows.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "input_query": sample.input_query,
                "expected_query": sample.expected_query,
                "predicted_query": predicted,
                "status": status,
                "latency_ms": round(latency_ms, 2),
            }
        )

        if show_each:
            print(f"  [{idx:>4}/{len(dataset)}] split={sample.split:<7} status={status:<13} latency={round(latency_ms, 2)}ms")
            print(f"      input:    {sample.input_query}")
            print(f"      predicted:{predicted}")
            print(f"      expected: {sample.expected_query}")
        elif idx % 50 == 0:
            print(f"  processed {idx}/{len(dataset)}")

    summary = {}
    for split, s in split_stats.items():
        acc = (s["correct"] / s["total"] * 100.0) if s["total"] else 0.0
        summary[split] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(acc, 2),
        }

    total_acc = total_correct / len(dataset) * 100.0
    avg_latency = total_latency / len(dataset)

    print(
        f"  medium:  {summary['medium']['correct']}/{summary['medium']['total']} ({summary['medium']['accuracy']}%)\n"
        f"  hard:    {summary['hard']['correct']}/{summary['hard']['total']} ({summary['hard']['accuracy']}%)\n"
        f"  extreme: {summary['extreme']['correct']}/{summary['extreme']['total']} ({summary['extreme']['accuracy']}%)\n"
        f"  clean:   {summary['clean']['correct']}/{summary['clean']['total']} ({summary['clean']['accuracy']}%)\n"
        f"  total:   {total_correct}/{len(dataset)} ({round(total_acc, 2)}%) | avg latency: {round(avg_latency, 2)} ms"
    )

    return {
        "model": model_name,
        "summary": summary,
        "overall": {
            "correct": total_correct,
            "total": len(dataset),
            "accuracy": round(total_acc, 2),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "rows": rows,
    }


def save_json(results: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"long_query_stress_results_{stamp}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_csv(model_result: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_result["model"].replace("/", "_")
    out_path = output_dir / f"long_query_stress_rows_{model_name}_{stamp}.csv"

    fieldnames = [
        "sample_id", "split", "input_query", "expected_query",
        "predicted_query", "status", "latency_ms",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(model_result["rows"])
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-query stress test for ByT5 models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["byt5-base", "byt5-small"],
        help="Models to evaluate (default: byt5-base byt5-small)",
    )
    parser.add_argument("--seed", type=int, default=20260305, help="Random seed for deterministic dataset")
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=384,
        help="Inference max_length for long-query benchmark (default: 384)",
    )
    parser.add_argument(
        "--show-each",
        action="store_true",
        help="Print every sample step-by-step in terminal",
    )
    parser.add_argument("--save-json", action="store_true", help="Save aggregated results as JSON")
    parser.add_argument("--save-csv", action="store_true", help="Save per-sample rows as CSV for each model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building deterministic dataset ...")
    dataset = generate_dataset(seed=args.seed)
    print(
        f"Dataset ready: total={len(dataset)} | "
        f"medium={MEDIUM_COUNT}, hard={HARD_COUNT}, extreme={EXTREME_COUNT}, clean={CLEAN_COUNT}"
    )

    corrector = TypoCorrector()
    all_results = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "model_max_length": args.model_max_length,
        "counts": {
            "medium": MEDIUM_COUNT,
            "hard": HARD_COUNT,
            "extreme": EXTREME_COUNT,
            "clean": CLEAN_COUNT,
            "total": TOTAL_COUNT,
        },
        "evaluation_rule": "strict_binary_only: correct | not_corrected",
        "results": [],
    }

    start = time.perf_counter()
    for model_name in args.models:
        model_result = evaluate_model(
            model_name,
            dataset,
            corrector,
            model_max_length=args.model_max_length,
            show_each=args.show_each,
        )
        all_results["results"].append(model_result)

    elapsed = time.perf_counter() - start
    all_results["elapsed_sec"] = round(elapsed, 2)
    print(f"\nCompleted in {round(elapsed, 2)} seconds")

    output_dir = Path(__file__).parent / "outputs"
    if args.save_json:
        path = save_json(all_results, output_dir)
        print(f"JSON saved: {path}")

    if args.save_csv:
        for model_result in all_results["results"]:
            path = save_csv(model_result, output_dir)
            print(f"CSV saved: {path}")


if __name__ == "__main__":
    main()
