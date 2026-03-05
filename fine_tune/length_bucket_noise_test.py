#!/usr/bin/env python3
"""
Length-bucket stress test for ByT5 typo correction models.

Buckets:
  - 1-4 words
  - 5-7 words
  - 8-10 words
  - 10+ words

Noise is proportional to query length and severity.
Evaluation is strict binary only: correct | not_corrected.

Usage:
  python length_bucket_noise_test.py --models byt5-base byt5-small --save-json --save-csv
  python length_bucket_noise_test.py --show-each --model-max-length 384
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
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.corrector import TypoCorrector


BUCKETS: List[Tuple[str, int, int | None]] = [
    ("1-4", 1, 4),
    ("5-7", 5, 7),
    ("8-10", 8, 10),
    ("10+", 11, None),
]

SEVERITIES = ["clean", "medium", "hard", "extreme"]


@dataclass
class Sample:
    sample_id: int
    bucket: str
    severity: str
    input_query: str
    expected_query: str


BRANDS = [
    "apple", "samsung", "sony", "dell", "hp", "lenovo", "asus", "acer",
    "msi", "nvidia", "amd", "logitech", "razer", "corsair", "bose",
]
PRODUCTS = [
    "laptop", "tablet", "monitor", "router", "earbuds", "headphones", "keyboard",
    "mouse", "smartphone", "webcam", "microphone", "ssd", "drone", "camera",
]
QUALIFIERS = [
    "wireless", "gaming", "portable", "budget", "premium", "quiet", "lightweight",
    "durable", "oled", "4k", "bluetooth", "fast", "rgb", "compact",
]
USE_CASE = [
    "travel", "office", "gaming", "streaming", "editing", "study", "work",
]
CONNECTORS = ["for", "with", "and", "under", "best", "for"]


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def mutate_word(word: str, rng: random.Random, intensity: int) -> str:
    if len(word) < 3 or not any(c.isalpha() for c in word):
        return word

    chars = list(word)
    for _ in range(max(1, intensity)):
        op = rng.choice(["swap", "delete", "replace", "insert"])
        pos = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if not pos:
            return "".join(chars)

        if op == "swap" and len(pos) >= 2:
            p = rng.choice(pos[:-1])
            chars[p], chars[p + 1] = chars[p + 1], chars[p]
        elif op == "delete" and len(chars) >= 3:
            p = rng.choice(pos)
            del chars[p]
        elif op == "replace":
            p = rng.choice(pos)
            chars[p] = rng.choice(string.ascii_lowercase)
        elif op == "insert":
            p = rng.choice(pos)
            chars.insert(p, rng.choice(string.ascii_lowercase))

    return "".join(chars)


def build_clean_query(word_count: int, rng: random.Random) -> str:
    tokens = [
        rng.choice(BRANDS),
        rng.choice(PRODUCTS),
        rng.choice(QUALIFIERS),
        rng.choice(USE_CASE),
        rng.choice(QUALIFIERS),
        rng.choice(CONNECTORS),
        rng.choice(PRODUCTS),
        rng.choice(USE_CASE),
        rng.choice(QUALIFIERS),
        rng.choice(BRANDS),
        "under",
        str(rng.choice([500, 700, 1000, 1500])),
        "dollars",
        "with",
        rng.choice(["warranty", "reviews", "shipping", "support"]),
    ]

    while len(tokens) < word_count:
        tokens.append(rng.choice(QUALIFIERS + PRODUCTS + USE_CASE))

    return " ".join(tokens[:word_count])


def apply_noise(clean_query: str, severity: str, rng: random.Random) -> str:
    if severity == "clean":
        return clean_query

    tokens = clean_query.split()
    n_words = len(tokens)

    ratios = {
        "medium": 0.25,
        "hard": 0.45,
        "extreme": 0.70,
    }
    intensity_map = {
        "medium": 1,
        "hard": 2,
        "extreme": 3,
    }

    edits = max(1, round(n_words * ratios[severity]))
    edits = min(edits, n_words)
    idxs = rng.sample(range(n_words), edits)

    for i in idxs:
        tokens[i] = mutate_word(tokens[i], rng, intensity_map[severity])

    if severity in {"hard", "extreme"} and n_words >= 5 and rng.random() < 0.45:
        drop_i = rng.randint(1, len(tokens) - 2)
        del tokens[drop_i]

    if severity == "extreme":
        if rng.random() < 0.55 and len(tokens) >= 4:
            merge_i = rng.randint(1, len(tokens) - 2)
            tokens[merge_i] = tokens[merge_i] + tokens[merge_i + 1]
            del tokens[merge_i + 1]
        if rng.random() < 0.55:
            insert_i = rng.randint(0, len(tokens))
            tokens.insert(insert_i, "".join(rng.choices(string.ascii_lowercase + string.digits, k=3)))

    return " ".join(tokens)


def sample_word_count(min_words: int, max_words: int | None, rng: random.Random) -> int:
    if max_words is None:
        return rng.randint(min_words, 16)
    return rng.randint(min_words, max_words)


def generate_dataset(seed: int, per_bucket: int) -> List[Sample]:
    rng = random.Random(seed)
    dataset: List[Sample] = []

    per_severity = per_bucket // 4
    remainder = per_bucket - per_severity * 4

    for bucket_name, min_words, max_words in BUCKETS:
        plan = {sev: per_severity for sev in SEVERITIES}
        plan["clean"] += remainder

        for severity in SEVERITIES:
            for _ in range(plan[severity]):
                wc = sample_word_count(min_words, max_words, rng)
                clean_query = build_clean_query(wc, random.Random(rng.randint(0, 10**9)))
                noisy_query = apply_noise(clean_query, severity, random.Random(rng.randint(0, 10**9)))

                dataset.append(
                    Sample(
                        sample_id=len(dataset) + 1,
                        bucket=bucket_name,
                        severity=severity,
                        input_query=noisy_query,
                        expected_query=clean_query,
                    )
                )

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

    bucket_summary = {
        b[0]: {"correct": 0, "total": 0} for b in BUCKETS
    }
    bucket_severity_summary: Dict[str, Dict[str, Dict[str, int]]] = {
        b[0]: {s: {"correct": 0, "total": 0} for s in SEVERITIES} for b in BUCKETS
    }

    rows = []
    total_correct = 0
    total_latency = 0.0

    for idx, sample in enumerate(dataset, start=1):
        out = corrector.correct(sample.input_query, model=model_name)
        predicted = out.get("corrected_query", sample.input_query)
        latency_ms = float(out.get("latency_ms", 0.0))

        is_correct = normalize(predicted) == normalize(sample.expected_query)
        status = "correct" if is_correct else "not_corrected"

        bucket_summary[sample.bucket]["total"] += 1
        bucket_severity_summary[sample.bucket][sample.severity]["total"] += 1
        if is_correct:
            bucket_summary[sample.bucket]["correct"] += 1
            bucket_severity_summary[sample.bucket][sample.severity]["correct"] += 1
            total_correct += 1

        total_latency += latency_ms

        rows.append(
            {
                "sample_id": sample.sample_id,
                "bucket": sample.bucket,
                "severity": sample.severity,
                "input_query": sample.input_query,
                "expected_query": sample.expected_query,
                "predicted_query": predicted,
                "status": status,
                "latency_ms": round(latency_ms, 2),
            }
        )

        if show_each:
            print(
                f"  [{idx:>4}/{len(dataset)}] bucket={sample.bucket:<4} severity={sample.severity:<7} "
                f"status={status:<13} latency={round(latency_ms, 2)}ms"
            )
            print(f"      input:    {sample.input_query}")
            print(f"      predicted:{predicted}")
            print(f"      expected: {sample.expected_query}")
        elif idx % 50 == 0:
            print(f"  processed {idx}/{len(dataset)}")

    bucket_report = {}
    for bucket, c in bucket_summary.items():
        acc = (c["correct"] / c["total"] * 100.0) if c["total"] else 0.0
        bucket_report[bucket] = {
            "correct": c["correct"],
            "total": c["total"],
            "accuracy": round(acc, 2),
        }

    bucket_severity_report: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for bucket, sev_data in bucket_severity_summary.items():
        bucket_severity_report[bucket] = {}
        for severity, c in sev_data.items():
            acc = (c["correct"] / c["total"] * 100.0) if c["total"] else 0.0
            bucket_severity_report[bucket][severity] = {
                "correct": c["correct"],
                "total": c["total"],
                "accuracy": round(acc, 2),
            }

    total_acc = total_correct / len(dataset) * 100.0
    avg_latency = total_latency / len(dataset)

    print("\n  Bucket accuracy:")
    for b in [x[0] for x in BUCKETS]:
        item = bucket_report[b]
        print(f"    {b:>4}: {item['correct']}/{item['total']} ({item['accuracy']}%)")
    print(f"  total: {total_correct}/{len(dataset)} ({round(total_acc, 2)}%) | avg latency: {round(avg_latency, 2)} ms")

    return {
        "model": model_name,
        "bucket_summary": bucket_report,
        "bucket_severity_summary": bucket_severity_report,
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
    out_path = output_dir / f"length_bucket_results_{stamp}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_csv(model_result: Dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_result["model"].replace("/", "_")
    out_path = output_dir / f"length_bucket_rows_{model_name}_{stamp}.csv"
    fieldnames = [
        "sample_id", "bucket", "severity", "input_query", "expected_query",
        "predicted_query", "status", "latency_ms",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(model_result["rows"])
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Length-bucket typo correction benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["byt5-base", "byt5-small"],
        help="Models to evaluate (default: byt5-base byt5-small)",
    )
    parser.add_argument("--seed", type=int, default=20260305, help="Random seed")
    parser.add_argument("--per-bucket", type=int, default=200, help="Total samples per bucket (default: 200)")
    parser.add_argument("--model-max-length", type=int, default=384, help="Inference max_length (default: 384)")
    parser.add_argument("--show-each", action="store_true", help="Print every sample in terminal")
    parser.add_argument("--save-json", action="store_true", help="Save aggregated results JSON")
    parser.add_argument("--save-csv", action="store_true", help="Save per-row CSV for each model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building length-bucket dataset ...")
    dataset = generate_dataset(seed=args.seed, per_bucket=args.per_bucket)
    print(f"Dataset ready: total={len(dataset)} | per_bucket={args.per_bucket} | buckets={', '.join(b[0] for b in BUCKETS)}")

    corrector = TypoCorrector()
    all_results = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "model_max_length": args.model_max_length,
        "per_bucket": args.per_bucket,
        "buckets": [b[0] for b in BUCKETS],
        "severity_levels": SEVERITIES,
        "evaluation_rule": "strict_binary_only: correct | not_corrected",
        "results": [],
    }

    start = time.perf_counter()
    for model_name in args.models:
        model_result = evaluate_model(
            model_name=model_name,
            dataset=dataset,
            corrector=corrector,
            model_max_length=args.model_max_length,
            show_each=args.show_each,
        )
        all_results["results"].append(model_result)

    elapsed = time.perf_counter() - start
    all_results["elapsed_sec"] = round(elapsed, 2)
    print(f"\nCompleted in {round(elapsed, 2)} seconds")

    output_dir = Path(__file__).parent / "outputs"
    if args.save_json:
        p = save_json(all_results, output_dir)
        print(f"JSON saved: {p}")
    if args.save_csv:
        for r in all_results["results"]:
            p = save_csv(r, output_dir)
            print(f"CSV saved: {p}")


if __name__ == "__main__":
    main()
