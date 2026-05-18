"""
benchmark_universal.py
======================

Per-model profiling script referenced by spec section 5.4.3 and the
PERF-CR-007 row in performance_testing.md.

Loads the requested Correction model via the existing TypoCorrector
orchestrator, runs N warmup + M measured /correct calls in-process, and
emits a JSON line per model containing:

  - model
  - samples
  - p50_ms
  - p95_ms
  - throughput_rps
  - vram_peak_mb         (if CUDA available, else null)

Output is written to stdout (so existing CI plumbing can pipe it) AND to
correction_baseline.json at the script's directory if --out is supplied.

Usage
-----
  python benchmark_universal.py
  python benchmark_universal.py --models BYT5-Large-V3 T5-Large-V2.1
  python benchmark_universal.py --iterations 50 --warmup 5
  python benchmark_universal.py --out correction_baseline.json

This script is intentionally lightweight: it is not a substitute for
Locust under sustained load. It exists so PERF-CR-007 can run in CI and
so the +10 percent pass band has a recorded baseline file. For the M6
production baseline, increase --iterations and run on the deployment GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional


# Make `app.corrector` importable regardless of CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


_DEFAULT_MODELS = ["BYT5-Large-V3"]
_DEFAULT_QUERIES = [
    "iphn 15 pro",
    "addidas snekers",
    "samsng galxy s24",
    "lapyop charger",
    "wireles mose",
    "blutooth speker",
    "cofee makker",
    "runing shose for men",
    "labtop bag 15 inch",
    "phone caes for iphone",
]


def _percentile(samples: list[float], p: int) -> Optional[float]:
    if not samples:
        return None
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _vram_peak_mb() -> Optional[float]:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        return round(peak_bytes / (1024 * 1024), 2)
    except Exception:
        return None


def _reset_vram_peak() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def benchmark_model(
    corrector: Any,
    model_name: str,
    queries: list[str],
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    """Run warmup + measured iterations against `model_name` via TypoCorrector.correct."""
    _reset_vram_peak()

    # Warmup
    for i in range(warmup):
        q = queries[i % len(queries)]
        try:
            corrector.correct(q, model=model_name)
        except Exception:  # pragma: no cover - safety
            # Surface model-load failures to the caller.
            raise

    samples_ms: list[float] = []
    t_total_start = time.perf_counter()
    for i in range(iterations):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        corrector.correct(q, model=model_name)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    total_seconds = time.perf_counter() - t_total_start

    return {
        "model": model_name,
        "samples": iterations,
        "warmup_discarded": warmup,
        "p50_ms": _percentile(samples_ms, 50),
        "p95_ms": _percentile(samples_ms, 95),
        "mean_ms": round(statistics.fmean(samples_ms), 3) if samples_ms else None,
        "throughput_rps": round(iterations / total_seconds, 3) if total_seconds > 0 else None,
        "vram_peak_mb": _vram_peak_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal Correction model benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=_DEFAULT_MODELS,
        help="Model names from app.corrector._MODEL_REGISTRY (e.g. BYT5-Large-V3 T5-Large-V2.1).",
    )
    parser.add_argument("--iterations", type=int, default=10, help="Measured /correct calls per model.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup calls (discarded).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the JSON result. Defaults to stdout-only.",
    )
    args = parser.parse_args()

    # Import lazily so --help works without weights present.
    try:
        from app.corrector import TypoCorrector  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"failed to import TypoCorrector: {exc!r}"}))
        return 2

    corrector = TypoCorrector()
    queries = _DEFAULT_QUERIES

    results: list[dict[str, Any]] = []
    for model in args.models:
        try:
            res = benchmark_model(corrector, model, queries, args.iterations, args.warmup)
        except Exception as exc:
            res = {"model": model, "error": repr(exc)}
        results.append(res)

    payload: dict[str, Any] = {
        "schema": "correction_baseline.v1",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "queries_pool_size": len(queries),
        "results": results,
    }

    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _HERE / out_path
        out_path.write_text(serialized, encoding="utf-8")

    # Non-zero exit if every model errored.
    if results and all("error" in r for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
