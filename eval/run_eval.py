"""
run_eval.py
===========
Run the 487-query benchmark against the Correction API for each model.

Prereq:
    1. Start the Correction API in another terminal:
           cd Correction
           python api.py        # serves on http://localhost:5001
    2. Build the dataset (one-time):
           python eval/build_dataset.py
    3. Run this:
           python eval/run_eval.py

Output:
    eval/results.json     — all model x query rows with timing
    eval/results.csv      — flat CSV (Excel/Sheets-friendly)

Resilience:
    - 3 retries per request with exponential backoff
    - per-model warm-up call (so the first real query isn't penalised by the
      first-load latency from the corrector's lazy model loading)
    - skips models that fail to warm up after 5 attempts
    - resumable: if results.json already exists, skips already-done
      (model, query_id) pairs
"""

import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

API_URL = os.environ.get("CORRECTION_API_URL", "http://localhost:5001/correct")
HEALTH_URL = os.environ.get("CORRECTION_HEALTH_URL", "http://localhost:5001/health")
DATASET = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "results.json")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

MODELS = [
    "BYT5-Large-V3",
    "T5-Large-V2.1",
    "qwen-3.5-2b",
    "byt5-base",
    "byt5-small",
]

REQUEST_TIMEOUT_S = 60          # generous for first-load
MAX_RETRIES = 3
WARMUP_MAX_TRIES = 5

WARMUP_QUERY = "tst querry"     # something that *should* trigger model load


def http_post(payload, timeout=REQUEST_TIMEOUT_S):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health_check():
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("status") == "healthy"
    except Exception as e:
        print(f"[health] FAIL: {e}")
        return False


def warmup_model(model):
    print(f"[warmup] loading {model} ...", flush=True)
    last_err = None
    for attempt in range(WARMUP_MAX_TRIES):
        try:
            t0 = time.perf_counter()
            r = http_post({"query": WARMUP_QUERY, "model": model}, timeout=600)
            dt = (time.perf_counter() - t0) * 1000
            print(
                f"[warmup] {model} READY in {dt:7.0f} ms "
                f"-> resolved={r.get('model_used')!r} corrected={r.get('corrected_query')!r}",
                flush=True,
            )
            return True
        except Exception as e:
            last_err = e
            print(
                f"[warmup] {model} attempt {attempt+1}/{WARMUP_MAX_TRIES} failed: {e}",
                flush=True,
            )
            time.sleep(2 + attempt * 2)
    print(f"[warmup] {model} GAVE UP. last error: {last_err}", flush=True)
    return False


def call_with_retry(model, query):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            r = http_post({"query": query, "model": model})
            wall_ms = (time.perf_counter() - t0) * 1000
            return r, wall_ms, None
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = "<no body>"
            last_err = f"HTTP {e.code}: {body[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(0.4 * (attempt + 1))
    return None, 0.0, last_err


def load_dataset():
    with open(DATASET, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results():
    if not os.path.exists(RESULTS_JSON):
        return []
    try:
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def main():
    if not health_check():
        print("[fatal] API not healthy at", HEALTH_URL)
        print("        Start it first:  python api.py")
        sys.exit(1)

    dataset = load_dataset()
    existing = load_existing_results()
    seen = {(r["model"], r["id"]) for r in existing}
    print(f"[init] {len(dataset)} queries, {len(MODELS)} models, {len(existing)} cached results")

    # Warm up first
    live_models = []
    for m in MODELS:
        if warmup_model(m):
            live_models.append(m)
        else:
            print(f"[warn] skipping {m} (warmup failed)")
    if not live_models:
        print("[fatal] no models warmed up; aborting")
        sys.exit(2)

    results = list(existing)
    total_calls = len(live_models) * len(dataset)
    done_calls = sum(1 for r in existing if r["model"] in live_models)
    t_start = time.time()

    for m in live_models:
        for i, q in enumerate(dataset):
            key = (m, q["id"])
            if key in seen:
                continue

            r, wall_ms, err = call_with_retry(m, q["query"])
            done_calls += 1

            if err is not None:
                row = {
                    "model": m,
                    "id": q["id"],
                    "query": q["query"],
                    "expected": q["expected"],
                    "length": q["length"],
                    "difficulty": q["difficulty"],
                    "category": q["category"],
                    "corrected": "",
                    "model_used": "",
                    "changed": False,
                    "latency_ms": 0.0,
                    "wall_ms": wall_ms,
                    "error": err,
                }
            else:
                row = {
                    "model": m,
                    "id": q["id"],
                    "query": q["query"],
                    "expected": q["expected"],
                    "length": q["length"],
                    "difficulty": q["difficulty"],
                    "category": q["category"],
                    "corrected": r.get("corrected_query", ""),
                    "model_used": r.get("model_used", ""),
                    "changed": bool(r.get("changed", False)),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "wall_ms": wall_ms,
                    "error": "",
                }

            results.append(row)

            # Live progress every 10 rows
            if done_calls % 10 == 0 or done_calls == total_calls:
                elapsed = time.time() - t_start
                rate = done_calls / elapsed if elapsed > 0 else 0
                eta = (total_calls - done_calls) / rate if rate > 0 else 0
                print(
                    f"[run] {done_calls:4d}/{total_calls:4d}  "
                    f"({100*done_calls/total_calls:5.1f}%)  "
                    f"model={m}  id={q['id']}  "
                    f"rate={rate:5.2f} q/s  eta={eta/60:5.1f} min",
                    flush=True,
                )

            # Periodic checkpoint
            if done_calls % 50 == 0:
                with open(RESULTS_JSON, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # Final write
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fields = list(results[0].keys()) if results else []
    if fields:
        with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(r)

    print(f"[done] wrote {RESULTS_JSON} ({len(results)} rows)")
    print(f"[done] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
