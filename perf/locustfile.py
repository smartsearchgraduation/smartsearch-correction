"""
Locust performance scaffolding for the Correction service (FastAPI).

Targets the running Correction FastAPI service (uvicorn). The service must
be started with model weights already loaded on the RTX 5070 Ti — these runs
CANNOT execute in CI because CI hardware has no GPU and the ByT5-Large-V3
weights (~2.4 GB) are not packaged.

Deliverable rows targeted by this file
--------------------------------------
The pytest suite covers single-call latency under mocked weights. Locust is
here to drive the rows where the deliverable explicitly calls for either a
cold-start probe or sustained concurrency that pytest skips.

  PERF-CR-001   single-query latency (SymSpell-equivalent path)
                  --users 1 -t 60s
  PERF-CR-002   single-query latency (ByT5 active model)
                  --users 1 -t 60s
  PERF-CR-003   cold-start / first-token probe (one-shot)
                  --users 1 -t 10s     <-- short window on a fresh process
  PERF-CR-005   masking-aware single-query latency
                  --users 1 -t 60s
  PERF-CR-004   concurrent throughput (M6 manual run only;
                  IGNORED IN CI per Scope Limitation)
                  --users 8 -t 60s

Run commands
------------
Single-user latency (PERF-CR-001 / 002 / 005):

    cd Correction
    locust -f perf/locustfile.py --headless \\
        -u 1 -r 1 -t 60s \\
        --host http://localhost:8001 \\
        --csv perf/out/cr_single

Cold-start probe (PERF-CR-003) — restart the FastAPI process first:

    cd Correction
    locust -f perf/locustfile.py --headless \\
        -u 1 -r 1 -t 10s \\
        --host http://localhost:8001 \\
        --csv perf/out/cr_coldstart

Concurrent throughput (PERF-CR-004) — manual M6 only, GPU saturation expected:

    cd Correction
    locust -f perf/locustfile.py --headless \\
        -u 8 -r 8 -t 60s \\
        --host http://localhost:8001 \\
        --csv perf/out/cr_concurrent

Notes
-----
- The eval_t5.jsonl fixture referenced in the deliverable is not on disk in
  this workspace, so SAMPLE_QUERIES below is an in-file substitute. Replace
  with file-loaded samples once the eval fixture lands.
- --csv dumps per-task latency rows so an offline tool can compute p50/p95
  without locust's web UI.
- /correct expects {"query": ..., "model": ...}. Adjust ACTIVE_MODEL if the
  active model on disk differs from "byt5-large-v3".
"""

import os
import random

from locust import HttpUser, task, between

# ---------------------------------------------------------------------------
# Optional Prometheus integration (gated by ImportError so locust still works
# even if prometheus_client is absent). The harness sets LOCUST_PROM_PORT
# from the run_all_perf.bat. Defaults to 9302 (Correction slot).
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, start_http_server
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _PROM_AVAILABLE = False

_PROM_PORT = int(os.environ.get("LOCUST_PROM_PORT", "9302"))

if _PROM_AVAILABLE:
    REQ_LATENCY = Histogram(
        "locust_request_duration_seconds",
        "Locust request latency",
        ["endpoint", "method", "status"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    REQ_TOTAL = Counter(
        "locust_request_total",
        "Locust request count",
        ["endpoint", "method", "status"],
    )

    from locust import events as _events

    @_events.init.add_listener
    def _start_prom(environment, **kwargs):  # noqa: D401
        try:
            start_http_server(_PROM_PORT)
        except OSError:
            pass

    @_events.request.add_listener
    def _record_request(
        request_type,
        name,
        response_time,
        response_length,
        response,
        context,
        exception,
        start_time,
        url,
        **kwargs,
    ):
        status = "fail" if exception else "ok"
        REQ_LATENCY.labels(
            endpoint=name, method=request_type, status=status
        ).observe(response_time / 1000.0)
        REQ_TOTAL.labels(
            endpoint=name, method=request_type, status=status
        ).inc()


# ---------------------------------------------------------------------------
# In-file fixture (eval_t5.jsonl is missing — see header note)
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    "runing shoes",                # single-typo
    "wireles bluetoth headphones", # multi-typo, brand-bearing
    "iphne 14 pro max",            # brand-bearing, masked path
    "samsng galaxy s23",           # brand-bearing, masked path
    "coffe machine",               # single-typo
    "winter jacet",                # single-typo
    "lightweght laptop",           # single-typo
    "office chair ergonomics",     # clean
    "leather walet",               # single-typo
    "smartwatch black",            # clean
    "gaming mouse rgb",            # clean
    "hidrolik pres",               # foreign — model-agnostic path
]

# Active model name as wired in the running Correction service. Update this
# when promoting a new checkpoint.
ACTIVE_MODEL = "byt5-large-v3"


class CorrectionUser(HttpUser):
    """
    Simulated client of the Correction FastAPI service.

    Weight skew is intentional: /correct dominates so the GPU stays busy;
    /health is a cheap liveness sample to confirm the service stays up
    under load.
    """

    wait_time = between(0.0, 0.1)

    @task(10)
    def correct(self):
        """POST /correct — main hot path (PERF-CR-001/002/003/005)."""
        query = random.choice(SAMPLE_QUERIES)
        with self.client.post(
            "/correct",
            json={"query": query, "model": ACTIVE_MODEL},
            name="POST /correct",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code in (400, 422):
                # Validation rejection is acceptable under load.
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")

    @task(1)
    def health(self):
        """GET /health — liveness sampler."""
        self.client.get("/health", name="GET /health")
