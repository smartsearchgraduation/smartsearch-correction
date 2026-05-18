# Correction — Locust performance scaffolding

This directory contains the Locust load-generator script for the Correction
FastAPI service. It is supplemental to the pytest suite at
`Correction/tests/integration/test_performance.py`; the pytest suite uses
mocked weights, while this script targets the real GPU-loaded service.

These runs CANNOT execute in CI — the Correction service requires the
ByT5-Large-V3 weights (~2.4 GB) loaded on an RTX 5070 Ti.

## Deliverable rows covered

| Row(s)                              | Scenario                  | Locust args               |
|-------------------------------------|---------------------------|---------------------------|
| PERF-CR-001                         | SymSpell single-call P95  | `-u 1 -r 1 -t 60s`        |
| PERF-CR-002                         | ByT5 single-call P95      | `-u 1 -r 1 -t 60s`        |
| PERF-CR-003                         | cold-start probe          | `-u 1 -r 1 -t 10s`        |
| PERF-CR-005                         | masking-aware single-call | `-u 1 -r 1 -t 60s`        |
| PERF-CR-004 (manual M6 only)        | concurrent throughput     | `-u 8 -r 8 -t 60s`        |

PERF-CR-004 is IGNORED IN CI per the deliverable's Scope Limitation note;
it is runnable manually at M6 against the production GPU host.

## Run commands

Single-user latency runs:

```
cd Correction
locust -f perf/locustfile.py --headless \
    -u 1 -r 1 -t 60s \
    --host http://localhost:8001 \
    --csv perf/out/cr_single
```

Cold-start probe (restart the FastAPI process first, then run immediately):

```
cd Correction
locust -f perf/locustfile.py --headless \
    -u 1 -r 1 -t 10s \
    --host http://localhost:8001 \
    --csv perf/out/cr_coldstart
```

Concurrent throughput (manual M6 only):

```
cd Correction
locust -f perf/locustfile.py --headless \
    -u 8 -r 8 -t 60s \
    --host http://localhost:8001 \
    --csv perf/out/cr_concurrent
```

## Prerequisites

- Correction FastAPI service running on `http://localhost:8001`.
- ByT5-Large-V3 weights present on disk and loaded on the RTX 5070 Ti.
- The active model name in `locustfile.py:ACTIVE_MODEL` matches the model
  the running service has loaded (default `byt5-large-v3`).
- Brand-masking dictionary present if testing the masking path
  (PERF-CR-005).
- The `eval_t5.jsonl` evaluation fixture is NOT currently on disk in this
  workspace; the locustfile substitutes an in-file `SAMPLE_QUERIES` list.
  Replace with file-loaded samples once the eval fixture lands.

## Output

`--csv perf/out/cr_single` (or whichever name) produces the standard locust
CSV bundle. Compute offline p50/p95 from `*_stats_history.csv` for each row.
