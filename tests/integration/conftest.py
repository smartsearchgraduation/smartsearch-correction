"""Integration-test conftest for the Correction service.

The 5.4.3 perf rows are gated on either a live GPU + model weights
(BYT5-Large-V3 / T5-Large-V2.1), the Locust binary, or files that are not
on disk during case-writing (`eval_t5.jsonl`, `benchmark_universal.py`).
Each test in this directory therefore declares its own `@pytest.mark.skip`
with the spec-defined reason, and this conftest just makes the parent
`Correction/tests` importable for any future helpers.

Stub eviction
-------------
The parent ``Correction/tests/conftest.py`` installs lightweight stubs for
``torch`` / ``transformers`` / ``peft`` so unit tests can run on a GPU-less
CI box. Integration perf rows (PERF-CR-001/002/004/005/006/007) require the
real CUDA-enabled stack: ``torch.cuda.is_available()`` must reflect the
actual hardware, not the stub's hard-coded ``False``. Pytest loads
conftest.py files top-down, so by the time we run we are guaranteed the
parent already installed its stubs; we drop them here BEFORE any
``tests/integration/`` test module is imported, then force-import the real
packages so the real ``torch`` lands in ``sys.modules`` and downstream
``import torch`` lookups grab it.

Unit-test scope is unaffected: this file is only loaded when pytest
collects something under ``tests/integration/``.
"""
from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Evict the parent conftest's torch / transformers / peft stubs so the real
# packages can be imported below. A real torch module exposes ``__version__``
# and ``__file__``; the stub does neither, so we use that as the
# discriminator. Iterate over a snapshot of sys.modules because we mutate it.
# ---------------------------------------------------------------------------
_STUB_ROOTS = ("torch", "transformers", "peft")
for _mod_name in list(sys.modules):
    _root = _mod_name.split(".", 1)[0]
    if _root in _STUB_ROOTS:
        _m = sys.modules.get(_mod_name)
        if _m is not None and not hasattr(_m, "__version__") and not hasattr(
            _m, "__file__"
        ):
            # SimpleNamespace-backed stub from tests/conftest.py — drop it.
            del sys.modules[_mod_name]
        elif isinstance(_m, types.ModuleType) and getattr(
            _m, "__file__", None
        ) is None:
            # Bare ModuleType placeholder (e.g. transformers.utils stub).
            del sys.modules[_mod_name]

# Force-import the real packages now so any downstream ``import torch`` in
# integration test modules resolves to the real CUDA-enabled wheel rather
# than re-triggering the parent stub install (which is guarded by
# ``"torch" not in sys.modules`` and would otherwise re-install the stub).
try:
    import torch  # noqa: F401
except Exception:
    # Real torch genuinely missing — leave the slot empty so the perf
    # tests' own try/except records ``_GPU_AVAILABLE = False`` and skips
    # with the documented reason.
    pass
try:
    import transformers  # noqa: F401
except Exception:
    pass
try:
    import peft  # noqa: F401
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
