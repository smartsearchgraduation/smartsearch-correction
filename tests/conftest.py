"""Test bootstrap."""
from __future__ import annotations
import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if "app" not in sys.modules:
    fake_app = types.ModuleType("app")
    fake_app.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = fake_app

# ---------------------------------------------------------------------------
# Install torch / transformers / peft stubs BEFORE any test module is imported
# so that model files (which do `import torch` at module load) succeed without
# GPU dependencies.
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    _torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    _torch.device = lambda x: x
    _torch.no_grad = lambda: __import__("contextlib").nullcontext()
    sys.modules["torch"] = _torch

if "transformers" not in sys.modules:
    _tx = types.ModuleType("transformers")

    class _Stub:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def to(self, *a, **kw):
            return self

        def eval(self, *a, **kw):
            return self

    for _n in (
        "T5ForConditionalGeneration",
        "AutoTokenizer",
        "T5Tokenizer",
        "AutoModelForCausalLM",
        "PreTrainedTokenizer",
        "AutoModel",
        "AutoConfig",
        "BitsAndBytesConfig",
    ):
        setattr(_tx, _n, _Stub)
    sys.modules["transformers"] = _tx
    sys.modules["transformers.utils"] = types.ModuleType("transformers.utils")

if "peft" not in sys.modules:
    _peft = types.ModuleType("peft")

    class _PStub:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

    _peft.PeftModel = _PStub
    _peft.PeftConfig = _PStub
    sys.modules["peft"] = _peft

_TMP_BASE = Path(__file__).resolve().parent / ".tmp"


@pytest.fixture
def isolated_tmp():
    _TMP_BASE.mkdir(parents=True, exist_ok=True)
    p = _TMP_BASE / uuid.uuid4().hex
    p.mkdir(parents=True, exist_ok=True)
    try:
        yield p
    finally:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
