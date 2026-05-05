"""Shared torch / transformers / peft stubs for model tests (no GPU needed)."""
from __future__ import annotations
import sys
import types


def install() -> None:
    """Idempotently install lightweight stubs so model files import cleanly."""
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_mod.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        torch_mod.device = lambda x: x
        torch_mod.no_grad = lambda: __import__("contextlib").nullcontext()
        sys.modules["torch"] = torch_mod

    if "transformers" not in sys.modules:
        tx = types.ModuleType("transformers")

        class _Stub:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def to(self, *a, **kw):
                return self

            def eval(self, *a, **kw):
                return self

        for n in (
            "T5ForConditionalGeneration",
            "AutoTokenizer",
            "T5Tokenizer",
            "AutoModelForCausalLM",
            "PreTrainedTokenizer",
            "AutoModel",
            "AutoConfig",
            "BitsAndBytesConfig",
        ):
            setattr(tx, n, _Stub)
        sys.modules["transformers"] = tx
        sys.modules["transformers.utils"] = types.ModuleType("transformers.utils")

    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")

        class _PStub:
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

        peft.PeftModel = _PStub
        peft.PeftConfig = _PStub
        sys.modules["peft"] = peft
