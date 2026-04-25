"""Test bootstrap."""
from __future__ import annotations
import shutil, sys, types, uuid
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if "app" not in sys.modules:
    fake_app = types.ModuleType("app")
    fake_app.__path__ = [str(ROOT / "app")]
    sys.modules["app"] = fake_app

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
