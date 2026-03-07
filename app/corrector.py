"""
TypoCorrector – E-commerce Search Query Spell Correction Orchestrator

Single-model architecture using ByT5 (byte-level typo correction).
Model is lazy-loaded on first use to keep startup fast.

Usage:
    from app.corrector import TypoCorrector

    corrector = TypoCorrector()

    # Simple – just the corrected string
    text = corrector.correct_query("iphnoe 15 pro")   # → "iphone 15 pro"

    # Full – with metadata
    result = corrector.correct("iphnoe 15 pro")
"""

import logging
import time
from typing import Optional

from .models.byt5 import ByT5Corrector
from .models.base import BaseCorrector
from .models.qwen import QwenCorrector

logger = logging.getLogger(__name__)

# Model registry: name -> (path_suffix, description)
_MODEL_REGISTRY = {
    "byt5-base": ("byt5-typo-best", "ByT5-base fine-tuned (H100)"),
    "byt5-small": ("byt5-typo-final", "ByT5-small fine-tuned"),
    "byt5-large": ("byt5-large/best", "ByT5-large fine-tuned"),
    "qwen-3.5-2b": ("models/qwen3.5-2b", "Qwen 3.5 2B (guarded typo-corrector)"),
}

_DEFAULT_MODEL = "byt5-base"


class TypoCorrector:
    """
    Central orchestrator for spell correction.

    Supports multiple ByT5 variants (small / base) with lazy loading.
    """

    def __init__(self):
        import os
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._models: dict[str, BaseCorrector] = {}
        logger.info("TypoCorrector initialised (available models: %s)", list(_MODEL_REGISTRY.keys()))

    def _get_model(self, model_name: str | None) -> BaseCorrector:
        """Get or lazily create a model instance."""
        name = model_name if model_name in _MODEL_REGISTRY else _DEFAULT_MODEL
        if name not in self._models:
            import os
            model_ref = _MODEL_REGISTRY[name][0]

            if name.startswith("qwen"):
                qwen_path = os.path.join(self._base_dir, model_ref)
                if os.path.exists(qwen_path):
                    self._models[name] = QwenCorrector(model_name_or_path=qwen_path)
                else:
                    self._models[name] = QwenCorrector(model_name_or_path="Qwen/Qwen3.5-2B")
            else:
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = ByT5Corrector(model_path=path)

            self._models[name].name = name
        return self._models[name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_query(self, query: str, model: Optional[str] = None) -> str:
        """Return only the corrected string."""
        if not query or not query.strip():
            return query
        result = self.correct(query, model=model)
        return result["corrected_query"]

    def correct(self, query: str, model: Optional[str] = None) -> dict:
        """
        Full correction with metadata.

        Args:
            query: User's search query.
            model: Model name ("byt5-base" or "byt5-small"). None = default.
        """
        m = self._get_model(model)
        return m.correct(query)

    def correct_batch(self, queries: list[str], model: Optional[str] = None) -> dict:
        """Correct a batch and return aggregated stats."""
        start = time.perf_counter()
        results = [self.correct(q, model=model) for q in queries]
        total_ms = (time.perf_counter() - start) * 1000

        return {
            "results": results,
            "batch_stats": {
                "total_queries": len(queries),
                "total_latency_ms": round(total_ms, 2),
                "avg_latency_ms": round(total_ms / len(queries), 2) if queries else 0,
            },
        }

    def list_models(self) -> list[dict]:
        """Describe available models (for /models endpoint)."""
        return [
            {
                "name": name,
                "description": desc,
                "architecture": "ByT5 (encoder-decoder, byte-level)",
                "loaded": name in self._models and self._models[name].is_loaded(),
            }
            for name, (_, desc) in _MODEL_REGISTRY.items()
        ]
