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

logger = logging.getLogger(__name__)


class TypoCorrector:
    """
    Central orchestrator for spell correction.

    Uses ByT5 byte-level model for character-aware typo correction.
    """

    def __init__(self):
        self._model = ByT5Corrector()
        logger.info("TypoCorrector initialised (model=byt5)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_query(self, query: str) -> str:
        """Return only the corrected string."""
        if not query or not query.strip():
            return query
        result = self.correct(query)
        return result["corrected_query"]

    def correct(self, query: str, model: Optional[str] = None) -> dict:
        """
        Full correction with metadata.

        Args:
            query: User's search query.
            model: Ignored (kept for API compatibility).

        Returns:
            {
                "original_query": str,
                "corrected_query": str,
                "changed": bool,
                "model_used": str,
                "latency_ms": float,
            }
        """
        return self._model.correct(query)

    def correct_batch(self, queries: list[str], model: Optional[str] = None) -> dict:
        """Correct a batch and return aggregated stats."""
        start = time.perf_counter()
        results = [self.correct(q) for q in queries]
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
                "name": "byt5",
                "description": "ByT5 fine-tuned – byte-level typo correction",
                "architecture": "ByT5 (encoder-decoder, byte-level)",
                "loaded": self._model.is_loaded(),
            }
        ]
