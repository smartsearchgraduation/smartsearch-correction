"""
TypoCorrector V2 – E-commerce Search Query Spell Correction Orchestrator

Enhanced version with T5-large support and brand lookup integration.
Single-model architecture with optional RAG-based brand protection.
Model is lazy-loaded on first use to keep startup fast.

Usage:
    from app.corrector_v2 import TypoCorrector

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
from .models.t5_large import T5LargeCorrector
from .models.base import BaseCorrector
from .models.qwen import QwenCorrector

logger = logging.getLogger(__name__)

# Model registry: name -> (path_suffix, description)
_MODEL_REGISTRY = {
    # ByT5 variants
    "byt5-base":      ("models/byt5-typo-best",                          "ByT5-base fine-tuned"),
    "byt5-small":     ("models/byt5-typo-final",                         "ByT5-small fine-tuned"),
    "byt5-large":     ("models/byt5-large/best",                         "ByT5-large fine-tuned"),
    # T5-Large variants
    "T5-Large-V2":    ("models/t5-large-typo/v2/t5_correction_v2-1",     "T5-Large v2.1 fine-tuned"),
    "T5-Large-V2.1":  ("models/t5-large-typo/v2/t5_correction_v2-1",     "T5-Large v2.1 + FastText/FAISS pipeline"),
    # LLM
    "qwen-3.5-2b":   ("models/qwen3.5-2b",                              "Qwen 3.5 2B (guarded typo-corrector)"),
}

_DEFAULT_MODEL = "byt5-base"


class TypoCorrector:
    """
    Central orchestrator for spell correction with optional brand protection.

    Supports multiple T5 variants (T5-large default) with lazy loading.
    Optionally integrates with BrandLookup RAG module for brand-aware corrections.
    """

    def __init__(self, enable_brand_lookup: bool = False):
        import os
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._models: dict[str, BaseCorrector] = {}
        self._brand_lookup = None
        self._enable_brand_lookup = enable_brand_lookup

        # Try to import BrandLookup if enabled
        if enable_brand_lookup:
            try:
                from .brand_lookup import BrandLookup
                self._brand_lookup = BrandLookup()
                logger.info("BrandLookup RAG module loaded")
            except ImportError:
                logger.warning("BrandLookup not available; brand protection disabled")
                self._enable_brand_lookup = False

        logger.info(
            "TypoCorrector V2 initialised (default model: %s, brand lookup: %s, available models: %s)",
            _DEFAULT_MODEL,
            self._enable_brand_lookup,
            list(_MODEL_REGISTRY.keys()),
        )

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
            elif name == "T5-Large-V2.1":
                # Full pipeline: T5-Large + FastText/FAISS fallback
                from .models.t5_pipeline import T5LargePipelineCorrector
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = T5LargePipelineCorrector(model_path=path)
            elif name == "T5-Large-V2":
                # Standalone T5-Large v2.1 (no fallback)
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = T5LargeCorrector(model_path=path)
            elif name.startswith("byt5"):
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = ByT5Corrector(model_path=path)
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
        Full correction with metadata and optional brand protection.

        Args:
            query: User's search query.
            model: Model name ("t5-large", "byt5-base", etc.). None = default (t5-large).

        Returns:
            {
                "original_query": str,
                "corrected_query": str,
                "changed": bool,
                "model_used": str,
                "latency_ms": float,
                "confidence": float (0.0-1.0),
                "brand_protected": bool,
            }
        """
        m = self._get_model(model)
        original_query = query

        # Step 1: Identify and protect known brands/units
        protected_query = original_query
        brand_regions = {}
        if self._enable_brand_lookup and self._brand_lookup:
            try:
                protected_query, brand_regions = self._brand_lookup.protect_brands(original_query)
            except Exception as e:
                logger.warning("Brand protection failed: %s", e)

        # Step 2: Send the (possibly protected) query to the model
        result = m.correct(protected_query)
        model_output = result["corrected_query"]

        # Step 3: Merge results – apply brand corrections back
        final_output = model_output
        if self._enable_brand_lookup and self._brand_lookup and brand_regions:
            try:
                final_output = self._brand_lookup.apply_brand_corrections(
                    original_query, model_output
                )
            except Exception as e:
                logger.warning("Brand correction merge failed: %s", e)

        # Step 4: Confidence scoring via edit distance
        confidence = self._compute_confidence(original_query, final_output)

        # Build final result
        final_result = {
            "original_query": original_query,
            "corrected_query": final_output,
            "changed": final_output.lower().strip() != original_query.lower().strip(),
            "model_used": m.name,
            "latency_ms": result["latency_ms"],
            "confidence": round(confidence, 3),
            "brand_protected": len(brand_regions) > 0 if self._enable_brand_lookup else False,
        }

        return final_result

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
        models = []
        for name, (_, desc) in _MODEL_REGISTRY.items():
            if name.startswith("qwen"):
                arch = "Qwen (causal LM, instruction-tuned)"
                model_type = "llm"
            elif name == "T5-Large-V2.1":
                arch = "T5-large v2.1 + FastText/FAISS pipeline"
                model_type = "pipeline"
            elif name == "T5-Large-V2":
                arch = "T5-large v2.1 (encoder-decoder, token-level)"
                model_type = "seq2seq"
            elif name.startswith("byt5"):
                arch = "ByT5 (encoder-decoder, byte-level)"
                model_type = "seq2seq"
            else:
                arch = "ByT5 (encoder-decoder, byte-level)"
                model_type = "seq2seq"

            models.append({
                "name": name,
                "description": desc,
                "architecture": arch,
                "type": model_type,
                "loaded": name in self._models and self._models[name].is_loaded(),
                "default": name == _DEFAULT_MODEL,
            })
        return models

    def get_default_model(self) -> str:
        """Return the default model name."""
        return _DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(original: str, corrected: str) -> float:
        """
        Compute confidence in the correction using edit distance.

        If the model output is very similar to the original (edit distance < 2),
        consider it low confidence. Otherwise, scale confidence based on similarity.

        Returns a value between 0.0 and 1.0.
        """
        from difflib import SequenceMatcher

        # Quick check: if strings are already equal, high confidence
        if original.lower().strip() == corrected.lower().strip():
            return 1.0

        # Compute edit distance approximation via SequenceMatcher ratio
        ratio = SequenceMatcher(None, original.lower(), corrected.lower()).ratio()

        # If very similar (ratio > 0.95), it's a minor change – lower confidence
        if ratio > 0.95:
            return 0.4

        # Otherwise, confidence = ratio (0.0 to 1.0)
        return ratio
