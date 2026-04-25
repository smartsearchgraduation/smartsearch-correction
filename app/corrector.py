"""
TypoCorrector - E-commerce Search Query Spell Correction Orchestrator with brand masking.
"""

import logging
import time
from typing import Optional

from .models.byt5 import ByT5Corrector
from .models.base import BaseCorrector
from .models.qwen import QwenCorrector
from .models.t5_large import T5LargeCorrector
from .masking import MaskingPipeline

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    "byt5-base":      ("models/byt5-typo-best",                          "ByT5-base fine-tuned"),
    "byt5-small":     ("models/byt5-typo-final",                         "ByT5-small fine-tuned"),
    "byt5-large":     ("models/byt5-large/best",                         "ByT5-large fine-tuned"),
    "BYT5-Large-V3":  ("models/byt5-large-v3",                           "ByT5-Large v3 fine-tuned"),
    "T5-Large-V2":    ("models/t5-large-typo/v2/t5_correction_v2-1",     "T5-Large v2.1 fine-tuned"),
    "T5-Large-V2.1":  ("models/t5-large-typo/v2/t5_correction_v2-1",     "T5-Large v2.1 + FastText/FAISS pipeline"),
    "qwen-3.5-2b":   ("models/qwen3.5-2b",                              "Qwen 3.5 2B (guarded typo-corrector)"),
}

_DEFAULT_MODEL = "BYT5-Large-V3"


class TypoCorrector:
    """Central orchestrator for spell correction with brand masking."""

    def __init__(self):
        import os
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._models: dict[str, BaseCorrector] = {}

        try:
            self._mask_pipeline = MaskingPipeline()
            logger.info("Brand masking pipeline ready: %s", self._mask_pipeline.stats())
        except Exception as e:
            logger.warning("Brand masking pipeline disabled: %s", e)
            self._mask_pipeline = None

        logger.info("TypoCorrector initialised (available models: %s)", list(_MODEL_REGISTRY.keys()))

    def _get_model(self, model_name):
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
                from .models.t5_pipeline import T5LargePipelineCorrector
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = T5LargePipelineCorrector(model_path=path)
            elif name == "T5-Large-V2":
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = T5LargeCorrector(model_path=path)
            elif name.startswith("byt5") or name.startswith("BYT5"):
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = ByT5Corrector(model_path=path)
            else:
                path = os.path.join(self._base_dir, model_ref)
                self._models[name] = ByT5Corrector(model_path=path)

            self._models[name].name = name
        return self._models[name]

    def correct_query(self, query, model=None):
        if not query or not query.strip():
            return query
        result = self.correct(query, model=model)
        return result["corrected_query"]

    def correct(self, query, model=None):
        """Full correction. Brand masking wraps every model except T5-Large-V2.1."""
        m = self._get_model(model)
        resolved_name = m.name

        skip_masking = (
            self._mask_pipeline is None
            or not self._mask_pipeline.is_healthy()
            or (resolved_name and "v2.1" in resolved_name.lower())
        )

        if skip_masking:
            return m.correct(query)

        masked_query, mask_map = self._mask_pipeline.mask(query)
        result = m.correct(masked_query)
        if mask_map:
            corrected_masked = result.get("corrected_query", masked_query)
            result["corrected_query"] = self._mask_pipeline.unmask(corrected_masked, mask_map)
            result["changed"] = (
                result["corrected_query"].lower().strip() != query.lower().strip()
            )
            result["original_query"] = query
        return result

    def correct_batch(self, queries, model=None):
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

    def list_models(self):
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

    def get_default_model(self):
        return _DEFAULT_MODEL
