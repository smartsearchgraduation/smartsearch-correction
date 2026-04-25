"""
TypoCorrector V3 – Final Pipeline Orchestrator

4-stage pipeline:
  Stage 0: Pre-processing  (lowercase, unicode normalisation, entity protection)
  Stage 1: T5-Large        (primary, context-aware correction)
  Stage 2: FastText+FAISS  (fallback, word-level correction when T5 is not confident)
  Stage 3: Post-processing (restore protected entities, validate output)

Usage:
    from app.corrector_v3 import TypoCorrector

    corrector = TypoCorrector()
    result = corrector.correct("corsiar k70 keybord")
    # → {
    #     "original_query":  "corsiar k70 keybord",
    #     "corrected_query": "corsair k70 keyboard",
    #     "changed":         True,
    #     "correction_source": "t5-large",
    #     "confidence":      0.923,
    #     "latency_ms":      207.4,
    #     "suggestions":     [],
    #   }
"""

import logging
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry (kept for /models endpoint compatibility)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "t5-large": ("models/t5-large-typo", "T5-large fine-tuned (primary)"),
    "byt5-base": ("models/byt5-typo-best", "ByT5-base fine-tuned"),
    "byt5-small": ("models/byt5-typo-final", "ByT5-small fine-tuned"),
    "byt5-large": ("models/byt5-large/best", "ByT5-large fine-tuned"),
    "qwen-3.5-2b": ("models/qwen3.5-2b", "Qwen 3.5 2B (guarded typo-corrector)"),
}

_DEFAULT_MODEL = "t5-large"

# FastText model path relative to project root
_FASTTEXT_MODEL = "models/fasttext-ecommerce.bin"


class TypoCorrector:
    """
    Final pipeline orchestrator with confidence-based routing.

    T5-Large is the primary corrector (context-aware).
    FastText+FAISS is the fallback (word-level, brand-name–aware).
    BrandLookup provides a whitelist of protected entities (O(1) lookups).
    """

    def __init__(self):
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._data_dir = os.path.join(self._base_dir, "data")

        # Lazy-loaded components
        self._t5 = None
        self._fasttext = None
        self._brand_lookup = None
        self._whitelist: set = set()

        # Initialise BrandLookup eagerly (it's pure-Python, no ML weights)
        self._init_brand_lookup()

        logger.info(
            "TypoCorrector V3 initialised (default: %s, whitelist: %d entries)",
            _DEFAULT_MODEL,
            len(self._whitelist),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_brand_lookup(self):
        try:
            from .brand_lookup import BrandLookup
            self._brand_lookup = BrandLookup()
            self._whitelist = self._build_whitelist()
        except Exception as e:
            logger.warning("BrandLookup init failed: %s — whitelist empty", e)

    def _build_whitelist(self) -> set:
        """Build a lowercase whitelist from all BrandLookup protected entities."""
        whitelist = set()
        if self._brand_lookup is None:
            return whitelist
        for entry in self._brand_lookup.all_protected:
            whitelist.add(entry.lower())
        return whitelist

    def _get_t5(self):
        """Lazy-load T5-Large."""
        if self._t5 is None:
            from .models.t5_large import T5LargeCorrector
            path = os.path.join(self._base_dir, "models", "t5-large-typo")
            self._t5 = T5LargeCorrector(model_path=path)
            self._t5.name = "t5-large"
        return self._t5

    def _get_fasttext(self):
        """Lazy-load FastText+FAISS fallback."""
        if self._fasttext is None:
            from .models.fasttext_fallback import FastTextFallback
            ft_path = os.path.join(self._base_dir, _FASTTEXT_MODEL)
            self._fasttext = FastTextFallback(
                model_path=ft_path,
                vocab_path=self._data_dir,
            )
            try:
                self._fasttext.load()
            except Exception as e:
                logger.warning("FastText load failed: %s — fallback disabled", e)
        return self._fasttext

    # ------------------------------------------------------------------
    # Stage 0: Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, query: str) -> tuple[str, set]:
        """
        Normalise the query and identify protected tokens.

        Returns:
            (clean_query, protected_set)
            where protected_set contains lowercased tokens that must not be changed.
        """
        # Lowercase + strip
        clean = query.lower().strip()

        # Unicode normalisation (NFKC: decomposes ligatures, normalises whitespace-like chars)
        clean = unicodedata.normalize("NFKC", clean)

        # Collapse multiple whitespace
        clean = " ".join(clean.split())

        # Identify tokens that are in the whitelist
        protected: set = set()
        for token in clean.split():
            if token in self._whitelist or token.upper() in {w.upper() for w in self._whitelist}:
                protected.add(token)

        return clean, protected

    # ------------------------------------------------------------------
    # Stage 1: T5-Large confidence check
    # ------------------------------------------------------------------

    @staticmethod
    def _t5_is_confident(original: str, output: str) -> bool:
        """
        Return True only when T5's output is trustworthy.

        Reject when:
        - T5 returned the query unchanged      (couldn't fix it)
        - Similarity ratio < 0.50             (likely hallucination)
        - Word count changed by more than 1   (added/removed words)
        """
        orig_norm = original.lower().strip()
        out_norm = output.lower().strip()

        # No change → T5 gave up
        if orig_norm == out_norm:
            return False

        # Similarity check (SequenceMatcher is fast for short strings)
        ratio = SequenceMatcher(None, orig_norm, out_norm).ratio()
        if ratio < 0.50:
            return False

        # Word-count stability
        if abs(len(orig_norm.split()) - len(out_norm.split())) > 1:
            return False

        return True

    # ------------------------------------------------------------------
    # Stage 2: FastText merge helper
    # ------------------------------------------------------------------

    def _merge_corrections(self, t5_output: str, ft_result: dict) -> str:
        """
        Merge T5 output with FastText word-level corrections.

        Strategy: FastText wins on individual words it auto-corrected
        (it is more reliable for brand/product names). For words FastText
        left unchanged, prefer the T5 output.
        """
        t5_tokens = t5_output.split()

        # Build a lookup: original_token → FastText corrected token
        ft_corrections: dict[str, str] = {}
        for corr in ft_result.get("word_corrections", []):
            if corr["action"] == "auto_correct":
                ft_corrections[corr["original"].lower()] = corr["corrected"]

        if not ft_corrections:
            return t5_output

        # Reconstruct using the original (clean) query tokens as index
        orig_tokens = ft_result.get("original_query", "").split()

        result_tokens = []
        for i, t5_tok in enumerate(t5_tokens):
            orig_tok = orig_tokens[i].lower() if i < len(orig_tokens) else ""
            if orig_tok in ft_corrections:
                result_tokens.append(ft_corrections[orig_tok])
            else:
                result_tokens.append(t5_tok)

        return " ".join(result_tokens)

    # ------------------------------------------------------------------
    # Stage 3: Post-processing
    # ------------------------------------------------------------------

    def _postprocess(self, original_clean: str, corrected: str, protected: set) -> str:
        """
        Restore protected entities and validate output integrity.

        Checks:
        - Restore any protected token that T5/FastText incorrectly changed
        - Output must not be >30% longer than input
        - Numbers/digits present in the original must be preserved
        - Word count must not grow by more than 1
        """
        orig_tokens = original_clean.split()
        corr_tokens = corrected.split()

        # If word counts diverge too much, fall back to original
        if abs(len(orig_tokens) - len(corr_tokens)) > 1:
            return original_clean

        # Token-level: restore protected positions
        result_tokens = []
        for i, corr_tok in enumerate(corr_tokens):
            if i < len(orig_tokens) and orig_tokens[i].lower() in protected:
                result_tokens.append(orig_tokens[i])  # restore protected
            else:
                result_tokens.append(corr_tok)

        result = " ".join(result_tokens)

        # Validate: output must not be >30% longer than input
        if len(result) > len(original_clean) * 1.30:
            return original_clean

        # Validate: preserve all digit sequences from original
        orig_numbers = re.findall(r"\d+", original_clean)
        result_numbers = re.findall(r"\d+", result)
        if sorted(orig_numbers) != sorted(result_numbers):
            return original_clean

        return result

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(
        original: str,
        corrected: str,
        source: str,
        latency_ms: float,
        suggestions: list = None,
    ) -> dict:
        suggestions = suggestions or []
        changed = corrected.lower().strip() != original.lower().strip()

        # Confidence: similarity ratio between original and corrected
        if not changed:
            confidence = 1.0
        else:
            confidence = round(
                SequenceMatcher(None, original.lower(), corrected.lower()).ratio(), 3
            )

        # Format suggestions as plain strings for the API
        suggestion_strings = [
            f"Did you mean '{s['corrected']}'?" for s in suggestions
            if isinstance(s, dict) and "corrected" in s
        ]

        return {
            "original_query": original,
            "corrected_query": corrected,
            "changed": changed,
            "correction_source": source,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
            "suggestions": suggestion_strings,
        }

    # ------------------------------------------------------------------
    # Main correction pipeline
    # ------------------------------------------------------------------

    def correct(self, query: str, model: Optional[str] = None) -> dict:
        """
        Run the full 4-stage correction pipeline.

        Args:
            query: User's raw search query.
            model: Ignored in V3 (pipeline is fixed). Kept for API compatibility.

        Returns:
            {
                "original_query":    str,
                "corrected_query":   str,
                "changed":           bool,
                "correction_source": str,   # "t5-large" | "fasttext" | "t5-large+fasttext"
                "confidence":        float,
                "latency_ms":        float,
                "suggestions":       list[str],
            }
        """
        if not query or not query.strip():
            return self._build_response(query, query, "none", 0.0)

        pipeline_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 0: Pre-processing
        # ------------------------------------------------------------------
        clean_query, protected = self._preprocess(query)

        # ------------------------------------------------------------------
        # Stage 1: T5-Large (primary)
        # ------------------------------------------------------------------
        t5 = self._get_t5()
        t5_result = t5.correct(clean_query)
        t5_output = t5_result["corrected_query"]
        t5_ms = t5_result["latency_ms"]

        if self._t5_is_confident(clean_query, t5_output):
            final = self._postprocess(clean_query, t5_output, protected)
            total_ms = (time.perf_counter() - pipeline_start) * 1000
            return self._build_response(query, final, "t5-large", total_ms)

        # ------------------------------------------------------------------
        # Stage 2: FastText+FAISS fallback
        # ------------------------------------------------------------------
        ft = self._get_fasttext()
        source = "t5-large+fasttext"

        if not ft.is_loaded():
            # FastText unavailable — use T5 output even if low confidence,
            # or fall back to original if T5 didn't change anything
            if t5_output.lower().strip() != clean_query.lower().strip():
                final = self._postprocess(clean_query, t5_output, protected)
                source = "t5-large"
            else:
                final = clean_query
                source = "none"

            total_ms = (time.perf_counter() - pipeline_start) * 1000
            return self._build_response(query, final, source, total_ms)

        ft_result = ft.correct_query(clean_query, self._whitelist)
        ft_ms = ft_result["latency_ms"]

        # Merge strategy:
        #   • T5 changed something (low confidence) → merge with FastText overrides
        #   • T5 unchanged → use FastText result directly
        if t5_output.lower().strip() != clean_query.lower().strip():
            merged = self._merge_corrections(t5_output, ft_result)
        else:
            merged = ft_result["corrected_query"]
            # If FastText also changed nothing, we have nothing to offer
            if merged.lower().strip() == clean_query.lower().strip():
                source = "none"

        final = self._postprocess(clean_query, merged, protected)
        total_ms = (time.perf_counter() - pipeline_start) * 1000

        return self._build_response(
            query, final, source, total_ms,
            suggestions=ft_result.get("suggestions", []),
        )

    def correct_query(self, query: str, model: Optional[str] = None) -> str:
        """Return only the corrected string (convenience wrapper)."""
        if not query or not query.strip():
            return query
        return self.correct(query, model=model)["corrected_query"]

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
            elif name.startswith("t5-large"):
                arch = "T5-large (encoder-decoder, token-level)"
                model_type = "seq2seq"
            else:
                arch = "ByT5 (encoder-decoder, byte-level)"
                model_type = "seq2seq"

            loaded = False
            if name == "t5-large" and self._t5 is not None:
                loaded = self._t5.is_loaded()

            models.append({
                "name": name,
                "description": desc,
                "architecture": arch,
                "type": model_type,
                "loaded": loaded,
                "default": name == _DEFAULT_MODEL,
            })

        # Add FastText as a special fallback entry
        ft_loaded = self._fasttext is not None and self._fasttext.is_loaded()
        models.append({
            "name": "fasttext-fallback",
            "description": "FastText + FAISS word-level fallback",
            "architecture": "FastText skipgram + FAISS flat index",
            "type": "embedding",
            "loaded": ft_loaded,
            "default": False,
        })

        return models

    def get_default_model(self) -> str:
        """Return the default model name."""
        return _DEFAULT_MODEL
