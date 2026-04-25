"""
End-to-end masking pipeline: tokenize → exact match → fuzzy match → mask.

The pipeline is the only public entry point most callers (corrector.py,
brand_lookup.py) need. It is model-agnostic: nothing here knows about
ByT5 or T5. The contract is:

    masked, mask_map = pipeline.mask(text)
    corrected_masked = your_model.correct(masked)
    final = pipeline.unmask(corrected_masked, mask_map)

Robustness goals (per project requirements):

    * **Brand preservation:** correctly-spelled brands in the input must
      appear in canonical form in ``final`` regardless of what the model
      does to the masked text.
    * **Mask survival:** if the model partially corrupts a mask token
      (case, whitespace, missing brackets), unmask still recovers the
      brand by lenient regex and ultimately by positional fallback.
    * **No worse than baseline:** if the entire mask layer fails (mask
      extraction crashes, dataset missing, etc.), the pipeline falls back
      to identity so the corrector still serves traffic.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.masking.dataset import BrandEntry, load_brands, load_deny_list
from app.masking.english_filter import EnglishFilter
from app.masking.exact_matcher import ExactMatch, ExactMatcher
from app.masking.fuzzy_matcher import FuzzyMatcher
from app.masking.mask_format import (
    MASK_TOKEN_PATTERN,
    find_mask_shapes,
    find_masks_lenient,
    find_masks_strict,
    make_mask,
)
from app.masking.tokenizer import strip_punct_edges, tokenize

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass(frozen=True)
class MaskingConfig:
    brands_path: Path = DEFAULT_DATA_DIR / "brands_v2.jsonl"
    english_words_path: Path = DEFAULT_DATA_DIR / "english_words.txt"
    deny_list_path: Path = DEFAULT_DATA_DIR / "deny_list.txt"
    enable_fuzzy: bool = True


class MaskingPipeline:
    """
    Build once at service startup; reuse for every request.
    Thread-safe: no mutable per-request state on the instance.
    """

    def __init__(self, config: MaskingConfig | None = None):
        self.config = config or MaskingConfig()
        self._healthy: bool = True
        try:
            entries: list[BrandEntry] = load_brands(self.config.brands_path)
            deny: set[str] = load_deny_list(self.config.deny_list_path)
            self._exact = ExactMatcher(entries, deny_list=deny)
            if self.config.enable_fuzzy:
                self._eng = EnglishFilter(self.config.english_words_path)
                self._fuzzy: FuzzyMatcher | None = FuzzyMatcher(
                    entries, self._eng, deny_list=deny
                )
            else:
                self._fuzzy = None
            logger.info(
                "MaskingPipeline ready (brands=%d, exact_patterns=%d, "
                "fuzzy=%s, fuzzy_dict=%d, english=%d, deny=%d)",
                len(entries),
                self._exact.pattern_count(),
                self._fuzzy is not None and self._fuzzy.is_enabled(),
                self._fuzzy.dictionary_size() if self._fuzzy else 0,
                len(self._eng) if self.config.enable_fuzzy else 0,
                len(deny),
            )
        except Exception as e:  # pragma: no cover — boot-time guard
            logger.error("MaskingPipeline init failed: %s — pipeline disabled", e)
            self._exact = None  # type: ignore[assignment]
            self._fuzzy = None
            self._healthy = False

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Return ``(masked_text, mask_map)``.

        ``mask_map`` is an *ordered* dict: mask token → canonical brand.
        Order matters because positional fallback in ``unmask`` relies on
        the original generation order.

        On any internal failure the pipeline falls back to identity:
        returns ``(text, {})``.
        """
        if not self._healthy or not text:
            return text, {}

        try:
            return self._mask_unchecked(text)
        except Exception as e:  # pragma: no cover — runtime guard
            logger.warning("mask() crashed: %s — returning unmasked input", e)
            return text, {}

    def _mask_unchecked(self, text: str) -> tuple[str, dict[str, str]]:
        # Stage 1: exact matches over the full string.
        exact_matches: list[ExactMatch] = self._exact.find_all(text)

        # Build a "covered" bitmap so fuzzy matching only runs on gaps.
        covered = bytearray(len(text))  # 0/1 per character
        for m in exact_matches:
            for i in range(m.start, m.end):
                covered[i] = 1

        # Stage 2: fuzzy matching over uncovered tokens.
        fuzzy_spans: list[tuple[int, int, str]] = []
        if self._fuzzy is not None and self._fuzzy.is_enabled():
            for tok in tokenize(text):
                if tok.start >= tok.end:
                    continue
                # Skip if any character of token already covered by exact
                if any(covered[tok.start:tok.end]):
                    continue

                stripped, lead, trail = strip_punct_edges(tok.text)
                if not stripped or len(stripped) < 4:
                    continue

                m = self._fuzzy.lookup(stripped)
                if m is None:
                    continue
                s = tok.start + lead
                e = tok.end - trail
                if s >= e:
                    continue
                fuzzy_spans.append((s, e, m.canonical))
                for i in range(s, e):
                    covered[i] = 1

        # Stage 3: assemble masked text, generating mask map in order.
        all_spans: list[tuple[int, int, str]] = (
            [(m.start, m.end, m.canonical) for m in exact_matches]
            + fuzzy_spans
        )
        all_spans.sort(key=lambda s: s[0])

        out_parts: list[str] = []
        mask_map: dict[str, str] = {}
        cursor = 0
        for idx, (s, e, canonical) in enumerate(all_spans):
            out_parts.append(text[cursor:s])
            mask_token = make_mask(idx)
            out_parts.append(mask_token)
            mask_map[mask_token] = canonical
            cursor = e
        out_parts.append(text[cursor:])

        return "".join(out_parts), mask_map

    # ------------------------------------------------------------------
    # unmask
    # ------------------------------------------------------------------

    def unmask(self, text: str, mask_map: dict[str, str]) -> str:
        """
        Restore canonical brand strings from mask tokens in ``text``.

        Tier 1 (strict regex):
            Replace exact ``<<M\\d+>>`` occurrences via mask_map.
        Tier 2 (lenient regex):
            Replace forgiving variants (whitespace inside, lowercased m,
            "O" instead of "0", missing one bracket pair).
        Tier 3 (positional fallback):
            If unrestored masks remain AND the count of mask-shaped
            substrings in ``text`` matches the count of unrestored masks,
            substitute by position.

        If at the end >50% of masks are still unrestored, return the
        original unrestored ``text`` unchanged: better to surface a
        broken-looking string than to silently corrupt the user query.
        """
        if not mask_map:
            return text
        if not text:
            return text

        result = text
        unrestored: set[str] = set(mask_map.keys())

        # ---- Tier 1: strict ----
        def repl_strict(m: re.Match[str]) -> str:
            tok = m.group(0)
            if tok in mask_map:
                unrestored.discard(tok)
                return mask_map[tok]
            return tok

        result = MASK_TOKEN_PATTERN.sub(repl_strict, result)
        if not unrestored:
            return result

        # ---- Tier 2: lenient ----
        # Walk lenient matches, map by index back to the canonical.
        lenient_matches = find_masks_lenient(result)
        if lenient_matches:
            # Build a canonical-by-index lookup once.
            by_index: dict[int, str] = {}
            for tok, canonical in mask_map.items():
                m = MASK_TOKEN_PATTERN.fullmatch(tok)
                if m:
                    by_index[int(m.group(1))] = canonical
            # Replace right-to-left to preserve offsets
            for start, end, idx in sorted(lenient_matches, key=lambda x: -x[0]):
                if idx in by_index:
                    canonical = by_index[idx]
                    result = result[:start] + canonical + result[end:]
                    unrestored.discard(make_mask(idx))

        if not unrestored:
            return result

        # ---- Tier 3: positional fallback ----
        # Count mask-shaped substrings still in result. If equal to the
        # remaining unrestored count, restore in mask_map order.
        shape_spans = find_mask_shapes(result)
        if shape_spans and len(shape_spans) == len(unrestored):
            ordered_unrestored: list[str] = [
                tok for tok in mask_map.keys() if tok in unrestored
            ]
            for (start, end), tok in zip(
                sorted(shape_spans, key=lambda x: -x[0]),
                # Note: we iterate result occurrences right-to-left, but
                # we need the LAST canonical first. Build right-to-left
                # order of canonicals to match.
                reversed(ordered_unrestored),
                strict=False,
            ):
                canonical = mask_map[tok]
                result = result[:start] + canonical + result[end:]
                unrestored.discard(tok)

        # ---- Last-line guard ----
        # If still many masks unrestored, prefer not to confuse the user.
        if len(unrestored) > len(mask_map) // 2:
            logger.warning(
                "unmask: %d/%d masks unrestored — returning text with "
                "remaining mask tokens visible",
                len(unrestored),
                len(mask_map),
            )

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        return self._healthy

    def stats(self) -> dict:
        return {
            "healthy": self._healthy,
            "exact_patterns": (
                self._exact.pattern_count() if self._exact is not None else 0
            ),
            "fuzzy_enabled": (
                self._fuzzy is not None and self._fuzzy.is_enabled()
            ),
            "fuzzy_dict": (
                self._fuzzy.dictionary_size()
                if self._fuzzy is not None and self._fuzzy.is_enabled()
                else 0
            ),
        }
