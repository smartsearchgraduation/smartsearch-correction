"""
SymSpell-based fuzzy matcher for typo'd brand names.

Conservative by design: false positives (a normal English word being
masked as a brand) corrupt the user's query and are NOT recoverable by
the downstream correction model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from app.masking.dataset import BrandEntry
from app.masking.english_filter import EnglishFilter

logger = logging.getLogger(__name__)

try:
    from symspellpy import SymSpell, Verbosity  # type: ignore
    _HAS_SYMSPELL = True
except ImportError:
    _HAS_SYMSPELL = False
    SymSpell = None  # type: ignore
    Verbosity = None  # type: ignore


@dataclass(frozen=True)
class FuzzyMatch:
    canonical: str
    distance: int


class FuzzyMatcher:
    def __init__(
        self,
        entries,
        english_filter,
        deny_list,
        max_edit_distance=2,
    ):
        self._eng = english_filter
        self._deny = {d.lower() for d in deny_list}
        self._max_ed = max_edit_distance
        self._lookup = {}
        self._enabled = _HAS_SYMSPELL

        if not _HAS_SYMSPELL:
            self._sym = None
            return

        self._sym = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )

        for entry in entries:
            if not entry.symspell_eligible:
                continue
            key = entry.canonical.lower().strip()
            if not key:
                continue
            if key in self._deny:
                continue
            if not all(c.isalpha() or c.isspace() for c in key):
                continue
            # IMPORTANT: do NOT filter canonicals by english_filter.
            # Many e-commerce brands (Razer, Bose, Adobe, ...) appear in
            # older English wordlists. The english check is only on input.
            if key in self._lookup:
                continue
            self._sym.create_dictionary_entry(key, count=1)
            self._lookup[key] = (entry.canonical, entry.min_match_length)

    @staticmethod
    def _adaptive_ed(token):
        n = len(token)
        if n < 4:
            return 0
        return 1

    def lookup(self, token):
        if not self._enabled or self._sym is None or not token:
            return None
        token_clean = token.lower()
        if not all(c.isalpha() or c.isspace() for c in token_clean):
            return None
        if len(token_clean) < 4:
            return None
        if token_clean in self._deny:
            return None
        if self._eng.is_real_word(token_clean):
            return None
        ed = self._adaptive_ed(token_clean)
        if ed == 0:
            return None
        suggestions = self._sym.lookup(
            token_clean, Verbosity.CLOSEST,
            max_edit_distance=ed, include_unknown=False,
        )
        if not suggestions:
            return None
        best = suggestions[0]
        if best.distance == 0:
            return None
        canonical_display, min_len = self._lookup.get(best.term, (None, 0))
        if canonical_display is None:
            return None
        if len(token_clean) < min_len:
            return None
        ratio = len(token_clean) / len(best.term)
        if ratio < 0.6 or ratio > 1.6:
            return None
        return FuzzyMatch(canonical=canonical_display, distance=int(best.distance))

    def is_enabled(self):
        return self._enabled

    def dictionary_size(self):
        return len(self._lookup)
