"""
Aho-Corasick exact matcher for brand/product canonical forms and aliases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from app.masking.dataset import BrandEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExactMatch:
    start: int
    end: int
    canonical: str
    matched_text: str


try:
    import ahocorasick
    _HAS_AC = True
except ImportError:
    _HAS_AC = False
    logger.warning("pyahocorasick not installed; ExactMatcher uses slow fallback.")


class ExactMatcher:
    """Aho-Corasick over all brand aliases (case-insensitive)."""

    def __init__(self, entries: Iterable[BrandEntry], deny_list: set[str]):
        self._deny = {d.lower() for d in deny_list}
        self._patterns: dict[str, tuple[str, int]] = {}

        for entry in entries:
            if entry.canonical.lower() in self._deny:
                continue
            for alias in entry.exact_aliases:
                key = alias.lower().strip()
                if not key:
                    continue
                if key in self._deny:
                    continue
                value = (entry.canonical, len(key))
                existing = self._patterns.get(key)
                if existing is None or len(value[0]) > len(existing[0]):
                    self._patterns[key] = value

        if _HAS_AC and self._patterns:
            self._automaton = ahocorasick.Automaton()
            for key, value in self._patterns.items():
                self._automaton.add_word(key, value)
            self._automaton.make_automaton()
        else:
            # Empty pattern set OR pyahocorasick missing.
            self._automaton = None

    def find_all(self, text: str) -> list[ExactMatch]:
        if not text:
            return []

        text_lower = text.lower()
        raw: list[tuple[int, int, str]] = []

        if _HAS_AC and self._automaton is not None:
            for end_idx, value in self._automaton.iter(text_lower):
                canonical, alias_len = value
                start_idx = end_idx - alias_len + 1
                end_idx_excl = end_idx + 1
                if not _is_word_boundary(text_lower, start_idx, end_idx_excl):
                    continue
                raw.append((start_idx, end_idx_excl, canonical))
        else:
            # Slow fallback or empty pattern set
            for key, (canonical, _) in self._patterns.items():
                start = 0
                klen = len(key)
                while True:
                    idx = text_lower.find(key, start)
                    if idx < 0:
                        break
                    end = idx + klen
                    if _is_word_boundary(text_lower, idx, end):
                        raw.append((idx, end, canonical))
                    start = idx + 1

        return _resolve_overlaps(raw, text)

    def pattern_count(self) -> int:
        return len(self._patterns)


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Match must start and end at a word boundary."""
    before_ok = (start == 0) or not text[start - 1].isalnum()
    after_ok = (end == len(text)) or not text[end].isalnum()
    return before_ok and after_ok


def _resolve_overlaps(matches: list[tuple[int, int, str]], original_text: str) -> list[ExactMatch]:
    """Longest match wins; ties broken by leftmost-start."""
    if not matches:
        return []
    matches.sort(key=lambda m: (-(m[1] - m[0]), m[0]))
    chosen: list[tuple[int, int, str]] = []
    used = [False] * len(original_text)
    for start, end, canonical in matches:
        if any(used[start:end]):
            continue
        for i in range(start, end):
            used[i] = True
        chosen.append((start, end, canonical))
    chosen.sort(key=lambda m: m[0])
    return [
        ExactMatch(start=s, end=e, canonical=c, matched_text=original_text[s:e])
        for s, e, c in chosen
    ]
