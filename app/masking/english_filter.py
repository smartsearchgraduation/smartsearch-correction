"""
English wordlist filter — guards SymSpell against real-English collisions.

Example:
    "apply" is edit-distance 1 from "Apple". Without this filter, SymSpell
    would mask "apply" as if it were a typo of "Apple", corrupting the
    user's actual word.

Decision rule:
    If a candidate token is itself a real English word, do NOT flag it
    as a brand typo — the user knew what they were typing.
"""
from __future__ import annotations

from pathlib import Path


class EnglishFilter:
    """Lowercased English wordlist; in-memory hash set lookup."""

    def __init__(self, wordlist_path: Path):
        self._words: set[str] = set()
        if wordlist_path.exists():
            for line in wordlist_path.read_text(encoding="utf-8").splitlines():
                w = line.strip().lower()
                if w:
                    self._words.add(w)

    def is_real_word(self, token: str) -> bool:
        return token.lower() in self._words

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, token: str) -> bool:
        return self.is_real_word(token)
