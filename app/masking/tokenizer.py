"""
Whitespace + punctuation tokenizer with character offset tracking.

Returns Token objects that record (text, start, end) so the masking
pipeline can replace exact ranges in the original string without losing
the surrounding characters.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    text: str   # token surface (may include attached punctuation)
    start: int  # char offset in the original string (inclusive)
    end: int    # char offset (exclusive)


def tokenize(text: str) -> list[Token]:
    """
    Split on whitespace, preserving offsets. Punctuation attached to a
    word remains part of the same token (e.g. "iphone," is one token).

    Stripping leading/trailing punctuation is the matcher's job, since
    only the matcher knows what counts as a brand boundary.
    """
    tokens: list[Token] = []
    n = len(text)
    i = 0
    while i < n:
        # Skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        tokens.append(Token(text=text[start:i], start=start, end=i))
    return tokens


def strip_punct_edges(token: str) -> tuple[str, int, int]:
    """
    Return ``(stripped, leading, trailing)`` where ``stripped`` is the
    token with leading and trailing non-alphanumeric chars removed and
    ``leading``/``trailing`` are the counts of removed chars on each side.

    Apostrophes in the middle of a token (e.g. "men's") are preserved.
    """
    n = len(token)
    lead = 0
    while lead < n and not (token[lead].isalnum()):
        lead += 1
    trail = 0
    while trail < n - lead and not (token[n - 1 - trail].isalnum()):
        trail += 1
    return token[lead:n - trail], lead, trail
