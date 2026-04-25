"""
Mask token format and multi-tier restoration utilities.

Format: ``<<M0>>``, ``<<M1>>``, ...
Chosen because:
  - "<<" and ">>" almost never appear in English search queries
  - Pattern is easy to parse with a strict regex
  - Short enough that ByT5 (byte-level) is unlikely to add/remove characters

Restoration tiers (used by MaskingPipeline.unmask):
  1. STRICT  – exact `<<M\\d+>>` regex match.
  2. LENIENT – tolerates whitespace inside, "O" instead of "0", lowercase
               "m", and partial bracket loss like "<M0>".
  3. POSITIONAL – if N masks were generated and N (any) mask-shaped tokens
                  remain in the output, restore by index regardless of
                  exact format.

Tier 3 is the last line of defense. It guarantees that even severe model
corruption of mask tokens does not turn into a missing brand in the
final output.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Strict pattern — what we EMIT and what we expect to find on the way back
# ---------------------------------------------------------------------------

MASK_TOKEN_PATTERN = re.compile(r"<<M(\d+)>>")

# ---------------------------------------------------------------------------
# Lenient pattern — tolerant of common ByT5 / model corruption modes
#
# Matches things like:
#     <<M0>>     (correct)
#     << M0 >>   (whitespace)
#     <<m0>>     (lowercase m)
#     <<MO>>     (O instead of 0; treat O/o -> 0 by convention)
#     < <M0> >   (single brackets each side)
#     <M0>       (single brackets)
#
# Group 1 captures the index payload (digits, or the letter O/o which we
# coerce to 0 -- we only ever do this when we have exactly one mask anyway).
# ---------------------------------------------------------------------------

MASK_TOKEN_LENIENT = re.compile(
    r"""
    <\s*<?\s*           # one or two opening angles, optional whitespace
    [Mm]                # M (or lowercased m)
    \s*
    ([0-9Oo]+)          # the index — digits, or O/o (corrupted 0)
    \s*
    >?\s*>              # one or two closing angles
    """,
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Generic mask-shaped pattern — final fallback for positional restoration
#
# Matches "anything that looks like our mask shape", regardless of the
# actual contents. Used by tier 3 when even the lenient pattern fails to
# match an emitted mask token.
# ---------------------------------------------------------------------------

MASK_SHAPE_FALLBACK = re.compile(r"<+\s*[A-Za-z]?\s*[0-9OoIlL]+\s*>+")


def make_mask(index: int) -> str:
    """Generate the mask token for a given index."""
    return f"<<M{index}>>"


def find_masks_strict(text: str) -> list[tuple[int, int, int]]:
    """Return ``[(start, end, index), ...]`` for all strict matches."""
    return [
        (m.start(), m.end(), int(m.group(1)))
        for m in MASK_TOKEN_PATTERN.finditer(text)
    ]


def find_masks_lenient(text: str) -> list[tuple[int, int, int]]:
    """
    Return ``[(start, end, index), ...]`` for lenient matches.

    The captured group is coerced to int. Letters O/o map to digit 0.
    """
    out: list[tuple[int, int, int]] = []
    for m in MASK_TOKEN_LENIENT.finditer(text):
        raw = m.group(1)
        digits = "".join("0" if c in "Oo" else c for c in raw)
        try:
            idx = int(digits)
        except ValueError:
            continue
        out.append((m.start(), m.end(), idx))
    return out


def find_mask_shapes(text: str) -> list[tuple[int, int]]:
    """
    Return ``[(start, end), ...]`` for any mask-shaped substring,
    regardless of contents. Used for positional restoration.
    """
    return [(m.start(), m.end()) for m in MASK_SHAPE_FALLBACK.finditer(text)]
