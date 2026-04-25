"""
Dataset loader for brands_v2.jsonl and the deny_list.txt.

Both files live under ``Correction/data/`` and are produced by:
    scripts/build_brand_dataset.py
    scripts/download_scowl.py
    data/deny_list.txt (manually curated)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BrandEntry:
    """One brand/model entry from brands_v2.jsonl."""
    canonical: str               # display form, e.g. "iPhone 15 Pro"
    brand: str                   # parent brand, e.g. "iPhone"
    category: str                # rough category, e.g. "smartphone"
    exact_aliases: tuple[str, ...]  # case variants for exact match
    symspell_eligible: bool      # is fuzzy match safe for this entry?
    min_match_length: int        # minimum input length for fuzzy match


def load_brands(path: Path) -> list[BrandEntry]:
    """
    Load brand entries from a JSONL file. Returns an empty list if the
    file is missing — callers must handle that case (boot the pipeline
    in degraded mode so the API still serves requests).
    """
    if not path.exists():
        return []
    out: list[BrandEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            # Skip malformed lines rather than crashing the service
            continue
        try:
            out.append(BrandEntry(
                canonical=d["canonical"],
                brand=d.get("brand", d["canonical"]),
                category=d.get("category", "general"),
                exact_aliases=tuple(d.get("exact_aliases", [d["canonical"]])),
                symspell_eligible=bool(d.get("symspell_eligible", False)),
                min_match_length=int(d.get("min_match_length", max(4, len(d["canonical"]) - 1))),
            ))
        except KeyError:
            # Required field missing — skip this entry
            continue
    return out


def load_deny_list(path: Path) -> set[str]:
    """
    Load the deny list. Lines starting with `#` are comments. Returns a
    set of lowercased brand names that should NEVER be masked.
    """
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s.lower())
    return out
