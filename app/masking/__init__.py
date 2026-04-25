"""
Brand/model name masking pipeline for the correction service.

Public API:
    MaskingPipeline.mask(text) -> (masked_text, mask_map)
    MaskingPipeline.unmask(corrected_text, mask_map) -> restored_text

The pipeline is model-agnostic. It runs BEFORE any correction model
(ByT5, T5, Qwen, ...) and AFTER. It guarantees that brand and model names
spelled correctly in the input remain correct in the output, even if the
underlying spelling-correction model would otherwise corrupt them.

Architecture:
    ExactMatcher  (Aho-Corasick) -- high-confidence whole-string brand matches
    FuzzyMatcher  (SymSpell)     -- typo'd brand recovery (e.g. "iphne" -> "iPhone")
    EnglishFilter (wordlist)     -- guards SymSpell against real-English collisions
    MaskingPipeline              -- orchestrates the above

Mask token format: ``<<M0>>``, ``<<M1>>``, ...
The unmask step is failure-tolerant: if a model corrupts mask tokens
(case change, whitespace, dropped brackets), restoration falls back to
lenient regex and finally to positional restoration so that user queries
are NEVER worse than before masking was applied.
"""
from app.masking.pipeline import MaskingPipeline, MaskingConfig

__all__ = ["MaskingPipeline", "MaskingConfig"]
