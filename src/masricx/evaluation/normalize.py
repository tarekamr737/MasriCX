"""The two fixed MasriCX evaluation normalization modes."""

from __future__ import annotations

import re

from masricx.data.text import audit_normalize

__all__ = ["normalize_for_evaluation", "normalize_normalized", "normalize_raw"]

_WHITESPACE = re.compile(r"\s+")


def normalize_raw(text: str) -> str:
    """Raw mode: trim and collapse whitespace, with no other rewriting."""
    return _WHITESPACE.sub(" ", text).strip()


def normalize_normalized(text: str) -> str:
    """Normalized mode: fixed conservative MasriCX normalization."""
    return audit_normalize(text)


def normalize_for_evaluation(text: str, mode: str) -> str:
    if mode == "raw":
        return normalize_raw(text)
    if mode == "normalized":
        return normalize_normalized(text)
    raise ValueError("mode must be 'raw' or 'normalized'")
