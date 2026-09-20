"""Deterministic text cleaning, audit normalization, and language token statistics.

Pure functions on strings; no I/O, no randomness. Shared by the loaders and the
audit engine so the definitions are consistent everywhere.

Definitions (fixed for MasriCX Phase 1):

- Raw cleaning: Unicode NFKC, strip, collapse internal whitespace runs to a
  single space.
- Audit/dedupe normalization: raw cleaning plus lowercase Latin letters,
  removal of Arabic diacritics (harakat, sukun, shadda, tanwin, small-letter
  marks) and tatweel, and mapping of punctuation/symbols to spaces. Letters and
  decimal digits are preserved in all scripts. No transliteration, no deletion
  of English lexical tokens, no rewriting of Egyptian dialect into MSA.
- Arabic token: contains at least one Arabic letter and no Latin letter.
- English token: contains at least one Latin letter and no Arabic letter.
- Mixed token (e.g. Latin+Arabic in one token): counted separately; contributes
  to both language *presence* but never as the sole evidence for a token class.
- Code-switched sample: at least one meaningful Arabic token and one meaningful
  Latin token. One-letter Latin tokens do not count as meaningful English.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "TOKEN_THRESHOLD_CHARS",
    "AuditTokenStats",
    "TokenLanguage",
    "audit_normalize",
    "classify_token",
    "clean_transcript",
    "count_arabic_letters",
    "count_english_letters",
    "is_arabic_letter",
    "is_latin_letter",
    "is_unusually_long_transcript",
    "normalized_length",
    "normalized_whitespace_token_count",
    "token_audit_stats",
]

ARABIC_BLOCK_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)

LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
)

# Arabic combining marks / diacritics (harakat, sukun, shadda, tanwin, marks).
_ARABIC_DIACRITICS: frozenset[str] = frozenset(
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655"
    "\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f\u0670"
    "\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06df\u06e0\u06e1\u06e2"
    "\u06e3\u06e4\u06e7\u06e8\u06ea\u06eb\u06ec\u06ed"
)
_TATWEEL: str = "\u0640"

_WHITESPACE_RUN: re.Pattern[str] = re.compile(r"\s+")

_MIN_TOKEN_MEANINGFUL_LENGTH: int = 2


def is_arabic_letter(ch: str) -> bool:
    """A character from an Arabic block that is a letter (not a diacritic/mark)."""
    point = ord(ch)
    if not any(low <= point <= high for low, high in ARABIC_BLOCK_RANGES):
        return False
    if ch in _ARABIC_DIACRITICS or ch == _TATWEEL:
        return False
    return unicodedata.category(ch).startswith("L")


def is_latin_letter(ch: str) -> bool:
    point = ord(ch)
    return any(low <= point <= high for low, high in LATIN_RANGES)


def count_english_letters(text: str) -> int:
    return sum(1 for ch in text if is_latin_letter(ch))


def count_arabic_letters(text: str) -> int:
    return sum(1 for ch in text if is_arabic_letter(ch))


def count_decimal_digits(text: str) -> int:
    return sum(1 for ch in text if ch.isdecimal())


def count_symbols(text: str) -> int:
    """Non-alphanumeric printable symbols (punctuation, currency, math, etc.)."""
    return sum(
        1
        for ch in text
        if not ch.isspace()
        and not ch.isdecimal()
        and not is_arabic_letter(ch)
        and not is_latin_letter(ch)
    )


class TokenLanguage(Enum):
    AR_ONLY = "AR_ONLY"
    EN_ONLY = "EN_ONLY"
    MIXED = "MIXED"
    OTHER = "OTHER"


def classify_token(token: str) -> TokenLanguage:
    """Token-level language class using the fixed MasriCX definitions."""
    stripped = token.strip()
    if not stripped:
        return TokenLanguage.OTHER
    has_ar = any(is_arabic_letter(c) for c in stripped)
    has_en = any(is_latin_letter(c) for c in stripped)
    if has_ar and has_en:
        return TokenLanguage.MIXED
    if has_ar:
        return TokenLanguage.AR_ONLY
    if has_en:
        return TokenLanguage.EN_ONLY
    return TokenLanguage.OTHER


def _meaningful_token_counts(tokens: list[str]) -> tuple[int, int, int, int]:
    """Return (ar_only, en_only_meaningful, other, mixed) counts."""
    ar_only = en_other = mixed = 0
    en_meaningful = 0
    for tok in tokens:
        cls = classify_token(tok)
        if cls is TokenLanguage.AR_ONLY:
            ar_only += 1
        elif cls is TokenLanguage.EN_ONLY:
            en_other += 1
            # One-letter Latin tokens are not meaningful English evidence
            # (e.g. article "a", stray initials).
            if len(tok) > 1:
                en_meaningful += 1
        elif cls is TokenLanguage.MIXED:
            mixed += 1
    return ar_only, en_meaningful, en_other, mixed


@dataclass(frozen=True)
class AuditTokenStats:
    """Token-level language statistics for one transcript."""

    token_count: int
    ar_only_tokens: int
    en_only_tokens: int
    mixed_tokens: int
    other_tokens: int
    en_only_meaningful_tokens: int  # EN_ONLY with length > 1
    code_switched: bool


def token_audit_stats(transcript: str) -> AuditTokenStats:
    from_audit = audit_normalize(transcript)
    tokens = from_audit.split()
    counts = [classify_token(t) for t in tokens]
    ar_only = counts.count(TokenLanguage.AR_ONLY)
    en_only = counts.count(TokenLanguage.EN_ONLY)
    mixed = counts.count(TokenLanguage.MIXED)
    other = counts.count(TokenLanguage.OTHER)
    en_meaningful = sum(
        1 for t, c in zip(tokens, counts, strict=True) if c is TokenLanguage.EN_ONLY and len(t) > 1
    )
    return AuditTokenStats(
        token_count=len(tokens),
        ar_only_tokens=ar_only,
        en_only_tokens=en_only,
        mixed_tokens=mixed,
        other_tokens=other,
        en_only_meaningful_tokens=en_meaningful,
        code_switched=ar_only > 0 and en_meaningful > 0,
    )


def clean_transcript(text: str) -> str:
    """Raw cleaning: Unicode NFKC, strip, collapse whitespace."""
    return _WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _translate_char(ch: str) -> str:
    if ch in _ARABIC_DIACRITICS or ch == _TATWEEL:
        return ""
    if is_arabic_letter(ch):
        return ch
    if is_latin_letter(ch):
        return ch.lower()
    if ch.isdecimal():
        return ch
    if ch.isspace():
        return " "
    return " "


def audit_normalize(text: str) -> str:
    """Normalization for audit/dedupe (see module docstring).

    Keeps letters and decimal digits in every script; lowercases Latin;
    removes Arabic diacritics and tatweel; maps punctuation/symbols to spaces;
    collapses resulting whitespace.
    """
    cleaned = unicodedata.normalize("NFKC", text)
    out = "".join(_translate_char(c) for c in cleaned)
    return _WHITESPACE_RUN.sub(" ", out).strip()


TOKEN_THRESHOLD_CHARS: int = 50


def normalized_whitespace_token_count(normalized: str) -> int:
    return len(normalized.split())


def normalized_length(normalized: str) -> int:
    return len(normalized)


def is_unusually_long_transcript(
    transcript: str,
    max_tokens: int = 50,
    max_normalized_chars: int = 300,
) -> bool:
    """Unusually long: >50 whitespace tokens OR >300 normalized characters."""
    n = normalized_whitespace_token_count(transcript)
    if n > max_tokens:
        return True
    return normalized_length(transcript) > max_normalized_chars


def code_switched_category(transcript: str) -> str:
    """Return 'AR_EN_CODE_SWITCHED' | 'AR_ONLY' | 'EN_ONLY' | 'OTHER'."""
    stats = token_audit_stats(transcript)
    if stats.ar_only_tokens > 0 and stats.en_only_meaningful_tokens > 0:
        return "AR_EN_CODE_SWITCHED"
    if stats.ar_only_tokens > 0:
        return "AR_ONLY"
    if stats.en_only_meaningful_tokens > 0:
        return "EN_ONLY"
    return "OTHER"
