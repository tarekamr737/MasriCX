"""Deterministic streaming aggregation of per-example audit facts.

Fold-only design: :class:`AuditAggregator` consumes :class:`ExampleAudit`
records one at a time and retains only compact counters, hashed fingerprints,
and normalized text needed for duplicate summaries. No waveform arrays are
retained, so memory stays bounded for streaming audits of 45K examples.
"""

from __future__ import annotations

import re
from collections import Counter

from masricx.data.audio_metrics import duration_stats
from masricx.data.example_audit import ExampleAudit
from masricx.data.near_duplicates import NearDuplicateIndex
from masricx.data.text import is_arabic_letter, is_latin_letter

__all__ = ["NUMBER_BUCKETS_DOC", "AuditAggregator", "number_bucket"]

NUMBER_SLOT_PATTERN: re.Pattern[str] = re.compile(r"\d+(?:[.,]\d+)?")

NUMBER_BUCKET_INTEGRAL_1 = "integer_digits_1"
NUMBER_BUCKET_INTEGRAL_2_3 = "integer_digits_2_3"
NUMBER_BUCKET_INTEGRAL_4_PLUS = "integer_digits_4_plus"
NUMBER_BUCKET_DECIMAL = "decimal"

NUMBER_BUCKETS_DOC = (
    "privacy-safe structural number buckets (raw numeric tokens are PII-"
    "sensitive and never serialized): 'integer_digits_1' = integer of 1 digit, "
    "'integer_digits_2_3' = integer of 2-3 digits, 'integer_digits_4_plus' = "
    "integer of 4+ digits, 'decimal' = number containing a decimal point or "
    "comma. Each matched number token contributes to exactly one bucket: "
    "'decimal' if it contains '.' or ',', otherwise by integer digit count."
)


def number_bucket(token: str) -> str:
    """Deterministic PII-safe structural bucket for one numeric token."""
    if "." in token or "," in token:
        return NUMBER_BUCKET_DECIMAL
    digits = sum(1 for ch in token if ch.isdecimal())
    if digits <= 1:
        return NUMBER_BUCKET_INTEGRAL_1
    if digits <= 3:
        return NUMBER_BUCKET_INTEGRAL_2_3
    return NUMBER_BUCKET_INTEGRAL_4_PLUS


_SYMBOL_CHARS = frozenset(
    "&%$#@*+=<>|/\\~^_{}[]()\"'`!?.,;:-\u060c\u061b\u061f\u2026\u00ab\u00bb\u00ab\u2019"
)


class AuditAggregator:
    """Fold per-example audit facts into the final report payload."""

    def __init__(self, audio_available: bool = True, unavailable_note: str = "") -> None:
        self._durations: list[float] = []
        self._sample_rates: Counter[int] = Counter()
        self._empty_transcripts = 0
        self._silent = 0
        self._clipped = 0
        self._corrupted = 0
        self._mismatch = 0
        self._unusual_long = 0
        self._audio_examples = 0
        self._audio_available = audio_available
        self._unavailable_note = unavailable_note
        self._dupe_text = NearDuplicateIndex()
        self._dupe_audio = NearDuplicateIndex()
        self._lang: Counter[str] = Counter()
        self._numbers: Counter[str] = Counter()
        self._symbols: Counter[str] = Counter()
        self._ar_letter_chars = 0
        self._en_letter_chars = 0
        self._examples = 0

    def add(self, ex: ExampleAudit) -> None:
        self._examples += 1
        if ex.duration_seconds is not None and ex.duration_seconds >= 0:
            self._durations.append(ex.duration_seconds)
        if ex.sample_rate is not None:
            self._sample_rates[ex.sample_rate] += 1
            self._audio_examples += 1
        if ex.empty_transcript:
            self._empty_transcripts += 1
        if ex.silent:
            self._silent += 1
        if ex.clipped:
            self._clipped += 1
        if ex.corrupted:
            self._corrupted += 1
        if ex.mismatch:
            self._mismatch += 1
        if ex.unusual_long:
            self._unusual_long += 1
        self._lang[ex.language_category] += 1
        for ch in ex.transcript_raw:
            if is_arabic_letter(ch):
                self._ar_letter_chars += 1
            elif is_latin_letter(ch):
                self._en_letter_chars += 1
            elif ch in _SYMBOL_CHARS:
                self._symbols[ch] += 1
        for token in NUMBER_SLOT_PATTERN.findall(ex.transcript_raw):
            self._numbers[number_bucket(token)] += 1
        self._dupe_text.add(ex.sample_id, ex.normalized)
        if ex.audio_fingerprint is not None:
            self._dupe_audio.add(ex.sample_id, ex.audio_fingerprint)

    def summary(self) -> dict[str, object]:
        """Final aggregates (deterministic; no audio, no PII)."""
        stats = duration_stats(self._durations)
        empty_note = "0 durations recorded (no positive-duration examples measured)"
        if not self._durations and self._examples > 0:
            duration_obj: dict[str, object] = {"note": empty_note}
        else:
            duration_obj = {
                "total_seconds": round(stats.total_seconds, 3),
                "min": round(stats.min, 3),
                "p25": round(stats.p25, 3),
                "median": round(stats.median, 3),
                "p75": round(stats.p75, 3),
                "p95": round(stats.p95, 3),
                "max": round(stats.max, 3),
            }
        return {
            "examples": self._examples,
            "audio_metrics_available": self._audio_available,
            "audio_metrics_note": self._unavailable_note,
            "audio_examples_measured": self._audio_examples,
            "duration": duration_obj,
            "sample_rates": dict(self._sample_rates),
            "empty_transcripts": self._empty_transcripts,
            "corrupted_audio": self._corrupted,
            "silent_clips": self._silent,
            "clipped_audio": self._clipped,
            "mismatch_outliers": self._mismatch,
            "unusually_long_transcripts": self._unusual_long,
            "language_categories": dict(self._lang),
            "arabic_letter_ratio": self._letter_ratio(True),
            "latin_letter_ratio": self._letter_ratio(False),
            "number_frequency": dict(
                self._numbers.most_common()
            ),  # structural buckets only, never raw tokens
            "number_buckets_doc": NUMBER_BUCKETS_DOC,
            "symbol_frequency": dict(self._symbols.most_common(50)),
            "transcript_duplicates": self._dupe_text.summary(),
            "audio_duplicates": self._dupe_audio.summary(),
        }

    def _letter_ratio(self, arabic: bool) -> float:
        total = self._ar_letter_chars + self._en_letter_chars
        if total == 0:
            return 0.0
        return round((self._ar_letter_chars if arabic else self._en_letter_chars) / total, 6)
