"""Deterministic, dependency-free core metrics for MasriCX evaluation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from masricx.data.text import TokenLanguage, classify_token
from masricx.evaluation.normalize import normalize_for_evaluation

__all__ = [
    "ErrorCounts",
    "MetricValue",
    "character_error_rate",
    "english_term_recall",
    "number_accuracy",
    "real_time_factor",
    "word_error_rate",
]

_NUMBER_TOKEN = re.compile(r"[^\W_]+(?:[.,\u066b\u066c/-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def rate(self) -> float | None:
        if self.reference_units == 0:
            return None if self.insertions == 0 else 1.0
        return (self.substitutions + self.deletions + self.insertions) / self.reference_units


@dataclass(frozen=True)
class MetricValue:
    matched: int
    reference: int

    @property
    def value(self) -> float | None:
        return self.matched / self.reference if self.reference else None


def _errors(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    rows: list[list[tuple[int, int, int, int]]] = [
        [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    ]
    for i, ref in enumerate(reference, start=1):
        row = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, start=1):
            if ref == hyp:
                row.append(rows[i - 1][j - 1])
                continue
            sub = rows[i - 1][j - 1]
            delete = rows[i - 1][j]
            insert = row[j - 1]
            candidates = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            row.append(min(candidates, key=lambda item: (item[0], item[3], item[2], item[1])))
        rows.append(row)
    _, substitutions, deletions, insertions = rows[-1][-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def word_error_rate(reference: str, hypothesis: str, *, mode: str = "raw") -> ErrorCounts:
    ref = normalize_for_evaluation(reference, mode).split()
    hyp = normalize_for_evaluation(hypothesis, mode).split()
    return _errors(ref, hyp)


def character_error_rate(reference: str, hypothesis: str, *, mode: str = "raw") -> ErrorCounts:
    ref = list(normalize_for_evaluation(reference, mode).replace(" ", ""))
    hyp = list(normalize_for_evaluation(hypothesis, mode).replace(" ", ""))
    return _errors(ref, hyp)


def _english_terms(text: str) -> list[str]:
    tokens = normalize_for_evaluation(text, "normalized").split()
    return [token for token in tokens if classify_token(token) is TokenLanguage.EN_ONLY]


def english_term_recall(reference: str, hypothesis: str) -> MetricValue:
    reference_terms = Counter(_english_terms(reference))
    hypothesis_terms = Counter(_english_terms(hypothesis))
    matched = sum((reference_terms & hypothesis_terms).values())
    return MetricValue(matched, sum(reference_terms.values()))


def _canonical_number_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _NUMBER_TOKEN.finditer(normalized):
        token = match.group(0)
        if not any(ch.isdecimal() for ch in token):
            continue
        tokens.append("".join(str(unicodedata.digit(ch)) if ch.isdecimal() else ch for ch in token))
    return tokens


def number_accuracy(reference: str, hypothesis: str) -> MetricValue:
    reference_numbers = Counter(_canonical_number_tokens(reference))
    hypothesis_numbers = Counter(_canonical_number_tokens(hypothesis))
    matched = sum((reference_numbers & hypothesis_numbers).values())
    return MetricValue(matched, sum(reference_numbers.values()))


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float | None:
    if processing_seconds < 0 or audio_seconds < 0:
        raise ValueError("durations must be non-negative")
    return processing_seconds / audio_seconds if audio_seconds else None
