from __future__ import annotations

import pytest

from masricx.evaluation.metrics import (
    character_error_rate,
    english_term_recall,
    number_accuracy,
    real_time_factor,
    word_error_rate,
)
from masricx.evaluation.normalize import (
    normalize_for_evaluation,
    normalize_normalized,
    normalize_raw,
)


def test_raw_normalization_changes_only_whitespace() -> None:
    assert normalize_raw("  Hello،   عَالَم  ") == "Hello، عَالَم"


def test_normalized_mode_is_conservative_and_keeps_languages_and_numbers() -> None:
    assert normalize_normalized("  Hello، عَالَمــ 123  ") == "hello عالم 123"
    assert normalize_for_evaluation(" A ", "normalized") == "a"
    with pytest.raises(ValueError, match="mode"):
        normalize_for_evaluation("x", "unknown")


def test_word_and_character_error_counts() -> None:
    wer = word_error_rate("one two three", "one too extra", mode="raw")
    assert (wer.substitutions, wer.deletions, wer.insertions, wer.rate) == (2, 0, 0, 2 / 3)
    cer = character_error_rate("abc", "adc", mode="raw")
    assert cer.substitutions == 1
    assert cer.rate == pytest.approx(1 / 3)


def test_normalized_wer_ignores_case_diacritics_and_punctuation() -> None:
    assert word_error_rate("Hello، عَالَم", "hello عالم", mode="normalized").rate == 0
    assert word_error_rate("", "invented", mode="raw").rate == 1.0
    assert word_error_rate("", "", mode="raw").rate is None


def test_english_term_recall_uses_all_reference_occurrences() -> None:
    score = english_term_recall(
        "اعمل restart للـ router وبعدها افتح الـ application restart",
        "restart router",
    )
    assert (score.matched, score.reference, score.value) == (2, 4, 0.5)


def test_number_accuracy_handles_unicode_digits_and_alphanumeric_models() -> None:
    score = number_accuracy("هات ١٢ و package5 و A-10", "هات 12 و package5 و A-11")
    assert (score.matched, score.reference) == (2, 3)
    assert score.value == pytest.approx(2 / 3)


def test_real_time_factor() -> None:
    assert real_time_factor(2.0, 10.0) == 0.2
    assert real_time_factor(1.0, 0.0) is None
    with pytest.raises(ValueError):
        real_time_factor(-1.0, 2.0)
