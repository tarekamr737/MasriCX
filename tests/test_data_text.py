"""Tests for text cleaning, normalization, and language classification."""

from __future__ import annotations

import pytest

from masricx.data.text import (
    audit_normalize,
    clean_transcript,
    code_switched_category,
    count_arabic_letters,
    count_english_letters,
    is_unusually_long_transcript,
    token_audit_stats,
)


class TestCleanTranscript:
    def test_nfkc_strip_collapse(self) -> None:
        assert clean_transcript(
            "  \u0645\u0631\u062d\u0628\u0627   hello\t world \u0645\u0631\u062d\u0628\u0627  "
        ) == ("\u0645\u0631\u062d\u0628\u0627 hello world \u0645\u0631\u062d\u0628\u0627")

    def test_empty_and_none(self) -> None:
        assert clean_transcript("") == ""
        assert clean_transcript("   ") == ""


class TestAuditNormalize:
    def test_removes_diacritics_and_tatweel(self) -> None:
        s = "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627 \u0640\u0627\u0644\u0644\u0647\u0640"
        # tatweel removed, diacritics removed, letters preserved
        out = audit_normalize(s)
        for diacritic in "\u064e\u0652\u064b\u0640":
            assert diacritic not in out
        assert "\u0645\u0631\u062d\u0628\u0627" in out

    def test_lowercases_latin_keeps_digits(self) -> None:
        # Period maps to a space (punctuation); digits preserved.
        assert audit_normalize("Hello World 12.5") == "hello world 12 5"

    def test_punctuation_to_spaces(self) -> None:
        assert audit_normalize("a, b; c!") == "a b c"

    def test_preserves_arabic_and_english(self) -> None:
        out = audit_normalize("\u0627\u0644\u0633\u0644\u0627\u0645 Welcome")
        assert "\u0627\u0644\u0633\u0644\u0627\u0645" in out
        assert "welcome" in out

    def test_no_transliteration_or_deletion(self) -> None:
        out = audit_normalize("Wa Alaikum Assalam \u0648\u0639\u0644\u064a\u0643\u0645")
        assert (
            out == "wa alaikum assalam \u0648\u0639\u0644\u064a\u0643\u0645"
        )  # English kept (lowercased), Arabic untouched


class TestLetterCounts:
    def test_arabic_letters_exclude_diacritics(self) -> None:
        s = "\u0645\u064e\u0631\u062d\u0628\u0627 # four letters plus fatha"
        assert count_arabic_letters(s) == 5

    def test_latin_letters(self) -> None:
        assert count_english_letters("Hello") == 5
        assert count_english_letters("\u0645\u0631\u062d\u0628\u0627") == 0


class TestTokenClassification:
    def test_ar_only_token(self) -> None:
        stats = token_audit_stats("\u0645\u0631\u062d\u0628\u0627")
        assert stats.ar_only_tokens == 1
        assert not stats.code_switched

    def test_en_only_one_letter_not_meaningful(self) -> None:
        stats = token_audit_stats("\u0645\u0631\u062d\u0628\u0627 a")
        assert stats.en_only_tokens == 1
        assert stats.en_only_meaningful_tokens == 0
        assert not stats.code_switched

    def test_two_letter_latin_is_meaningful(self) -> None:
        stats = token_audit_stats("\u0645\u0631\u062d\u0628\u0627 ok")
        assert stats.en_only_meaningful_tokens == 1
        assert stats.code_switched

    def test_mixed_token_counted_separately(self) -> None:
        # One token mixing Arabic and Latin letters.
        mixed = "\u0645ok"
        stats = token_audit_stats(mixed)
        assert stats.mixed_tokens == 1
        assert stats.ar_only_tokens == 0
        assert stats.en_only_tokens == 0

    def test_code_switch_category(self) -> None:
        assert (
            code_switched_category("Welcome \u0645\u0631\u062d\u0628\u0627")
            == "AR_EN_CODE_SWITCHED"
        )
        assert code_switched_category("\u0645\u0631\u062d\u0628\u0627") == "AR_ONLY"
        assert code_switched_category("hello there") == "EN_ONLY"
        assert code_switched_category("") == "OTHER"

    def test_numbers_are_other(self) -> None:
        # "12", "3" and "5" (period split the decimal into three digit tokens
        # after normalization maps '.' to a space).
        stats = token_audit_stats("12 3.5")
        assert stats.other_tokens == 3


class TestUnusuallyLong:
    def test_token_boundary(self) -> None:
        assert not is_unusually_long_transcript(" ".join(["ab"] * 50))
        assert is_unusually_long_transcript(" ".join(["ab"] * 51))

    def test_char_boundary(self) -> None:
        s = "a" * 300
        assert not is_unusually_long_transcript(s)
        assert is_unusually_long_transcript("a" * 301)


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_transcript_stats(empty: str) -> None:
    stats = token_audit_stats(empty)
    assert stats.token_count == 0
    assert not stats.code_switched
