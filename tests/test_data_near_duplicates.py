"""Tests for near-duplicate detection: determinism, idempotency, bounds."""

from __future__ import annotations

import pytest

import masricx.data.near_duplicates as near_duplicates_mod
from masricx.data.near_duplicates import (
    MAX_BUCKET_MEMBERS,
    MAX_TOTAL_COMPARISONS,
    MIN_NEAR_DUP_CHARS,
    NearDuplicateIndex,
    bigrams,
    jaccard_bigrams,
)


def _ids(n: int) -> list[str]:
    return [f"id{i:05d}" for i in range(n)]


class TestBasics:
    def test_bigrams(self) -> None:
        assert bigrams("abc") == {"ab", "bc"}
        assert bigrams("a") == set()
        assert bigrams("") == set()

    def test_jaccard_identical_and_disjoint(self) -> None:
        assert jaccard_bigrams("hello world", "hello world") == 1.0
        assert jaccard_bigrams("abcdef", "uvwxyz") == 0.0


class TestExactDuplicates:
    def test_exact_duplicate_groups(self) -> None:
        idx = NearDuplicateIndex()
        text = "repeated identical normalized text sample"
        for sid in _ids(3):
            idx.add(sid, text)
        s = idx.summary()
        assert s["exact_duplicate_groups"] == 1  # one text repeated
        assert s["exact_duplicate_extra_instances"] == 2

    def test_short_texts_never_candidates(self) -> None:
        idx = NearDuplicateIndex()
        idx.add("a", "abcdefgh")  # 9 chars < 10
        idx.add("b", "abcdefgh")
        s = idx.summary()
        assert s["unique_texts_indexed"] == 0
        assert s["near_duplicate_pair_count"] == 0


class TestNearDuplicates:
    def test_near_pair_detected(self) -> None:
        idx = NearDuplicateIndex()
        base = "the quick brown fox jumps over the lazy dog today"
        variant = "the quick brown fox jumps over the lazy dogs today"
        idx.add("a", base)
        idx.add("b", variant)
        pairs = idx.near_duplicate_pairs()
        assert len(pairs) == 1
        assert {pairs[0][0], pairs[0][1]} == {"a", "b"}
        assert pairs[0][2] >= 0.90

    def test_distinct_unrelated_not_reported(self) -> None:
        idx = NearDuplicateIndex()
        idx.add("a", "completely unrelated letters here now")
        idx.add("b", "totally different subject matter indeed")
        assert idx.near_duplicate_pairs() == []

    def test_determinism_across_insertion_order(self) -> None:
        # Deliberately disjoint texts so no pair crosses the 0.90 threshold.
        words = [
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
            "juliet",
        ]
        texts = [words[i % 10] + " " + words[(i * 3) % 10] + " zzz" + str(i) for i in range(30)]
        idx_a = NearDuplicateIndex()
        for i, t in enumerate(texts):
            idx_a.add(f"id{i}", t)
        idx_b = NearDuplicateIndex()
        for i in reversed(range(len(texts))):
            idx_b.add(f"id{i}", texts[i])
        assert idx_a.near_duplicate_pairs() == idx_b.near_duplicate_pairs()
        ta = idx_a.summary()
        tb = idx_b.summary()
        assert ta["pair_comparisons"] == tb["pair_comparisons"]
        assert ta["near_duplicate_pairs_reported"] == tb["near_duplicate_pairs_reported"]

    def test_repeated_calls_idempotent(self) -> None:
        idx = NearDuplicateIndex()
        base = "shared transcript for idempotency testing"
        variant = "shared transcript for idempotency testing!"
        idx.add("a", base)
        idx.add("b", variant)
        first = idx.near_duplicate_pairs()
        second = idx.near_duplicate_pairs()
        third_summary = idx.summary()
        fourth_summary = idx.summary()
        assert first == second
        assert third_summary == fourth_summary
        assert third_summary["near_duplicate_pair_count"] == len(first)
        assert third_summary["pair_comparisons"] == fourth_summary["pair_comparisons"]

    def test_adversarial_oversized_bucket_capped(self) -> None:
        # Every text shares the ultra-common bigram 'th', blowing one bucket
        # past MAX_BUCKET_MEMBERS; that bucket must be skipped deterministically
        # instead of generating quadratic work.
        n = 400  # far exceeds the bucket cap
        idx = NearDuplicateIndex()
        for i in range(n):
            idx.add(f"id{i}", f"the value one hundred and twenty three {i} xyzzy")
        s = idx.summary()
        bound = s["candidate_bound"]
        assert isinstance(bound, dict)
        assert bound["buckets_skipped_over_cap"] >= 1
        # Work stays bounded by the cap, not by n*(n-1)/2 over the shared bigram.
        assert s["pair_comparisons"] < n * (n - 1) // 2
        assert s["comparison_budget_exhausted"] is False

    def test_comparison_budget_hard_cap(self) -> None:
        # Force budget pressure and verify the hard cap is respected.
        n = 300
        idx = NearDuplicateIndex()
        for i in range(n):
            idx.add(f"id{i}", f"abc def ghi jkl mno {i * 7 % 13} pqr stu vwx")
        s = idx.summary()
        assert s["pair_comparisons"] <= MAX_TOTAL_COMPARISONS
        bound = s["candidate_bound"]
        assert isinstance(bound, dict)
        assert bound["max_total_comparisons"] == MAX_TOTAL_COMPARISONS

    def test_exhausted_budget_deterministic_across_insertion_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With a tiny budget the enumeration cutoff itself must not depend on
        # insertion order: members are processed in canonical sample_id order.
        monkeypatch.setattr(near_duplicates_mod, "MAX_TOTAL_COMPARISONS", 3)
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        texts = [f"{words[i % 8]} near dup candidate {i} shared context words" for i in range(12)]
        idx_a = NearDuplicateIndex()
        for i, t in enumerate(texts):
            idx_a.add(f"id{i}", t)
        idx_b = NearDuplicateIndex()
        for i in reversed(range(len(texts))):
            idx_b.add(f"id{i}", texts[i])
        assert idx_a.near_duplicate_pairs() == idx_b.near_duplicate_pairs()
        ta = idx_a.summary()
        tb = idx_b.summary()
        assert ta["pair_comparisons"] == tb["pair_comparisons"] == 3
        assert ta["comparison_budget_exhausted"] is True
        assert ta["near_duplicate_pairs_reported"] == tb["near_duplicate_pairs_reported"]

    def test_pair_orientation_canonical(self) -> None:
        idx = NearDuplicateIndex()
        base = "canonical orientation regression test text"
        variant = "canonical orientation regression test text!"
        idx.add("zz", base)
        idx.add("aa", variant)
        pairs = idx.near_duplicate_pairs()
        assert len(pairs) == 1
        # Orientation determined by sorted sample ids, not insertion order.
        assert pairs[0][0] < pairs[0][1]

    def test_max_pairs_reported_cap(self) -> None:
        idx = NearDuplicateIndex()
        # 30 texts in 15 near-identical pairs across disjoint letter sets.
        for i in range(30):
            tail = "a" if i % 2 == 0 else "b"
            idx.add(f"p{i}", f"template text alpha {tail} idx {i}")
        pairs = idx.near_duplicate_pairs()
        assert len(pairs) <= 20

    def test_bucket_member_cap_exposed(self) -> None:
        assert MAX_BUCKET_MEMBERS > 0

    def test_min_chars_boundary(self) -> None:
        assert MIN_NEAR_DUP_CHARS == 10
