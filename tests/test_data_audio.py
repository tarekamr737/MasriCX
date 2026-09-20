"""Tests for audio metrics: quantiles, silence, clipping, hashing stability.

Uses plain Python lists / array('f') as waveforms so the module under test
stays dependency-light (numpy/soundfile are lazy, training-env only).
"""

from __future__ import annotations

from array import array

import pytest

from masricx.data.audio_metrics import (
    CLIP_ABS_THRESHOLD,
    DurationStats,
    audio_fingerprint,
    audio_flags,
    chars_per_second,
    duration_stats,
    is_duration_mismatch,
    quantile,
)


class TestQuantiles:
    def test_min_p25_median_p75_p95_max(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = duration_stats(vals)
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert stats.median == 5.5
        assert stats.p25 == pytest.approx(3.25)
        assert stats.p75 == pytest.approx(7.75)
        assert stats.p95 == pytest.approx(9.55)
        assert stats.total_seconds == 55.0
        assert stats.count == 10

    def test_empty_and_single(self) -> None:
        assert duration_stats([]).count == 0
        single = duration_stats([2.5])
        assert single.median == 2.5
        assert single.p95 == 2.5

    def test_quantile_deterministic(self) -> None:
        vals = sorted([3.0, 1.0, 2.0])
        assert quantile(vals, 0.5) == 2.0

    def test_dataclass_shape(self) -> None:
        assert isinstance(duration_stats([1.0]), DurationStats)


class TestSilenceClipping:
    def test_silent_peak_threshold(self) -> None:
        assert audio_flags([0.0] * 100).silent
        assert audio_flags([1e-5] * 100).silent
        assert not audio_flags([1e-3] * 100).silent

    def test_silent_peak_exactly_at_threshold(self) -> None:
        assert audio_flags([1e-4] * 100).silent  # peak <= 1e-4 counts as silent

    def test_clipping_at_0_1pct(self) -> None:
        wave = [0.0] * 10000
        for i in range(10):  # exactly 0.1%
            wave[i] = 1.0
        flags = audio_flags(wave)
        assert flags.clipped
        assert not flags.silent

    def test_no_clipping_below_threshold(self) -> None:
        wave = [0.0] * 10000
        for i in range(9):  # 0.09% < 0.1%
            wave[i] = 1.0
        assert not audio_flags(wave).clipped

    def test_abs_threshold_used(self) -> None:
        wave = [0.0] * 1000
        wave[0] = wave[1] = -1.0  # negative clipping
        assert audio_flags(wave).clipped
        assert abs(-1.0) >= CLIP_ABS_THRESHOLD

    def test_corrupted_nonfinite(self) -> None:
        assert audio_flags([0.1, float("nan")]).corrupted
        assert audio_flags([0.1, float("inf")]).corrupted

    def test_empty_wave_silent(self) -> None:
        flags = audio_flags([])
        assert flags.silent
        assert not flags.corrupted


class TestAudioFingerprintStability:
    def test_identical_waveform_identical_hash(self) -> None:
        wave = [i / 1000 - 0.5 for i in range(1000)]
        assert audio_fingerprint(16000, wave) == audio_fingerprint(16000, wave)

    def test_different_sample_rate_different_hash(self) -> None:
        wave = [i / 1000 - 0.5 for i in range(1000)]
        assert audio_fingerprint(16000, wave) != audio_fingerprint(8000, wave)

    def test_different_content_different_hash(self) -> None:
        assert audio_fingerprint(16000, [1.0] * 100) != audio_fingerprint(16000, [0.9] * 100)

    def test_hash_is_sha256_hex(self) -> None:
        fp = audio_fingerprint(16000, [0.0] * 10)
        assert len(fp) == 64
        int(fp, 16)  # hex parse

    def test_container_independence(self) -> None:
        values = [i / 1000 - 0.5 for i in range(1000)]
        as_list = audio_fingerprint(16000, values)
        as_array = audio_fingerprint(16000, array("f", values))
        as_floats = audio_fingerprint(16000, [float(v) for v in values])
        assert as_list == as_array == as_floats


class TestMismatch:
    def test_too_few_chars_per_second(self) -> None:
        # 2 chars over 10 seconds -> 0.2 cps < 0.5 => mismatch
        assert is_duration_mismatch("ab", 10.0)

    def test_too_many_chars_per_second(self) -> None:
        # 310 chars over 10 seconds -> 31 cps > 30 => mismatch
        assert is_duration_mismatch("a" * 310, 10.0)

    def test_within_bounds(self) -> None:
        assert not is_duration_mismatch("a" * 10, 10.0)  # 1.0 cps
        assert not is_duration_mismatch("a" * 300, 10.0)  # 30 cps boundary

    def test_empty_or_zero_duration_not_mismatch(self) -> None:
        assert not is_duration_mismatch("", 5.0)
        assert not is_duration_mismatch("hello", 0.0)
        assert not is_duration_mismatch("hello", -1.0)

    def test_chars_per_second(self) -> None:
        assert chars_per_second(100, 10.0) == pytest.approx(10.0)
        assert chars_per_second(100, 0.0) == 0.0
