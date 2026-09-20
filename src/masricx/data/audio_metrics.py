"""Audio-level deterministic audit metrics.

Threshold definitions fixed for MasriCX Phase 1 (documented, conservative):

- Silence: peak amplitude <= 1e-4.
- Clipped: at least 0.1% of samples have absolute amplitude >= 0.999.
- Mismatch outliers: characters/second < 0.5 or > 30 for non-empty samples
  with positive duration (characters measured on the raw-cleaned transcript).
- Corrupted audio: decoding failure or non-finite waveform values.

These functions are pure and testable on synthetic waveforms.
"""

from __future__ import annotations

import hashlib
import math
import sys
from array import array
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "CLIP_ABS_THRESHOLD",
    "CLIP_SAMPLE_FRACTION_THRESHOLD",
    "MISMATCH_CPS_MAX",
    "MISMATCH_CPS_MIN",
    "SILENCE_PEAK_THRESHOLD",
    "AudioFlags",
    "DurationStats",
    "audio_fingerprint",
    "audio_flags",
    "chars_per_second",
    "duration_stats",
    "is_duration_mismatch",
    "quantile",
]

SILENCE_PEAK_THRESHOLD = 1e-4
CLIP_SAMPLE_FRACTION_THRESHOLD = 0.001  # 0.1% of samples
CLIP_ABS_THRESHOLD = 0.999
MISMATCH_CPS_MIN = 0.5
MISMATCH_CPS_MAX = 30.0


@dataclass(frozen=True)
class AudioFlags:
    silent: bool
    clipped: bool
    corrupted: bool


@dataclass(frozen=True)
class DurationStats:
    count: int
    total_seconds: float
    min: float
    p25: float
    median: float
    p75: float
    p95: float
    max: float


def audio_flags(samples: Iterable[float]) -> AudioFlags:
    """Deterministic silence/clipping/corruption detection.

    Accepts any iterable of floats (list, array('f'), numpy array, etc.)
    without requiring numpy imports in this module.
    """
    values = list(samples)
    if not values:
        return AudioFlags(silent=True, clipped=False, corrupted=False)
    if not all(math.isfinite(v) for v in values):
        return AudioFlags(silent=False, clipped=False, corrupted=True)
    peak = max(abs(v) for v in values)
    silent = peak <= SILENCE_PEAK_THRESHOLD
    clipped_count = sum(1 for v in values if abs(v) >= CLIP_ABS_THRESHOLD)
    clipped = (clipped_count / len(values)) >= CLIP_SAMPLE_FRACTION_THRESHOLD
    return AudioFlags(silent=silent, clipped=clipped, corrupted=False)


def audio_fingerprint(sample_rate: int, samples: Iterable[float]) -> str:
    """Stable SHA-256 over sampling rate + canonical float32 PCM bytes.

    Canonicalization: each sample narrowed to float32 precision via
    ``array('f')`` (little-endian), independent of the container's dtype.
    Identical PCM produces identical hashes; the sample-rate prefix prevents
    cross-rate collisions.
    """
    canon = array("f", (float(v) for v in samples))
    if sys.byteorder == "big":
        canon.byteswap()
    prefix = f"sr{sample_rate}|".encode()
    return hashlib.sha256(prefix + canon.tobytes()).hexdigest()


def quantile(sorted_values: list[float], q: float) -> float:
    """Deterministic linear-interpolation quantile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac)


def duration_stats(durations_seconds: list[float]) -> DurationStats:
    """Min/p25/median/p75/p95/max from an accumulated duration list."""
    vals = sorted(durations_seconds)
    if not vals:
        return DurationStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total = float(sum(vals))
    return DurationStats(
        count=len(vals),
        total_seconds=total,
        min=vals[0],
        p25=quantile(vals, 0.25),
        median=quantile(vals, 0.50),
        p75=quantile(vals, 0.75),
        p95=quantile(vals, 0.95),
        max=vals[-1],
    )


def chars_per_second(transcript_chars: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return transcript_chars / duration_seconds


def is_duration_mismatch(transcript: str, duration_seconds: float) -> bool:
    """Conservative mismatch: chars/sec < 0.5 or > 30, non-empty, duration > 0.

    Characters counted after raw cleaning (NFKC/strip/collapse); empty
    transcripts are counted by the separate empty-transcript metric and are not
    mismatch outliers by definition here.
    """
    from masricx.data.text import clean_transcript

    cleaned = clean_transcript(transcript)
    if not cleaned or duration_seconds <= 0:
        return False
    cps = chars_per_second(len(cleaned), duration_seconds)
    return cps < MISMATCH_CPS_MIN or cps > MISMATCH_CPS_MAX
