"""Deterministic signal-only telephone degradations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["bandpass", "codec_roundtrip", "narrowband_roundtrip", "telephone_test_degrade"]

FloatArray = NDArray[np.float32]


def _mono_float32(samples: object) -> FloatArray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("audio must be a mono 1-D waveform")
    return np.clip(array, -1.0, 1.0).astype(np.float32, copy=False)


def bandpass(
    samples: object, sample_rate: int, low_hz: float = 300.0, high_hz: float = 3400.0
) -> FloatArray:
    """Zero-phase FFT bandpass, preserving waveform length."""
    audio = _mono_float32(samples)
    if sample_rate <= 0 or not 0 <= low_hz < high_hz <= sample_rate / 2:
        raise ValueError("invalid sample rate or bandpass frequencies")
    if not len(audio):
        return audio.copy()
    spectrum = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    spectrum[(frequencies < low_hz) | (frequencies > high_hz)] = 0
    return np.clip(np.fft.irfft(spectrum, n=len(audio)), -1.0, 1.0).astype(np.float32)


def _resample_linear(audio: FloatArray, source_rate: int, target_rate: int) -> FloatArray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if not len(audio) or source_rate == target_rate:
        return audio.copy()
    target_length = max(1, round(len(audio) * target_rate / source_rate))
    source_positions = np.arange(len(audio), dtype=np.float64)
    target_positions = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def narrowband_roundtrip(
    samples: object, sample_rate: int, narrowband_rate: int = 8000
) -> FloatArray:
    audio = _mono_float32(samples)
    return _resample_linear(
        _resample_linear(audio, sample_rate, narrowband_rate), narrowband_rate, sample_rate
    )


def codec_roundtrip(samples: object, codec: str = "mulaw") -> FloatArray:
    """Deterministic 8-bit companding simulation for µ-law or A-law."""
    audio = _mono_float32(samples)
    magnitude = np.abs(audio)
    if codec == "mulaw":
        mu = 255.0
        compressed = np.sign(audio) * np.log1p(mu * magnitude) / np.log1p(mu)
        quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
        restored = np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu
    elif codec == "alaw":
        a = 87.6
        denominator = 1.0 + np.log(a)
        compressed_magnitude = np.where(
            magnitude < 1.0 / a,
            a * magnitude / denominator,
            (1.0 + np.log(a * np.maximum(magnitude, 1.0 / a))) / denominator,
        )
        compressed = np.sign(audio) * compressed_magnitude
        quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
        qmag = np.abs(quantized)
        restored_magnitude = np.where(
            qmag < 1.0 / denominator,
            qmag * denominator / a,
            np.exp(qmag * denominator - 1.0) / a,
        )
        restored = np.sign(quantized) * restored_magnitude
    else:
        raise ValueError("codec must be 'mulaw' or 'alaw'")
    result: FloatArray = np.asarray(np.clip(restored, -1.0, 1.0), dtype=np.float32)
    return result


def telephone_test_degrade(samples: object, sample_rate: int) -> FloatArray:
    """Fixed evaluation degradation: bandpass plus 8 kHz round-trip."""
    return narrowband_roundtrip(bandpass(samples, sample_rate), sample_rate, 8000)
