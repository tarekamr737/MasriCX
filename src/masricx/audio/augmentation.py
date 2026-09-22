"""Seeded training-time signal augmentation; transcript labels never enter this API."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from masricx.audio.telephony import bandpass, codec_roundtrip, narrowband_roundtrip

__all__ = ["AugmentationTrace", "augment_training_audio"]

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class AugmentationTrace:
    selected: bool
    seed: int
    transforms: tuple[str, ...]


def _example_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _probability(config: Mapping[str, object], key: str) -> float:
    value = _number(config.get(key, 0.0), key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _speed(audio: FloatArray, rate: float) -> FloatArray:
    if not len(audio):
        return audio.copy()
    target_length = max(1, round(len(audio) / rate))
    return np.interp(
        np.linspace(0, len(audio) - 1, target_length),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def augment_training_audio(
    samples: object,
    sample_rate: int,
    sample_id: str,
    config: Mapping[str, object],
) -> tuple[FloatArray, AugmentationTrace]:
    """Apply at most three independently proposed transforms after selection."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono 1-D waveform")
    seed_value = config.get("seed", 42)
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ValueError("seed must be an integer")
    seed = _example_seed(seed_value, sample_id)
    rng = random.Random(seed)
    if not bool(config.get("enabled", False)) or rng.random() >= _probability(
        config, "apply_probability"
    ):
        return audio.copy(), AugmentationTrace(False, seed, ())
    candidates = [
        ("bandpass", "telephone_bandpass_probability"),
        ("narrowband", "narrowband_resample_probability"),
        ("codec", "mulaw_alaw_probability"),
        ("gain", "gain_variation_probability"),
        ("clipping", "clipping_probability"),
        ("noise", "noise_probability"),
        ("speed", "speed_perturbation_probability"),
    ]
    chosen = [name for name, key in candidates if rng.random() < _probability(config, key)]
    if not chosen:
        chosen = ["bandpass"]
    chosen = chosen[:3]
    output = audio.copy()
    np_rng = np.random.default_rng(seed)
    for name in chosen:
        if name == "bandpass":
            output = bandpass(output, sample_rate)
        elif name == "narrowband":
            output = narrowband_roundtrip(output, sample_rate)
        elif name == "codec":
            output = codec_roundtrip(output, "mulaw" if rng.random() < 0.5 else "alaw")
        elif name == "gain":
            output = output * np.float32(rng.uniform(0.7, 1.2))
        elif name == "clipping":
            threshold = rng.uniform(0.65, 0.9)
            output = np.clip(output, -threshold, threshold) / threshold
        elif name == "noise":
            rms = float(np.sqrt(np.mean(np.square(output)))) if len(output) else 0.0
            sigma = max(rms, 1e-4) * rng.uniform(0.005, 0.02)
            output = output + np_rng.normal(0.0, sigma, len(output)).astype(np.float32)
        elif name == "speed":
            bounds = config.get("speed_rate_range", [0.95, 1.05])
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("speed_rate_range must contain [min, max]")
            output = _speed(
                output,
                rng.uniform(
                    _number(bounds[0], "speed_rate_range[0]"),
                    _number(bounds[1], "speed_rate_range[1]"),
                ),
            )
    return np.clip(output, -1.0, 1.0).astype(np.float32), AugmentationTrace(
        True, seed, tuple(chosen)
    )
