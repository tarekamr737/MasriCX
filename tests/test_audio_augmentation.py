from __future__ import annotations

import numpy as np
import pytest

from masricx.audio.augmentation import augment_training_audio
from masricx.audio.telephony import (
    bandpass,
    codec_roundtrip,
    narrowband_roundtrip,
    telephone_test_degrade,
)


def _tone(frequency: float = 1000.0, seconds: float = 0.1) -> np.ndarray:
    time = np.arange(round(16000 * seconds)) / 16000
    return (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_bandpass_preserves_in_band_and_rejects_out_of_band() -> None:
    in_band = bandpass(_tone(1000), 16000)
    out_band = bandpass(_tone(5000), 16000)
    assert np.sqrt(np.mean(in_band**2)) > 0.3
    assert np.sqrt(np.mean(out_band**2)) < 1e-4
    with pytest.raises(ValueError):
        bandpass(_tone(), 16000, 3400, 300)


def test_narrowband_codec_and_fixed_test_degradation_are_deterministic() -> None:
    audio = _tone()
    assert narrowband_roundtrip(audio, 16000).shape == audio.shape
    assert np.array_equal(
        telephone_test_degrade(audio, 16000), telephone_test_degrade(audio, 16000)
    )
    for codec in ("mulaw", "alaw"):
        result = codec_roundtrip(audio, codec)
        assert result.shape == audio.shape
        assert np.max(np.abs(result)) <= 1.0
    with pytest.raises(ValueError, match="codec"):
        codec_roundtrip(audio, "bad")


def _config() -> dict[str, object]:
    return {
        "enabled": True,
        "seed": 42,
        "apply_probability": 1.0,
        "telephone_bandpass_probability": 1.0,
        "narrowband_resample_probability": 1.0,
        "mulaw_alaw_probability": 1.0,
        "gain_variation_probability": 1.0,
        "clipping_probability": 1.0,
        "noise_probability": 1.0,
        "speed_perturbation_probability": 1.0,
        "speed_rate_range": [0.95, 1.05],
    }


def test_training_augmentation_is_seeded_and_caps_transform_count() -> None:
    audio = _tone()
    first, trace_a = augment_training_audio(audio, 16000, "sample-1", _config())
    second, trace_b = augment_training_audio(audio, 16000, "sample-1", _config())
    assert np.array_equal(first, second)
    assert trace_a == trace_b
    assert trace_a.selected
    assert 1 <= len(trace_a.transforms) <= 3


def test_disabled_pipeline_is_exact_identity_copy() -> None:
    audio = _tone()
    result, trace = augment_training_audio(
        audio, 16000, "sample", {"enabled": False, "apply_probability": 0.35}
    )
    assert np.array_equal(result, audio)
    assert result is not audio
    assert not trace.selected


def test_seeded_selection_rate_matches_config_budget() -> None:
    config = _config()
    config["apply_probability"] = 0.35
    config.update(
        {key: 0.0 for key in config if key.endswith("_probability") and key != "apply_probability"}
    )
    selected = sum(
        augment_training_audio([], 16000, f"sample-{index}", config)[1].selected
        for index in range(2000)
    )
    assert 0.30 <= selected / 2000 <= 0.40


def test_probability_validation_fails_closed() -> None:
    config = _config()
    config["apply_probability"] = 1.1
    with pytest.raises(ValueError, match="apply_probability"):
        augment_training_audio(_tone(), 16000, "sample", config)
