from __future__ import annotations

import numpy as np
import pytest

from masricx.inference.transcribe import _load_audio, build_parser


def test_inference_parser_matches_required_interface() -> None:
    args = build_parser().parse_args(
        [
            "--audio",
            "sample.wav",
            "--model",
            "org/model",
            "--revision",
            "a" * 40,
        ]
    )
    assert args.audio.name == "sample.wav"
    assert args.model == "org/model"
    assert args.revision == "a" * 40


def test_load_audio_reads_mono_and_stereo(tmp_path: object) -> None:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "stereo.wav"
    stereo = np.column_stack([np.ones(160, dtype=np.float32), np.zeros(160, dtype=np.float32)])
    soundfile.write(path, stereo, 16000)
    audio, sample_rate = _load_audio(path)
    assert sample_rate == 16000
    assert audio.shape == (160,)
    assert np.mean(audio) == pytest.approx(0.5, abs=1e-3)
