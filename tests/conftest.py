"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def valid_config_yaml() -> str:
    return """\
experiment:
  name: test-e1
  experiment: E1
model:
  id: openai/whisper-large-v3-turbo
  load_in_8bit: true
  dtype: float16
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj]
training:
  learning_rate: 1.0e-4
  epochs: 2
  warmup_ratio: 0.05
  weight_decay: 0.01
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
  eval_strategy: steps
  eval_steps: 250
  save_strategy: steps
  save_steps: 250
  save_total_limit: 3
  seed: 42
dataset:
  id: Seif-Eldeen-Sameh/asr_codeswitched_dataset
  revision: TBD
  license: TBD
  splits:
    train: 90
    validation: 5
    test: 5
    seed: 42
augmentation:
  enabled: false
  seed: 42
  telephone_bandpass_probability: 0.0
  narrowband_resample_probability: 0.0
  mulaw_alaw_probability: 0.0
  gain_variation_probability: 0.0
  clipping_probability: 0.0
  noise_probability: 0.0
  speed_perturbation_probability: 0.0
  speed_rate_range: [0.95, 1.05]
  apply_probability: 0.0
  preserve_transcript: true
telephone_test:
  enabled: true
  seed: 42
  bandpass_hz: [300, 3400]
  reference_resample_khz: 8
"""


@pytest.fixture()
def write_config(tmp_path: Path):
    def _write(text: str, name: str = "config.yaml") -> Path:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    return _write
