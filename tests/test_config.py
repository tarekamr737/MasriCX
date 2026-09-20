"""Unit tests for config loading/validation (offline, no model/dataset access)."""

from __future__ import annotations

from pathlib import Path

import pytest

from masricx.config.loader import ConfigError, load_config, validate_files
from masricx.config.schema import ExperimentRoot


def test_valid_config_loads(write_config, valid_config_yaml) -> None:
    cfg = load_config(write_config(valid_config_yaml))
    assert isinstance(cfg, ExperimentRoot)
    assert cfg.model.id == "openai/whisper-large-v3-turbo"
    assert cfg.lora.r == 16
    assert cfg.training.seed == 42


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


def test_broken_yaml_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("experiment: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="parse"):
        load_config(p)


def test_missing_required_top_level_key(write_config, valid_config_yaml) -> None:
    lines = valid_config_yaml.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("lora:"):
            skipping = True
            continue
        if skipping:
            if line and not line.startswith(" "):
                skipping = False
            else:
                continue
        out.append(line)
    text = "\n".join(out)
    with pytest.raises(ConfigError, match="missing required top-level keys"):
        load_config(write_config(text, "no_lora.yaml"))


def test_unknown_top_level_key_rejected(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml + "\nrogue_section:\n  x: 1\n"
    with pytest.raises(ConfigError, match="unknown top-level keys"):
        load_config(write_config(bad))


def test_tbd_revision_becomes_none(write_config, valid_config_yaml) -> None:
    cfg = load_config(write_config(valid_config_yaml))
    assert cfg.dataset.revision is None
    assert cfg.dataset.license is None


def test_splits_must_sum_to_100(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("train: 90", "train: 80")
    with pytest.raises(ConfigError, match="sum to 100"):
        load_config(write_config(bad))


def test_8bit_required(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("load_in_8bit: true", "load_in_8bit: false")
    with pytest.raises(ConfigError, match="8-bit"):
        load_config(write_config(bad))


def test_invalid_target_modules_rejected(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("target_modules: [q_proj, v_proj]", "target_modules: [w_proj]")
    with pytest.raises(ConfigError, match="target_modules"):
        load_config(write_config(bad))


def test_lora_alpha_must_be_ge_r(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("alpha: 32", "alpha: 8")
    with pytest.raises(ConfigError, match=r"alpha must be >= lora\.r"):
        load_config(write_config(bad))


def test_eval_steps_required_for_steps_strategy(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("  eval_steps: 250\n", "")
    with pytest.raises(ConfigError, match="eval_steps"):
        load_config(write_config(bad))


def test_fp16_required_with_gradient_checkpointing(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("  fp16: true", "  fp16: false")
    with pytest.raises(ConfigError, match="fp16"):
        load_config(write_config(bad))


def test_egyspeak_pseudo_label_must_be_true(tmp_path: Path) -> None:
    text = """\
experiment:
  name: test-e3
  experiment: E3
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
  splits: {train: 90, validation: 5, test: 5, seed: 42}
egyspeak:
  id: MohamedGomaa30/EGYSpeak
  is_pseudo_labelled: false
  conditions: [0, 10, 25]
"""
    p = tmp_path / "e3.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="pseudo"):
        load_config(p)


def test_e3_requires_egyspeak_section(tmp_path: Path) -> None:
    text = """\
experiment:
  name: test-e3
  experiment: E3
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
  id: x/y
  splits: {train: 90, validation: 5, test: 5, seed: 42}
"""
    p = tmp_path / "e3_no_section.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="requires the 'egyspeak' section"):
        load_config(p)


def test_validate_files_returns_all(repo_root: Path) -> None:
    paths = [
        repo_root / "configs/pilot.yaml",
        repo_root / "configs/codeswitch.yaml",
        repo_root / "configs/telephone.yaml",
        repo_root / "configs/egyspeak_ablation.yaml",
    ]
    results = validate_files(paths)
    assert len(results) == 4
    assert all(isinstance(r, ExperimentRoot) for r in results)


def test_nonnegative_splits(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("train: 90", "train: -10").replace(
        "validation: 5", "validation: 15"
    )
    with pytest.raises(ConfigError, match=r"greater than or equal to 0|Input should be valid"):
        load_config(write_config(bad))


def test_exact_model_id_required(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace(
        "id: openai/whisper-large-v3-turbo", "id: openai/whisper-large-v3"
    )
    with pytest.raises(ConfigError, match="exactly"):
        load_config(write_config(bad))


def test_exact_dtype_float16_required(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("dtype: float16", "dtype: bfloat16")
    with pytest.raises(ConfigError):
        load_config(write_config(bad))


def test_lora_targets_must_be_exactly_q_v_proj(write_config, valid_config_yaml) -> None:
    extra_module = valid_config_yaml.replace(
        "target_modules: [q_proj, v_proj]",
        "target_modules: [q_proj, v_proj, k_proj]",
    )
    with pytest.raises(ConfigError, match=r"lora\.target_modules"):
        load_config(write_config(extra_module))

    reordered = valid_config_yaml.replace(
        "target_modules: [q_proj, v_proj]",
        "target_modules: [v_proj, q_proj]",
    )
    assert load_config(write_config(reordered, "reordered.yaml")) is not None


def test_apply_probability_zero_when_disabled(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("apply_probability: 0.0", "apply_probability: 0.35")
    with pytest.raises(ConfigError, match=r"must be 0.0 when enabled=false"):
        load_config(write_config(bad))


def test_apply_probability_range_when_enabled(write_config, valid_config_yaml) -> None:
    enabled = (
        valid_config_yaml.replace("experiment: E1", "experiment: E2")
        .replace("enabled: false", "enabled: true")
        .replace(
            "telephone_bandpass_probability: 0.0",
            "telephone_bandpass_probability: 0.30",
        )
        .replace(
            "narrowband_resample_probability: 0.0",
            "narrowband_resample_probability: 0.30",
        )
        .replace("mulaw_alaw_probability: 0.0", "mulaw_alaw_probability: 0.15")
        .replace("gain_variation_probability: 0.0", "gain_variation_probability: 0.20")
        .replace("clipping_probability: 0.0", "clipping_probability: 0.10")
        .replace("noise_probability: 0.0", "noise_probability: 0.15")
        .replace(
            "speed_perturbation_probability: 0.0",
            "speed_perturbation_probability: 0.10",
        )
        .replace("apply_probability: 0.0", "apply_probability: 0.35")
    )
    assert load_config(write_config(enabled, "enabled_ok.yaml")) is not None

    too_low = enabled.replace("apply_probability: 0.35", "apply_probability: 0.10")
    with pytest.raises(ConfigError, match=r"\[0\.30, 0\.40\]"):
        load_config(write_config(too_low, "too_low.yaml"))

    too_high = enabled.replace("apply_probability: 0.35", "apply_probability: 0.80")
    with pytest.raises(ConfigError, match=r"\[0\.30, 0\.40\]"):
        load_config(write_config(too_high, "too_high.yaml"))


def test_augmentation_and_telephone_test_seeds_required(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace("  enabled: false\n  seed: 42", "  enabled: false")
    with pytest.raises(ConfigError, match="augmentation"):
        load_config(write_config(bad))

    bad_test = valid_config_yaml.replace(
        "telephone_test:\n  enabled: true\n  seed: 42", "telephone_test:\n  enabled: true"
    )
    with pytest.raises(ConfigError, match=r"telephone_test.seed"):
        load_config(write_config(bad_test, "no_tt_seed.yaml"))


def test_speed_rate_range_exact(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace(
        "speed_rate_range: [0.95, 1.05]", "speed_rate_range: [0.9, 1.1]"
    )
    with pytest.raises(ConfigError, match="exactly"):
        load_config(write_config(bad))


def test_e1_requires_augmentation_disabled(write_config, valid_config_yaml) -> None:
    bad = (
        valid_config_yaml.replace("enabled: false", "enabled: true")
        .replace("telephone_bandpass_probability: 0.0", "telephone_bandpass_probability: 0.30")
        .replace(
            "narrowband_resample_probability: 0.0",
            "narrowband_resample_probability: 0.30",
        )
        .replace("mulaw_alaw_probability: 0.0", "mulaw_alaw_probability: 0.15")
        .replace("gain_variation_probability: 0.0", "gain_variation_probability: 0.20")
        .replace("clipping_probability: 0.0", "clipping_probability: 0.10")
        .replace("noise_probability: 0.0", "noise_probability: 0.15")
        .replace("speed_perturbation_probability: 0.0", "speed_perturbation_probability: 0.10")
        .replace("apply_probability: 0.0", "apply_probability: 0.35")
    )
    with pytest.raises(ConfigError, match="E1 is clean-data"):
        load_config(write_config(bad))


def _make_e2(valid_config_yaml: str, kind: str) -> str:
    return valid_config_yaml.replace("experiment: E1", f"experiment: {kind}")


def test_e2_requires_augmentation_enabled(write_config, valid_config_yaml) -> None:
    text = _make_e2(valid_config_yaml, "E2")
    with pytest.raises(ConfigError, match="E2 requires telephone augmentation"):
        load_config(write_config(text, "e2_clean.yaml"))


def test_egyspeak_conditions_exact(tmp_path: Path) -> None:
    base = """\
experiment:
  name: test-e3
  experiment: E3
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
  id: x/y
  splits: {train: 90, validation: 5, test: 5, seed: 42}
egyspeak:
  id: MohamedGomaa30/EGYSpeak
  is_pseudo_labelled: true
  include_percent: 0
  conditions: [0, 10, 25]
"""
    p = tmp_path / "e3_ok.yaml"
    p.write_text(base, encoding="utf-8")
    assert isinstance(load_config(p), ExperimentRoot)

    bad_conditions = base.replace("conditions: [0, 10, 25]", "conditions: [0, 20]")
    p2 = tmp_path / "e3_bad_conditions.yaml"
    p2.write_text(bad_conditions, encoding="utf-8")
    with pytest.raises(ConfigError, match="exactly"):
        load_config(p2)

    bad_include = base.replace(
        "include_percent: 0", "include_percent: 25"
    )  # 25 is valid; use an invalid one
    bad_include = bad_include.replace("include_percent: 25", "include_percent: 50")
    p3 = tmp_path / "e3_bad_include.yaml"
    p3.write_text(bad_include, encoding="utf-8")
    with pytest.raises(ConfigError, match="include_percent"):
        load_config(p3)

    bad_fraction = base.replace(
        "  conditions: [0, 10, 25]",
        "  conditions: [0, 10, 25]\n  max_fraction_of_primary: 0.5",
    )
    p4 = tmp_path / "e3_bad_fraction.yaml"
    p4.write_text(bad_fraction, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"at most 0.25"):
        load_config(p4)


def test_non_42_seed_rejected(write_config, valid_config_yaml) -> None:
    bad = valid_config_yaml.replace(
        "  save_total_limit: 3\n  seed: 42", "  save_total_limit: 3\n  seed: 999"
    )
    with pytest.raises(ConfigError, match="seed"):
        load_config(write_config(bad))


def test_pilot_candidates_exact(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs/pilot.yaml")
    got = sorted((c.run, c.lora_rank, c.learning_rate) for c in cfg.pilot_candidates)  # type: ignore[union-attr]
    assert got == sorted([("P1", 8, 1.0e-4), ("P2", 16, 1.0e-4), ("P3", 16, 5.0e-5)])


def test_pilot_candidates_wrong_combo_rejected(write_config, valid_config_yaml) -> None:
    text = (
        valid_config_yaml.replace("experiment: E1", "experiment: E1")
        + "\npilot_candidates:\n  - run: P1\n    lora_rank: 8\n    learning_rate: 2.0e-4\n"
    )
    with pytest.raises(ConfigError, match="pilot_candidates"):
        load_config(write_config(text, "bad_pilot.yaml"))


def test_programming_errors_not_swallowed(write_config, valid_config_yaml) -> None:
    class Boom(Exception):
        pass

    # A non-ValidationError leaking from the schema should propagate, not be
    # converted into ConfigError.
    import masricx.config.loader as loader_mod

    original = loader_mod.ExperimentRoot.model_validate

    def broken_validate(*args: object, **kwargs: object) -> None:
        raise Boom("programmer error")

    loader_mod.ExperimentRoot.model_validate = broken_validate  # type: ignore[method-assign]
    try:
        with pytest.raises(Boom):
            load_config(write_config(valid_config_yaml, "boom.yaml"))
    finally:
        loader_mod.ExperimentRoot.model_validate = original  # type: ignore[method-assign]


def test_validation_error_is_caught_as_config_error(write_config, valid_config_yaml) -> None:
    with pytest.raises(ConfigError):
        # pydantic ValidationError is raised and wrapped.
        load_config(write_config(valid_config_yaml.replace("epochs: 2", "epochs: 0")))
