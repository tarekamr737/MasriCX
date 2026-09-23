"""LoRA configuration and lazy PEFT model preparation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

__all__ = ["enable_whisper_input_grads", "lora_config_dict", "prepare_lora_model"]


def enable_whisper_input_grads(model: Any) -> Any:
    """Keep Whisper encoder activations differentiable under checkpointing."""

    def make_inputs_require_grad(module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    return model


def _integer(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"lora.{key} must be a positive integer")
    return value


def _float(config: Mapping[str, object], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"lora.{key} must be numeric")
    return float(value)


def lora_config_dict(config: Mapping[str, object]) -> dict[str, object]:
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) for item in targets)
    ):
        raise ValueError("lora.target_modules must be a non-empty string list")
    dropout = _float(config, "dropout")
    if not 0 <= dropout < 1:
        raise ValueError("lora.dropout must be in [0, 1)")
    return {
        "r": _integer(config, "r"),
        "lora_alpha": _integer(config, "alpha"),
        "lora_dropout": dropout,
        "target_modules": targets,
        "bias": "none",
        "task_type": "SEQ_2_SEQ_LM",
    }


def prepare_lora_model(model: Any, config: Mapping[str, object]) -> Any:
    """Prepare an already-loaded 8-bit model; imports PEFT only at execution."""
    try:
        peft = importlib.import_module("peft")
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise RuntimeError("PEFT is required for training; install requirements.txt") from exc
    prepared = peft.prepare_model_for_kbit_training(model)
    enable_whisper_input_grads(prepared)
    return peft.get_peft_model(prepared, peft.LoraConfig(**lora_config_dict(config)))
