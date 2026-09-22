"""Fail-closed checkpoint discovery and completeness validation."""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["checkpoint_step", "latest_complete_checkpoint", "validate_checkpoint"]

_CHECKPOINT = re.compile(r"checkpoint-(\d+)$")
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_REQUIRED = (
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "experiment_metadata.json",
    "training_config.json",
    "metrics.json",
)


def checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT.search(path.name)
    if match is None:
        raise ValueError(f"not a checkpoint-N directory: {path}")
    return int(match.group(1))


def validate_checkpoint(path: Path) -> tuple[bool, tuple[str, ...]]:
    missing = [name for name in _REQUIRED if not (path / name).is_file()]
    if not any((path / name).is_file() for name in _ADAPTER_WEIGHTS):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    return not missing, tuple(missing)


def latest_complete_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_dir() or _CHECKPOINT.fullmatch(path.name) is None:
            continue
        complete, _ = validate_checkpoint(path)
        if complete:
            candidates.append((checkpoint_step(path), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]
