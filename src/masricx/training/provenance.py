"""Experiment provenance artifacts required in final outputs and checkpoints."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from masricx.training.plan import TrainingPlan

__all__ = ["experiment_metadata", "git_state", "write_provenance"]


def git_state(repo_root: Path = Path(".")) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "TBD", True
    return commit, dirty


def experiment_metadata(
    plan: TrainingPlan, training_examples: int, *, started_at: str | None = None
) -> dict[str, Any]:
    commit, dirty = git_state()
    return {
        "experiment": plan.experiment,
        "git_commit": commit,
        "git_dirty": dirty,
        "base_model": plan.model_id,
        "base_model_revision": plan.model_revision,
        "dataset_id": plan.dataset_id,
        "dataset_revision": plan.dataset_revision,
        "seed": plan.seed,
        "learning_rate": plan.training.get("learning_rate"),
        "lora_rank": plan.lora.get("r"),
        "lora_alpha": plan.lora.get("alpha"),
        "augmentation": "telephone-v1" if plan.augmentation.get("enabled") is True else "none",
        "training_examples": training_examples,
        "started_at": started_at or datetime.now(UTC).isoformat(),
    }


def write_provenance(
    directory: Path, plan: TrainingPlan, metadata: dict[str, Any], metrics: object
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "experiment_metadata.json": metadata,
        "training_config.json": plan.as_dict(),
        "metrics.json": metrics,
    }
    for name, payload in files.items():
        (directory / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
