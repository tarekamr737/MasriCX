"""Network-free validation and materialization of a MasriCX training plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from masricx.training.lora import lora_config_dict

__all__ = ["TrainingPlan", "build_training_plan"]


@dataclass(frozen=True)
class TrainingPlan:
    experiment: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    output_dir: str
    split_dir: str
    seed: int
    load_in_8bit: bool
    fp16: bool
    lora: dict[str, object]
    training: dict[str, object]
    augmentation: dict[str, object]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping: {key}")
    return value


def build_training_plan(config_path: Path, output_dir: Path, split_dir: Path) -> TrainingPlan:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training config must be a mapping")
    experiment = _mapping(config, "experiment")
    model = _mapping(config, "model")
    dataset = _mapping(config, "dataset")
    training = _mapping(config, "training")
    lora = _mapping(config, "lora")
    augmentation = _mapping(config, "augmentation")
    seed = training.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training.seed must be an integer")
    revision = dataset.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("dataset.revision must be a pinned 40-character commit")
    model_revision = model.get("revision")
    if not isinstance(model_revision, str) or len(model_revision) != 40:
        raise ValueError("model.revision must be a pinned 40-character commit")
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"):
        if not (split_dir / name).is_file():
            raise ValueError(f"missing fixed split file: {split_dir / name}")
    return TrainingPlan(
        experiment=str(experiment.get("name")),
        model_id=str(model.get("id")),
        model_revision=model_revision,
        dataset_id=str(dataset.get("id")),
        dataset_revision=revision,
        output_dir=str(output_dir),
        split_dir=str(split_dir),
        seed=seed,
        load_in_8bit=model.get("load_in_8bit") is True,
        fp16=training.get("fp16") is True,
        lora=(lora_config_dict(lora) and dict(lora)),
        training=dict(training),
        augmentation=dict(augmentation),
    )
