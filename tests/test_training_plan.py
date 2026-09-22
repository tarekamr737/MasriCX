from __future__ import annotations

import json
from pathlib import Path

import pytest

from masricx.training.checkpoint import latest_complete_checkpoint, validate_checkpoint
from masricx.training.executor import _select_manifest
from masricx.training.lora import lora_config_dict
from masricx.training.plan import build_training_plan
from masricx.training.train import main


def test_lora_config_is_exact_and_fail_closed() -> None:
    config = lora_config_dict(
        {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
    )
    assert config["task_type"] == "SEQ_2_SEQ_LM"
    assert config["target_modules"] == ["q_proj", "v_proj"]
    with pytest.raises(ValueError):
        lora_config_dict({"r": 0, "alpha": 32, "dropout": 0.05, "target_modules": []})


def _complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "experiment_metadata.json",
        "training_config.json",
        "metrics.json",
    ):
        (path / name).write_text("x", encoding="utf-8")


def test_checkpoint_discovery_skips_incomplete_and_selects_latest(tmp_path: Path) -> None:
    _complete_checkpoint(tmp_path / "checkpoint-10")
    _complete_checkpoint(tmp_path / "checkpoint-20")
    (tmp_path / "checkpoint-30").mkdir()
    assert latest_complete_checkpoint(tmp_path) == tmp_path / "checkpoint-20"
    complete, missing = validate_checkpoint(tmp_path / "checkpoint-30")
    assert not complete
    assert "trainer_state.json" in missing


def test_training_plan_and_cli_dry_run_are_network_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"):
        (split_dir / name).write_text("{}\n", encoding="utf-8")
    plan = build_training_plan(Path("configs/pilot.yaml"), tmp_path / "output", split_dir)
    assert plan.model_id == "openai/whisper-large-v3-turbo"
    assert plan.model_revision == "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    assert plan.dataset_revision == "712de01079517771f95bcdecee68ca232b979628"
    assert plan.load_in_8bit
    assert (
        main(
            [
                "--config",
                "configs/pilot.yaml",
                "--output-dir",
                str(tmp_path / "output"),
                "--split-dir",
                str(split_dir),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["resume_from"] is None
    assert payload["experiment"] == "pilot-lora"


def test_plan_requires_persisted_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing fixed split"):
        build_training_plan(Path("configs/codeswitch.yaml"), tmp_path / "output", tmp_path)


def test_manifest_selection_carries_stable_sample_ids() -> None:
    class FakeDataset:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        def select(self, indices: list[int]) -> FakeDataset:
            return FakeDataset([self.rows[index] for index in indices])

        def add_column(self, name: str, values: list[str]) -> FakeDataset:
            for row, value in zip(self.rows, values, strict=True):
                row[name] = value
            return self

    selected = _select_manifest(
        FakeDataset([{"value": 0}, {"value": 1}, {"value": 2}]),
        [{"index": 2, "sample_id": "stable-2"}, {"index": 0, "sample_id": "stable-0"}],
    )
    assert selected.rows == [
        {"value": 2, "_masricx_sample_id": "stable-2"},
        {"value": 0, "_masricx_sample_id": "stable-0"},
    ]
