from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from masricx.training.callbacks import build_integrity_callback
from masricx.training.plan import build_training_plan
from masricx.training.provenance import experiment_metadata, write_provenance


def _plan(tmp_path: Path) -> object:
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"):
        (splits / name).write_text("{}\n", encoding="utf-8")
    return build_training_plan(Path("configs/codeswitch.yaml"), tmp_path / "output", splits)


def test_provenance_contains_required_fields_and_writes_three_files(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    metadata = experiment_metadata(plan, 123, started_at="2026-01-01T00:00:00+00:00")
    assert metadata["base_model_revision"] == "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    assert metadata["dataset_revision"] == "712de01079517771f95bcdecee68ca232b979628"
    assert metadata["training_examples"] == 123
    assert metadata["started_at"] == "2026-01-01T00:00:00+00:00"
    target = tmp_path / "checkpoint-1"
    write_provenance(target, plan, metadata, [{"eval_wer": 0.5}])
    for name in ("experiment_metadata.json", "training_config.json", "metrics.json"):
        assert (target / name).is_file()
        json.loads((target / name).read_text(encoding="utf-8"))


def test_integrity_callback_stops_nonfinite_and_repeated_wer_degradation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    class FakeTransformers:
        class TrainerCallback:
            pass

    callback = build_integrity_callback(FakeTransformers, plan, {})
    control = SimpleNamespace()
    with pytest.raises(RuntimeError, match="non-finite"):
        callback.on_log(None, None, control, {"loss": float("nan")})
    callback.on_log(None, None, control, {"eval_wer": 0.1})
    callback.on_log(None, None, control, {"eval_wer": 0.2})
    with pytest.raises(RuntimeError, match="degraded"):
        callback.on_log(None, None, control, {"eval_wer": 0.3})
