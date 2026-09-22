from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from masricx.evaluation.prediction_artifact import ensure_prediction_provenance
from masricx.evaluation.run_inference import BASELINE_MODELS, _completed, _manifest, main


def test_required_baselines_are_fixed() -> None:
    assert BASELINE_MODELS == (
        "openai/whisper-large-v3-turbo",
        "Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched",
        "NAMAA-Space/EgypTalk-ASR-v2",
    )


def test_manifest_limit_and_resume_ids(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps({"sample_id": f"id-{i}", "index": i}) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    assert len(_manifest(path, 2)) == 2
    assert _completed(path) == {"id-0", "id-1", "id-2"}


def test_cli_is_dry_run_by_default(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "predictions.jsonl"
    assert (
        main(["--model", BASELINE_MODELS[0], "--revision", "a" * 40, "--output", str(output)]) == 0
    )
    assert not output.exists()


def test_prediction_provenance_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    sidecar = ensure_prediction_provenance(output, {"model": "a", "schema_version": 1})
    assert json.loads(sidecar.read_text(encoding="utf-8"))["model"] == "a"
    ensure_prediction_provenance(output, {"model": "a", "schema_version": 1})
    with pytest.raises(ValueError, match="provenance mismatch"):
        ensure_prediction_provenance(output, {"model": "b", "schema_version": 1})


def test_existing_predictions_require_provenance(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing provenance"):
        ensure_prediction_provenance(output, {"model": "a"})


def test_baseline_config_matches_required_models() -> None:
    payload = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    models = payload["models"]
    assert tuple(item["id"] for item in models) == BASELINE_MODELS
    assert all(len(item["revision"]) == 40 for item in models)
    assert {item["framework"] for item in models} == {"transformers", "nemo"}
