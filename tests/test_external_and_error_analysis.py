from __future__ import annotations

import json
from pathlib import Path

from masricx.evaluation.error_analysis import build_error_analysis, classify_error
from masricx.evaluation.external_inference import main as external_main


def test_external_cli_is_evaluation_only_dry_run(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "external.jsonl"
    assert (
        external_main(
            [
                "--model",
                "org/model",
                "--revision",
                "a" * 40,
                "--split",
                "test",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert not output.exists()


def test_error_classification_is_deterministic_and_privacy_safe() -> None:
    records = [
        {
            "sample_id": "one",
            "reference": "اتصل على ١٢ وافتح router",
            "hypothesis": "اتصل على 13",
            "audio_seconds": 0.8,
        },
        {
            "sample_id": "two",
            "reference": "مرحبا",
            "hypothesis": "hello hello hello hello hello hello hello hello hello hello",
            "audio_seconds": 2.0,
        },
    ]
    assert set(classify_error(records[0])) == {
        "english_term_error",
        "number_error",
        "short_utterance_failure",
    }
    report = build_error_analysis(records, per_category=1)
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["category_counts"]["number_error"] == 1
    assert report["category_counts"]["repetition"] == 1
    assert report["category_counts"]["hallucination_candidate"] == 1
    assert "اتصل" not in serialized
    assert "hello" not in serialized


def test_correct_record_has_no_error_categories() -> None:
    assert (
        classify_error({"sample_id": "ok", "reference": "hello عالم", "hypothesis": "hello عالم"})
        == ()
    )
