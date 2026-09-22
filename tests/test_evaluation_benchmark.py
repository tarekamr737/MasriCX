from __future__ import annotations

import json
from pathlib import Path

import pytest

from masricx.evaluation.benchmark import (
    EvaluationRecord,
    evaluate_records,
    render_benchmark,
    write_benchmark,
)


def test_aggregate_metrics_and_buckets_are_micro_averaged() -> None:
    report = evaluate_records(
        [
            EvaluationRecord(
                "cs",
                "اعمل restart ١٢",
                "اعمل restart 12",
                audio_seconds=2.0,
                processing_seconds=1.0,
            ),
            EvaluationRecord("ar", "مرحبا", "اهلا extra", silent=True),
        ]
    )
    metrics = report["metrics"]
    assert report["examples"] == 2
    assert metrics["english_term_recall"] == {"matched": 1, "reference": 1, "value": 1.0}
    assert metrics["number_accuracy"] == {"matched": 1, "reference": 1, "value": 1.0}
    # Digit script is preserved by normalized WER; Number Accuracy separately
    # recognizes Arabic-Indic and ASCII decimal digits as equivalent.
    assert metrics["code_switched_wer"]["rate"] == pytest.approx(1 / 3)
    assert metrics["rtf"] == 0.5
    assert metrics["rtf_timed_examples"] == 1
    assert metrics["hallucination_flagged"] == 1
    assert metrics["hallucination_reasons"] == {"invented_english": 1, "silent_speech": 1}
    assert set(report["buckets"]) == {"AR_EN_CODE_SWITCHED", "AR_ONLY"}


def test_hallucination_repetition_and_excessive_output_are_screening_flags() -> None:
    hypothesis = "hello hello hello one two three four five six seven eight nine ten"
    report = evaluate_records([EvaluationRecord("x", "قصير", hypothesis, audio_seconds=0.5)])
    reasons = report["metrics"]["hallucination_reasons"]
    assert reasons["repeated_phrase"] == 1
    assert reasons["invented_english"] == 1
    assert reasons["excessive_output"] == 1
    assert "manual review" in report["hallucination_policy"]


def test_empty_report_uses_null_not_fabricated_zero_rates() -> None:
    report = evaluate_records([])
    assert report["examples"] == 0
    assert report["metrics"]["wer_raw"]["rate"] is None
    assert report["metrics"]["hallucination_rate"] is None
    assert report["metrics"]["rtf"] is None


def test_negative_timing_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative duration"):
        evaluate_records([EvaluationRecord("bad", "a", "a", 1.0, -0.1)])


def test_renderer_and_writer_are_deterministic_and_text_free(tmp_path: Path) -> None:
    report = evaluate_records(
        [EvaluationRecord("private-id", "secret reference", "secret hypothesis")]
    )
    markdown = render_benchmark(report)
    assert "wer_normalized" in markdown
    assert "secret" not in markdown
    json_path = tmp_path / "benchmark.json"
    md_path = tmp_path / "benchmark.md"
    write_benchmark(report, json_path, md_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["examples"] == 1
    assert md_path.read_text(encoding="utf-8") == markdown
    assert json_path.read_text(encoding="utf-8").endswith("\n")
