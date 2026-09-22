"""Aggregate saved ASR predictions into deterministic benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from masricx.data.text import TokenLanguage, classify_token, code_switched_category
from masricx.evaluation.metrics import (
    ErrorCounts,
    character_error_rate,
    english_term_recall,
    number_accuracy,
    word_error_rate,
)
from masricx.evaluation.normalize import normalize_normalized

__all__ = ["EvaluationRecord", "evaluate_records", "render_benchmark", "write_benchmark"]


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    reference: str
    hypothesis: str
    audio_seconds: float | None = None
    processing_seconds: float | None = None
    silent: bool = False


def _merge(values: Sequence[ErrorCounts]) -> ErrorCounts:
    return ErrorCounts(
        substitutions=sum(value.substitutions for value in values),
        deletions=sum(value.deletions for value in values),
        insertions=sum(value.insertions for value in values),
        reference_units=sum(value.reference_units for value in values),
    )


def _language_error(reference: str, hypothesis: str, category: TokenLanguage) -> ErrorCounts:
    ref = [
        token
        for token in normalize_normalized(reference).split()
        if classify_token(token) is category
    ]
    hyp = [
        token
        for token in normalize_normalized(hypothesis).split()
        if classify_token(token) is category
    ]
    return _token_errors(ref, hyp)


def _token_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Reuse the public WER implementation with collision-safe token encoding."""
    separator = "\u241f"
    if any(separator in token for token in (*reference, *hypothesis)):
        raise ValueError("reserved evaluation separator occurs in a token")
    return word_error_rate(" ".join(reference), " ".join(hypothesis), mode="raw")


def _repeated(tokens: Sequence[str]) -> bool:
    for width in range(1, min(4, len(tokens) // 3) + 1):
        for start in range(len(tokens) - (3 * width) + 1):
            phrase = tokens[start : start + width]
            if (
                phrase
                == tokens[start + width : start + 2 * width]
                == tokens[start + 2 * width : start + 3 * width]
            ):
                return True
    return False


def _english(text: str) -> Counter[str]:
    return Counter(
        token
        for token in normalize_normalized(text).split()
        if classify_token(token) is TokenLanguage.EN_ONLY
    )


def _hallucination_reasons(row: EvaluationRecord) -> tuple[str, ...]:
    ref = normalize_normalized(row.reference)
    hyp = normalize_normalized(row.hypothesis)
    tokens = hyp.split()
    reasons: list[str] = []
    if row.silent and hyp:
        reasons.append("silent_speech")
    if _repeated(tokens):
        reasons.append("repeated_phrase")
    if _english(row.hypothesis) - _english(row.reference):
        reasons.append("invented_english")
    if (len(tokens) >= 10 and len(hyp) > max(20, 3 * len(ref))) or (
        row.audio_seconds is not None and row.audio_seconds < 1.0 and len(tokens) >= 10
    ):
        reasons.append("excessive_output")
    return tuple(reasons)


def _error_payload(counts: ErrorCounts) -> dict[str, int | float | None]:
    return {**asdict(counts), "rate": counts.rate}


def evaluate_records(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    """Compute micro-averaged metrics without serializing text or audio."""
    rows = list(records)
    errors: dict[str, list[ErrorCounts]] = defaultdict(list)
    buckets: dict[str, list[ErrorCounts]] = defaultdict(list)
    english_matched = english_reference = number_matched = number_reference = 0
    hallucination_reasons: Counter[str] = Counter()
    hallucinated = 0
    processing_total = audio_total = 0.0
    timed_rows = 0
    for row in rows:
        raw = word_error_rate(row.reference, row.hypothesis, mode="raw")
        normalized = word_error_rate(row.reference, row.hypothesis, mode="normalized")
        errors["wer_raw"].append(raw)
        errors["wer_normalized"].append(normalized)
        errors["cer_raw"].append(character_error_rate(row.reference, row.hypothesis, mode="raw"))
        errors["cer_normalized"].append(
            character_error_rate(row.reference, row.hypothesis, mode="normalized")
        )
        errors["arabic_token_wer"].append(
            _language_error(row.reference, row.hypothesis, TokenLanguage.AR_ONLY)
        )
        errors["english_token_wer"].append(
            _language_error(row.reference, row.hypothesis, TokenLanguage.EN_ONLY)
        )
        category = code_switched_category(row.reference)
        buckets[category].append(normalized)
        if category == "AR_EN_CODE_SWITCHED":
            errors["code_switched_wer"].append(normalized)
        english = english_term_recall(row.reference, row.hypothesis)
        english_matched += english.matched
        english_reference += english.reference
        numbers = number_accuracy(row.reference, row.hypothesis)
        number_matched += numbers.matched
        number_reference += numbers.reference
        reasons = _hallucination_reasons(row)
        hallucinated += int(bool(reasons))
        hallucination_reasons.update(reasons)
        if row.processing_seconds is not None and row.audio_seconds is not None:
            if row.processing_seconds < 0 or row.audio_seconds < 0:
                raise ValueError(f"negative duration for sample {row.sample_id}")
            processing_total += row.processing_seconds
            audio_total += row.audio_seconds
            timed_rows += 1
    merged = {name: _error_payload(_merge(values)) for name, values in errors.items()}
    for required in (
        "wer_raw",
        "wer_normalized",
        "cer_raw",
        "cer_normalized",
        "code_switched_wer",
        "arabic_token_wer",
        "english_token_wer",
    ):
        merged.setdefault(required, _error_payload(_merge([])))
    return {
        "schema_version": 1,
        "examples": len(rows),
        "metrics": {
            **merged,
            "english_term_recall": {
                "matched": english_matched,
                "reference": english_reference,
                "value": english_matched / english_reference if english_reference else None,
            },
            "number_accuracy": {
                "matched": number_matched,
                "reference": number_reference,
                "value": number_matched / number_reference if number_reference else None,
            },
            "hallucination_rate": hallucinated / len(rows) if rows else None,
            "hallucination_flagged": hallucinated,
            "hallucination_reasons": dict(sorted(hallucination_reasons.items())),
            "rtf": processing_total / audio_total if audio_total else None,
            "rtf_timed_examples": timed_rows,
        },
        "buckets": {
            name: _error_payload(_merge(values)) for name, values in sorted(buckets.items())
        },
        "hallucination_policy": "Automated screening only; manual review is required for every flagged sample.",
    }


def _percent(value: object) -> str:
    return f"{100 * value:.2f}%" if isinstance(value, (int, float)) else "TBD"


def render_benchmark(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "# MasriCX Benchmark Results",
        "",
        f"Evaluated examples: {report.get('examples', 0)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name in (
        "wer_raw",
        "wer_normalized",
        "cer_raw",
        "cer_normalized",
        "code_switched_wer",
        "arabic_token_wer",
        "english_token_wer",
    ):
        lines.append(f"| {name} | {_percent(metrics.get(name, {}).get('rate'))} |")
    for name in ("english_term_recall", "number_accuracy"):
        lines.append(f"| {name} | {_percent(metrics.get(name, {}).get('value'))} |")
    lines += [
        f"| hallucination_rate | {_percent(metrics.get('hallucination_rate'))} |",
        f"| RTF | {metrics.get('rtf') if metrics.get('rtf') is not None else 'TBD'} |",
        "",
        "## Normalized WER by reference-language bucket",
        "",
        "| Bucket | WER | Reference words |",
        "|---|---:|---:|",
    ]
    for category, value in sorted(report.get("buckets", {}).items()):
        lines.append(
            f"| {category} | {_percent(value.get('rate'))} | {value.get('reference_units')} |"
        )
    lines += ["", "## Hallucination policy", "", str(report.get("hallucination_policy")), ""]
    return "\n".join(lines)


def write_benchmark(report: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(render_benchmark(report), encoding="utf-8", newline="\n")


def _load_jsonl(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationRecord(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid prediction record at {path}:{line_number}: {exc}"
                ) from exc
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate saved MasriCX prediction JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/BENCHMARK_RESULTS.md"))
    args = parser.parse_args(argv)
    write_benchmark(
        evaluate_records(_load_jsonl(args.predictions)), args.output_json, args.output_md
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
