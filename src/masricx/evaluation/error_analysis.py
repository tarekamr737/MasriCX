"""Privacy-safe deterministic candidate selection for manual ASR error review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from masricx.evaluation.metrics import english_term_recall, number_accuracy, word_error_rate
from masricx.evaluation.normalize import normalize_normalized

CATEGORIES = (
    "english_term_error",
    "number_error",
    "hallucination_candidate",
    "repetition",
    "short_utterance_failure",
    "long_utterance_failure",
    "other_word_error",
)


def _repetition(text: str) -> bool:
    tokens = normalize_normalized(text).split()
    return any(
        tokens[index] == tokens[index + 1] == tokens[index + 2] for index in range(len(tokens) - 2)
    )


def classify_error(record: dict[str, Any]) -> tuple[str, ...]:
    reference = str(record.get("reference") or "")
    hypothesis = str(record.get("hypothesis") or "")
    wer = word_error_rate(reference, hypothesis, mode="normalized").rate
    if not wer:
        return ()
    categories: list[str] = []
    english = english_term_recall(reference, hypothesis)
    if english.reference and english.matched < english.reference:
        categories.append("english_term_error")
    numbers = number_accuracy(reference, hypothesis)
    if numbers.reference and numbers.matched < numbers.reference:
        categories.append("number_error")
    ref_norm, hyp_norm = normalize_normalized(reference), normalize_normalized(hypothesis)
    if (not ref_norm and hyp_norm) or len(hyp_norm) > max(20, 3 * len(ref_norm)):
        categories.append("hallucination_candidate")
    if _repetition(hypothesis):
        categories.append("repetition")
    duration = record.get("audio_seconds")
    if isinstance(duration, (int, float)) and duration < 1.0:
        categories.append("short_utterance_failure")
    if isinstance(duration, (int, float)) and duration > 15.0:
        categories.append("long_utterance_failure")
    return tuple(categories or ["other_word_error"])


def build_error_analysis(records: list[dict[str, Any]], per_category: int = 20) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    candidates: dict[str, list[dict[str, object]]] = {name: [] for name in CATEGORIES}
    for record in records:
        categories = classify_error(record)
        if not categories:
            continue
        counts.update(categories)
        sample_id = str(record["sample_id"])
        rate = word_error_rate(
            str(record.get("reference") or ""),
            str(record.get("hypothesis") or ""),
            mode="normalized",
        ).rate
        item: dict[str, object] = {
            "sample_id": sample_id,
            "normalized_wer": rate,
            "categories": categories,
        }
        for category in categories:
            candidates[category].append(item)
    for category, items in candidates.items():
        items.sort(
            key=lambda item: hashlib.sha256(f"{category}:{item['sample_id']}".encode()).hexdigest()
        )
        candidates[category] = items[:per_category]
    return {
        "schema_version": 1,
        "records": len(records),
        "category_counts": {name: counts[name] for name in CATEGORIES},
        "review_candidates": candidates,
        "privacy": "IDs, categories, and error rates only; no transcript or audio serialized.",
        "status": "automated candidates only; manual review still required",
    }


def render_error_analysis(report: dict[str, Any]) -> str:
    lines = [
        "# MasriCX Error Analysis Candidates",
        "",
        f"Records evaluated: {report['records']}",
        "",
        "Automated candidate selection only; manual review is still required.",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in report["category_counts"].items())
    lines += ["", str(report["privacy"]), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic ASR error-review candidates")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/error_analysis.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/ERROR_ANALYSIS_RESULTS.md"))
    parser.add_argument("--per-category", type=int, default=20)
    args = parser.parse_args(argv)
    records = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line
    ]
    report = build_error_analysis(records, args.per_category)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    args.output_md.write_text(render_error_analysis(report), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
