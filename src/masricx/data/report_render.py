"""Report rendering for the data audit: JSON + deterministic Markdown.

Outputs contain deterministic counts and example identifiers only: never raw
audio, never Personally Identifiable Information, never waveform content.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = ["render_markdown", "write_json", "write_reports"]

_METADATA_SECTIONS: tuple[str, ...] = ("registry", "governance")


def _fmt(value: object) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(rows: list[tuple[str, object]]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines.extend(f"| {k} | {_fmt(v)} |" for k, v in rows)
    return "\n".join(lines)


def render_markdown(report: Mapping[str, object]) -> str:
    """Deterministic Markdown rendering of the canonical report structure."""
    lines = ["# MasriCX Data Audit", ""]

    # Registry/governance section (verified card metadata).
    meta = report.get("registry", {})
    if isinstance(meta, Mapping):
        lines += [
            "## Card-declared/verified metadata",
            "",
            "Source retrieval date: " + _fmt(meta.get("retrieved")) + "",
            "",
            "| Dataset | Role | Revision | Declared license | Rows | Pseudo-labelled | Release blocker |",
            "|---|---|---|---|---|---|---|",
        ]
        tables = meta.get("datasets", [])
        if isinstance(tables, list):
            for ds in tables:
                if isinstance(ds, Mapping):
                    lines.append(
                        " | ".join(
                            [
                                "",
                                _fmt(ds.get("id")),
                                _fmt(ds.get("role")),
                                _fmt(ds.get("revision")),
                                _fmt(ds.get("license")),
                                _fmt(ds.get("row_count")),
                                _fmt(ds.get("pseudo_labelled")),
                                _fmt(ds.get("release_blocker")),
                                "",
                            ]
                        )
                    )
        lines.append("")

    # Measured section.
    measured = report.get("measured", {})
    lines += ["## Measured audit results", ""]
    if not isinstance(measured, Mapping) or not measured:
        lines += ["**NOT RUN** - no measured values in this report.", ""]
        return "\n".join(lines) + "\n"
    if not measured.get("audio_metrics_available", True):
        note = measured.get("audio_metrics_note", "") or "audio decoding skipped"
        lines += [
            "**Audio metrics unavailable** (" + _fmt(note) + "). Audio flags, "
            "audio hashes, and sampling rates are unavailable, NOT zero.",
            "",
        ]
    lines += [
        _table(
            [
                ("examples", measured.get("examples")),
                ("total duration (s)", measured.get("duration", {}).get("total_seconds")),
                ("duration p25 (s)", measured.get("duration", {}).get("p25")),
                ("median duration (s)", measured.get("duration", {}).get("median")),
                ("duration p75 (s)", measured.get("duration", {}).get("p75")),
                ("duration p95 (s)", measured.get("duration", {}).get("p95")),
                ("min duration (s)", measured.get("duration", {}).get("min")),
                ("max duration (s)", measured.get("duration", {}).get("max")),
                ("sampling-rate distribution", measured.get("sample_rates")),
                ("empty transcripts", measured.get("empty_transcripts")),
                ("corrupted audio", measured.get("corrupted_audio")),
                ("silent clips", measured.get("silent_clips")),
                ("clipped audio", measured.get("clipped_audio")),
                ("mismatch outliers", measured.get("mismatch_outliers")),
                ("unusually long transcripts", measured.get("unusually_long_transcripts")),
                ("Arabic letter ratio", measured.get("arabic_letter_ratio")),
                ("Latin letter ratio", measured.get("latin_letter_ratio")),
                ("language categories", measured.get("language_categories")),
            ]
        ),
        "",
        "### Numbers and symbols",
        "",
        _table(
            [
                ("number frequency by structural bucket", measured.get("number_frequency")),
                ("symbol frequency (top 50)", measured.get("symbol_frequency")),
            ]
        ),
        "",
        "### Duplicates",
        "",
        _table(
            [
                (
                    "exact duplicate transcript groups",
                    measured.get("transcript_duplicates", {}).get("exact_duplicate_groups"),
                ),
                (
                    "exact duplicate transcript extra instances",
                    measured.get("transcript_duplicates", {}).get(
                        "exact_duplicate_extra_instances"
                    ),
                ),
                (
                    "near-duplicate pairs reported",
                    measured.get("transcript_duplicates", {}).get("near_duplicate_pair_count"),
                ),
                ("near-duplicate method", measured.get("transcript_duplicates", {}).get("method")),
                (
                    "exact duplicate audio groups",
                    measured.get("audio_duplicates", {}).get("exact_duplicate_groups"),
                ),
                (
                    "exact duplicate audio extra instances",
                    measured.get("audio_duplicates", {}).get("exact_duplicate_extra_instances"),
                ),
            ]
        ),
        "",
        "No audio or PII is included in this report: counts and IDs only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_json(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def write_reports(
    report: Mapping[str, object],
    json_path: Path,
    md_path: Path,
) -> None:
    write_json(report, json_path)
    md = render_markdown(report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
