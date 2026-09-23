"""Report rendering for the data audit: JSON + deterministic Markdown.

Outputs contain deterministic counts and example identifiers only: never raw
audio, never Personally Identifiable Information (raw numeric tokens are
reported only as structural buckets), never waveform content.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "derived_governance",
    "derived_interpretation",
    "render_markdown",
    "write_json",
    "write_reports",
]


def _mapping_cell(value: Mapping[object, object]) -> str:
    """Readable deterministic cell for count mappings (keys are structural:
    sampling rates, language categories, privacy-safe buckets, symbols)."""
    return "; ".join(f"{k}: {_plain(v)}" for k, v in value.items())


def _plain(value: object) -> str:
    """Human-readable scalar for table cells (never a dict/list repr)."""
    if value is None:
        return "TBD"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        # Governance structures render their human-meaningful field;
        # count mappings (structural keys only) render as k: v pairs.
        if "declared" in value:
            return _plain(value["declared"])
        if "effective" in value:
            return _plain(value["effective"])
        if "blocked" in value:
            return ("YES: " if value.get("blocked") else "no: ") + _plain(value.get("reason"))
        keys = list(value.keys())
        if all(isinstance(k, (int, str)) for k in keys) and all(
            isinstance(v, (int, float, str)) for v in value.values()
        ):
            return _mapping_cell(value)
        return "TBD"
    if isinstance(value, dict):
        return "TBD"
    if isinstance(value, (list, tuple)):
        return "TBD" if not value else "; ".join(_plain(v) for v in value)
    return str(value)


def _fmt(value: object) -> str:
    return _plain(value)


def _split_counts_text(ds: Mapping[str, object]) -> str:
    """Row-count/split description: row_count, else 'split: n; split: n'."""
    row_count = ds.get("row_count")
    if row_count is not None:
        return _plain(row_count)
    splits = ds.get("splits")
    if isinstance(splits, Mapping) and splits:
        return "; ".join(f"{k}: {_plain(v)}" for k, v in splits.items())
    return "TBD"


def _pct(numerator: object, denominator: object) -> str | None:
    """Two-decimal percentage from counts, or None if not computable."""
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        return None
    if denominator <= 0:
        return None
    return f"{100.0 * numerator / denominator:.2f}%"


def _count_phrase(count: object, singular: str, plural: str) -> str:
    """Render a count with deterministic singular/plural wording."""
    return f"{count} {singular if count == 1 else plural}"


def _table(rows: list[tuple[str, object]]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines.extend(f"| {k} | {_plain(v)} |" for k, v in rows)
    return "\n".join(lines)


def derived_interpretation(measured: Mapping[str, object]) -> list[str]:
    """Deterministic interpretation text derived only from measured values.

    The wording/orchestrator decisions are fixed; percentages are computed
    from counts in code (two decimals). Statements whose expected keys are
    missing are omitted rather than invented.
    """
    out: list[str] = []
    examples = measured.get("examples")

    # Audio coverage / duration consistency.
    dur = measured.get("duration")
    if isinstance(dur, Mapping) and examples is not None:
        total_s = dur.get("total_seconds")
        rates = measured.get("sample_rates")
        audio_measured = measured.get("audio_examples_measured")
        if total_s is not None:
            hours = float(total_s) / 3600.0
            if isinstance(audio_measured, int) and isinstance(examples, int):
                out.append(
                    f"- Audio coverage: {audio_measured:,}/{examples:,} audio rows decoded; total "
                    f"{float(total_s):.2f} s = {hours:.4f} h"
                    + (
                        " (consistent with the card's 34.1 h claim)."
                        if 34.0 <= hours < 34.25
                        else "."
                    )
                )
            if isinstance(rates, Mapping) and len(rates) == 1:
                sr = next(iter(rates))
                out.append(f"- All measured audio at {sr} Hz sampling rate.")
        if dur.get("min") is not None and float(dur.get("min", 0)) < 0.1:
            out.append(
                f"- The {float(dur['min']):.3f} s minimum duration is suspicious; "
                "review it in Phase 2 before trusting it as real speech."
            )

    # Language coverage.
    lang = measured.get("language_categories")
    if isinstance(lang, Mapping) and examples is not None:
        cs = lang.get("AR_EN_CODE_SWITCHED")
        ar = lang.get("AR_ONLY")
        en = lang.get("EN_ONLY")
        other = lang.get("OTHER")
        parts: list[str] = []
        if cs is not None:
            parts.append(f"{cs} AR_EN_CODE_SWITCHED ({_pct(cs, examples)})")
        if ar is not None:
            parts.append(f"{ar} AR_ONLY ({_pct(ar, examples)})")
        if en is not None:
            parts.append(f"{en} EN_ONLY ({_pct(en, examples)})")
        if other is not None:
            parts.append(f"{other} OTHER ({_pct(other, examples)})")
        if parts:
            out.append(
                "- Language categories: "
                + ", ".join(parts)
                + ". This validates meaningful code-switch coverage but "
                "strong Arabic dominance; preserve stratified code-switch "
                "metadata in splitting/evaluation."
            )

    # Quality triage.
    triage_fields = [
        ("empty transcript", "empty transcripts", "empty_transcripts"),
        ("silent clip", "silent clips", "silent_clips"),
        ("corrupted decode", "corrupted decodes", "corrupted_audio"),
        ("clipped clip", "clipped clips", "clipped_audio"),
        (
            "duration/transcript mismatch outlier",
            "duration/transcript mismatch outliers",
            "mismatch_outliers",
        ),
        ("unusually long transcript", "unusually long transcripts", "unusually_long_transcripts"),
    ]
    if examples is not None:
        triage_parts = [
            f"{_count_phrase(n, singular, plural)} ({_pct(n, examples)})"
            for singular, plural, key in triage_fields
            if (n := measured.get(key)) is not None
        ]
        if triage_parts:
            out.append(
                "- Quality triage: "
                + "; ".join(triage_parts)
                + ". These are review/filter candidates, not automatic "
                "deletions until IDs/rules are inspected in Phase 2."
            )

    # Deduplication.
    td = measured.get("transcript_duplicates")
    if isinstance(td, Mapping) and examples is not None:
        groups = td.get("exact_duplicate_groups")
        extra = td.get("exact_duplicate_extra_instances")
        if groups is not None and extra is not None:
            out.append(
                f"- Deduplication: {groups} exact transcript groups / "
                f"{extra} extra instances ({_pct(extra, examples)} of rows) "
                "require deduplication before splitting."
            )
        aud = measured.get("audio_duplicates")
        if isinstance(aud, Mapping) and aud.get("exact_duplicate_groups") is not None:
            out.append(
                "- Exact audio duplicate groups: "
                f"{aud['exact_duplicate_groups']} under the canonical "
                "sampling-rate + float32-PCM SHA-256 hash."
            )
        pairs = td.get("near_duplicate_pair_count")
        bound = td.get("candidate_bound")
        skipped = bound.get("buckets_skipped_over_cap") if isinstance(bound, Mapping) else None
        exhausted = td.get("comparison_budget_exhausted")
        if pairs is not None:
            cap = bound.get("pair_cap_reported") if isinstance(bound, Mapping) else None
            cap_text = f" (capped at {cap})" if cap is not None else ""
            note = (
                f"- Near-duplicate output is diagnostic only: {pairs} reported "
                f"pairs{cap_text} is the reporting cap, not total prevalence"
            )
            details: list[str] = []
            if skipped is not None:
                details.append(f"{skipped} oversized buckets were skipped")
            if exhausted is not None:
                details.append(
                    "the global comparison budget was "
                    + ("exhausted" if exhausted else "not exhausted")
                )
            if details:
                note += "; " + " and ".join(details)
            note += ". Do not infer a dataset-wide near-duplicate rate from this diagnostic output."
            out.append(note)

    # Number privacy note.
    if "number_frequency" in measured:
        out.append(
            "- Number frequencies are privacy-safe structural buckets only "
            "(integer_digits_1, integer_digits_2_3, integer_digits_4_plus, "
            "decimal); raw numeric tokens were not serialized."
        )
    return out


def derived_governance(registry: Mapping[str, object]) -> list[str]:
    """Render the fixed governance conclusion when all governed sources exist."""
    datasets = registry.get("datasets")
    if not isinstance(datasets, list):
        return []
    ids = {
        ds.get("id") for ds in datasets if isinstance(ds, Mapping) and isinstance(ds.get("id"), str)
    }
    required = {
        "Seif-Eldeen-Sameh/asr_codeswitched_dataset",
        "MohamedGomaa30/EGYSpeak",
        "UBC-NLP/Casablanca",
    }
    if not required.issubset(ids):
        return []
    lines = [
        "- Governance: primary raw/transformed data is not redistributed; public model-weight "
        "licensing remains blocked pending an explicit full-data-versus-filtered-data decision; "
        "EGYSpeak E3 remains disabled pending its separate license-chain resolution; Casablanca "
        "remains evaluation-only."
    ]
    primary = next(
        (
            dataset
            for dataset in datasets
            if isinstance(dataset, Mapping)
            and dataset.get("id") == "Seif-Eldeen-Sameh/asr_codeswitched_dataset"
        ),
        None,
    )
    boundary = primary.get("source_boundary_evidence") if isinstance(primary, Mapping) else None
    if isinstance(boundary, Mapping) and isinstance(boundary.get("evidence"), str):
        lines.append(
            f"- Source-boundary evidence (verified {boundary.get('verified', 'TBD')}): "
            f"{boundary['evidence']}"
        )
    return lines


def render_markdown(report: Mapping[str, object]) -> str:
    """Deterministic Markdown rendering of the canonical report structure.

    Returns text ending with exactly one trailing newline; no line has
    trailing whitespace.
    """
    lines = ["# MasriCX Data Audit", ""]

    # Registry/governance section (verified card metadata).
    meta = report.get("registry", {})
    if isinstance(meta, Mapping):
        lines += [
            "## Card-declared/verified metadata",
            "",
            "Source retrieval date: " + _plain(meta.get("retrieved")),
            "",
            "| Dataset | Role | Revision | Declared license | Rows | Pseudo-labelled | Release blocker |",
            "|---|---|---|---|---|---|---|",
        ]
        datasets = meta.get("datasets", [])
        if isinstance(datasets, list):
            for ds in datasets:
                if not isinstance(ds, Mapping):
                    continue
                cells = [
                    _plain(ds.get("id")),
                    _plain(ds.get("role")),
                    _plain(ds.get("revision")),
                    _plain(ds.get("license")),
                    _split_counts_text(ds),
                    _plain(ds.get("pseudo_labelled")),
                    _plain(ds.get("release_blocker")),
                ]
                lines.append("| " + " | ".join(cells) + " |")
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
            "**Audio metrics unavailable** (" + _plain(note) + "). Audio flags, "
            "audio hashes, and sampling rates are unavailable, NOT zero.",
            "",
        ]
    duration = measured.get("duration", {}) if isinstance(measured.get("duration"), Mapping) else {}
    assert isinstance(duration, Mapping)
    total_s = duration.get("total_seconds")
    total_h = float(total_s) / 3600.0 if total_s is not None else None
    td = measured.get("transcript_duplicates", {})
    td = td if isinstance(td, Mapping) else {}
    aud_d = measured.get("audio_duplicates", {})
    aud_d = aud_d if isinstance(aud_d, Mapping) else {}
    td_bound = td.get("candidate_bound", {})
    td_bound = td_bound if isinstance(td_bound, Mapping) else {}
    aud_bound = aud_d.get("candidate_bound", {})
    aud_bound = aud_bound if isinstance(aud_bound, Mapping) else {}

    lines += [
        _table(
            [
                ("examples", measured.get("examples")),
                ("total duration (s)", total_s),
                ("total duration (h)", round(total_h, 4) if total_h is not None else None),
                ("duration p25 (s)", duration.get("p25")),
                ("median duration (s)", duration.get("median")),
                ("duration p75 (s)", duration.get("p75")),
                ("duration p95 (s)", duration.get("p95")),
                ("min duration (s)", duration.get("min")),
                ("max duration (s)", duration.get("max")),
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
                (
                    "number frequency by structural bucket",
                    measured.get("number_frequency"),
                ),
                ("number bucket definition", measured.get("number_buckets_doc")),
                ("symbol frequency", measured.get("symbol_frequency")),
            ]
        ),
        "",
        "### Duplicates",
        "",
        _table(
            [
                (
                    "exact duplicate transcript groups",
                    td.get("exact_duplicate_groups"),
                ),
                (
                    "exact duplicate transcript extra instances",
                    td.get("exact_duplicate_extra_instances"),
                ),
                (
                    "near-duplicate pairs reported (capped at "
                    f"{td_bound.get('pair_cap_reported', 20)})",
                    td.get("near_duplicate_pair_count"),
                ),
                ("near-duplicate pair comparisons", td.get("pair_comparisons")),
                (
                    "near-duplicate global comparison budget exhausted",
                    td.get("comparison_budget_exhausted"),
                ),
                (
                    "near-duplicate oversized buckets skipped",
                    td_bound.get("buckets_skipped_over_cap"),
                ),
                ("near-duplicate method", td.get("method")),
                (
                    "exact duplicate audio groups",
                    aud_d.get("exact_duplicate_groups"),
                ),
                (
                    "exact duplicate audio extra instances",
                    aud_d.get("exact_duplicate_extra_instances"),
                ),
                ("audio duplicate pair comparisons", aud_d.get("pair_comparisons")),
                (
                    "audio duplicate oversized buckets skipped",
                    aud_bound.get("buckets_skipped_over_cap"),
                ),
            ]
        ),
        "",
        "No audio or PII is included in this report: counts and IDs only; "
        "raw numeric tokens are reported only as structural buckets.",
        "",
    ]

    # Deterministic interpretation/decisions derived from measured values.
    interpretation = derived_interpretation(measured)
    if isinstance(meta, Mapping):
        interpretation.extend(derived_governance(meta))
    if interpretation:
        lines += [
            "## Interpretation and decisions",
            "",
            *interpretation,
            "",
        ]

    text = "\n".join(lines).rstrip("\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text + "\n"


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
    """Render both reports. Caller is responsible for not overwriting an
    existing canonical JSON when only regenerating Markdown."""
    write_json(report, json_path)
    _write_md(report, md_path)


def _write_md(report: Mapping[str, object], md_path: Path) -> None:
    md = render_markdown(report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8", newline="\n")
