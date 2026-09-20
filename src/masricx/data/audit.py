"""MasriCX deterministic data audit engine and CLI.

Usage:

    python -m masricx.data.audit --config configs/codeswitch.yaml
    python -m masricx.data.audit --config configs/codeswitch.yaml \
        --max-examples 500 --streaming

Deterministic audit: for each example it computes compact facts (transcript
language stats, duration quantiles, silence/clipping flags, hashes for exact
duplicate audio), aggregates them in bounded memory, and renders
``artifacts/data_audit.json`` plus ``reports/DATA_AUDIT.md``.

- Full audit decodes audio by default.
- ``--no-audio`` marks audio metrics **unavailable**, not zero; the loader is
  asked to disable audio decoding (``decode_audio=False``) so no waveform is
  ever decoded or touched.
- Outputs contain no audio, no PII: counts, statistics, and sample IDs only.
- Near-duplicate detection uses per-bigram hash bucketing (candidate
  generation; buckets over the member cap are skipped) + exact bigram
  Jaccard, with an explicit worst-case comparison bound (see
  ``masricx.data.near_duplicates``); it is NOT guaranteed subquadratic nor
  exhaustive.
- Phase 1 scope: this CLI audits the PRIMARY code-switch dataset only. If the
  config names a different registered dataset, it fails loudly rather than
  silently loading it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from masricx.data.aggregators import AuditAggregator
from masricx.data.audio_metrics import audio_fingerprint, audio_flags, is_duration_mismatch
from masricx.data.example_audit import ExampleAudit
from masricx.data.load_codeswitch import (
    CODESWITCH_DEFAULT_CONFIG,
    load_codeswitch_examples,
)
from masricx.data.report_render import write_reports
from masricx.data.text import (
    audit_normalize,
    clean_transcript,
    code_switched_category,
    is_unusually_long_transcript,
)

__all__ = ["audit_example", "build_parser", "main", "run_audit"]

DEFAULT_JSON_PATH = Path("artifacts/data_audit.json")
DEFAULT_MD_PATH = Path("reports/DATA_AUDIT.md")

_REGISTRY_ID_BY_DATASET: dict[str, dict[str, str]] = {
    "Seif-Eldeen-Sameh/asr_codeswitched_dataset": {},
    "MohamedGomaa30/EGYSpeak": {},
    "UBC-NLP/Casablanca": {},
}

# Phase 1: this CLI audits only the primary dataset; other registered datasets
# fail loudly instead of being silently loaded (their loaders exist and are
# individually governance-gated).
_PRIMARY_AUDIT_ONLY = frozenset({"Seif-Eldeen-Sameh/asr_codeswitched_dataset"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m masricx.data.audit",
        description="Deterministic offline data audit for MasriCX datasets.",
    )
    parser.add_argument("--config", required=True, help="MasriCX experiment config YAML")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Audit at most N examples (0 audits none; useful for smoke runs)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream examples instead of materializing the split",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio decoding; audio metrics are reported as unavailable, not zero",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Path for the JSON report (default artifacts/data_audit.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="Path for the Markdown report (default reports/DATA_AUDIT.md)",
    )
    return parser


def audit_example(
    sample_id: str,
    index: int,
    transcript_raw: str,
    duration_seconds: float | None,
    sample_rate: int | None,
    samples: Any = None,
) -> ExampleAudit:
    """Pure per-example fact extraction testable on synthetic records.

    ``samples`` is a 1-D waveform (array-like) or None when audio is skipped.
    """
    cleaned = clean_transcript(transcript_raw or "")
    normalized = audit_normalize(transcript_raw or "")
    category = code_switched_category(cleaned)
    duration = duration_seconds if duration_seconds is not None else None

    silent = clipped = corrupted = None
    fingerprint = None
    if samples is not None:
        values = list(samples)
        flags = audio_flags(values)
        silent, clipped, corrupted = flags.silent, flags.clipped, flags.corrupted
        sr = sample_rate if sample_rate is not None else 16000
        fingerprint = audio_fingerprint(sr, values)
        if duration is None and values:
            # Derived duration from the waveform; it is stored in the record
            # AND used for mismatch evaluation below.
            duration = len(values) / float(sr or 16000)

    mismatch = (
        is_duration_mismatch(cleaned, duration) if duration is not None and duration > 0 else False
    )
    empty = not cleaned
    from masricx.data.text import token_audit_stats

    toks = token_audit_stats(cleaned)
    return ExampleAudit(
        sample_id=sample_id,
        index=index,
        transcript_raw=cleaned,
        duration_seconds=duration,
        sample_rate=sample_rate,
        silent=silent,
        clipped=clipped,
        corrupted=corrupted,
        audio_fingerprint=fingerprint,
        language_category=category,
        normalized=normalized,
        char_count=len(cleaned),
        token_count=toks.token_count,
        ar_only_tokens=toks.ar_only_tokens,
        en_only_tokens=toks.en_only_tokens,
        mixed_tokens=toks.mixed_tokens,
        other_tokens=toks.other_tokens,
        en_meaningful_tokens=toks.en_only_meaningful_tokens,
        code_switched=toks.code_switched,
        unusual_long=is_unusually_long_transcript(cleaned),
        mismatch=mismatch,
        empty_transcript=empty,
    )


def _registry_from_yaml(config_path: Path) -> dict[str, Any]:
    """Registry dataset metadata from ``configs/data_sources.yaml``."""
    registry_path = Path("configs/data_sources.yaml")
    if not registry_path.exists():
        registry_path = config_path.parent / "data_sources.yaml"
    if not registry_path.exists():
        return {"retrieved": None, "datasets": []}
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    return {
        "retrieved": data.get("retrieved"),
        "datasets": data.get("datasets", []),
    }


def run_audit(
    config_path: Path,
    max_examples: int | None = None,
    streaming: bool = False,
    no_audio: bool = False,
) -> dict[str, object]:
    """Run the audit for the dataset named in a MasriCX experiment config."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    repo_id: str = dataset_cfg["id"]
    revision_raw: Any = dataset_cfg.get("revision")
    revision: str = str(revision_raw) if revision_raw else ""
    if revision is None or revision == "TBD":
        raise SystemExit(
            f"dataset.revision in {config_path} is {revision!r}: pin the exact "
            "commit sha (see configs/data_sources.yaml) before auditing"
        )

    if repo_id not in _REGISTRY_ID_BY_DATASET:
        raise SystemExit(
            f"dataset {repo_id!r} is not in the MasriCX registry "
            "(configs/data_sources.yaml); refusing to audit unknown data sources"
        )
    if repo_id not in _PRIMARY_AUDIT_ONLY:
        raise SystemExit(
            f"dataset {repo_id!r} is registered but this Phase 1 audit CLI "
            "audits only the primary code-switch dataset "
            "(Seif-Eldeen-Sameh/asr_codeswitched_dataset); refusing to "
            f"silently load {repo_id!r}"
        )

    loader_fn = load_codeswitch_examples

    aggregator = AuditAggregator(
        audio_available=not no_audio,
        unavailable_note=(
            "audio decoding disabled with --no-audio; audio metrics "
            "(flags, hashes, sampling rates) are unavailable, not zero"
        )
        if no_audio
        else "",
    )

    if max_examples is not None and max_examples == 0:
        examples: list[dict[str, Any]] = []
    else:
        examples = loader_fn(  # type: ignore[assignment]
            revision=revision,
            split="train",
            config=CODESWITCH_DEFAULT_CONFIG,
            streaming=streaming,
            max_examples=max_examples,
            decode_audio=not no_audio,
        )

    for index, ex in enumerate(examples):
        samples = None
        duration: float | None = None
        sample_rate = None
        if not no_audio:
            # Strictly no access to the audio field in no-audio mode.
            audio = ex.get("audio")
            if audio is not None and isinstance(audio, dict):
                samples = audio.get("array")
                sample_rate = audio.get("sampling_rate")
                if samples is not None and hasattr(samples, "__len__"):
                    duration = len(samples) / float(sample_rate or 16000)
        rec = audit_example(
            sample_id=str(ex["sample_id"]),
            index=index,
            transcript_raw=str(ex.get("transcript") or ""),
            duration_seconds=duration,
            sample_rate=sample_rate,
            samples=samples,
        )
        aggregator.add(rec)

    registry = _registry_from_yaml(config_path)
    report: dict[str, object] = {
        "registry": registry,
        "measured": aggregator.summary(),
        "dataset": {"id": repo_id, "revision": revision, "config": CODESWITCH_DEFAULT_CONFIG},
        "options": {
            "max_examples": max_examples,
            "streaming": streaming,
            "no_audio": no_audio,
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(
        config_path=args.config,
        max_examples=args.max_examples,
        streaming=args.streaming,
        no_audio=args.no_audio,
    )
    write_reports(report, args.output_json, args.output_md)
    print(f"audit report written: {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
