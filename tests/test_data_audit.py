"""Audit engine tests on synthetic records and report rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from masricx.data.aggregators import AuditAggregator, number_bucket
from masricx.data.audit import audit_example, build_parser, run_audit
from masricx.data.report_render import (
    derived_governance,
    derived_interpretation,
    render_markdown,
    write_json,
    write_reports,
)

CODESWITCH_REVISION = "712de01079517771f95bcdecee68ca232b979628"


class TestAuditExample:
    def test_synthetic_record(self) -> None:
        rec = audit_example(
            sample_id="sid",
            index=0,
            transcript_raw="Hello  مرحبا",
            duration_seconds=2.0,
            sample_rate=16000,
            samples=[0.0] * (32000),
        )
        assert rec.sample_id == "sid"
        assert rec.duration_seconds == 2.0
        assert rec.language_category == "AR_EN_CODE_SWITCHED"  # Hello + مرحبا
        assert rec.char_count == len("Hello مرحبا")
        assert not rec.empty_transcript

    def test_no_audio_flags_unavailable_not_zero(self) -> None:
        rec = audit_example("sid", 0, "text", 1.0, 16000, samples=None)
        assert rec.silent is None
        assert rec.clipped is None
        assert rec.corrupted is None
        assert rec.audio_fingerprint is None

    def test_audio_flags_flow_through(self) -> None:
        wave = [0.0] * (2000)
        rec = audit_example("sid", 0, "hello", 0.125, 16000, samples=wave)
        assert rec.silent is True
        assert rec.audio_fingerprint is not None

    def test_mismatch_boundary(self) -> None:
        rec = audit_example("sid", 0, "a", 10.0, 16000)  # 0.1 cps < 0.5
        assert rec.mismatch
        rec_ok = audit_example("sid", 0, "a" * 10, 10.0, 16000)
        assert not rec_ok.mismatch

    def test_unusual_long_boundary(self) -> None:
        rec = audit_example("sid", 0, " ".join(["ab"] * 51), 1.0, 16000)
        assert rec.unusual_long
        rec_ok = audit_example("sid", 0, " ".join(["ab"] * 50), 1.0, 16000)
        assert not rec_ok.unusual_long

    def test_empty_transcript(self) -> None:
        rec = audit_example("sid", 0, "   ", 1.0, 16000)
        assert rec.empty_transcript


class TestAggregator:
    def test_summary_shape_and_no_audio_no_waveform(self) -> None:
        agg = AuditAggregator()
        for i in range(5):
            agg.add(audit_example(f"id{i}", i, f"sample transcript number {i}", 2.0 + i, 16000))
        s = agg.summary()
        assert s["examples"] == 5
        assert s["audio_metrics_available"] is True
        assert s["sample_rates"] == {16000: 5}
        # No waveform-like values (no 'array' keys anywhere in summary).
        assert "array" not in json.dumps(s)

    def test_no_audio_marks_unavailable(self) -> None:
        agg = AuditAggregator(
            audio_available=False, unavailable_note="audio metrics unavailable: disabled"
        )
        agg.add(audit_example("id0", 0, "text", 3.0, 16000, samples=None))
        s = agg.summary()
        assert s["audio_metrics_available"] is False
        assert "unavailable" in str(s["audio_metrics_note"])
        # Explicitly NOT zero: no silent/clipped counts present as 0 that
        # suggest measurement; we emit flags only via examples' fields and
        # keep counters at their default sentinel meaning-not-measured.
        assert s["silent_clips"] == 0  # counter but note declares unavailability

    def test_duplicate_counts(self) -> None:
        agg = AuditAggregator()
        wave = [0.0] * (100)
        for i in range(3):
            rec = audit_example(f"id{i}", i, "same text here", 1.0, 16000, samples=wave)
            agg.add(rec)
        s = agg.summary()
        assert s["transcript_duplicates"]["exact_duplicate_groups"] == 1
        assert s["transcript_duplicates"]["exact_duplicate_extra_instances"] == 2
        assert s["audio_duplicates"]["exact_duplicate_groups"] == 1

    def test_language_category_counts(self) -> None:
        agg = AuditAggregator()
        agg.add(audit_example("a", 0, "مرحبا", 1.0, 16000))
        agg.add(audit_example("b", 1, "hello there", 1.0, 16000))
        agg.add(audit_example("c", 2, "hello مرحبا", 1.0, 16000))
        s = agg.summary()
        assert s["language_categories"] == {
            "AR_ONLY": 1,
            "EN_ONLY": 1,
            "AR_EN_CODE_SWITCHED": 1,
        }


@pytest.fixture()
def audit_config(tmp_path: Path) -> Path:
    cfg = {
        "dataset": {
            "id": "Seif-Eldeen-Sameh/asr_codeswitched_dataset",
            "revision": CODESWITCH_REVISION,
        }
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


class TestRunAudit:
    def test_no_audio_smoke(
        self, audit_config: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Much lighter than loading data: patch the loader to return synthetic rows.
        rows = [
            {
                "sample_id": f"sid{i}",
                "audio": {"array": [0.0] * 100, "sampling_rate": 16000},
                "transcript": f"مرحبا world {i}",
                "index": i,
            }
            for i in range(3)
        ]

        captured_kwargs: dict = {}

        def fake_loader(**kw):
            captured_kwargs.update(kw)
            return iter(rows)

        monkeypatch.setattr(
            "masricx.data.audit.load_codeswitch_examples", fake_loader, raising=False
        )
        report = run_audit(audit_config, no_audio=True)
        measured = report["measured"]
        assert measured["examples"] == 3
        assert measured["audio_metrics_available"] is False
        assert "unavailable" in str(measured["audio_metrics_note"])
        # decode_audio=False must reach the loader in no-audio mode.
        assert captured_kwargs["decode_audio"] is False

    def test_no_audio_never_reads_audio_field(
        self, audit_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows: list[dict] = [
            {
                "sample_id": f"sid{i}",
                "audio": {"array": [9.0] * 50, "sampling_rate": 16000},
                "transcript": f"مرحبا world {i}",
                "index": i,
            }
            for i in range(2)
        ]
        monkeypatch.setattr(
            "masricx.data.audit.load_codeswitch_examples", lambda **kw: iter(rows), raising=False
        )
        report = run_audit(audit_config, no_audio=True)
        measured = report["measured"]
        # Audio-dependent fields are unavailable, not measured as zero-by-decode.
        assert measured["audio_examples_measured"] == 0
        assert "unavailable" in str(measured["audio_metrics_note"])
        # No audio hashes were computed.
        assert measured["audio_duplicates"]["unique_texts_indexed"] == 0

    def test_no_audio_uses_decode_false_in_loader(
        self, audit_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict = {}

        def fake_loader(**kw):
            calls.update(kw)
            return iter([])

        monkeypatch.setattr(
            "masricx.data.audit.load_codeswitch_examples", fake_loader, raising=False
        )
        run_audit(audit_config, no_audio=True)
        assert calls["decode_audio"] is False

    def test_decode_true_when_audio_enabled(
        self, audit_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict = {}

        def fake_loader(**kw):
            calls.update(kw)
            return iter([])

        monkeypatch.setattr(
            "masricx.data.audit.load_codeswitch_examples", fake_loader, raising=False
        )
        run_audit(audit_config, no_audio=False)
        assert calls["decode_audio"] is True

    def test_with_fake_audio(self, audit_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [
            {
                "sample_id": f"sid{i}",
                "audio": {"array": [0.0] * 100, "sampling_rate": 16000},
                "transcript": f"مرحبا world {i}",
                "index": i,
            }
            for i in range(2)
        ]

        monkeypatch.setattr(
            "masricx.data.audit.load_codeswitch_examples", lambda **kw: iter(rows), raising=False
        )
        report = run_audit(audit_config)
        measured = report["measured"]
        assert measured["audio_metrics_available"] is True
        assert measured["sample_rates"] == {16000: 2}
        assert measured["duration"]["total_seconds"] > 0

    def test_primary_only_enforced(self, tmp_path: Path) -> None:
        # EGYSpeak is registered but is NOT the primary dataset: the CLI must
        # fail loudly, never silently load it.
        cfg = {
            "dataset": {
                "id": "MohamedGomaa30/EGYSpeak",
                "revision": "59c40fc6833382a743c180f50addcda5edaa47c3",
            }
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(SystemExit, match="primary"):
            run_audit(p, no_audio=True)

    def test_primary_only_enforced_casablanca(self, tmp_path: Path) -> None:
        cfg = {
            "dataset": {
                "id": "UBC-NLP/Casablanca",
                "revision": "8951b1b88e28c1107142ced57967b8d16350951d",
            }
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(SystemExit, match="primary"):
            run_audit(p, no_audio=True)

    def test_tbd_revision_rejected(self, tmp_path: Path) -> None:
        cfg = {"dataset": {"id": "Seif-Eldeen-Sameh/asr_codeswitched_dataset", "revision": "TBD"}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(SystemExit, match="revision"):
            run_audit(p, no_audio=True)

    def test_unknown_dataset_rejected(self, tmp_path: Path) -> None:
        cfg = {"dataset": {"id": "unknown/repo", "revision": CODESWITCH_REVISION}}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(SystemExit, match="registry"):
            run_audit(p, no_audio=True)


class TestRendering:
    def _report(self) -> dict[str, object]:
        return {
            "registry": {
                "retrieved": "2026-09-20",
                "datasets": [
                    {
                        "id": "unit/testdata",
                        "role": "primary_training",
                        "revision": "abc1234567890",
                        "license": "MIT (caveated)",
                        "row_count": 3,
                        "pseudo_labelled": False,
                        "release_blocker": "source review",
                    }
                ],
            },
            "measured": {
                "examples": 3,
                "audio_metrics_available": True,
                "audio_metrics_note": "",
                "audio_examples_measured": 3,
                "duration": {
                    "total_seconds": 7.5,
                    "min": 2.0,
                    "p25": 2.5,
                    "median": 2.5,
                    "p75": 2.5,
                    "p95": 2.5,
                    "max": 3.0,
                },
                "sample_rates": {"16000": 3},
                "empty_transcripts": 0,
                "corrupted_audio": 0,
                "silent_clips": 1,
                "clipped_audio": 0,
                "mismatch_outliers": 1,
                "unusually_long_transcripts": 0,
                "language_categories": {"AR_EN_CODE_SWITCHED": 3},
                "arabic_letter_ratio": 0.5,
                "latin_letter_ratio": 0.5,
                "number_frequency": {"integer_digits_1": 2},
                "number_buckets_doc": "structural buckets only",
                "symbol_frequency": {",": 2},
                "transcript_duplicates": {
                    "method": "per-bigram bucketing",
                    "exact_duplicate_groups": 1,
                    "exact_duplicate_extra_instances": 1,
                    "near_duplicate_pair_count": 1,
                    "near_duplicate_pairs_reported": [
                        {"id_a": "sid1", "id_b": "sid2", "similarity": 0.95}
                    ],
                },
                "audio_duplicates": {
                    "exact_duplicate_groups": 1,
                    "exact_duplicate_extra_instances": 1,
                },
            },
        }

    def test_markdown_contains_hand_worked_values(self) -> None:
        md = render_markdown(self._report())
        assert "## Measured audit results" in md
        assert "Card-declared/verified metadata" in md
        assert "2026-09-20" in md
        assert "7.5" in md
        assert "| examples | 3 |" in md
        assert "silent clips | 1" in md
        assert "exact duplicate transcript groups | 1" in md
        # No PII: transcript contents must not appear.
        assert "مرحبا" not in md  # hand-worked report has no transcript text

    def test_markdown_not_run(self) -> None:
        md = render_markdown({"registry": {"retrieved": "2026-09-20"}, "measured": {}})
        assert "NOT RUN" in md

    def test_no_audio_note_in_markdown(self) -> None:
        report = self._report()
        measured = report["measured"]
        assert isinstance(measured, dict)
        measured["audio_metrics_available"] = False
        measured["audio_metrics_note"] = "audio decoding skipped (--no-audio)"
        md = render_markdown(report)
        assert "Audio metrics unavailable" in md
        assert "UNAVAILABLE, NOT ZERO".lower() in md.lower()

    def test_json_write_roundtrip(self, tmp_path: Path) -> None:
        report = self._report()
        p = tmp_path / "out.json"
        write_json(report, p)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded == report

    def test_write_reports_paths(self, tmp_path: Path) -> None:
        report = self._report()
        jp = tmp_path / "artifacts" / "data_audit.json"
        mp = tmp_path / "reports" / "DATA_AUDIT.md"
        write_reports(report, jp, mp)
        assert jp.exists()
        assert mp.exists()
        assert "Measured audit results" in mp.read_text(encoding="utf-8")


class TestCLI:
    def test_parser_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/codeswitch.yaml",
                "--max-examples",
                "5",
                "--streaming",
                "--no-audio",
                "--output-json",
                "a.json",
                "--output-md",
                "b.md",
            ]
        )
        assert args.max_examples == 5
        assert args.streaming
        assert args.no_audio
        assert args.output_json == Path("a.json")
        assert args.output_md == Path("b.md")

    def test_defaults(self) -> None:
        args = build_parser().parse_args(["--config", "x.yaml"])
        assert args.max_examples is None
        assert not args.streaming

        assert not args.no_audio


class TestPrivacySafeNumberBuckets:
    def test_number_bucket_assignment(self) -> None:
        assert number_bucket("7") == "integer_digits_1"
        assert number_bucket("42") == "integer_digits_2_3"
        assert number_bucket("300") == "integer_digits_2_3"
        assert number_bucket("0100112345") == "integer_digits_4_plus"
        assert number_bucket("3.5") == "decimal"
        assert number_bucket("1,500") == "decimal"

    def test_phone_like_number_absent_from_serialized_summary(self) -> None:
        agg = AuditAggregator()
        agg.add(audit_example("a", 0, "call me on 0100112345 now", 2.0, 16000))
        agg.add(audit_example("b", 1, "total 1,250.50 pounds", 2.0, 16000))
        s = agg.summary()
        serialized = json.dumps(s, ensure_ascii=False)
        assert "0100112345" not in serialized
        assert "1,250.50" not in serialized
        assert "1250" not in serialized
        # 1,250.50 -> tones are matched per regex tokens; both are PII-safe buckets.
        assert s["number_frequency"] == {
            "decimal": 1,
            "integer_digits_2_3": 1,
            "integer_digits_4_plus": 1,
        }

    def test_raw_numbers_never_in_markdown(self, tmp_path: Path) -> None:
        from masricx.data.report_render import render_markdown

        agg = AuditAggregator()
        agg.add(audit_example("a", 0, "phone 0100112345", 2.0, 16000))
        md = render_markdown({"measured": agg.summary()})
        assert "0100112345" not in md
        assert "integer_digits_4_plus" in md


class TestDerivedDuration:
    def test_duration_derived_from_samples_when_missing(self) -> None:
        samples = [0.0] * 64000  # 4.0 seconds at 16 kHz
        rec = audit_example("sid", 0, "hello", None, 16000, samples=samples)
        assert rec.duration_seconds == pytest.approx(4.0)
        assert rec.sample_rate == 16000

    def test_derived_duration_used_for_mismatch(self) -> None:
        # 1 char over 8 seconds -> 0.125 cps < 0.5 => mismatch, using the
        # derived duration (explicit duration_seconds=None).
        samples = [0.1] * 128000
        rec = audit_example("sid", 0, "a", None, 16000, samples=samples)
        assert rec.duration_seconds == pytest.approx(8.0)
        assert rec.mismatch is True

    def test_explicit_duration_wins(self) -> None:
        samples = [0.1] * 16000
        rec = audit_example("sid", 0, "a" * 100, 10.0, 16000, samples=samples)
        assert rec.duration_seconds == 10.0
        assert rec.mismatch is False  # 10 cps within bounds


class TestMetadataRendering:
    def _registry_report(self) -> dict[str, object]:
        return {
            "registry": {
                "retrieved": "2026-09-20",
                "datasets": [
                    {
                        "id": "unit/withrows",
                        "role": "primary_training",
                        "revision": "a" * 40,
                        "license": {"declared": "MIT (aggregate card metadata)", "caveat": "mixed"},
                        "row_count": 45189,
                        "pseudo_labelled": False,
                        "release_blocker": {
                            "blocked": True,
                            "reason": "final model publication blocked pending source review",
                        },
                    },
                    {
                        "id": "unit/splits",
                        "role": "external_evaluation_only",
                        "revision": "b" * 40,
                        "license": {"declared": "CC-BY-NC-ND-4.0"},
                        "splits": {"validation": 846, "test": 846},
                        "pseudo_labelled": False,
                        "release_blocker": {
                            "blocked": True,
                            "reason": "redistribution blocked by NC-ND",
                        },
                    },
                ],
            },
            "measured": {},
        }

    def test_license_rendered_as_plain_text_not_dict(self) -> None:
        md = render_markdown(self._registry_report())
        assert "MIT (aggregate card metadata)" in md
        assert "{" not in md.splitlines()[7]
        assert "'declared'" not in md

    def test_release_blocker_readable_yes_reason(self) -> None:
        md = render_markdown(self._registry_report())
        assert "YES: final model publication blocked pending source review" in md
        assert "{'blocked'" not in md

    def test_split_counts_fallback_when_no_row_count(self) -> None:
        md = render_markdown(self._registry_report())
        assert "validation: 846; test: 846" in md
        assert "45189" in md

    def test_rows_have_no_stray_or_trailing_spaces(self) -> None:
        md = render_markdown(self._registry_report())
        for line in md.split("\n"):
            assert line == line.rstrip(), f"trailing whitespace: {line!r}"
            if line.startswith("|"):
                assert not line.endswith(" | ") or line.endswith("|")


class TestInterpretation:
    def _measured(self) -> dict[str, object]:
        return {
            "examples": 45189,
            "audio_metrics_available": True,
            "audio_examples_measured": 45189,
            "duration": {
                "total_seconds": 122806.01,
                "min": 0.021,
                "p25": 1.56,
                "median": 2.2,
                "p75": 2.8,
                "p95": 7.23,
                "max": 24.94,
            },
            "sample_rates": {"16000": 45189},
            "empty_transcripts": 1,
            "silent_clips": 1,
            "corrupted_audio": 0,
            "clipped_audio": 66,
            "mismatch_outliers": 53,
            "unusually_long_transcripts": 168,
            "language_categories": {
                "AR_EN_CODE_SWITCHED": 10451,
                "AR_ONLY": 34550,
                "EN_ONLY": 170,
                "OTHER": 18,
            },
            "transcript_duplicates": {
                "exact_duplicate_groups": 684,
                "exact_duplicate_extra_instances": 2022,
                "near_duplicate_pair_count": 20,
                "pair_comparisons": 191239,
                "comparison_budget_exhausted": False,
                "candidate_bound": {
                    "buckets_skipped_over_cap": 1017,
                    "pair_cap_reported": 20,
                },
            },
            "audio_duplicates": {"exact_duplicate_groups": 0},
            "number_frequency": {"integer_digits_1": 283},
        }

    def test_percentages_computed_not_hardcoded(self) -> None:
        s = derived_interpretation(self._measured())
        text = "\n".join(s)
        assert "23.13%" in text  # 10451/45189
        assert "76.46%" in text  # 34550/45189
        assert "0.38%" in text  # 170/45189
        assert "0.04%" in text  # 18/45189
        assert "4.47%" in text  # 2022/45189

    def test_hours_derived_from_seconds(self) -> None:
        s = derived_interpretation(self._measured())
        text = "\n".join(s)
        assert "34.1128 h" in text  # 122806.01/3600
        assert "122806.01 s" in text

    def test_decision_wording_present(self) -> None:
        text = "\n".join(derived_interpretation(self._measured()))
        assert "preserve stratified code-switch metadata" in text
        assert "review/filter candidates, not automatic deletions" in text
        assert "require deduplication before splitting" in text
        assert "diagnostic only" in text
        assert "not total prevalence" in text
        assert "1017 oversized buckets were skipped" in text
        assert "budget was not exhausted" in text
        assert "privacy-safe structural buckets" in text
        assert "raw numeric tokens were not serialized" in text

    def test_capped_near_duplicate_caveat(self) -> None:
        text = "\n".join(derived_interpretation(self._measured()))
        assert "capped at 20" in text
        assert "Do not infer a dataset-wide near-duplicate rate" in text

    def test_missing_keys_omit_statements(self) -> None:
        m = self._measured()
        del m["language_categories"]
        del m["duration"]
        del m["transcript_duplicates"]
        del m["number_frequency"]
        s = derived_interpretation(m)
        text = "\n".join(s)
        assert "Language categories" not in text
        assert "Audio coverage" not in text
        assert "Near-duplicate" not in text
        assert "Number frequencies" not in text
        assert len(s) >= 1  # quality triage still derived from counts

    def test_min_duration_suspicion_reported(self) -> None:
        text = "\n".join(derived_interpretation(self._measured()))
        assert "0.021 s minimum duration is suspicious" in text


class TestRenderingInvariants:
    def test_exactly_one_trailing_newline(self) -> None:
        md = render_markdown(TestRendering()._report())
        assert md.endswith("\n")
        assert not md.endswith("\n\n")

    def test_no_trailing_whitespace_anywhere(self) -> None:
        for report in (TestRendering()._report(), TestMetadataRendering()._registry_report()):
            md = render_markdown(report)
            for line in md.split("\n"):
                assert line == line.rstrip(), f"trailing whitespace: {line!r}"

    def test_full_report_has_interpretation_section(self) -> None:
        report = TestRendering()._report()
        md = render_markdown(report)
        assert "## Interpretation and decisions" in md


class TestFinalInterpretationReview:
    def test_audio_coverage_uses_measured_and_total(self) -> None:
        measured = TestInterpretation()._measured()
        measured["audio_examples_measured"] = 45189
        text = "\n".join(derived_interpretation(measured))
        assert "45,189/45,189 audio rows decoded" in text

    def test_audio_coverage_not_inferred_without_measured_count(self) -> None:
        measured = TestInterpretation()._measured()
        del measured["audio_examples_measured"]
        text = "\n".join(derived_interpretation(measured))
        assert "audio rows decoded" not in text

    def test_singular_and_plural_triage_wording(self) -> None:
        measured = TestInterpretation()._measured()
        text = "\n".join(derived_interpretation(measured))
        assert "1 empty transcript (0.00%)" in text
        assert "1 silent clip (0.00%)" in text
        assert "66 clipped clips (0.15%)" in text

    def test_governance_conclusion_from_registry(self) -> None:
        registry = {
            "datasets": [
                {"id": "Seif-Eldeen-Sameh/asr_codeswitched_dataset"},
                {"id": "MohamedGomaa30/EGYSpeak"},
                {"id": "UBC-NLP/Casablanca"},
            ]
        }
        text = "\n".join(derived_governance(registry))
        assert "not redistributed" in text
        assert "model-weight licensing remains blocked" in text
        assert "full-data-versus-filtered-data decision" in text
        assert "EGYSpeak E3 remains disabled" in text
        assert "Casablanca remains evaluation-only" in text

    def test_governance_omitted_for_incomplete_registry(self) -> None:
        assert derived_governance({"datasets": [{"id": "UBC-NLP/Casablanca"}]}) == []
