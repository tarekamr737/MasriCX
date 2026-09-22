from __future__ import annotations

import json
from pathlib import Path

import pytest

from masricx.data.deduplicate import SplitRecord, deduplicate_examples, transcript_fingerprint
from masricx.data.split import assign_splits, verify_leakage, write_splits


def _example(sample_id: str, index: int, text: str, samples: list[float]) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "index": index,
        "transcript": text,
        "audio": {"array": samples, "sampling_rate": 16000},
    }


def test_transcript_fingerprint_uses_dedupe_normalization() -> None:
    assert transcript_fingerprint("  Hello، عَالَم  ") == transcript_fingerprint("hello عالم")
    assert transcript_fingerprint(" ــ ") is None


def test_dedupe_uses_transcript_and_audio_and_keeps_lowest_index() -> None:
    result = deduplicate_examples(
        [
            _example("later", 2, "Hello، عالم", [0.1, 0.2]),
            _example("earlier", 1, "hello عالم", [0.3, 0.4]),
            _example("audio-a", 3, "نص أول", [0.5, 0.6]),
            _example("audio-b", 4, "different text", [0.5, 0.6]),
        ]
    )
    assert result.input_count == 4
    assert result.removed_count == 2
    assert result.duplicate_groups == 2
    assert result.transcript_collision_groups == 1
    assert result.audio_collision_groups == 1
    assert [record.sample_id for record in result.records] == ["earlier", "audio-a"]


def _records(count: int = 100) -> list[SplitRecord]:
    categories = ("AR_ONLY", "AR_EN_CODE_SWITCHED", "EN_ONLY", "OTHER")
    return [
        SplitRecord(
            sample_id=f"id-{index:03d}",
            index=index,
            transcript_sha256=f"text-{index:03d}",
            audio_sha256=f"audio-{index:03d}",
            language_category=categories[index % len(categories)],
            dedupe_group_sha256=f"group-{index:03d}",
        )
        for index in range(count)
    ]


def test_split_is_deterministic_stratified_and_keeps_linked_pairs_together() -> None:
    records = _records()
    pairs = [("id-000", "id-099")]
    first = assign_splits(records, seed=42, linked_pairs=pairs)
    second = assign_splits(list(reversed(records)), seed=42, linked_pairs=pairs)
    assert {name: [record.sample_id for record in rows] for name, rows in first.items()} == {
        name: [record.sample_id for record in rows] for name, rows in second.items()
    }
    assert sum(len(rows) for rows in first.values()) == 100
    assert all(first[name] for name in ("train", "validation", "test"))
    verify_leakage(first, pairs)


def test_verify_leakage_rejects_exact_hash_crossing() -> None:
    records = _records(2)
    duplicate = SplitRecord(
        sample_id=records[1].sample_id,
        index=records[1].index,
        transcript_sha256=records[0].transcript_sha256,
        audio_sha256=records[1].audio_sha256,
        language_category=records[1].language_category,
        dedupe_group_sha256=records[1].dedupe_group_sha256,
    )
    with pytest.raises(ValueError, match="exact transcript leakage"):
        verify_leakage({"train": [records[0]], "validation": [duplicate], "test": []})


def test_write_splits_contains_no_transcript_or_audio_payload(tmp_path: Path) -> None:
    dedupe = deduplicate_examples([_example("one", 0, "hello عالم", [0.1, 0.2])])
    splits = assign_splits(dedupe.records)
    metadata = write_splits(
        tmp_path,
        splits,
        dedupe,
        dataset_id="unit/data",
        revision="a" * 40,
        seed=42,
        ratios={"train": 0.9, "validation": 0.05, "test": 0.05},
        linked_pairs=[],
    )
    assert metadata["policy"]["speaker_disjoint"] is False
    manifest = "".join((tmp_path / f"{name}.jsonl").read_text() for name in splits)
    assert "hello" not in manifest
    assert "array" not in manifest
    assert json.loads((tmp_path / "metadata.json").read_text())["deduplication"]["input_count"] == 1
