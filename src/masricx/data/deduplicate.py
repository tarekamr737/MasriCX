"""Deterministic exact deduplication for MasriCX split generation.

Rows are connected when either their non-empty normalized transcripts or their
decoded-audio fingerprints match.  One canonical row (lowest dataset index,
then sample ID) is retained from each connected component.  No transcript or
waveform is persisted by this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from masricx.data.audio_metrics import audio_fingerprint
from masricx.data.text import audit_normalize, code_switched_category

__all__ = ["DedupeResult", "SplitRecord", "deduplicate_examples", "transcript_fingerprint"]


def transcript_fingerprint(transcript: str) -> str | None:
    """SHA-256 of dedupe-normalized text; empty text has no dedupe key."""
    normalized = audit_normalize(transcript)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitRecord:
    """Privacy-safe metadata retained for one canonical dataset row."""

    sample_id: str
    index: int
    transcript_sha256: str | None
    audio_sha256: str | None
    language_category: str
    dedupe_group_sha256: str


@dataclass(frozen=True)
class DedupeResult:
    records: tuple[SplitRecord, ...]
    input_count: int
    removed_count: int
    duplicate_groups: int
    transcript_collision_groups: int
    audio_collision_groups: int


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self) -> int:
        value = len(self.parent)
        self.parent.append(value)
        return value

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _audio_key(example: dict[str, Any]) -> str | None:
    audio = example.get("audio")
    if not isinstance(audio, dict):
        return None
    samples = audio.get("array")
    if samples is None:
        return None
    return audio_fingerprint(int(audio.get("sampling_rate") or 16000), samples)


def deduplicate_examples(examples: Iterable[dict[str, Any]]) -> DedupeResult:
    """Retain one canonical row per exact transcript/audio component."""
    raw: list[tuple[str, int, str | None, str | None, str]] = []
    uf = _UnionFind()
    transcript_first: dict[str, int] = {}
    audio_first: dict[str, int] = {}
    transcript_collisions: set[str] = set()
    audio_collisions: set[str] = set()

    for fallback_index, example in enumerate(examples):
        sample_id = str(example["sample_id"])
        index = int(example.get("index", fallback_index))
        transcript = str(example.get("transcript") or "")
        text_hash = transcript_fingerprint(transcript)
        audio_hash = _audio_key(example)
        row = uf.add()
        raw.append((sample_id, index, text_hash, audio_hash, code_switched_category(transcript)))
        if text_hash is not None:
            if text_hash in transcript_first:
                uf.union(row, transcript_first[text_hash])
                transcript_collisions.add(text_hash)
            else:
                transcript_first[text_hash] = row
        if audio_hash is not None:
            if audio_hash in audio_first:
                uf.union(row, audio_first[audio_hash])
                audio_collisions.add(audio_hash)
            else:
                audio_first[audio_hash] = row

    groups: dict[int, list[int]] = {}
    for row in range(len(raw)):
        groups.setdefault(uf.find(row), []).append(row)

    records: list[SplitRecord] = []
    duplicate_groups = 0
    for members in groups.values():
        if len(members) > 1:
            duplicate_groups += 1
        canonical = min(members, key=lambda pos: (raw[pos][1], raw[pos][0]))
        sample_id, index, text_hash, audio_hash, category = raw[canonical]
        member_ids = "\n".join(sorted(raw[pos][0] for pos in members))
        group_hash = hashlib.sha256(member_ids.encode("utf-8")).hexdigest()
        records.append(SplitRecord(sample_id, index, text_hash, audio_hash, category, group_hash))
    records.sort(key=lambda record: (record.index, record.sample_id))
    return DedupeResult(
        records=tuple(records),
        input_count=len(raw),
        removed_count=len(raw) - len(records),
        duplicate_groups=duplicate_groups,
        transcript_collision_groups=len(transcript_collisions),
        audio_collision_groups=len(audio_collisions),
    )
