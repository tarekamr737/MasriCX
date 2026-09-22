"""Deterministic, leakage-aware MasriCX split generation and persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from masricx.data.deduplicate import DedupeResult, SplitRecord, deduplicate_examples
from masricx.data.load_codeswitch import CODESWITCH_DEFAULT_CONFIG, load_codeswitch_examples

__all__ = ["assign_splits", "build_parser", "main", "verify_leakage", "write_splits"]

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}


class _IdUnion:
    def __init__(self, ids: Iterable[str]) -> None:
        self.parent = {value: value for value in ids}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        if left not in self.parent or right not in self.parent:
            return
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _apportion(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}
    result = {name: int(raw[name]) for name in SPLIT_NAMES}
    remaining = total - sum(result.values())
    order = sorted(
        SPLIT_NAMES, key=lambda name: (-(raw[name] - result[name]), SPLIT_NAMES.index(name))
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must have exactly {SPLIT_NAMES}")
    if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1")


def assign_splits(
    records: Sequence[SplitRecord],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    linked_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, tuple[SplitRecord, ...]]:
    """Assign canonical rows deterministically, stratified by language class.

    ``linked_pairs`` are audit-reported near-duplicate constraints. They are
    co-located but never removed and are explicitly not treated as exhaustive.
    """
    _validate_ratios(ratios)
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("sample IDs must be unique")
    union = _IdUnion(by_id)
    for left, right in linked_pairs:
        union.union(left, right)
    components: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        components[union.find(record.sample_id)].append(record)

    total_targets = _apportion(len(records), ratios)
    categories = sorted({record.language_category for record in records})
    category_totals = Counter(record.language_category for record in records)
    category_targets = {
        category: _apportion(category_totals[category], ratios) for category in categories
    }
    assigned_total = Counter[str]()
    assigned_category: dict[str, Counter[str]] = defaultdict(Counter)
    output: dict[str, list[SplitRecord]] = {name: [] for name in SPLIT_NAMES}

    units = list(components.values())
    units.sort(
        key=lambda unit: hashlib.sha256(
            f"{seed}:".encode() + "\n".join(sorted(row.sample_id for row in unit)).encode()
        ).hexdigest()
    )
    for unit in units:
        unit_counts = Counter(record.language_category for record in unit)

        scores: dict[str, tuple[float, float, int]] = {}
        for name in SPLIT_NAMES:
            category_deficit = sum(
                category_targets[category][name] - assigned_category[category][name]
                for category in unit_counts
            )
            total_deficit = total_targets[name] - assigned_total[name]
            scores[name] = (category_deficit, total_deficit, -SPLIT_NAMES.index(name))
        destination = max(SPLIT_NAMES, key=scores.__getitem__)
        output[destination].extend(unit)
        assigned_total[destination] += len(unit)
        for category, count in unit_counts.items():
            assigned_category[category][destination] += count

    return {
        name: tuple(sorted(output[name], key=lambda row: (row.index, row.sample_id)))
        for name in SPLIT_NAMES
    }


def verify_leakage(
    splits: Mapping[str, Sequence[SplitRecord]],
    linked_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, int]:
    """Raise on ID, exact-text, exact-audio, or known-pair cross-split leakage."""
    ownership: dict[str, str] = {}
    keyed: dict[tuple[str, str], str] = {}
    for split_name, records in splits.items():
        for record in records:
            if record.sample_id in ownership:
                raise ValueError(f"sample ID repeated across splits: {record.sample_id}")
            ownership[record.sample_id] = split_name
            for kind, value in (
                ("transcript", record.transcript_sha256),
                ("audio", record.audio_sha256),
            ):
                if value is None:
                    continue
                key = (kind, value)
                previous = keyed.get(key)
                if previous is not None and previous != split_name:
                    raise ValueError(
                        f"exact {kind} leakage across {previous}/{split_name}: {value}"
                    )
                keyed[key] = split_name
    checked_pairs = 0
    for left, right in linked_pairs:
        if left in ownership and right in ownership:
            checked_pairs += 1
            if ownership[left] != ownership[right]:
                raise ValueError(f"reported near-duplicate pair crosses splits: {left}, {right}")
    return {"sample_ids": len(ownership), "exact_keys": len(keyed), "linked_pairs": checked_pairs}


def _write_jsonl(path: Path, records: Sequence[SplitRecord]) -> None:
    content = "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_splits(
    output_dir: Path,
    splits: Mapping[str, Sequence[SplitRecord]],
    dedupe: DedupeResult,
    *,
    dataset_id: str,
    revision: str,
    seed: int,
    ratios: Mapping[str, float],
    linked_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Persist privacy-safe JSONL manifests and deterministic metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage = verify_leakage(splits, linked_pairs)
    for name in SPLIT_NAMES:
        _write_jsonl(output_dir / f"{name}.jsonl", splits[name])
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {"id": dataset_id, "revision": revision},
        "policy": {
            "seed": seed,
            "ratios": dict(ratios),
            "normalization": "masricx.data.text.audit_normalize",
            "canonical_rule": "lowest dataset index, then sample_id",
            "speaker_disjoint": False,
            "speaker_disjoint_reason": "no reliable speaker/session/source ID is available",
            "near_duplicate_scope": "audit-reported diagnostic pairs only; not exhaustive",
        },
        "deduplication": {
            "input_count": dedupe.input_count,
            "retained_count": len(dedupe.records),
            "removed_count": dedupe.removed_count,
            "duplicate_groups": dedupe.duplicate_groups,
            "transcript_collision_groups": dedupe.transcript_collision_groups,
            "audio_collision_groups": dedupe.audio_collision_groups,
        },
        "splits": {
            name: {
                "count": len(splits[name]),
                "language_categories": dict(
                    sorted(Counter(r.language_category for r in splits[name]).items())
                ),
            }
            for name in SPLIT_NAMES
        },
        "leakage_check": leakage,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return metadata


def _reported_pairs(audit_path: Path) -> list[tuple[str, str]]:
    if not audit_path.exists():
        return []
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    pairs = (
        payload.get("measured", {})
        .get("transcript_duplicates", {})
        .get("near_duplicate_pairs_reported", [])
    )
    return [(str(pair["id_a"]), str(pair["id_b"])) for pair in pairs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic MasriCX splits")
    parser.add_argument("--config", type=Path, default=Path("configs/codeswitch.yaml"))
    parser.add_argument("--audit-json", type=Path, default=Path("artifacts/data_audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--max-examples", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    split_cfg = dataset["splits"]
    ratios = {name: float(split_cfg[name]) / 100.0 for name in SPLIT_NAMES}
    seed = int(split_cfg["seed"])
    examples = load_codeswitch_examples(
        revision=str(dataset["revision"]),
        config=CODESWITCH_DEFAULT_CONFIG,
        split="train",
        max_examples=args.max_examples,
        decode_audio=True,
    )
    dedupe = deduplicate_examples(examples)
    pairs = _reported_pairs(args.audit_json)
    splits = assign_splits(dedupe.records, seed=seed, ratios=ratios, linked_pairs=pairs)
    metadata = write_splits(
        args.output_dir,
        splits,
        dedupe,
        dataset_id=str(dataset["id"]),
        revision=str(dataset["revision"]),
        seed=seed,
        ratios=ratios,
        linked_pairs=pairs,
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
