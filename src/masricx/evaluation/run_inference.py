"""Resumable batch inference for E0/E1/E2 benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from masricx.audio.telephony import telephone_test_degrade
from masricx.data.audio_metrics import audio_flags
from masricx.data.load_codeswitch import CODESWITCH_REVISION, load_codeswitch_examples
from masricx.evaluation.baseline_runtime import load_runtime
from masricx.evaluation.prediction_artifact import (
    ensure_prediction_provenance,
    file_sha256,
)

BASELINE_MODELS = (
    "openai/whisper-large-v3-turbo",
    "Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched",
    "NAMAA-Space/EgypTalk-ASR-v2",
)


def _manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return records if limit is None else records[:limit]


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["sample_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def run_inference(
    *,
    model_id: str,
    model_revision: str,
    manifest_path: Path,
    output_path: Path,
    adapter: str | None = None,
    telephone: bool = False,
    max_examples: int | None = None,
    allow_cpu: bool = False,
) -> int:  # pragma: no cover - model integration
    wanted = _manifest(manifest_path, max_examples)
    wanted_by_index = {int(row["index"]): str(row["sample_id"]) for row in wanted}
    ensure_prediction_provenance(
        output_path,
        {
            "adapter": adapter,
            "dataset": "Seif-Eldeen-Sameh/asr_codeswitched_dataset",
            "dataset_revision": CODESWITCH_REVISION,
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "max_examples": max_examples,
            "model": model_id,
            "model_revision": model_revision,
            "schema_version": 1,
            "telephone": telephone,
        },
    )
    complete = _completed(output_path)
    runtime = load_runtime(model_id, model_revision, adapter=adapter, allow_cpu=allow_cpu)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for example in load_codeswitch_examples(revision=CODESWITCH_REVISION, decode_audio=True):
            index = int(example["index"])
            if index not in wanted_by_index:
                continue
            sample_id = wanted_by_index[index]
            if sample_id in complete:
                continue
            audio = example["audio"]
            samples = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            if telephone:
                samples = telephone_test_degrade(samples, sample_rate)
            hypothesis, elapsed = runtime.transcribe(
                samples, sample_rate, output_path.parent / "tmp"
            )
            duration = len(samples) / sample_rate
            record = {
                "sample_id": sample_id,
                "reference": str(example.get("transcript") or ""),
                "hypothesis": hypothesis,
                "audio_seconds": duration,
                "processing_seconds": elapsed,
                "silent": audio_flags(samples).silent,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable MasriCX benchmark inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telephone", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = {
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "manifest": str(args.manifest),
        "output": str(args.output),
        "telephone": args.telephone,
        "max_examples": args.max_examples,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.execute:
        written = run_inference(
            model_id=args.model,
            model_revision=args.revision,
            adapter=args.adapter,
            manifest_path=args.manifest,
            output_path=args.output,
            telephone=args.telephone,
            max_examples=args.max_examples,
            allow_cpu=args.allow_cpu,
        )
        print(f"wrote {written} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
