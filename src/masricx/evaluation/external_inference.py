"""Casablanca Egypt evaluation-only inference runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from masricx.data.audio_metrics import audio_flags
from masricx.data.load_casablanca import CASABLANCA_REVISION, load_casablanca_examples
from masricx.evaluation.baseline_runtime import load_runtime
from masricx.evaluation.prediction_artifact import ensure_prediction_provenance


def run_external(
    *,
    model_id: str,
    model_revision: str,
    output_path: Path,
    adapter: str | None = None,
    split: str = "test",
    max_examples: int | None = None,
    allow_cpu: bool = False,
) -> int:  # pragma: no cover - external model integration
    ensure_prediction_provenance(
        output_path,
        {
            "adapter": adapter,
            "config": "Egypt",
            "dataset": "UBC-NLP/Casablanca",
            "dataset_revision": CASABLANCA_REVISION,
            "max_examples": max_examples,
            "model": model_id,
            "model_revision": model_revision,
            "schema_version": 1,
            "split": split,
        },
    )
    runtime = load_runtime(model_id, model_revision, adapter=adapter, allow_cpu=allow_cpu)
    completed = set()
    if output_path.exists():
        completed = {
            str(json.loads(line)["sample_id"])
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for example in load_casablanca_examples(split=split, max_examples=max_examples):
            sample_id = str(example["sample_id"])
            if sample_id in completed:
                continue
            audio = example["audio"]
            samples = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = int(audio["sampling_rate"])
            hypothesis, elapsed = runtime.transcribe(
                samples, sample_rate, output_path.parent / "tmp"
            )
            record = {
                "sample_id": sample_id,
                "reference": str(example.get("transcript") or ""),
                "hypothesis": hypothesis,
                "audio_seconds": len(samples) / sample_rate,
                "processing_seconds": elapsed,
                "silent": audio_flags(samples).silent,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run evaluation-only Casablanca Egypt inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "dataset": "UBC-NLP/Casablanca",
                "config": "Egypt",
                "split": args.split,
                "model": args.model,
                "revision": args.revision,
                "output": str(args.output),
                "evaluation_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.execute:
        written = run_external(
            model_id=args.model,
            model_revision=args.revision,
            adapter=args.adapter,
            split=args.split,
            output_path=args.output,
            max_examples=args.max_examples,
            allow_cpu=args.allow_cpu,
        )
        print(f"wrote {written} external predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
