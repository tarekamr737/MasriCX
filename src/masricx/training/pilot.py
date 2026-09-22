"""Run one orchestrator-defined P1/P2/P3 pilot candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from masricx.training.executor import execute_training
from masricx.training.plan import TrainingPlan, build_training_plan
from masricx.training.train import _resolve_resume


def select_candidate(plan: TrainingPlan, config_path: Path, run: str) -> TrainingPlan:
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate = next(
        (item for item in raw.get("pilot_candidates", []) if item.get("run") == run), None
    )
    if candidate is None:
        raise ValueError(f"pilot candidate {run} is not defined")
    lora = dict(plan.lora)
    training = dict(plan.training)
    lora["r"] = int(candidate["lora_rank"])
    training["learning_rate"] = float(candidate["learning_rate"])
    return replace(
        plan,
        experiment=f"{plan.experiment}-{run.lower()}",
        lora=lora,
        training=training,
        output_dir=str(Path(plan.output_dir) / run.lower()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or execute a P1/P2/P3 LoRA pilot")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--run", choices=("P1", "P2", "P3"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/training/pilot"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = select_candidate(
        build_training_plan(args.config, args.output_dir, args.split_dir), args.config, args.run
    )
    resume = _resolve_resume(Path(plan.output_dir), args.resume_from, args.auto_resume)
    print(
        json.dumps(
            {**plan.as_dict(), "resume_from": str(resume) if resume else None},
            indent=2,
            sort_keys=True,
        )
    )
    if args.execute:
        execute_training(plan, resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
