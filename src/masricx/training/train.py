"""MasriCX LoRA training entry point.

The default dry-run validates all local research inputs without importing the
GPU stack or accessing the network. Actual GPU execution is explicit with
``--execute`` and is intended for the thin Kaggle notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masricx.training.checkpoint import latest_complete_checkpoint, validate_checkpoint
from masricx.training.executor import execute_training
from masricx.training.plan import build_training_plan

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or execute MasriCX LoRA training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".runtime/training"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument(
        "--execute", action="store_true", help="Import GPU stack and start training"
    )
    return parser


def _resolve_resume(output_dir: Path, explicit: Path | None, automatic: bool) -> Path | None:
    resume = explicit or (latest_complete_checkpoint(output_dir) if automatic else None)
    if resume is not None:
        complete, missing = validate_checkpoint(resume)
        if not complete:
            raise SystemExit(f"checkpoint is incomplete: {resume}; missing={','.join(missing)}")
    return resume


def _execute_not_yet_imported(plan: object, resume: Path | None) -> None:
    """Guard until the model/data adapter is invoked in the GPU environment."""
    del plan, resume
    raise SystemExit(
        "GPU execution adapter is not activated in this local environment. "
        "Use the verified Kaggle workflow after orchestrator authorization."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_training_plan(args.config, args.output_dir, args.split_dir)
    resume = _resolve_resume(args.output_dir, args.resume_from, args.auto_resume)
    payload = {**plan.as_dict(), "resume_from": str(resume) if resume else None}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.execute:
        execute_training(plan, resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
