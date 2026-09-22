"""Training metrics, research stop conditions, and checkpoint provenance callback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from masricx.evaluation.metrics import word_error_rate
from masricx.training.plan import TrainingPlan
from masricx.training.provenance import write_provenance


def compute_wer(processor: Any, prediction: Any) -> dict[str, float]:
    predictions = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    labels = prediction.label_ids.copy()
    labels[labels == -100] = processor.tokenizer.pad_token_id
    hypotheses = processor.batch_decode(predictions, skip_special_tokens=True)
    references = processor.batch_decode(labels, skip_special_tokens=True)
    counts = [
        word_error_rate(reference, hypothesis, mode="normalized")
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    edits = sum(item.substitutions + item.deletions + item.insertions for item in counts)
    reference_units = sum(item.reference_units for item in counts)
    return {"wer": edits / reference_units if reference_units else float("nan")}


def build_integrity_callback(
    transformers: Any,
    plan: TrainingPlan,
    metadata: dict[str, Any],
) -> Any:
    class IntegrityCallback(transformers.TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.eval_wer: list[float] = []

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del args, state, kwargs
            for key, value in (logs or {}).items():
                if (
                    key in {"loss", "eval_loss", "eval_wer"}
                    and isinstance(value, (int, float))
                    and not math.isfinite(value)
                ):
                    raise RuntimeError(f"stop condition: non-finite {key}={value}")
            wer = (logs or {}).get("eval_wer")
            if isinstance(wer, (int, float)):
                self.eval_wer.append(float(wer))
                if (
                    len(self.eval_wer) >= 3
                    and self.eval_wer[-3] < self.eval_wer[-2] < self.eval_wer[-1]
                ):
                    raise RuntimeError("stop condition: WER degraded across three evaluations")
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del kwargs
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_provenance(checkpoint, plan, metadata, state.log_history)
            return control

    return IntegrityCallback()
