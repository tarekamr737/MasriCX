"""GPU-only Whisper LoRA executor, imported only after explicit authorization."""

from __future__ import annotations

import importlib
import json
import random
import warnings
from pathlib import Path
from typing import Any, cast

from masricx.audio.augmentation import augment_training_audio
from masricx.training.callbacks import build_integrity_callback, compute_wer
from masricx.training.hub import HubCheckpointStore
from masricx.training.lora import prepare_lora_model
from masricx.training.plan import TrainingPlan
from masricx.training.provenance import experiment_metadata, git_state, write_provenance

__all__ = ["configure_generation", "execute_training"]


def configure_generation(model: Any) -> None:
    """Apply Whisper decoding controls through the supported generation config."""
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []


def _number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"training.{key} must be numeric")
    return float(value)


def _integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"training.{key} must be an integer")
    return value


def _manifest(split_dir: str, name: str) -> list[dict[str, Any]]:
    path = Path(split_dir) / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_manifest(source: Any, records: list[dict[str, Any]]) -> Any:
    selected = source.select([int(record["index"]) for record in records])
    return selected.add_column(
        "_masricx_sample_id", [str(record["sample_id"]) for record in records]
    )


def _pilot_subset(dataset: Any, seed: int, target_hours: float = 5.0) -> Any:
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    selected: list[int] = []
    seconds = 0.0
    for position in order:
        audio = dataset[position]["audio"]
        selected.append(position)
        seconds += len(audio["array"]) / float(audio["sampling_rate"])
        if seconds >= target_hours * 3600:
            break
    return dataset.select(selected)


def _seeded_subset(dataset: Any, seed: int, max_examples: int) -> Any:
    if len(dataset) <= max_examples:
        return dataset
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    return dataset.select(order[:max_examples])


def execute_training(
    plan: TrainingPlan,
    resume: Path | None,
    checkpoint_repo: str | None = None,
    restore_remote: bool = False,
    max_steps: int | None = None,
) -> None:  # pragma: no cover
    """Execute on CUDA; refuses accidental local CPU training."""
    try:
        datasets = importlib.import_module("datasets")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise SystemExit("GPU dependencies are missing; install requirements.txt") from exc
    if not torch.cuda.is_available():
        raise SystemExit("training requires CUDA; refusing accidental CPU execution")
    checkpoint_store = (
        HubCheckpointStore(checkpoint_repo, plan.experiment, git_state()[0])
        if checkpoint_repo
        else None
    )
    if resume is None and restore_remote and checkpoint_store is not None:
        resume = checkpoint_store.restore_latest(Path(plan.output_dir))
    processor = transformers.WhisperProcessor.from_pretrained(
        plan.model_id, revision=plan.model_revision
    )
    quantization = transformers.BitsAndBytesConfig(load_in_8bit=plan.load_in_8bit)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        plan.model_id,
        revision=plan.model_revision,
        quantization_config=quantization,
        device_map="auto",
    )
    placements = set(getattr(model, "hf_device_map", {}).values())
    if placements.intersection({"cpu", "disk"}):
        raise RuntimeError("training refuses a model offloaded to CPU or disk")
    configure_generation(model)
    model = prepare_lora_model(model, plan.lora)
    source = datasets.load_dataset(
        plan.dataset_id,
        "default",
        split="train",
        revision=plan.dataset_revision,
    )
    train_set = _select_manifest(source, _manifest(plan.split_dir, "train"))
    validation_set = _select_manifest(source, _manifest(plan.split_dir, "validation"))
    if "pilot" in plan.experiment:
        train_set = _pilot_subset(train_set, plan.seed, plan.pilot_target_hours or 5.0)
        validation_set = _seeded_subset(
            validation_set, plan.seed, plan.pilot_validation_examples or 128
        )
    metadata = experiment_metadata(plan, len(train_set))
    write_provenance(Path(plan.output_dir), plan, metadata, [])

    def preprocess(example: dict[str, Any], augment: bool) -> dict[str, Any]:
        audio = example["audio"]
        samples = audio["array"]
        if augment and plan.augmentation.get("enabled") is True:
            samples, _ = augment_training_audio(
                samples,
                int(audio["sampling_rate"]),
                str(example["_masricx_sample_id"]),
                plan.augmentation,
            )
        text = next(
            str(example[key])
            for key in ("transcript", "transcription", "text", "sentence")
            if key in example
        )
        features = processor.feature_extractor(
            samples, sampling_rate=int(audio["sampling_rate"])
        ).input_features[0]
        return {"input_features": features, "labels": processor.tokenizer(text).input_ids}

    train_set = train_set.map(
        lambda row: preprocess(row, True), remove_columns=train_set.column_names
    )
    validation_set = validation_set.map(
        lambda row: preprocess(row, False), remove_columns=validation_set.column_names
    )

    class Collator:
        def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
            batch = processor.feature_extractor.pad(
                [{"input_features": row["input_features"]} for row in rows],
                return_tensors="pt",
            )
            labels = processor.tokenizer.pad(
                [{"input_ids": row["labels"]} for row in rows], return_tensors="pt"
            )
            label_ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
            if (label_ids[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
                label_ids = label_ids[:, 1:]
            batch["labels"] = label_ids
            return cast(dict[str, Any], batch)

    values = plan.training
    arguments = transformers.Seq2SeqTrainingArguments(
        output_dir=plan.output_dir,
        learning_rate=_number(values["learning_rate"], "learning_rate"),
        num_train_epochs=_number(values["epochs"], "epochs"),
        warmup_ratio=_number(values["warmup_ratio"], "warmup_ratio"),
        weight_decay=_number(values["weight_decay"], "weight_decay"),
        per_device_train_batch_size=_integer(
            values["per_device_train_batch_size"], "per_device_train_batch_size"
        ),
        per_device_eval_batch_size=_integer(
            values["per_device_train_batch_size"], "per_device_train_batch_size"
        ),
        gradient_accumulation_steps=_integer(
            values["gradient_accumulation_steps"], "gradient_accumulation_steps"
        ),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        fp16=plan.fp16,
        eval_strategy=str(values["eval_strategy"]),
        eval_steps=_integer(values["eval_steps"], "eval_steps"),
        save_strategy=str(values["save_strategy"]),
        save_steps=_integer(values["save_steps"], "save_steps"),
        save_total_limit=_integer(values["save_total_limit"], "save_total_limit"),
        logging_strategy="steps",
        logging_steps=(
            1
            if max_steps is not None
            else _integer(values.get("logging_steps", 25), "logging_steps")
        ),
        seed=plan.seed,
        predict_with_generate=True,
        remove_unused_columns=False,
        report_to=[],
        label_names=["labels"],
        max_steps=max_steps if max_steps is not None else -1,
    )
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=validation_set,
        data_collator=Collator(),
        processing_class=processor.feature_extractor,
        compute_metrics=lambda prediction: compute_wer(processor, prediction),
        callbacks=[build_integrity_callback(transformers, plan, metadata, checkpoint_store)],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"MatMul8bitLt: inputs will be cast from torch\.float32 to float16.*",
            category=UserWarning,
        )
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(plan.output_dir)
    processor.save_pretrained(plan.output_dir)
    final_metrics = {
        "train": result.metrics,
        "log_history": trainer.state.log_history,
    }
    write_provenance(Path(plan.output_dir), plan, metadata, final_metrics)
    if checkpoint_store is not None:
        checkpoint_store.upload_final(Path(plan.output_dir))
