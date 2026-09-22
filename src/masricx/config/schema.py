"""Typed schema for MasriCX experiment configuration files.

Validation is value-based and offline: no model or dataset access is performed.
Every field is checked against policies fixed by the orchestrator in
``MasriCX_AGENT_SPEC_DELEGATED.md``. Unknown/experiment-specific facts remain
``TBD``/``null`` elsewhere in the repo; they are not validated here.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AllowedExperiment = Literal["E0", "E1", "E2", "E3"]

# Phase 0 fixed scientific decisions (orchestrator-owned).
REQUIRED_MODEL_ID = "openai/whisper-large-v3-turbo"
REQUIRED_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
REQUIRED_LORA_TARGET_MODULES: frozenset[str] = frozenset({"q_proj", "v_proj"})
REQUIRED_SPEED_RATE_RANGE: tuple[float, float] = (0.95, 1.05)
REQUIRED_EGYSPEAK_CONDITIONS: tuple[int, ...] = (0, 10, 25)
MAX_EGYSPEAK_FRACTION_OF_PRIMARY = 0.25

__all__ = [
    "ALLOWED_TARGET_MODULES",
    "AugmentationConfig",
    "ConfigRootFields",
    "DatasetConfig",
    "EgyspeakConfig",
    "ExperimentMeta",
    "ExperimentRoot",
    "LoraConfig",
    "ModelConfig",
    "PilotCandidate",
    "PilotSubsetConfig",
    "SplitConfig",
    "TelephoneTestConfig",
    "TrainingConfig",
]

ALLOWED_TARGET_MODULES: frozenset[str] = REQUIRED_LORA_TARGET_MODULES


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(_StrictModel):
    """Base-model loading options. Fixed Phase 0 decisions: 8-bit loading and
    float16 dtype on exactly ``openai/whisper-large-v3-turbo``."""

    id: str
    revision: str = REQUIRED_MODEL_REVISION
    load_in_8bit: bool
    dtype: Literal["float16"]

    @field_validator("id")
    @classmethod
    def _exact_model_id(cls, v: str) -> str:
        if v != REQUIRED_MODEL_ID:
            raise ValueError(f"model.id must be exactly {REQUIRED_MODEL_ID!r}, got {v!r}")
        return v

    @field_validator("revision")
    @classmethod
    def _pinned_model_revision(cls, v: str) -> str:
        if v != REQUIRED_MODEL_REVISION:
            raise ValueError(
                f"model.revision must be pinned to {REQUIRED_MODEL_REVISION!r}, got {v!r}"
            )
        return v

    @field_validator("load_in_8bit")
    @classmethod
    def _require_8bit(cls, v: bool) -> bool:
        if not v:
            raise ValueError("model.load_in_8bit must be true: MasriCX uses 8-bit base loading")
        return v


class LoraConfig(_StrictModel):
    r: int = Field(ge=1, le=256)
    alpha: int = Field(ge=1, le=1024)
    dropout: float = Field(ge=0.0, le=0.5)
    target_modules: list[str]

    @field_validator("target_modules")
    @classmethod
    def _exact_targets(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("lora.target_modules contains duplicates")
        if not v:
            raise ValueError("lora.target_modules must not be empty")
        if set(v) != REQUIRED_LORA_TARGET_MODULES:
            raise ValueError(
                "lora.target_modules must be exactly q_proj and v_proj "
                "(order-insensitive), got "
                f"{sorted(v)}"
            )
        return v

    @model_validator(mode="after")
    def _alpha_at_least_r(self) -> "LoraConfig":
        if self.alpha < self.r:
            raise ValueError("lora.alpha must be >= lora.r")
        return self


class TrainingConfig(_StrictModel):
    learning_rate: float = Field(gt=0.0, lt=0.1)
    epochs: int = Field(ge=1, le=20)
    warmup_ratio: float = Field(ge=0.0, le=0.5)
    weight_decay: float = Field(ge=0.0, le=1.0)
    per_device_train_batch_size: int = Field(ge=1, le=256)
    gradient_accumulation_steps: int = Field(ge=1, le=256)
    gradient_checkpointing: bool
    fp16: bool
    eval_strategy: Literal["steps", "epoch", "no"]
    eval_steps: int | None = Field(default=None, ge=1)
    save_strategy: Literal["steps", "epoch", "no"]
    save_steps: int | None = Field(default=None, ge=1)
    save_total_limit: int | None = Field(default=None, ge=1)
    seed: Literal[42]

    @model_validator(mode="after")
    def _cross_field_step_requirements(self) -> "TrainingConfig":
        if self.eval_strategy == "steps" and self.eval_steps is None:
            raise ValueError("training.eval_steps is required when eval_strategy=steps")
        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("training.save_steps is required when save_strategy=steps")
        if self.gradient_checkpointing and not self.fp16:
            raise ValueError("training.fp16 must be true when gradient_checkpointing is enabled")
        return self


class SplitConfig(_StrictModel):
    """Deterministic fixed splits. Each percentage must be nonnegative and the
    three must sum to exactly 100."""

    train: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)
    seed: Literal[42]

    @model_validator(mode="after")
    def _percentages_sum(self) -> "SplitConfig":
        if self.train + self.validation + self.test != 100:
            raise ValueError(
                "splits must sum to 100 (train + validation + test), "
                f"got {self.train} + {self.validation} + {self.test}"
            )
        return self


class AugmentationConfig(_StrictModel):
    """Stochastic training-time telephone augmentation.

    ``apply_probability`` is the probability that an example receives the
    augmentation pipeline: 0.0 when disabled, within [0.30, 0.40] when
    enabled. Per-transform probabilities are conditional controls applied only
    to selected examples and retain their specification caps.
    """

    enabled: bool
    seed: Literal[42]
    telephone_bandpass_probability: float = Field(ge=0.0, le=0.3)
    narrowband_resample_probability: float = Field(ge=0.0, le=0.3)
    mulaw_alaw_probability: float = Field(ge=0.0, le=0.15)
    gain_variation_probability: float = Field(ge=0.0, le=0.2)
    clipping_probability: float = Field(ge=0.0, le=0.1)
    noise_probability: float = Field(ge=0.0, le=0.15)
    speed_perturbation_probability: float = Field(ge=0.0, le=0.1)
    speed_rate_range: tuple[float, float]
    apply_probability: float
    preserve_transcript: bool

    @field_validator("speed_rate_range")
    @classmethod
    def _exact_speed_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        low, high = v
        if (low, high) != REQUIRED_SPEED_RATE_RANGE:
            raise ValueError(f"augmentation.speed_rate_range must be exactly [0.95, 1.05], got {v}")
        return v

    @field_validator("preserve_transcript")
    @classmethod
    def _transcript_must_be_preserved(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "augmentation.preserve_transcript must be true: labels are never mutated"
            )
        return v

    @model_validator(mode="after")
    def _apply_probability_matches_enabled(self) -> "AugmentationConfig":
        if not self.enabled:
            if self.apply_probability != 0.0:
                raise ValueError("augmentation.apply_probability must be 0.0 when enabled=false")
        elif not (0.30 <= self.apply_probability <= 0.40):
            raise ValueError(
                "augmentation.apply_probability must be within [0.30, 0.40] when enabled=true"
            )
        return self


class TelephoneTestConfig(_StrictModel):
    """Deterministic fixed degradation used for the telephone evaluation set.

    Training-time stochastic augmentation and test-time fixed degradation are
    deliberately separate config sections so they can never be conflated.
    """

    enabled: bool
    seed: Literal[42]
    bandpass_hz: tuple[int, int]
    reference_resample_khz: int = Field(default=8, ge=8, le=16)

    @field_validator("bandpass_hz")
    @classmethod
    def _telephone_band(cls, v: tuple[int, int]) -> tuple[int, int]:
        low, high = v
        if low >= high:
            raise ValueError("telephone_test.bandpass_hz must be (low, high)")
        if not (200 <= low <= 400):
            raise ValueError("telephone_test.bandpass_hz low cutoff must be 200-400 Hz")
        if not (3000 <= high <= 4000):
            raise ValueError("telephone_test.bandpass_hz high cutoff must be 3000-4000 Hz")
        return v


class DatasetConfig(_StrictModel):
    """Primary dataset with pinned revision and documented license state."""

    id: str
    revision: str | None = None
    license: str | None = None
    splits: SplitConfig

    @field_validator("id")
    @classmethod
    def _nonempty_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset.id must be a non-empty repository id")
        return v


class EgyspeakConfig(_StrictModel):
    """EGYSpeak pseudo-labelled supplementary data (E3 ablation only).

    Pseudo-labelled machine-generated transcripts must never be treated as gold:
    ``is_pseudo_labelled=True`` is structurally required here. Conditions are
    exactly 0%/10%/25% filtered EGYSpeak and it may never exceed 25% of the
    primary dataset.
    """

    id: str
    is_pseudo_labelled: bool
    revision: str | None = None
    license: str | None = None
    include_percent: int
    conditions: list[int]
    max_fraction_of_primary: float = Field(default=MAX_EGYSPEAK_FRACTION_OF_PRIMARY, ge=0.0)

    @field_validator("is_pseudo_labelled")
    @classmethod
    def _must_declare_pseudo(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "egyspeak.is_pseudo_labelled must be true: EGYSpeak transcripts are "
                "machine-generated and may never be treated as gold ground truth"
            )
        return v

    @field_validator("conditions")
    @classmethod
    def _conditions_exact(cls, v: list[int]) -> list[int]:
        if tuple(sorted(v)) != REQUIRED_EGYSPEAK_CONDITIONS:
            raise ValueError(f"egyspeak.conditions must be exactly [0, 10, 25], got {sorted(v)}")
        return v

    @field_validator("include_percent")
    @classmethod
    def _include_percent_in_conditions(cls, v: int) -> int:
        if v not in REQUIRED_EGYSPEAK_CONDITIONS:
            raise ValueError(f"egyspeak.include_percent must be one of [0, 10, 25], got {v}")
        return v

    @field_validator("max_fraction_of_primary")
    @classmethod
    def _fraction_cap(cls, v: float) -> float:
        if v > MAX_EGYSPEAK_FRACTION_OF_PRIMARY:
            raise ValueError(
                "egyspeak.max_fraction_of_primary must be at most 0.25: "
                "EGYSpeak may never dominate the primary training set"
            )
        return v


class PilotCandidate(_StrictModel):
    """One P-run of the pilot hyperparameter study (spec section 20)."""

    run: str
    lora_rank: int = Field(ge=1, le=256)
    learning_rate: float = Field(gt=0.0, lt=0.1)


class PilotSubsetConfig(_StrictModel):
    """Optional pilot-subset sizing (pilot.yaml)."""

    target_hours: float = Field(gt=0.0)
    seed: Literal[42] = 42


class ExperimentMeta(_StrictModel):
    name: str
    experiment: AllowedExperiment
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("experiment.name must be non-empty")
        return v


class ExperimentRoot(_StrictModel):
    """Top-level required structure of every MasriCX experiment config."""

    experiment: ExperimentMeta
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig
    dataset: DatasetConfig
    pilot_subset: PilotSubsetConfig | None = None
    pilot_candidates: list[PilotCandidate] | None = None
    augmentation: AugmentationConfig | None = None
    telephone_test: TelephoneTestConfig | None = None
    egyspeak: EgyspeakConfig | None = None

    @model_validator(mode="after")
    def _experiment_distinctions(self) -> "ExperimentRoot":
        kind = self.experiment.experiment

        if kind == "E1":
            if self.augmentation is None:
                raise ValueError("E1 configs must contain an explicit augmentation section")
            if self.augmentation.enabled:
                raise ValueError("E1 is clean-data training: augmentation.enabled must be false")
        elif kind == "E2":
            if self.augmentation is None or not self.augmentation.enabled:
                raise ValueError("E2 requires telephone augmentation (augmentation.enabled=true)")
        elif kind == "E3":
            if self.egyspeak is None:
                raise ValueError("E3 requires the pseudo-labelled egyspeak section")

        return self

    @model_validator(mode="after")
    def _pilot_candidates_exact(self) -> "ExperimentRoot":
        if self.pilot_candidates is None:
            return self
        spec_runs = [
            ("P1", 8, 1.0e-4),
            ("P2", 16, 1.0e-4),
            ("P3", 16, 5.0e-5),
        ]
        got = sorted((c.run, c.lora_rank, c.learning_rate) for c in self.pilot_candidates)
        if got != sorted(spec_runs):
            raise ValueError(
                "pilot_candidates must be exactly the specification runs "
                "(P1: r=8, lr=1e-4), (P2: r=16, lr=1e-4), (P3: r=16, lr=5e-5), "
                f"got {got}"
            )
        return self


ConfigRootFields = frozenset(ExperimentRoot.model_fields.keys())

# Experiments that require their dedicated section to be present.
_EXPERIMENTS_REQUIRING_SECTION: dict[str, tuple[str, ...]] = {
    "E3": ("egyspeak",),
}
