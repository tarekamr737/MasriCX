"""MasriCX configuration loading and validation.

Offline, typed, minimal: reads YAML experiment configs and rejects missing
required structure and unsafe/invalid core values without any model or dataset
access. Used by CI as a quality gate.

CLI:
    python -m masricx.config validate configs/pilot.yaml configs/codeswitch.yaml
"""

from masricx.config.loader import ConfigError, load_config, validate_file, validate_files
from masricx.config.schema import (
    ALLOWED_TARGET_MODULES,
    AugmentationConfig,
    DatasetConfig,
    EgyspeakConfig,
    ExperimentMeta,
    ExperimentRoot,
    LoraConfig,
    ModelConfig,
    PilotSubsetConfig,
    SplitConfig,
    TelephoneTestConfig,
    TrainingConfig,
)

__all__ = [
    "ALLOWED_TARGET_MODULES",
    "AugmentationConfig",
    "ConfigError",
    "DatasetConfig",
    "EgyspeakConfig",
    "ExperimentMeta",
    "ExperimentRoot",
    "LoraConfig",
    "ModelConfig",
    "PilotSubsetConfig",
    "SplitConfig",
    "TelephoneTestConfig",
    "TrainingConfig",
    "load_config",
    "validate_file",
    "validate_files",
]
