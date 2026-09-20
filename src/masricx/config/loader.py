"""YAML config loading, structural validation, and CLI.

Offline only: reads YAML files and validates them against
:mod:`masricx.config.schema`. Never imports torch/transformers/datasets and
never touches the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from masricx.config.schema import (
    _EXPERIMENTS_REQUIRING_SECTION,
    ALLOWED_TARGET_MODULES,
    ConfigRootFields,
    ExperimentRoot,
)

REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "experiment",
    "model",
    "lora",
    "training",
    "dataset",
)

_TBD_SENTINEL = "TBD"


class ConfigError(Exception):
    """Raised when a config file is missing, unparseable, or invalid."""


def _replace_tbd(data: Any) -> Any:
    """Replace scalar 'TBD' placeholders with None so they pass optional fields
    while remaining forbidden where real values are required (None fails the
    str/bool/number validators of required fields)."""
    if isinstance(data, dict):
        return {k: _replace_tbd(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace_tbd(v) for v in data]
    if isinstance(data, str) and data.strip() == _TBD_SENTINEL:
        return None
    return data


def load_config(path: str | Path) -> ExperimentRoot:
    """Load and validate one experiment config file.

    Raises:
        ConfigError: file missing, unreadable, unparseable, or invalid.
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"config file not found: {file}")

    try:
        raw: Any = yaml.safe_load(file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read/parse {file}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{file}: top-level YAML must be a mapping")

    _check_required_keys(raw, file)
    _check_known_top_level_keys(raw, file)
    _check_target_modules(raw, file)
    _check_experiment_sections(raw, file)

    try:
        return ExperimentRoot.model_validate(_replace_tbd(raw))
    except ValidationError as exc:
        raise ConfigError(f"{file}: invalid config: {exc}") from exc


def _check_required_keys(raw: dict[str, Any], file: Path) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"{file}: missing required top-level keys: {missing}")


def _check_known_top_level_keys(raw: dict[str, Any], file: Path) -> None:
    unknown = [k for k in raw if k not in ConfigRootFields]
    if unknown:
        raise ConfigError(f"{file}: unknown top-level keys: {unknown}")


def _check_target_modules(raw: dict[str, Any], file: Path) -> None:
    lora = raw.get("lora")
    if isinstance(lora, dict):
        targets = lora.get("target_modules")
        if isinstance(targets, list):
            bad = [t for t in targets if t not in ALLOWED_TARGET_MODULES]
            if bad:
                raise ConfigError(
                    f"{file}: unknown lora.target_modules: {bad}; "
                    f"allowed: {sorted(ALLOWED_TARGET_MODULES)}"
                )


def _check_experiment_sections(raw: dict[str, Any], file: Path) -> None:
    experiment = raw.get("experiment")
    name = None
    if isinstance(experiment, dict):
        exp_name = experiment.get("experiment")
        if isinstance(exp_name, str):
            name = exp_name
    required_sections = _EXPERIMENTS_REQUIRING_SECTION.get(name, ()) if name is not None else ()
    for section in required_sections:
        if section not in raw:
            raise ConfigError(f"{file}: experiment '{name}' requires the '{section}' section")


def validate_file(path: str | Path) -> ExperimentRoot:
    """Validate one config file and return the parsed ExperimentRoot."""
    return load_config(path)


def validate_files(paths: list[str | Path]) -> list[ExperimentRoot]:
    """Validate multiple config files; raises ConfigError on the first failure."""
    return [load_config(p) for p in paths]


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="masricx.config",
        description=("Validate MasriCX experiment configs (offline; no model/dataset access)."),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate one or more experiment YAML configs").add_argument(
        "configs", nargs="+", help="paths to experiment YAML files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.command != "validate":  # pragma: no cover - argparse enforces choices
        raise ConfigError(f"unsupported command: {args.command}")
    try:
        for path in args.configs:
            load_config(path)
            print(f"OK: {path}")
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
