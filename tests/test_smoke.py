"""Smoke tests: package imports and config CLI behave correctly offline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import masricx
import masricx.config


def test_package_imports() -> None:
    assert masricx.__version__ == "0.1.0"
    assert hasattr(masricx.config, "load_config")


def _run_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "masricx.config", *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        check=False,
    )


def test_config_cli_validate_help(repo_root: Path) -> None:
    proc = _run_cli(repo_root, "validate", "--help")
    assert proc.returncode == 0
    assert "validate" in proc.stdout


def test_config_cli_validate_all_configs(repo_root: Path) -> None:
    configs = [
        str(repo_root / "configs" / name)
        for name in (
            "pilot.yaml",
            "codeswitch.yaml",
            "telephone.yaml",
            "egyspeak_ablation.yaml",
        )
    ]
    proc = _run_cli(repo_root, "validate", *configs)
    assert proc.returncode == 0, proc.stderr
    assert sum("OK:" in line for line in proc.stdout.splitlines()) == 4


def test_config_cli_rejects_missing_file(repo_root: Path, tmp_path: Path) -> None:
    proc = _run_cli(repo_root, "validate", str(tmp_path / "x.yaml"))
    assert proc.returncode == 1
    assert "ERROR" in proc.stderr


def test_subpackage_imports() -> None:
    import masricx.audio
    import masricx.data
    import masricx.evaluation
    import masricx.inference
    import masricx.training

    assert masricx.audio is not None


def test_runtime_dependencies_declared(repo_root: Path) -> None:
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "PyYAML" in deps
    assert "pydantic" in deps
    assert "platformdirs" not in deps
    assert "TBD" not in text.split("[project.urls]", 1)[-1] if "[project.urls]" in text else True


def test_python_range_supports_current_kaggle_runtime(repo_root: Path) -> None:
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.13"' in text


def test_project_urls_are_public_and_resolved(repo_root: Path) -> None:
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.urls]" in text
    assert "https://github.com/tarekamr737/MasriCX" in text
    assert "TBD" not in text.split("[project.urls]", 1)[1].split("[", 1)[0]


def test_requirements_includes_pydantic(repo_root: Path) -> None:
    req = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    assert "pydantic>=" in req
    assert "PyYAML" in req
    assert "platformdirs" not in req


def test_ci_installs_project_with_dev_deps(repo_root: Path) -> None:
    ci = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'pip install -e ".[dev]"' in ci
    assert "torch" not in ci.replace("timeout-minutes", "")
    assert "transformers" not in ci
    assert "datasets" not in ci
