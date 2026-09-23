from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from masricx.training.hub import HubCheckpointStore


class _Api:
    def __init__(self, files: dict[str, str] | None = None, *, private: bool = True) -> None:
        self.repo_requests: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.files = files or {}
        self.private = private

    def repo_info(self, **kwargs: Any) -> SimpleNamespace:
        self.repo_requests.append(kwargs)
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kwargs: Any) -> None:
        self.uploads.append(kwargs)

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        return list(self.files)

    def hf_hub_download(self, *, filename: str, local_dir: Path, **kwargs: Any) -> str:
        del kwargs
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.files[filename], encoding="utf-8")
        return str(destination)


def _write(path: Path, names: tuple[str, ...]) -> None:
    path.mkdir(parents=True)
    for name in names:
        (path / name).write_text("test", encoding="utf-8")


def test_checkpoint_store_preflights_repo_and_uploads_only_complete_checkpoint(
    tmp_path: Path,
) -> None:
    api = _Api()
    store = HubCheckpointStore("owner/training", "pilot-lora-p1", "abc123", api=api)
    checkpoint = tmp_path / "checkpoint-100"
    _write(
        checkpoint,
        (
            "adapter_config.json",
            "adapter_model.safetensors",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "experiment_metadata.json",
            "training_config.json",
            "metrics.json",
        ),
    )

    store.upload_checkpoint(checkpoint)

    assert api.repo_requests == [{"repo_id": "owner/training", "repo_type": "model", "token": True}]
    assert api.uploads[0]["path_in_repo"] == "runs/abc123/pilot-lora-p1/checkpoint-100"
    assert api.uploads[0]["folder_path"] == checkpoint


def test_checkpoint_store_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    api = _Api()
    store = HubCheckpointStore("owner/training", "E1", "abc123", api=api)
    checkpoint = tmp_path / "checkpoint-100"
    _write(checkpoint, ("adapter_config.json",))

    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        store.upload_checkpoint(checkpoint)

    assert not api.uploads


def test_checkpoint_store_uploads_complete_final_adapter(tmp_path: Path) -> None:
    api = _Api()
    store = HubCheckpointStore("owner/training", "E2", "abc123", api=api)
    output = tmp_path / "e2"
    _write(
        output,
        (
            "adapter_config.json",
            "adapter_model.safetensors",
            "experiment_metadata.json",
            "training_config.json",
        ),
    )

    store.upload_final(output)

    assert api.uploads[0]["path_in_repo"] == "runs/abc123/E2/final"


def test_checkpoint_store_requires_owner_name() -> None:
    with pytest.raises(ValueError, match="OWNER/NAME"):
        HubCheckpointStore("invalid", "E1", "abc123", api=_Api())


def test_checkpoint_store_requires_private_repo() -> None:
    with pytest.raises(RuntimeError, match="must be private"):
        HubCheckpointStore("owner/public", "E1", "abc123", api=_Api(private=False))


def test_checkpoint_store_restores_latest_complete_checkpoint(tmp_path: Path) -> None:
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "experiment_metadata.json",
        "training_config.json",
        "metrics.json",
    )
    prefix = "runs/abc123/E1"
    files = {
        **{f"{prefix}/checkpoint-100/{name}": "old" for name in required},
        **{f"{prefix}/checkpoint-200/{name}": "latest" for name in required},
        "runs/other/E1/checkpoint-999/adapter_config.json": "unrelated",
    }
    store = HubCheckpointStore("owner/training", "E1", "abc123", api=_Api(files))

    restored = store.restore_latest(tmp_path / "output")

    assert restored == tmp_path / "output" / "checkpoint-200"
    assert (restored / "trainer_state.json").read_text(encoding="utf-8") == "latest"


def test_checkpoint_store_returns_none_without_matching_remote_checkpoint(
    tmp_path: Path,
) -> None:
    store = HubCheckpointStore("owner/training", "E1", "abc123", api=_Api())
    assert store.restore_latest(tmp_path / "output") is None
