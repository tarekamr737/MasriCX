"""Explicit Hugging Face checkpoint persistence for authorized GPU runs."""

from __future__ import annotations

import importlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from masricx.training.checkpoint import validate_checkpoint

__all__ = ["HubCheckpointStore"]

_REMOTE_CHECKPOINT = re.compile(r"checkpoint-(\d+)")


class HubCheckpointStore:
    """Upload complete checkpoints and final adapters to one private Hub repo."""

    def __init__(
        self,
        repo_id: str,
        experiment: str,
        git_commit: str,
        *,
        api: Any | None = None,
    ) -> None:
        if not repo_id or "/" not in repo_id:
            raise ValueError("checkpoint repo must use the OWNER/NAME form")
        if api is None:
            if not os.environ.get("HF_TOKEN"):
                raise RuntimeError("HF_TOKEN is required when checkpoint upload is enabled")
            hub = importlib.import_module("huggingface_hub")
            api = hub.HfApi(token=os.environ["HF_TOKEN"])
        self.api = api
        self.repo_id = repo_id
        self.prefix = f"runs/{git_commit}/{experiment}"
        info = self.api.repo_info(repo_id=repo_id, repo_type="model", token=True)
        if getattr(info, "private", None) is not True:
            raise RuntimeError(f"checkpoint repository must be private: {repo_id}")

    def restore_latest(self, output_dir: Path) -> Path | None:
        """Download and validate the newest checkpoint for this exact run."""
        prefix = f"{self.prefix}/"
        candidates: set[tuple[int, str]] = set()
        for remote_path in self.api.list_repo_files(
            repo_id=self.repo_id, repo_type="model", token=True
        ):
            if not remote_path.startswith(prefix):
                continue
            relative = remote_path.removeprefix(prefix)
            checkpoint_name = relative.split("/", 1)[0]
            match = _REMOTE_CHECKPOINT.fullmatch(checkpoint_name)
            if match is not None:
                candidates.add((int(match.group(1)), checkpoint_name))
        if not candidates:
            return None

        _, checkpoint_name = max(candidates)
        remote_prefix = f"{self.prefix}/{checkpoint_name}/"
        files = [
            path
            for path in self.api.list_repo_files(
                repo_id=self.repo_id, repo_type="model", token=True
            )
            if path.startswith(remote_prefix)
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hub-restore-", dir=output_dir) as temporary:
            staging = Path(temporary)
            for remote_path in files:
                self.api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_path,
                    repo_type="model",
                    token=True,
                    local_dir=staging,
                )
            source = staging / self.prefix / checkpoint_name
            complete, missing = validate_checkpoint(source)
            if not complete:
                raise RuntimeError(
                    f"remote checkpoint is incomplete {checkpoint_name}: "
                    f"missing={','.join(missing)}"
                )
            destination = output_dir / checkpoint_name
            if destination.exists():
                raise RuntimeError(f"refusing to overwrite local checkpoint: {destination}")
            shutil.copytree(source, destination)
        return destination

    def upload_checkpoint(self, checkpoint: Path) -> None:
        complete, missing = validate_checkpoint(checkpoint)
        if not complete:
            raise RuntimeError(
                f"refusing to upload incomplete checkpoint {checkpoint}: "
                f"missing={','.join(missing)}"
            )
        self._upload(checkpoint, f"{self.prefix}/{checkpoint.name}")

    def upload_final(self, output_dir: Path) -> None:
        required = ("adapter_config.json", "experiment_metadata.json", "training_config.json")
        missing = [name for name in required if not (output_dir / name).is_file()]
        if not any(
            (output_dir / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            missing.append("adapter_model.safetensors|adapter_model.bin")
        if missing:
            raise RuntimeError(
                f"refusing to upload incomplete final adapter {output_dir}: "
                f"missing={','.join(missing)}"
            )
        self._upload(output_dir, f"{self.prefix}/final")

    def _upload(self, folder: Path, path_in_repo: str) -> None:
        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=folder,
            path_in_repo=path_in_repo,
            commit_message=f"Upload {path_in_repo}",
            ignore_patterns=["checkpoint-*", "checkpoint-*/**"],
        )
