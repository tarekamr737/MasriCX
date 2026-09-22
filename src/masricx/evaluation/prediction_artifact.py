"""Fail-closed provenance sidecars for resumable prediction JSONL artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_prediction_provenance(output_path: Path, expected: Mapping[str, Any]) -> Path:
    """Create or validate an immutable provenance sidecar before appending."""
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    payload = dict(expected)
    if sidecar.exists():
        actual = json.loads(sidecar.read_text(encoding="utf-8"))
        if actual != payload:
            raise ValueError(
                f"prediction provenance mismatch for {output_path}; use a new output path"
            )
        return sidecar
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError(f"missing provenance sidecar for existing predictions: {output_path}")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sidecar
