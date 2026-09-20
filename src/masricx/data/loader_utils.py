"""Shared loader utilities: stable sample IDs and dataset-revision guardrails."""

from __future__ import annotations

import hashlib
from typing import Any

_REVISION_MIN_LENGTH = 8


def stable_sample_id(
    repo_id: str,
    revision: str,
    split: str,
    index: int,
) -> str:
    """Deterministic, collision-resistant sample ID from stable coordinates.

    Uses SHA-256 over ``repo_id|revision|split|index`` so IDs are independent
    of Python ``hash()`` seeding. They depend on the dataset's pinned row
    order (index coordinates): they are stable and reproducible for a pinned
    repo/revision/split, not "order-independent" in any stronger sense.
    """
    payload = f"{repo_id}|{revision}|{split}|{index}".encode()
    return hashlib.sha256(payload).hexdigest()


def require_revision(repo_id: str, revision: str | None) -> str:
    """Require a pinned revision (commit sha) when loading datasets."""
    if not revision or not revision.strip():
        raise ValueError(
            f"dataset {repo_id!r}: a pinned revision (commit sha) is required; "
            "do not load unpinned 'main' so audits and training are reproducible"
        )
    rev = revision.strip()
    if len(rev) < _REVISION_MIN_LENGTH:
        raise ValueError(
            f"dataset {repo_id!r}: revision {rev!r} looks too short to be a "
            "commit sha; pin the exact revision"
        )
    return rev


def extract_audio_arrays(example: dict[str, Any]) -> tuple[Any, Any]:
    """Return (audio_dict, transcript) for a HF datasets example.

    Field names supported: 'audio' + 'transcript' (asr_codeswitched_dataset,
    Casablanca) or 'file_name'/'transcription' (EGYSpeak metadata schemas).
    """
    audio = example.get("audio")
    transcript = None
    for key in ("transcript", "transcription", "text", "sentence"):
        if key in example:
            transcript = example[key]
            break
    return audio, transcript


def load_datasets_module() -> Any:
    """Lazy import of :mod:`datasets` with an actionable error message."""
    try:
        import datasets  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "The 'datasets' package is required to load MasriCX audio datasets. "
            "Install it with: pip install 'datasets>=2.20,<5.0' (training/"
            "evaluation environments only; CI never downloads data)."
        ) from exc
    return datasets


def load_soundfile_module() -> Any:
    """Lazy import of :mod:`soundfile` with an actionable error message."""
    try:
        import soundfile  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "The 'soundfile' package is required to decode audio files. "
            "Install it with: pip install 'soundfile>=0.12,<1' "
            "(only needed for audio-level audits)."
        ) from exc
    return soundfile
