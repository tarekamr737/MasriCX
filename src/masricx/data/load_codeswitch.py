"""Loader for the primary code-switched dataset (Seif-Eldeen-Sameh/asr_codeswitched_dataset).

Governance (verified 2026-09-20 by the orchestrator from the live dataset card):

- Public, ungated; config ``default``; split ``train``; ~45,189 rows.
- Aggregate card metadata says MIT, but ~12,480 clips derive from
  ``MohamedRashad/arabic-english-code-switching`` whose revision/source is
  tagged GPL and built partly from YouTube plus a now-unavailable predecessor.
  The aggregate must therefore never be described as uniformly MIT.
- Allowed Phase 1/2: audit and provisional internal research/training under
  the published terms. No redistribution of raw or transformed data. Final
  model publication/license claims remain blocked pending source-chain review.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from masricx.data.loader_utils import (
    extract_audio_arrays,
    load_datasets_module,
    require_revision,
    stable_sample_id,
)

__all__ = [
    "CODESWITCH_DEFAULT_CONFIG",
    "CODESWITCH_DEFAULT_SPLIT",
    "CODESWITCH_REPO_ID",
    "CODESWITCH_REVISION",
    "load_codeswitch_examples",
    "load_codeswitch_streaming",
]

CODESWITCH_REPO_ID = "Seif-Eldeen-Sameh/asr_codeswitched_dataset"
CODESWITCH_REVISION = "712de01079517771f95bcdecee68ca232b979628"
CODESWITCH_DEFAULT_CONFIG = "default"
CODESWITCH_DEFAULT_SPLIT = "train"


def load_codeswitch_examples(
    revision: str = CODESWITCH_REVISION,
    split: str = CODESWITCH_DEFAULT_SPLIT,
    config: str = CODESWITCH_DEFAULT_CONFIG,
    streaming: bool = False,
    max_examples: int | None = None,
    decode_audio: bool = True,
) -> Iterable[dict[str, Any]]:
    """Yield examples with stable IDs, preserving audio/transcript when decoded.

    The exact revision is pinned by default. Sample IDs are derived
    deterministically from repo/revision/split/index (the dataset's pinned row
    order, NOT Python ``hash()``): they are stable and reproducible for the
    pinned repo/revision/split, not "order-independent" in any stronger
    sense. With ``decode_audio=False`` the audio column is cast to
    ``datasets.Audio(decode=False)`` before iteration so no audio bytes are
    decoded or returned (raw audio is left as path/metadata only); the
    yielded ``audio`` value is then None from this loader's perspective and
    the audit must never touch decoded waveform data.
    """
    rev = require_revision(CODESWITCH_REPO_ID, revision)
    if max_examples is not None and max_examples < 0:
        raise ValueError(f"max_examples must be >= 0 or None, got {max_examples}")
    datasets = load_datasets_module()
    if streaming or max_examples is not None:
        ds = datasets.load_dataset(
            CODESWITCH_REPO_ID,
            config,
            split=split,
            revision=rev,
            streaming=True,
        )
        if not decode_audio:
            ds = _disable_audio_decode(ds, datasets)
        iterator: Iterator[dict[str, Any]] = iter(ds)
        for index, example in enumerate(iterator):
            if max_examples is not None and index >= max_examples:
                break
            yield _with_ids(CODESWITCH_REPO_ID, rev, split, index, example, decode_audio)
    else:
        ds = datasets.load_dataset(CODESWITCH_REPO_ID, config, split=split, revision=rev)
        if not decode_audio:
            ds = _disable_audio_decode(ds, datasets)
        for index, example in enumerate(ds):
            yield _with_ids(CODESWITCH_REPO_ID, rev, split, index, dict(example), decode_audio)


def _disable_audio_decode(ds: Any, datasets: Any) -> Any:
    """Cast the audio column to ``datasets.Audio(decode=False)`` so no audio
    is ever decoded.

    Fail-closed: raising an actionable error is required because silently
    continuing with decoding enabled would violate the no-audio contract.
    """
    cast_column = getattr(ds, "cast_column", None)
    audio_cls = getattr(datasets, "Audio", None)
    if cast_column is None or audio_cls is None:
        raise RuntimeError(
            "decode_audio=False requires ds.cast_column and datasets.Audio to "
            f"disable audio decoding, but they are unavailable "
            f"(cast_column={cast_column is not None}, Audio={audio_cls is not None}). "
            "This environment cannot honor no-audio mode: refusing to "
            "silently keep decoding. Install a compatible 'datasets' version "
            "or run without --no-audio."
        )
    return cast_column("audio", audio_cls(decode=False))


def load_codeswitch_streaming(
    revision: str = CODESWITCH_REVISION,
    split: str = CODESWITCH_DEFAULT_SPLIT,
    config: str = CODESWITCH_DEFAULT_CONFIG,
    max_examples: int | None = None,
    decode_audio: bool = True,
) -> Iterable[dict[str, Any]]:
    """Streaming variant (generator only; nothing is downloaded eagerly)."""
    return load_codeswitch_examples(
        revision=revision,
        split=split,
        config=config,
        streaming=True,
        max_examples=max_examples,
        decode_audio=decode_audio,
    )


def _with_ids(
    repo_id: str,
    revision: str,
    split: str,
    index: int,
    example: dict[str, Any],
    decode_audio: bool = True,
) -> dict[str, Any]:
    if decode_audio:
        audio, transcript = extract_audio_arrays(example)
    else:
        # no-audio mode: never touch example["audio"] or .get("audio") at all;
        # read the transcript only.
        audio = None
        transcript = None
        for key in ("transcript", "transcription", "text", "sentence"):
            if key in example:
                transcript = example[key]
                break
    return {
        "sample_id": stable_sample_id(repo_id, revision, split, index),
        "repo_id": repo_id,
        "revision": revision,
        "split": split,
        "index": index,
        "audio": audio,
        "transcript": transcript,
        "is_pseudo_labelled": False,
        "decode_audio": decode_audio,
    }
