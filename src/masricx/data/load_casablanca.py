"""Loader for the external Casablanca Egypt evaluation set (UBC-NLP/Casablanca).

Governance (verified 2026-09-20 by the orchestrator from the live dataset card):

- Revision ``8951b1b88e28c1107142ced57967b8d16350951d``; license
  CC-BY-NC-ND-4.0; config ``Egypt``; validation 846 rows, test 846 rows;
  fields audio, seg_id, transcription, gender, duration.
- EVALUATION ONLY: never train, augment, or tune on it. Validation may be used
  only for pipeline smoke/diagnosis without model selection; test is the final
  external benchmark. No redistribution of audio or modified derivatives;
  publish only aggregate metrics and brief non-sensitive examples with
  attribution. ``release_blocker`` for data redistribution (license restrictions).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from masricx.data.loader_utils import (
    extract_audio_arrays,
    load_datasets_module,
    require_revision,
    stable_sample_id,
)

__all__ = [
    "CASABLANCA_ALLOWED_SPLITS",
    "CASABLANCA_REPO_ID",
    "CASABLANCA_REVISION",
    "load_casablanca_examples",
]

CASABLANCA_REPO_ID = "UBC-NLP/Casablanca"
CASABLANCA_REVISION = "8951b1b88e28c1107142ced57967b8d16350951d"
CASABLANCA_EGYPT_CONFIG = "Egypt"
CASABLANCA_ALLOWED_SPLITS = frozenset({"validation", "test"})


def load_casablanca_examples(
    revision: str = CASABLANCA_REVISION,
    split: str = "test",
    config: str = CASABLANCA_EGYPT_CONFIG,
    max_examples: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield Casablanca Egypt examples restricted to validation/test.

    Rejects ``train`` or any other split. Raises on any augmentation request
    (there is no augmentation parameter: training use is structurally refused).
    """
    rev = require_revision(CASABLANCA_REPO_ID, revision)
    if split not in CASABLANCA_ALLOWED_SPLITS:
        raise ValueError(
            f"Casablanca is evaluation-only: split must be one of "
            f"{sorted(CASABLANCA_ALLOWED_SPLITS)}, got {split!r}. Training or "
            "augmentation on it is forbidden."
        )
    if max_examples is not None and max_examples < 0:
        raise ValueError(f"max_examples must be >= 0 or None, got {max_examples}")
    datasets = load_datasets_module()
    ds = datasets.load_dataset(CASABLANCA_REPO_ID, config, split=split, revision=rev)
    for index, example in enumerate(ds):
        if max_examples is not None and index >= max_examples:
            break
        audio, transcript = extract_audio_arrays(example)
        yield {
            "sample_id": stable_sample_id(CASABLANCA_REPO_ID, rev, split, index),
            "repo_id": CASABLANCA_REPO_ID,
            "revision": rev,
            "split": split,
            "index": index,
            "audio": audio,
            "transcript": transcript,
            "seg_id": example.get("seg_id"),
            "gender": example.get("gender"),
            "duration": example.get("duration"),
            "is_pseudo_labelled": False,
            "role": "external_eval_only",
        }
