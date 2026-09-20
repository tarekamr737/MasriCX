"""Per-example audit aggregation record (pure, incrementally updatable).

One :class:`ExampleAudit` is produced per example by the CLI from raw
fields; :class:`masricx.data.aggregators.AuditAggregator` folds them into the
final report payload. Everything is deterministic and audio/PII-free in output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ExampleAudit"]


@dataclass
class ExampleAudit:
    """Extracted, compact audit facts for one example (no waveforms retained)."""

    sample_id: str
    index: int
    transcript_raw: str
    duration_seconds: float | None
    sample_rate: int | None
    # Audio flags (None => audio metrics unavailable, e.g. --no-audio).
    silent: bool | None = None
    clipped: bool | None = None
    corrupted: bool | None = None
    audio_fingerprint: str | None = None
    # Audio bytes are never retained here; metrics are computed by the caller.
    language_category: str = ""
    normalized: str = ""
    char_count: int = 0
    token_count: int = 0
    ar_only_tokens: int = 0
    en_only_tokens: int = 0
    mixed_tokens: int = 0
    other_tokens: int = 0
    en_meaningful_tokens: int = 0
    code_switched: bool = False
    unusual_long: bool = False
    mismatch: bool = False
    empty_transcript: bool = False
    number_slots: list[str] = field(default_factory=list)
