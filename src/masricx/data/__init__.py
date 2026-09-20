"""Dataset loading, cleaning, normalization, and deterministic auditing.

Public API:
- ``load_codeswitch`` / ``load_egyspeak`` / ``load_casablanca`` loaders (governance-aware)
- ``audit`` CLI (``python -m masricx.data.audit --config configs/codeswitch.yaml``)
- ``text`` normalization / language classification helpers
"""

from masricx.data.load_casablanca import load_casablanca_examples
from masricx.data.load_codeswitch import load_codeswitch_examples
from masricx.data.load_egyspeak import load_egyspeak_examples
from masricx.data.text import audit_normalize, clean_transcript, code_switched_category

__all__ = [
    "audit_normalize",
    "clean_transcript",
    "code_switched_category",
    "load_casablanca_examples",
    "load_codeswitch_examples",
    "load_egyspeak_examples",
]
