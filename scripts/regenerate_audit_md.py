"""Regenerate reports/DATA_AUDIT.md from the canonical audit JSON.

Offline regeneration ONLY: reads artifacts/data_audit.json (unchanged,
byte-for-byte) and re-renders the Markdown through the project renderer.
Never touches the JSON, and never loads datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from masricx.data.report_render import _write_md

JSON_PATH = Path("artifacts/data_audit.json")
MD_PATH = Path("reports/DATA_AUDIT.md")


def regenerate(json_path: Path = JSON_PATH, md_path: Path = MD_PATH) -> None:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    _write_md(report, md_path)  # newline="\n": never writes CRLF
    print(f"regenerated {md_path} from {json_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    regenerate(root / JSON_PATH, root / MD_PATH)
    sys.exit(0)
