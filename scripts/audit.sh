#!/usr/bin/env bash
# Thin wrapper: dataset audit. Delegates to src/masricx/data/audit (Phase 1).
set -euo pipefail

CONFIG="${1:-configs/codeswitch.yaml}"
if [[ $# -gt 0 ]]; then shift; fi

python -m masricx.data.audit --config "${CONFIG}" "$@"
