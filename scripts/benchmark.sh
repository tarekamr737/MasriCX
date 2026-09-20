#!/usr/bin/env bash
# Thin wrapper: benchmark evaluation. Delegates to
# src/masricx/evaluation/benchmark (later phase).
set -euo pipefail

CONFIG="${1:-configs/codeswitch.yaml}"

python -m masricx.evaluation.benchmark --config "${CONFIG}"
