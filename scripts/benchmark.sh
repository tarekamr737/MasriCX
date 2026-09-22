#!/usr/bin/env bash
# Aggregate an existing prediction JSONL into benchmark artifacts.
set -euo pipefail

PREDICTIONS="${1:?usage: benchmark.sh <predictions.jsonl> [benchmark options]}"
shift

python -m masricx.evaluation.benchmark --predictions "${PREDICTIONS}" "$@"
