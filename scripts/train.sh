#!/usr/bin/env bash
# Thin wrapper for the implemented training entry point. Training remains a
# dry run unless the caller explicitly adds --execute.
set -euo pipefail

CONFIG="${1:?usage: train.sh <config.yaml> [training options]}"
shift

python -m masricx.training.train --config "${CONFIG}" "$@"
