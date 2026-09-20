#!/usr/bin/env bash
# Thin wrapper: training entry point. Delegates to
# src/masricx.training.train (later phase). Never runs training implicitly;
# callers must pass an explicit config.
set -euo pipefail

CONFIG="${1:?usage: train.sh <config.yaml>}"

python -m masricx.training.train --config "${CONFIG}"
