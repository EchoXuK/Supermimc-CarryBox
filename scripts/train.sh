#!/bin/bash
# Train with the box-carry recipe (scripts/presets.sh, PRESET[box_carry]).
# Usage: bash scripts/train.sh <log_dir> [extra train.py flags, e.g. --seed 1]
# PRESET_NAME=main selects the original LAFAN1 recipe. Python: $PYTHON, default .venv/bin/python.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/presets.sh
LOG_DIR=${1:?usage: bash scripts/train.sh <log_dir> [train.py flags]}; shift
mkdir -p "$LOG_DIR"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # the reference runs used 4 CPU threads
PYTHONPATH=src ${PYTHON:-.venv/bin/python} -u -m blinkmimic.train --log_dir "$LOG_DIR" ${PRESET[${PRESET_NAME:-box_carry}]} "$@" 2>&1 \
  | tee "$LOG_DIR/stdout.log"
