#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher currently targets macOS." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
PID_FILE="$DATA/training.pid"
LIVE_FILE="$DATA/live_state.json"
LOG_FILE="$DATA/training.log"
mkdir -p "$DATA"

if [[ -f "$PID_FILE" ]]; then
  read -r PID < "$PID_FILE" || true
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    STEP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("step", "unknown"))' "$LIVE_FILE" 2>/dev/null || echo unknown)"
    echo "Training already running at PID $PID, step $STEP"
    exit 0
  fi
fi

cd "$ROOT"
export TRAINING_MAX_THREADS="${TRAINING_MAX_THREADS:-2}"
export TRAINING_MIN_BATTERY_PCT="${TRAINING_MIN_BATTERY_PCT:-20}"
export TICKS_PER_SECOND="${TICKS_PER_SECOND:-30}"
export OMP_NUM_THREADS="$TRAINING_MAX_THREADS"
export MKL_NUM_THREADS="$TRAINING_MAX_THREADS"
export OPENBLAS_NUM_THREADS="$TRAINING_MAX_THREADS"

( exec nohup uv run python -m scripts.rl_tabula_rasa.train_selfplay --continuous ) \
  >> "$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE.tmp"
mv "$PID_FILE.tmp" "$PID_FILE"
echo "Started headless training at PID $PID. Log: $LOG_FILE"
