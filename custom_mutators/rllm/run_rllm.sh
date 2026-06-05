#!/usr/bin/env bash
set -e

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_executable() {
  local path="$1"
  local label="$2"
  if [ ! -x "$path" ]; then
    echo "Missing executable $label: $path" >&2
    exit 1
  fi
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RLLM_DIR="$SCRIPT_DIR"
AFLPP_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

DEFAULT_DATASET="dataset-dec22-u16-seeds"   # 100 valid seeds (already u16)
DEFAULT_CONFIG="configs/grpo.json"
DATASET="$DEFAULT_DATASET"
RUN_NAME=""
CONFIG="$DEFAULT_CONFIG"
DEBUG=0
AFL_FUZZ_BIN="$AFLPP_DIR/afl-fuzz"
TARGET_BIN="$HOME/Documents/data_store/engines/jerryscript/build/bin/jerry"

USAGE="Usage: $0 [-i dataset_dir] -o run_name [-c config_json] [-d]"

while getopts "i:o:c:d" opt; do
  case "$opt" in
    i) DATASET="$OPTARG" ;;
    o) RUN_NAME="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    d) DEBUG=1 ;;
    *) echo "$USAGE" ; exit 1 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "$USAGE"
  exit 1
fi

# Resolve relative config paths against the rllm dir so the script works
# from any CWD; absolute paths are honored as-is.
case "$CONFIG" in
  /*) CONFIG_PATH="$CONFIG" ;;
  *)  CONFIG_PATH="$RLLM_DIR/$CONFIG" ;;
esac
require_file "$CONFIG_PATH" "config file"

OUT_DIR="$HOME/Documents/data_store/out/$RUN_NAME"
DSROOT="$HOME/Documents/data_store/dataset"

require_executable "$AFL_FUZZ_BIN" "afl-fuzz"
require_executable "$TARGET_BIN" "jerryscript binary"

# Seeds are u16 token-id binaries (TLAFL/CovRL Option B), not JS source. If
# DATASET already names a preprocessed u16 dir (its .tokenizer.json marker
# exists), use it directly; otherwise treat DATASET as raw JS and build the
# ${DATASET}-u16 version on first run (idempotent via the marker).
if [ -f "$DSROOT/${DATASET}.tokenizer.json" ]; then
  DATASET_DIR="$DSROOT/$DATASET"
else
  require_dir "$DSROOT/$DATASET" "raw dataset directory"
  DATASET_DIR="$DSROOT/${DATASET}-u16"
  if [ ! -f "${DATASET_DIR}.tokenizer.json" ]; then
    echo "[run_rllm] preprocessing seeds: $DSROOT/$DATASET -> $DATASET_DIR"
    (cd "$RLLM_DIR" && python -m data.preprocess --input "$DSROOT/$DATASET" --output "$DATASET_DIR")
  fi
fi
require_dir "$DATASET_DIR" "preprocessed (u16) dataset directory"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$RLLM_DIR"
export AFL_PYTHON_MODULE=rllm
export AFL_CUSTOM_MUTATOR_ONLY=1
export AFL_DISABLE_TRIM=1
export AFL_NO_FASTRESUME=1
# Load-bearing for Option B: keep the mutator's u16 output in the queue
# instead of overwriting it with post_process's decoded JS source.
# Without this, the queue would store decoded JS and the queue-as-tokens
# invariant collapses. See docs/option_b_viability.md §2.
export AFL_POST_PROCESS_KEEP_ORIGINAL=1
export AFL_FRAMESHIFT_DISABLE=1
export AFL_PRELOAD="$RLLM_DIR/exit_hook.so"
export RLM_STDERR_FILE="$OUT_DIR/rllm_stderr.txt"
# Deliver a CATCHABLE signal (SIGUSR1=10) to the child on timeout so exit_hook.so
# can tag hangs (→ "timeout" class, punished). Keep the forkserver on SIGKILL.
# This changes only *which* signal kills on timeout — NOT the -t timeout duration.
export AFL_KILL_SIGNAL=10
export AFL_FORK_SERVER_KILL_SIGNAL=9
export RLLM_CONFIG="$CONFIG"

echo "[run_rllm] queue files are u16 token-id binary; use 'python -m data.decode <file>' to inspect"

# -d: surface AFL_DEBUG output, propagate target stderr, dump Python tracebacks
# on signals, and tee everything to <out>/debug.log so the run is auditable.
if [ "$DEBUG" -eq 1 ]; then
  export AFL_DEBUG=1
  export AFL_NO_UI=1
  export AFL_DEBUG_CHILD=1
  export PYTHONUNBUFFERED=1
  export PYTHONFAULTHANDLER=1

  mkdir -p "$OUT_DIR"
  DEBUG_LOG="$OUT_DIR/debug.log"
  echo "[run_rllm] debug mode -> $DEBUG_LOG"

  set -o pipefail
  "$AFL_FUZZ_BIN" \
    -i "$DATASET_DIR" \
    -o "$OUT_DIR" \
    -- "$TARGET_BIN" @@ 2>&1 | tee -a "$DEBUG_LOG"
else
  "$AFL_FUZZ_BIN" \
    -i "$DATASET_DIR" \
    -o "$OUT_DIR" \
    -- "$TARGET_BIN" @@
fi
