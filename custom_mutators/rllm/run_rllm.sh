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

DEFAULT_DATASET="final-dataset-dec22"
DEFAULT_CONFIG="configs/default.json"
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

DATASET_DIR="$HOME/Documents/data_store/dataset/$DATASET"
OUT_DIR="$HOME/Documents/data_store/out/$RUN_NAME"

require_executable "$AFL_FUZZ_BIN" "afl-fuzz"
require_dir "$DATASET_DIR" "dataset directory"
require_executable "$TARGET_BIN" "jerryscript binary"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$RLLM_DIR"
export AFL_PYTHON_MODULE=rllm
export AFL_CUSTOM_MUTATOR_ONLY=1
export AFL_DISABLE_TRIM=1
export AFL_NO_FASTRESUME=1
export AFL_PRELOAD="$RLLM_DIR/exit_hook.so"
export RLM_STDERR_FILE="$OUT_DIR/rllm_stderr.txt"
export RLLM_CONFIG="$CONFIG"

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
