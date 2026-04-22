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

DEFAULT_DATASET="final-dataset-dec22"
DATASET="$DEFAULT_DATASET"
RUN_NAME=""
CONFIG=""
AFL_FUZZ_BIN="$HOME/Documents/AFLplusplus/afl-fuzz"
EXIT_HOOK_SO="$HOME/Documents/AFLplusplus/custom_mutators/rlm_mutator/exit_hook.so"
DATASET_DIR="$HOME/Documents/data_store/dataset/$DATASET"
TARGET_BIN="$HOME/Documents/data_store/engines/jerryscript/build/bin/jerry"

while getopts "i:o:c:" opt; do
  case "$opt" in
    i) DATASET="$OPTARG" ;;
    o) RUN_NAME="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    *) echo "Usage: $0 [-i dataset_dir] -o run_name [-c config_json]" ; exit 1 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "Usage: $0 [-i dataset_dir] -o run_name [-c config_json]"
  exit 1
fi

DATASET_DIR="$HOME/Documents/data_store/dataset/$DATASET"

require_executable "$AFL_FUZZ_BIN" "afl-fuzz"
require_file "$EXIT_HOOK_SO" "exit_hook.so"
require_dir "$DATASET_DIR" "dataset directory"
require_executable "$TARGET_BIN" "jerryscript binary"

if [ -n "$CONFIG" ]; then
  require_file "$CONFIG" "config JSON"
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$HOME/Documents/AFLplusplus/custom_mutators/rlm_mutator"
export AFL_PYTHON_MODULE=rlm
export AFL_CUSTOM_MUTATOR_ONLY=1
export LD_PRELOAD="$EXIT_HOOK_SO"

if [ -n "$CONFIG" ]; then
  export RLM_CONFIG="$CONFIG"
fi

"$AFL_FUZZ_BIN" \
  -i "$DATASET_DIR" \
  -o "$HOME/Documents/data_store/out/$RUN_NAME" \
  -- "$TARGET_BIN" @@
