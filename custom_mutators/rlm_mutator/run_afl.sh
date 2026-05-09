#!/usr/bin/env bash
set -e
set -o pipefail

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
DEBUG=0
GDB=0
AFL_FUZZ_BIN="$HOME/Documents/AFLplusplus/afl-fuzz"
EXIT_HOOK_SO="$HOME/Documents/AFLplusplus/custom_mutators/rlm_mutator/exit_hook.so"
DATASET_DIR="$HOME/Documents/data_store/dataset/$DATASET"
TARGET_BIN="$HOME/Documents/data_store/engines/jerryscript/build/bin/jerry"

usage() {
  cat <<EOF
Usage: $0 [-i dataset_dir] -o run_name [-c config_json] [-d] [-g]

  -i dataset_dir   Seed corpus subdirectory under data_store/dataset/ (default: $DEFAULT_DATASET)
  -o run_name      Output run name under data_store/out/ (required)
  -c config_json   Optional path to RLM_CONFIG JSON
  -d               Debug mode: AFL_DEBUG, no-UI logs, Python tracebacks, core dumps
                   Output is teed to <out>/debug.log
  -g               Run afl-fuzz under gdb (implies -d). Backtrace printed on crash.
EOF
}

while getopts "i:o:c:dgh" opt; do
  case "$opt" in
    i) DATASET="$OPTARG" ;;
    o) RUN_NAME="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    d) DEBUG=1 ;;
    g) GDB=1; DEBUG=1 ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  usage
  exit 1
fi

DATASET_DIR="$HOME/Documents/data_store/dataset/$DATASET"
OUT_DIR="$HOME/Documents/data_store/out/$RUN_NAME"

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
export AFL_PRELOAD="$EXIT_HOOK_SO"
export RLM_EXIT_FILE="$OUT_DIR/rlm_exit"

if [ -n "$CONFIG" ]; then
  export RLM_CONFIG="$CONFIG"
fi

# --------------------------------------------------------------------------
# Debug mode
#
# Surfaces three independent failure surfaces:
#   AFL_DEBUG=1            — afl-fuzz internal diagnostics on stderr
#   AFL_NO_UI=1            — disables curses so AFL_DEBUG output is visible
#   AFL_DEBUG_CHILD=1      — forwards target stderr instead of swallowing it
#   PYTHONFAULTHANDLER=1   — Python prints a C traceback on SIGSEGV/SIGABRT
#   PYTHONUNBUFFERED=1     — flush Python stdout/stderr per line, no buffer
#   AFL_FORKSRV_INIT_TMOUT — bigger budget for slow Python startup under debug
#   ulimit -c unlimited    — drop core files so we can post-mortem with gdb
#
# All output is tee'd to $OUT_DIR/debug.log so the run is auditable after
# the fact even if the terminal scrolls past the crash.
# --------------------------------------------------------------------------
if [ "$DEBUG" -eq 1 ]; then
  export AFL_DEBUG=1
  export AFL_NO_UI=1
  export AFL_DEBUG_CHILD=1
  export PYTHONUNBUFFERED=1
  export PYTHONFAULTHANDLER=1
  export AFL_FORKSRV_INIT_TMOUT=120000

  ulimit -c unlimited || true

  mkdir -p "$OUT_DIR"
  DEBUG_LOG="$OUT_DIR/debug.log"
  echo "[run_afl] debug mode" | tee -a "$DEBUG_LOG"
  echo "[run_afl] log file:   $DEBUG_LOG" | tee -a "$DEBUG_LOG"
  echo "[run_afl] core files: $(pwd)/core.<pid>" | tee -a "$DEBUG_LOG"
  echo "[run_afl] start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DEBUG_LOG"
fi

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if [ "$GDB" -eq 1 ]; then
  command -v gdb >/dev/null 2>&1 || { echo "gdb not installed" >&2; exit 1; }
  echo "[run_afl] launching under gdb (backtrace on crash)" | tee -a "$DEBUG_LOG"
  gdb -batch \
      -ex "set pagination off" \
      -ex "set print thread-events off" \
      -ex "handle SIGPIPE nostop noprint pass" \
      -ex "handle SIGSEGV stop print" \
      -ex "run" \
      -ex "echo \\n=== backtrace (current thread) ===\\n" \
      -ex "bt full" \
      -ex "echo \\n=== backtrace (all threads) ===\\n" \
      -ex "thread apply all bt" \
      --args "$AFL_FUZZ_BIN" \
      -i "$DATASET_DIR" \
      -o "$OUT_DIR" \
      -- "$TARGET_BIN" @@ 2>&1 | tee -a "$DEBUG_LOG"
elif [ "$DEBUG" -eq 1 ]; then
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
