#!/usr/bin/env bash
set -e

DEFAULT_DATASET="final-dataset-dec22"
DATASET="$DEFAULT_DATASET"
RUN_NAME=""

while getopts "i:o:" opt; do
  case "$opt" in
    i) DATASET="$OPTARG" ;;
    o) RUN_NAME="$OPTARG" ;;
    *) echo "Usage: $0 [-i dataset_dir] -o run_name" ; exit 1 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "Usage: $0 [-i dataset_dir] -o run_name"
  echo "Example: $0 -o grpo_run1"
  echo "Example: $0 -i another-dataset -o grpo_run2"
  exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$HOME/Documents/AFLplusplus/custom_mutators/rlm_mutator"
export AFL_PYTHON_MODULE=rlm
export AFL_CUSTOM_MUTATOR_ONLY=1
export LD_PRELOAD="$HOME/Documents/AFLplusplus/custom_mutators/rlm_mutator/exit_hook.so"
export RLM_CONFIG="$HOME/Documents/AFLplusplus/custom_mutators/rlm_mutator/config.json"

"$HOME/Documents/AFLplusplus/afl-fuzz" \
  -i "$HOME/Documents/data_store/dataset/$DATASET" \
  -o "$HOME/Documents/data_store/out/$RUN_NAME" \
  -- "$HOME/Documents/data_store/engines/jerryscript/build/bin/jerry" @@