# RLM GRPO AFL++ Mutator

This directory contains an AFL++ Python custom mutator that:

- generates mutations with a seq2seq policy
- records policy actions taken during fuzzing
- assigns rewards from coverage plus exit status
- periodically finetunes the policy with GRPO on collected rollout data

The intended entry point is [run_afl.sh](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/run_afl.sh:1).

## What This Mutator Does

At a high level:

1. AFL++ selects a seed.
2. The mutator masks parts of the seed and asks the model to infill them.
3. The mutator records:
   - the masked input `x_t`
   - the generated output `y_t`
   - the policy log-probability
   - the GRPO `group_id`
4. AFL++ executes the mutated sample.
5. `post_run()` computes reward from coverage novelty and exit status.
6. After enough queue events, the mutator finetunes on the collected rollout buffer.

Relevant files:

- [rlm.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/rlm.py:1): AFL++ hook entrypoint
- [mutator.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/mutator.py:1): masking, generation, rollout logging
- [grpo.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/grpo.py:1): GRPO loss
- [base_trainer.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/base_trainer.py:1): HF trainer integration
- [rewarding.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/rewarding.py:1): online TF-IDF coverage reward
- [run_afl.sh](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/run_afl.sh:1): launcher
- [configs/grpo_run1.json](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/configs/grpo_run1.json:1): example config

## Prerequisites

You need:

- an AFL++ build with Python custom mutator support
- a working target binary
- the dataset directory AFL++ will use as seeds
- a Python environment containing the mutator dependencies
- `gcc` to build `exit_hook.so`

The current launcher expects these paths:

- `~/Documents/AFLplusplus/afl-fuzz`
- `~/Documents/AFLplusplus/custom_mutators/rlm_mutator/exit_hook.so`
- `~/Documents/data_store/dataset/$DATASET`
- `~/Documents/data_store/engines/jerryscript/build/bin/jerry`

If any of these are missing, `run_afl.sh` will stop with a readable error.

## Important Python Version Note

The AFL++ Python mutator API is not inherently tied to Python 3.12 specifically.
What matters is this:

- the Python environment you activate at runtime must match the Python version AFL++ was built against

If `afl-fuzz` was built against Python 3.12, then using Python 3.12 is correct.
If it was built against a different Python version, your env must match that version, or you must rebuild AFL++ inside the env you want to use.

To check what `afl-fuzz` is linked against:

```bash
ldd ~/Documents/AFLplusplus/afl-fuzz | grep libpython
```

If you see `libpython3.12`, then keeping [environment.yml](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/environment.yml:1) at Python 3.12 is the right move.

## Create The Conda Environment

Option 1: use the provided conda env file:

```bash
conda env create -f custom_mutators/rlm_mutator/environment.yml
conda activate rlm-grpo
```

Option 2: create the env manually, then install from `requirements.txt`:

```bash
conda create -n rlm-grpo python=3.12 -y
conda activate rlm-grpo
pip install -r custom_mutators/rlm_mutator/requirements.txt
```

If you are using a GPU machine and need a CUDA-specific PyTorch build, replace the generic `torch` package in:

- [environment.yml](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/environment.yml:1)
- [requirements.txt](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/requirements.txt:1)

with the correct build for that machine before creating the env.

## Build The Exit Hook

The mutator uses `exit_hook.so` to recover normal process exit codes from AFL++ child runs.

From this directory:

```bash
cd ~/Documents/AFLplusplus/custom_mutators/rlm_mutator
gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c
```

Verify it exists:

```bash
ls -l exit_hook.so
```

## Verify The Python Environment

Before launching AFL++, make sure the activated env can import the required packages:

```bash
python -c "import torch, transformers, accelerate, peft, numpy, tensorboard, sentencepiece, safetensors"
```

If this fails, fix the env before running AFL++.

## Verify AFL++ Can Load Python

The most common mismatch is:

- the env can import `torch`
- but `afl-fuzz` cannot start because it is linked against another `libpython`

If you see an error like:

```text
error while loading shared libraries: libpython3.x.so.1.0: cannot open shared object file
```

then you should either:

- activate the env AFL++ was built against

or:

- rebuild AFL++ inside the env you want to use

## Configure A Run

The example config is:

- [configs/grpo_run1.json](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/configs/grpo_run1.json:1)

Important fields:

- `algorithm`: should be `grpo`
- `group_size`: GRPO group size
- `train_batch_size`: must equal `group_size`
- `fuzz_count`: must be divisible by `group_size`
- `sample_method`: `greedy` or `contrastive`
- `enable_logging`: enables trainer and rollout logging
- `logging_steps`: trainer logging cadence

These invariants are enforced in [config.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/config.py:259), so invalid configs fail early.

## Run AFL++

From this directory:

```bash
conda activate rlm-grpo
cd ~/Documents/AFLplusplus/custom_mutators/rlm_mutator
./run_afl.sh -o grpo1 -c configs/grpo_run1.json
```

You can choose another dataset with `-i`:

```bash
./run_afl.sh -i another-dataset -o grpo2 -c configs/grpo_run1.json
```

The launcher automatically sets:

- `PYTHONPATH`
- `AFL_PYTHON_MODULE=rlm`
- `AFL_CUSTOM_MUTATOR_ONLY=1`
- `AFL_PRELOAD=exit_hook.so`
- `RLM_CONFIG` if `-c` is provided

## What Happens During A Run

The AFL++ lifecycle is:

1. `init()` loads config, trainer, tokenizer/model, reward state, and SHM.
2. `fuzz_count()` prepares the next seed and may trigger finetuning.
3. `fuzz()` generates mutations and records policy actions.
4. `post_run()` computes reward and patches the pending rollout record.
5. `maybe_finetune()` drains completed records and runs one HF `Trainer.train()` cycle.
6. `deinit()` performs one last finetune on any remaining completed records.

## Logging And Outputs

If `enable_logging` is `true`:

- HF Trainer step logs are emitted every `logging_steps`
- rollout summaries are logged before each finetune
- a CSV of per-sample rollout data is written to:

```text
/tmp/rlm_trainer/rollout_samples.csv
```

The trainer `output_dir` is currently hardcoded in [base_trainer.py](/Users/sebastianmatthews/Documents/mast/AFLplusplus/custom_mutators/rlm_mutator/base_trainer.py:63) as:

```text
/tmp/rlm_trainer
```

## GRPO-Specific Checks Already Implemented

The current code already enforces the main GRPO correctness conditions:

- `train_batch_size == group_size`
- `fuzz_count % group_size == 0`
- batches are grouped by `group_id`
- `group_id` is unique across seeds within the rollout stream
- contrastive search uses `do_sample=False`

## Common Failures

### `ModuleNotFoundError: No module named 'torch'`

Your active env is missing the mutator dependencies.

Fix:

```bash
conda activate rlm-grpo
pip install -r custom_mutators/rlm_mutator/requirements.txt
```

### `libpython3.x.so.1.0: cannot open shared object file`

Your active env does not match the Python version AFL++ was built against.

Fix:

- use the matching env
- or rebuild AFL++ inside the env you want

### `Missing exit_hook.so`

Build the shared library:

```bash
gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c
```

### `Missing dataset directory`

Make sure the dataset exists at:

```text
~/Documents/data_store/dataset/<dataset-name>
```

### `TrainingConfig.train_batch_size must equal GRPOConfig.group_size`

Set `train_batch_size` and `group_size` to the same value in the config.

### `AFLConfig.fuzz_count must be divisible by GRPOConfig.group_size`

Choose `fuzz_count` so each seed produces a whole number of GRPO groups.

## Recommended Startup Checklist

Run these in order:

```bash
conda activate rlm-grpo
python -c "import torch, transformers, accelerate, peft, numpy, tensorboard, sentencepiece, safetensors"
ldd ~/Documents/AFLplusplus/afl-fuzz | grep libpython
cd ~/Documents/AFLplusplus/custom_mutators/rlm_mutator
gcc -shared -fPIC -O2 -o exit_hook.so exit_hook.c
./run_afl.sh -o grpo1 -c configs/grpo_run1.json
```

If all of that works, you should be ready to run AFL++ with the GRPO mutator.
