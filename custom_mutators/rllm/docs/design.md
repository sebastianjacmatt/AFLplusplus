# rllm Design

Two layers. The core runs standalone as an AFL++ custom mutator. Training is layered on top and communicates through a narrow interface.

## Core (standalone)

- `rllm.py` — AFL++ API shim
- `mutator.py` — orchestrates the fuzzing campaign
- `data/masking.py` — masking strategy
- `model/llm.py` — model serving
- `model/tokenizer.py` — tokenization

These five files are a complete, runnable mutator with no training dependency.

## Training layer

- `training/covrl_trainer.py` — common interface between mutator and trainer
- `data/rollout.py` — builds training data from the AFL++ queue, scoring via `data/validity.py`
- `model/critic.py` — value network, CovRL-specific; not used by the GRPO trainer

The training layer reaches into the core only through `mutator.py` and `model/llm.py`. Everything else in training is self-contained.

## Supporting files

- `data/preprocessing.py` — preprocesses the seed queue before a run
- `data/validity.py` — extracts validity signal from program output
- `config.py` / `configs/default.json` — campaign configuration
- `run_rllm.sh` — standalone runner
- `exit_hook.c` / `exit_hook.so` / `Makefile` — captures validity at runtime, feeds `validity.py`
