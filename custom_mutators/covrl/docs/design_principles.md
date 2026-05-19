# Design Principles

## Goal

Implement the four experimental variants from the design doc in a way where:
- each component can be tested in isolation
- swapping collection strategy, reward source, or policy loss requires changing config, not code
- the two AFL entry-point files (`covrl.py`, `rllm.py`) contain no business logic

---

## AFL hook responsibilities

AFL calls hooks in this order per queue entry:

```
queue_get(filename)                    ← decide whether to fuzz; trigger finetuning
fuzz_count(buf)                        ← tokenize + mask buf; return CFG.fuzz_count
for each of fuzz_count:
    fuzz(buf, add_buf, max_size)       ← run LLM inference on prepared mask; return mutation
    [AFL executes target]
    post_run()                         ← read coverage; compute reward; maybe commit to D_T
    [if new coverage:]
        queue_new_entry(new, orig)     ← commit pending entry to D_T (CollectInteresting only)
deinit()                               ← flush rollout buffer; save checkpoint
```

**Each hook owns exactly one concern:**

| Hook | Owns |
|---|---|
| `queue_get` | Finetuning trigger (every N entries) |
| `fuzz_count` | Tokenisation + mask preparation; sets batch size |
| `fuzz` | One LLM inference call per mask sample |
| `post_run` | Coverage read + reward computation + rollout commit (CollectAll) |
| `queue_new_entry` | Rollout commit (CollectInteresting only) |
| `deinit` | Checkpoint save |

---

## Module structure

```
covrl/
├── covrl.py                  ← AFL entry point: CollectInteresting + FinetuneCovRL
├── rllm.py                   ← AFL entry point: CollectAll + FinetuneRLLM (PPO or GRPO)
│
├── config.py                 ← Config dataclass; loaded from JSON
├── mutator.py                ← Mutator class: owns masking + LLM inference
├── rollout.py                ← RolloutBuffer: accumulates (x, y, R) triples
├── base_trainer.py           ← BaseTrainer + CovRLTrainer + RLLMTrainer
│
├── masking.py                ← Tokenisation, T5 span masking, reconstruction
├── utils.py                  ← TF-IDF reward, IDF update, validity check
│
└── policy_gradient_algorithms/
    ├── ppo.py                ← PPO-clip loss
    └── grpo.py               ← GRPO loss (group-relative advantage)
```

---

## The two entry-point files are thin shims

`covrl.py` and `rllm.py` should contain only AFL hook functions that delegate to a shared `Mutator` instance. No logic lives here — only wiring.

```python
# covrl.py
from covrl.mutator import Mutator
from covrl.config import Config

_m = Mutator(Config.load(), collection="interesting", trainer="covrl")

def init(seed):         _m.init(seed)
def fuzz_count(buf):    return _m.fuzz_count(buf)
def fuzz(buf, add_buf, max_size): return _m.fuzz(buf, add_buf, max_size)
def post_run():         _m.post_run()
def queue_new_entry(new, orig): _m.queue_new_entry(new, orig)
def queue_get(filename): return _m.queue_get(filename)
def deinit():           _m.deinit()
```

```python
# rllm.py
from covrl.mutator import Mutator
from covrl.config import Config

_m = Mutator(Config.load(), collection="all", trainer="rllm")
# ... same hook wiring
```

The `collection` and `trainer` arguments are the only difference between the two files.

---

## The collection split lives in RolloutBuffer

The distinction between CollectInteresting and CollectAll is a single decision:
**when does a rollout entry get committed to D_T?**

`RolloutBuffer` has two methods:

```python
class RolloutBuffer:
    def add(self, x, y, R):
        """Stage a completed rollout. Always called from post_run."""

    def commit_last(self):
        """Move the last staged entry into D_T. Called from queue_new_entry (CollectInteresting)
        or immediately after add() (CollectAll)."""
```

`Mutator` calls `commit_last()` at different points depending on the `collection` arg passed at construction. Nothing else in the codebase changes between the two collection strategies.

---

## Training is fully shared

`BaseTrainer` defines the finetuning contract:

```python
class BaseTrainer:
    def finetune(self, dataset: RolloutBuffer) -> None:
        raise NotImplementedError
```

`CovRLTrainer` (used by `covrl.py`):
1. Mix corpus into dataset
2. Train rewarder (8-class critic) on D_T
3. If t > 0: train mutator with PPO loss, using critic predictions as R̂

`RLLMTrainer` (used by `rllm.py`):
1. Mix corpus into dataset
2. Train mutator with PPO or GRPO loss, using stored direct rewards as R̂

The policy loss is injected:

```python
class RLLMTrainer(BaseTrainer):
    def __init__(self, policy_loss):  # ppo.loss or grpo.loss
        ...
```

This means all four variants from the design doc map to:

| Variant | entry point | collection arg | trainer arg | policy_loss arg |
|---|---|---|---|---|
| CovRL | `covrl.py` | `"interesting"` | `"covrl"` | `ppo` |
| CovRL-All | `covrl.py` (with config override) | `"all"` | `"covrl"` | `ppo` |
| RLLM-PPO | `rllm.py` | `"all"` | `"rllm"` | `ppo` |
| RLLM-GRPO | `rllm.py` | `"all"` | `"rllm"` | `grpo` |

---

## Masking is stateful per fuzz_count/fuzz cycle

`fuzz_count(buf)` tokenises `buf` and prepares `CFG.fuzz_count` independent span masks.
Each subsequent `fuzz()` call pops one mask and runs LLM inference on it.
State is held in `Mutator`, not in module globals.

```
fuzz_count(buf)
  → Masking.tokenise(buf) → [mask_1, mask_2, ..., mask_N]
  → store in self._pending_masks

fuzz(buf, ...)
  → mask = self._pending_masks.pop()
  → LLM.generate(masked_input, mask)
  → return reconstructed bytes
```

This ensures `fuzz` is stateless from AFL's perspective (it can be called any number of times) while keeping mask preparation cheap (done once per seed).

---

## Testing strategy

Because each component is a class with a clear interface, each can be tested without AFL:

| Component | How to test |
|---|---|
| `Masking` | Unit test: given a token sequence, check masked input + label format |
| `RolloutBuffer` | Unit test: verify commit semantics for both collection strategies |
| `CovRLTrainer` | Integration test: construct a small D_T, run finetune, check loss decreases |
| `RLLMTrainer` | Same as above, swap policy loss |
| `ppo.loss` / `grpo.loss` | Unit test: known ratio + reward → expected loss value |
| Full mutator | End-to-end: mock AFL hooks, run a short fuzz loop, inspect D_T contents |

AFL integration tests (calling `covrl.py` / `rllm.py` directly via `AFL_PYTHON_MODULE`) should be separate from unit tests and only run against a real target binary.
