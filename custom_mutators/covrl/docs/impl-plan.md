# Implementation Plan

Plan for building the **online CovRL/RLLM trainer** on top of the existing
collection layer (masking + mutator + rollout), with explicit extension
points so the **faithful CovRL variant** (queue re-execution via afl-showmap)
later drops in additively, with zero edits to Phase 1 code.

Context: see `docs/lm-fuzz-interface.md` for the architectural rationale and
`docs/design.md` for the algorithm spec.

---

## Overview

The online path is mostly additive — the collection layer (masking / mutator /
rollout) is in place and unchanged. We add a `Rewarder`, a `CorpusSampler`, a
reward discretiser, two policy-loss functions, and the actual `CovRLTrainer` /
`RLLMTrainer` bodies. The faithful variant later adds an `ExecutionAdapter` +
`QueueReader` + `LLMMutationSampler` and a `CovRLFaithfulTrainer` — zero
changes to anything we build in Phase 1.

---

## Phase 1 — Online path (now)

### What gets added / changed

| File | Change | Designed-in extension point |
|---|---|---|
| `rewarding.py` *(new)* | `Rewarder` class — CovRL Eq. 3-6 (binary TF, `sqrt(M)`, `log(N/(1+DF))`, momentum-blended IDF). `compute(obs)` and `observe_seed(bitmap)` + `update_cycle()` hooks. Stub `read_coverage()` raises loudly so we know when to wire SHM. `CovRL8ClassDiscretiser` (hardcoded bins). | `compute()` takes an `ExecutionObservation` dataclass — same shape whether built from live SHM (online) or from `ExecutionAdapter` (faithful). |
| `corpus.py` *(new)* | `CorpusSampler(cfg, masking, executor=None)` — loads + tokenises corpus once, builds frozen program-token TF-IDF, `sample(n)` returns `list[dict]` records `{x_t, y_t, reward, group_id=-1}`. | The `executor=None` arg is the swap point. When provided, `_score(program)` runs original-program execution and returns real coverage TF-IDF instead of proxy. |
| `policy_gradient_algorithms/ppo.py` | `loss(ratios, R_hat, group_ids, cfg)` — PPO-clip: `-min(ρ·R̂, clip(ρ, 1±ε)·R̂).mean()`. | — |
| `policy_gradient_algorithms/grpo.py` | `loss(ratios, R_hat, group_ids, cfg)` — group-relative advantage `A = (R̂ - μ_g) / (σ_g + ε)`, corpus (`gid=-1`) gets `A=0`, same clipped objective. | Group-detection works on any record with `group_id`, regardless of source — faithful sampler can use the same loss. |
| `base_trainer.py` | Rewrite. `BaseTrainer` owns `Masking`, `LLMModel`, `CorpusSampler`, snapshots `prev_model` per cycle. `CovRLTrainer` adds persistent critic (`AutoModelForSequenceClassification`, same arch as actor, fresh head). `RLLMTrainer` uses direct reward. | `BaseTrainer.finetune(rollout)` signature unchanged but the body is free to ignore `rollout` and pull records from elsewhere — exactly what the faithful trainer does later. |
| `mutator.py` | Lazy `_build_trainer` (only on first `_finetune_pending`); pass `self.masking` + `self.model` to trainer constructor (so trainer reuses them, no second model load). | — |
| `utils.py` | Gut: remove `mix_corpus`, `calc_reward`, `_COV_TFIDF` globals. Their behaviour moves to `rewarding.py` + `corpus.py` as proper class state. | — |
| `config.py` | Add `critic_path`, `corpus_dir` validation (raise if empty when trainer needs it), `clip_epsilon` (PPO/GRPO clip bound), `kl_coef=0.0` (deferred but field exists). | — |

### How the trainer composes

```python
class BaseTrainer:
    def __init__(self, cfg, model, masking, policy_loss):
        self.cfg, self.model, self.masking = cfg, model, masking
        self.policy_loss = policy_loss
        self.corpus      = CorpusSampler(cfg, masking)   # executor=None → proxy reward
        self.prev_model  = None
        self._cycle      = 0

class CovRLTrainer(BaseTrainer):
    def __init__(self, cfg, model, masking, policy_loss):
        super().__init__(cfg, model, masking, policy_loss)
        self.critic = self._load_or_init_critic()        # persistent

    def finetune(self, rollout):
        records = rollout.flush()
        self._train_critic(RolloutDataset(records))      # rollouts only
        n_corpus = min(4 * len(records), self.corpus.size)
        ds = RolloutDataset(records + self.corpus.sample(n_corpus))
        self._snapshot_prev()
        self._train_mutator(ds, reward_fn=self._critic_reward)
        self._cycle += 1

class RLLMTrainer(BaseTrainer):
    def finetune(self, rollout):
        records = rollout.flush()
        n_corpus = min(4 * len(records), self.corpus.size)
        ds = RolloutDataset(records + self.corpus.sample(n_corpus))
        self._snapshot_prev()
        self._train_mutator(ds, reward_fn=lambda batch: batch["reward"])
        self._cycle += 1
```

### How the mutator integrates (online reward path)

`mutator.post_run` builds an `ExecutionObservation` from live state:

```python
def post_run(self):
    if not self._has_pending_sample: return
    self._has_pending_sample = False
    x, y, out, gid = self._last
    obs = ExecutionObservation(
        bitmap   = self.rewarder.snapshot_bitmap(),   # stub raises until SHM wired
        exit_code= None,
        validity = None,
    )
    R = self.rewarder.compute(obs)
    self.rollout.add(x, y, R, group_id=gid)
    if self.collection == "all":
        self.rollout.commit_last()
```

The faithful variant builds the **same** `ExecutionObservation` from
`ExecutionAdapter.execute(infilled_bytes)`. Same `Rewarder.compute()` call.
Same downstream code.

### Implementation order

1. `policy_gradient_algorithms/ppo.py` + `grpo.py` (no deps, easy to unit test)
2. `rewarding.py` — `Rewarder`, `CovRL8ClassDiscretiser`, `ExecutionObservation`; SHM reader as a `NotImplementedError` stub
3. `corpus.py` — `CorpusSampler` with proxy reward
4. `base_trainer.py` rewrite — `BaseTrainer`, `CovRLTrainer`, `RLLMTrainer`
5. `mutator.py` — lazy `_build_trainer`; pass `masking` + `model` through
6. `utils.py` cleanup — remove dead code
7. `config.py` — new fields + validators
8. Run script + config JSON for end-to-end smoke test (separate from the pure-mutation script we already planned)

### Extension points explicitly preserved

| Phase 2 needs… | Phase 1 leaves a hook |
|---|---|
| Trainer that ignores `rollout` and uses a sampler | `BaseTrainer.finetune(rollout)` signature — trainer body decides whether to drain |
| Real corpus coverage instead of proxy | `CorpusSampler(cfg, masking, executor=None)` — pass an executor to swap |
| Reward from a bitmap rather than live SHM | `Rewarder.compute(ExecutionObservation)` — observation is what changes, not the rewarder |
| Critic and actor share machinery | `Masking` + `LLMModel` are dependency-injected; future `LLMMutationSampler` takes the same instances |
| Loss functions group-aware | `policy_loss(ratios, R_hat, group_ids, cfg)` — `group_ids` already in the contract; GRPO uses it, PPO ignores it |

---

## Phase 2 — Faithful CovRL (later, purely additive)

### What gets added

| File | Adds |
|---|---|
| `execution.py` *(new)* | `ExecutionAdapter(showmap_path, target_path)` — `execute(input_bytes) -> ExecutionObservation`. Subprocess wrapper around `afl-showmap`. Stateless, thread-safe. |
| `queue_reader.py` *(new)* | `QueueReader(afl_out_dir)` — yields `(seed_id, bytes)` from `out/queue/`. Watches mtime so re-runs see new entries. |
| `samplers.py` *(new)* | `LLMMutationSampler(masking, model, executor)` — takes programs, runs mask + `LLMModel.batch_generate` + `executor.execute`, returns records. Reuses **exactly** the same machinery as fuzz-time mutator. |
| `base_trainer.py` | New subclass `CovRLFaithfulTrainer(BaseTrainer)`. No edits to existing trainers. |

### `CovRLFaithfulTrainer.finetune`

```python
class CovRLFaithfulTrainer(BaseTrainer):
    def __init__(self, cfg, model, masking, policy_loss):
        super().__init__(cfg, model, masking, policy_loss)
        self.critic    = self._load_or_init_critic()
        self.executor  = ExecutionAdapter(cfg.showmap_path, cfg.target_path)
        self.queue     = QueueReader(cfg.afl_out_dir)
        self.sampler   = LLMMutationSampler(masking, model, self.executor)
        self.corpus    = CorpusSampler(cfg, masking, executor=self.executor)  # real reward

    def finetune(self, _rollout):                       # ignores online buffer
        programs = self.queue.snapshot()
        records  = self.sampler.process(programs)       # mask + infill + execute
        self._train_critic(RolloutDataset(records))
        n_corpus = min(4 * len(records), self.corpus.size)
        ds = RolloutDataset(records + self.corpus.sample(n_corpus))
        self._snapshot_prev()
        self._train_mutator(ds, reward_fn=self._critic_reward)
        self._cycle += 1
```

Compare to `CovRLTrainer.finetune` above — same shape, four lines differ
(where `records` comes from, and corpus gets `executor=`).

### Optional: skip online collection when running faithful

To avoid paying for fuzz-time rollout storage we'll never use, add a single
config flag:

```python
# config.py
online_collection: bool = True   # False for faithful trainer
```

In `mutator.post_run`: if `not cfg.online_collection`, skip the `rollout.add`
call. Three lines. Means the faithful path can run with zero in-memory rollout
cost.

---

## Decision checkpoints before starting

1. **`utils.py` deletion vs deprecation** — gut now (clean) or leave a
   back-compat shim? **Recommend gut.** Nothing outside this directory imports
   it.
2. **`Rewarder.snapshot_bitmap()` behaviour** when SHM isn't wired — raise
   `NotImplementedError` (loud) or return zeros (silent)? **Recommend raise**
   — surface bugs, don't pre-defend.
3. **Critic backbone path** — same as actor (`cfg.model_path`) or separate
   (`cfg.critic_path`)? CovRL uses same. **Default to `cfg.critic_path or
   cfg.model_path`** so users can override.
4. **`GroupedBatchSampler` + corpus** — defer (raise if `policy_loss=grpo`
   and `corpus_mix_ratio > 0`) or implement the special-cased "corpus fills
   batch remainder" sampler now? **Recommend defer** with a clear error.
   Phase-2-and-a-half problem.
5. **HF Trainer vs custom training loop** — keep the lazy custom
   `_MutatorTrainer(Trainer)` subclass we have now, extended for critic. Or
   factor out HF Trainer entirely? **Stay with HF Trainer**; the lazy custom
   subclass is fine.

If those defaults look right (`#1 gut`, `#2 raise`, `#3 fallback`, `#4 defer`,
`#5 stay`), start with step 1 of the implementation order (`ppo.py` +
`grpo.py` losses) and work through. Otherwise flip the relevant choice first.