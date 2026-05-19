# LM-Fuzz Interface — Implementation Plan (v3)

Plan for building the **online CovRL/RLLM trainer** on top of the existing
collection layer (masking + mutator + rollout), with explicit extension
points so the **faithful CovRL variant** (queue re-execution via afl-showmap)
drops in later, with zero edits to Phase 1 code.

This is **v3**: incorporates three corrections after a full read of CovRL's
`inferencer.py`, `do_covrl.py`, `finetuner.py`, and `critic.py`. Changes from
v2 are flagged inline with **[v3 fix]**.

Context: see `docs/lm-fuzz-interface.md` for the architectural rationale and
`docs/design.md` for the algorithm spec.

---

## Three corrections from v2 (incorporated below)

1. **Actor is skipped on cycle 0.** `inferencer.py:70-75` gates
   `finetune_actor` on `if not is_first:`. The critic still trains on cycle
   0; only the actor update is suppressed. **[v3 fix]** in
   `CovRLTrainer.finetune`.

2. **D_T accumulates across cycles in CovRL.** `finetuner.py:283-285` does
   `pd.concat([self.mutation_dataset, dataset], ...)` every cycle, so the
   training set for both critic and mutator grows monotonically with the AFL
   queue. Our `flush()` drains, which is wrong for CovRL. **[v3 fix]**
   `CovRLTrainer` keeps a `_rollout_history: list[dict]` that
   `extend()`s from `rollout.flush()` each cycle and is what gets passed to
   both training calls. RLLM/GRPO keep cycle-local rollouts (standard RL
   practice) — `_rollout_history` is CovRL-only.

3. **Critic initialises from random weights, not a pretrained backbone.**
   `critic.py:6-18` constructs a default `T5Config()` and instantiates
   `T5EncoderModel(config)` — `model_path` is accepted as an argument but
   never used at construction. Persistence is by keeping the
   `self.critic` object alive across cycles and calling `save_model()` after
   each `train_critic()`. **[v3 fix]** drop `cfg.critic_path` from the
   "checkpoint at init" path. Critic loads from `T5Config()`; runs
   `fit/save/reuse` from cycle 0.

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
| `rewarding.py` *(new)* | `Rewarder` class — CovRL Eq. 3-6 (binary TF, `sqrt(M)`, `log(N/(1+DF))`, momentum-blended IDF, α=0.6). `compute(obs)` and `observe_seed(bitmap)` + `update_cycle()` hooks. Stub `read_coverage()` raises loudly so we know when to wire SHM. `CovRL8ClassDiscretiser` (hardcoded bins, matches `base_utils.score_to_label`). | `compute()` takes an `ExecutionObservation` dataclass — same shape whether built from live SHM (online) or from `ExecutionAdapter` (faithful). |
| `corpus.py` *(new)* | `CorpusSampler(cfg, masking, executor=None)` — loads + tokenises corpus once, builds frozen program-token TF-IDF, `sample(n)` returns `list[dict]` records `{x_t, y_t, reward, group_id=-1}`. Without replacement; `n = min(4·\|D_T\|, \|C\|)`. | `executor=None` is the swap point. When provided, `_score(program)` runs original-program execution and returns real coverage TF-IDF instead of proxy. |
| `policy_gradient_algorithms/ppo.py` | `loss(ratios, R_hat, group_ids, cfg)` — PPO-clip: `-min(ρ·R̂, clip(ρ, 1±ε)·R̂).mean()`. | — |
| `policy_gradient_algorithms/grpo.py` | `loss(ratios, R_hat, group_ids, cfg)` — group-relative advantage `A = (R̂ - μ_g) / (σ_g + ε)`, corpus (`gid=-1`) gets `A=0`, same clipped objective. | Group-detection works on any record with `group_id`, regardless of source — faithful sampler can use the same loss. |
| `base_trainer.py` | Rewrite. `BaseTrainer` owns `Masking`, `LLMModel`, `CorpusSampler`, snapshots `prev_model` per cycle. **[v3 fix]** `CovRLTrainer` adds **`_rollout_history`** (accumulates across cycles) and persistent critic initialised from `T5Config()`. **[v3 fix]** Mutator update gated on `self._cycle > 0`. `RLLMTrainer` uses direct reward, drains rollout per-cycle (no history). | `BaseTrainer.finetune(rollout)` signature unchanged but body is free to ignore `rollout` and pull records from elsewhere — exactly what the faithful trainer does later. |
| `mutator.py` | Lazy `_build_trainer` (only on first `_finetune_pending`); pass `self.masking` + `self.model` to trainer constructor (no second model load). | — |
| `utils.py` | Gut: remove `mix_corpus`, `calc_reward`, `_COV_TFIDF` globals. Their behaviour moves to `rewarding.py` + `corpus.py` as proper class state. | — |
| `config.py` | Add `corpus_dir` validation (raise if empty when trainer needs it), `clip_epsilon` (PPO/GRPO clip bound), `kl_coef=0.0` (deferred but field exists), `idf_alpha=0.6`. **[v3 fix]** Drop `critic_path` — critic always from `T5Config()`. | — |

### How the trainer composes — **[v3 fix]**

```python
class BaseTrainer:
    def __init__(self, cfg, model, masking, policy_loss):
        self.cfg, self.model, self.masking = cfg, model, masking
        self.policy_loss = policy_loss
        self.corpus      = CorpusSampler(cfg, masking)   # executor=None → proxy
        self.prev_model  = None
        self._cycle      = 0


class CovRLTrainer(BaseTrainer):
    def __init__(self, cfg, model, masking, policy_loss):
        super().__init__(cfg, model, masking, policy_loss)
        # [v3 fix #3] critic from random T5Config(), not a pretrained backbone.
        # Persistence is via keeping self.critic alive across finetune() calls.
        self.critic           = self._init_critic_from_config()
        # [v3 fix #2] D_T accumulates monotonically across cycles in CovRL.
        # finetuner.py:283-285 does pd.concat([mutation_dataset, dataset]).
        self._rollout_history: list[dict] = []

    def finetune(self, rollout):
        # [v3 fix #2] Drain into the growing history, don't train on the
        # current cycle's records alone.
        self._rollout_history.extend(rollout.flush())

        # Critic trains every cycle, including cycle 0.
        self._train_critic(RolloutDataset(self._rollout_history))

        # [v3 fix #1] Actor is skipped on cycle 0 (inferencer.py:70-75).
        if self._cycle > 0:
            n_corpus = min(4 * len(self._rollout_history), self.corpus.size)
            ds = RolloutDataset(
                self._rollout_history + self.corpus.sample(n_corpus)
            )
            self._snapshot_prev()
            self._train_mutator(ds, reward_fn=self._critic_reward)

        self._cycle += 1


class RLLMTrainer(BaseTrainer):
    # RLLM/GRPO use cycle-local rollouts — no _rollout_history.
    # Standard on-policy RL practice; matches the design doc's Finetune body.
    def finetune(self, rollout):
        records  = rollout.flush()
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

1. `policy_gradient_algorithms/ppo.py` + `grpo.py` (no deps, unit-testable)
2. `rewarding.py` — `Rewarder`, `CovRL8ClassDiscretiser`,
   `ExecutionObservation`; SHM reader as a `NotImplementedError` stub
3. `corpus.py` — `CorpusSampler` with proxy reward
4. `base_trainer.py` rewrite — `BaseTrainer`, `CovRLTrainer`
   (with `_rollout_history` and cycle-0 gate), `RLLMTrainer`
5. `mutator.py` — lazy `_build_trainer`; pass `masking` + `model` through
6. `utils.py` cleanup — remove dead code
7. `config.py` — new fields + validators
8. Run script + config JSON for end-to-end smoke test (separate from the
   pure-mutation script we already planned)

### Extension points explicitly preserved

| Phase 2 needs… | Phase 1 leaves a hook |
|---|---|
| Trainer that ignores `rollout` and uses a sampler | `BaseTrainer.finetune(rollout)` signature — trainer body decides whether to drain |
| Real corpus coverage instead of proxy | `CorpusSampler(cfg, masking, executor=None)` — pass an executor to swap |
| Reward from a bitmap rather than live SHM | `Rewarder.compute(ExecutionObservation)` — observation is what changes, not the rewarder |
| Critic and actor share machinery | `Masking` + `LLMModel` are dependency-injected; future `LLMMutationSampler` takes the same instances |
| Loss functions group-aware | `policy_loss(ratios, R_hat, group_ids, cfg)` — `group_ids` already in the contract; GRPO uses it, PPO ignores it |
| CovRL's accumulating D_T | `CovRLTrainer._rollout_history` already there; faithful CovRL drops the `rollout` arg and replaces the `extend()` source with `LLMMutationSampler.process(queue.snapshot())` |

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
        self.critic            = self._init_critic_from_config()
        self.executor          = ExecutionAdapter(cfg.showmap_path, cfg.target_path)
        self.queue             = QueueReader(cfg.afl_out_dir)
        self.sampler           = LLMMutationSampler(masking, model, self.executor)
        self.corpus            = CorpusSampler(cfg, masking, executor=self.executor)
        self._rollout_history  : list[dict] = []     # same accumulation as online CovRL

    def finetune(self, _rollout):                    # ignores online buffer
        programs = self.queue.snapshot()
        self._rollout_history.extend(self.sampler.process(programs))

        self._train_critic(RolloutDataset(self._rollout_history))

        if self._cycle > 0:
            n_corpus = min(4 * len(self._rollout_history), self.corpus.size)
            ds = RolloutDataset(
                self._rollout_history + self.corpus.sample(n_corpus)
            )
            self._snapshot_prev()
            self._train_mutator(ds, reward_fn=self._critic_reward)

        self._cycle += 1
```

Compare with `CovRLTrainer.finetune` — same shape, the only differences are
the source feeding `_rollout_history.extend(...)` and the corpus getting an
`executor=`.

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

## Decision checkpoints

1. **`utils.py`** → gut now (clean). Nothing outside this directory imports it.
2. **`Rewarder.snapshot_bitmap()`** stub → raise `NotImplementedError`
   (surface bugs, don't pre-defend).
3. **Critic backbone** → **[v3 fix]** `T5Config()` random init. No
   `cfg.critic_path` — CovRL never loaded one.
4. **GRPO + corpus mixing** → defer; raise loudly if config requests both.
   Phase-2-and-a-half problem.
5. **HF Trainer** → keep (lazy custom `_MutatorTrainer(Trainer)` subclass for
   `compute_loss`).

---

## Faithfulness scorecard (vs CovRL reference implementation)

| Behaviour | CovRL code | This plan |
|---|---|---|
| Critic trains every cycle (incl. cycle 0) | `inferencer.py:71` | `_train_critic` always called in `finetune()` |
| Actor skipped on cycle 0 | `inferencer.py:72-75` | **[v3 fix]** `if self._cycle > 0:` gate |
| D_T accumulates across cycles | `finetuner.py:283-285` | **[v3 fix]** `_rollout_history.extend(...)` |
| 8-class non-uniform bins | `base_utils.score_to_label` | `CovRL8ClassDiscretiser` matches exactly |
| IDF momentum α=0.6 | `update_idf(alpha=cfg.alpha)` | `Rewarder.update_cycle` blends with α |
| Corpus ratio 4·\|D_T\|, no replacement | `train_dataset.sample(min(4n, \|C\|))` | `min(4 * len(_history), corpus.size)` |
| Persistent critic | `self.critic` lives on `FineTuner` | `CovRLTrainer.critic` persists |
| Critic init | `T5Config()`, random weights | **[v3 fix]** `_init_critic_from_config()` |
| PPO loss = policy + CE | `ppo_loss + cur_outputs.loss.mean()` | `policy_loss + CE` in `_train_mutator` |
| Corpus reward source | real coverage via afl-showmap | **deliberate deviation**: proxy via program-token TF-IDF; flagged in `CorpusSampler._score`; upgradeable to real via `executor=` |

The only deliberate divergence is the corpus reward source — every other
CovRL behaviour ports faithfully. The corpus deviation is encapsulated in
one method (`CorpusSampler._score`) and one constructor arg (`executor=`),
so reproducing CovRL exactly later is a one-line change.

---

## Next step

If decision checkpoints stand as listed, implementation order step 1 begins:
`policy_gradient_algorithms/ppo.py` + `grpo.py`.
