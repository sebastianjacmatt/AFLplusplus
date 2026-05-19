# LM-Fuzz Interface — Implementation Plan (v4)

Plan for building the **online CovRL/RLLM trainer** on top of the existing
collection layer (Masking + Mutator + Rollout), with explicit extension points
so the **faithful CovRL variant** (queue re-execution via afl-showmap) drops
in later with zero edits to Phase 1 code.

This is **v4**: a full re-write after a second pass through CovRL's
`finetuner.py`, `inferencer.py`, `critic.py`, and `critic_dataset.py`. Five
additional corrections beyond the three v3 picked up. Changes since v3 are
flagged with **[v4 fix]**.

Context: see `docs/design.md` for the algorithm spec.

---

## Corrections incorporated into v4

The v3 fixes are still correct and retained:

1. **Actor is skipped on cycle 0** — `inferencer.py:71-74` gates
   `finetune_actor` on `if not is_first:`.
2. **D_T accumulates across cycles** — `finetuner.py:283-285`
   `pd.concat([mutation_dataset, dataset])`.
3. **Critic initialises from random `T5Config()` weights** —
   `critic.py:16-18`; the `model_path` arg is dead code.

New in v4:

4. **[v4 fix] Critic trains on the same dataset as the mutator** —
   `finetuner.py:117-119` `train_critic` reads `self.dataset`, which is
   `corpus + accumulated mutations` on cycle 1+ and `corpus only` on cycle 0.
   v3 trained the critic on `_rollout_history` alone; that was wrong. The
   critic and mutator now share one dataset built once per cycle.
5. **[v4 fix] Critic input is `concat(x_t, y_t)` on the sequence axis** —
   `critic_dataset.py:55-57`. Not two separate sequences. Attention mask
   is `ones(|x_t|) + ones(|y_t|)`.
6. **[v4 fix] Critic uses the encoder's CLS hidden state for classification**
   — `critic.py:51` `hidden_states[:, 0, :]`. Not mean-pooled.
7. **[v4 fix] Critic forward returns `SequenceClassifierOutput`, not a bare
   tuple.** HF Trainer expects `.loss`/`.logits` attributes; CovRL's raw
   `(loss, logits)` return likely has a quiet bug.
8. **[v4 fix] PPO clip ε = 0.2** — `finetuner.py:168` `clamp(ratio, 0.8, 1.2)`.
   Config default updated.

Deliberate deviations from CovRL (not fixes — design choices, called out so
they don't masquerade as faithful):

- **Corpus reward source** — proxy program-token TF-IDF, not real coverage
  via `afl-showmap`. One swap point: `CorpusSampler(..., executor=)`.
- **`_rollout_history` source** — LLM-generated rollouts only, not the full
  AFL queue. CovRL's `load_files(out_dir)` reads every queue entry; we only
  see what `Mutator.fuzz` produced. For single-mutator AFL++ runs the gap is
  small. Upgradeable via the same Phase 2 `QueueReader`.
- **Critic backbone size** — `T5Config()` default (d_model=512), not
  CodeT5+ encoder (d_model=768). Matches code, deviates from paper text.

---

## Layered architecture

```
┌──────────────────────────────────────────────────────────┐
│ Entry-points       covrl.py     rllm.py                  │  thin AFL shims
├──────────────────────────────────────────────────────────┤
│ Orchestrator       Mutator                               │  AFL hook impl
├──────────────────────────────────────────────────────────┤
│ Generation         Masking      LLMModel                 │  language model
├──────────────────────────────────────────────────────────┤
│ Data               RolloutBuffer (D_T staging)           │
│                    RolloutDataset / Collator / Sampler   │  HF wiring
├──────────────────────────────────────────────────────────┤
│ Training           BaseTrainer  CovRLTrainer  RLLMTrainer│
│                    CorpusSampler                         │
│                    rewarding/  policy_gradient_algorithms│
└──────────────────────────────────────────────────────────┘
```

Each layer has one contract; variants are configuration, not edits.
Import direction is strictly top-down — entry-points import everything,
training imports nothing above Data, Data imports nothing above Generation.

---

## Layer 1 — Masking

Owns tokenisation, T5 span corruption, target construction, generation budget
calculation, reconstruction.

```python
class Masking:
    tokenizer: PreTrainedTokenizer        # shared with LLMModel
    def tokenize(self, buf: bytes) -> list[int]
    def mask(self, tokens: list[int]) -> MaskedProgram
    def target_ids(self, mp: MaskedProgram) -> list[int]
    def generation_budget(self, mp, max_new_tokens_per_span) -> int
    def reconstruct(self, mp, generated_ids) -> list[int]
    def decode(self, mp, generated_ids) -> bytes
    def batch_decode(self, mps, generated_ids_list) -> list[bytes]
```

`MaskedProgram` carries `original_ids`, `input_ids`, `spans`.

Invariants: `target_ids(mp)` reproduces the T5 supervised target for `mp`,
used by both corpus mixing and critic-input construction. `reconstruct`
accepts cleaned decoder output; `LLMModel` does the cleaning.

---

## Layer 2 — LLMModel

Owns the HF seq2seq actor and one batched-generate primitive.

```python
class LLMModel:
    tokenizer: PreTrainedTokenizer        # supplied by Mutator from Masking
    model: PreTrainedModel
    device: str
    def batch_generate(self, xs, n_samples, max_new_tokens) -> list[list[int]]
    def save(self, path)
```

`batch_generate` runs one `model.generate(num_return_sequences=n_samples)`
across all `xs`. Returns `len(xs) * n_samples` cleaned token-id lists
(decoder_start stripped, truncated at first EOS). Order is HF-standard.

Cleaning happens here once; raw `y_t` in storage would make CE supervise
decoder-start emission.

---

## Layer 3 — RolloutBuffer (D_T staging)

Per-sample records, schema extensible. Stages with `add`; promotes with
`commit_last`; drains with `flush`.

```python
class RolloutBuffer:
    def add(self, x_t, y_t, reward, group_id, **extra) -> None
    def commit_last(self) -> None
    def flush(self) -> list[dict]
    def __len__(self) / __iter__(self)
```

Record shape:

```python
{
    "x_t":      list[int],            # encoder input ids (with sentinels)
    "y_t":      list[int],            # cleaned T5 target form
    "reward":   float,                # in {-1.0, -0.5} ∪ [0, 1]
    "group_id": int,                  # -1 for corpus, ≥0 for GRPO groups
}
```

Collection strategy is encoded *only* by when the orchestrator calls
`commit_last`:

| Collection | `commit_last` fires |
|---|---|
| `"all"` | in `post_run`, immediately after `add` |
| `"interesting"` | in `queue_new_entry` (AFL fires it only for interesting runs) |

---

## Layer 4 — Dataset / Collator / Sampler

Bridges `flush()` output into HF batches.

```python
class RolloutDataset(torch.utils.data.Dataset):       # wraps list[dict]
class RolloutCollator:                                 # pads, builds batch
class GroupedBatchSampler(torch.utils.data.Sampler):   # complete GRPO groups
```

**Critic and actor share `RolloutCollator`** — but read different fields out
of the same batch. Both forms are produced by the same collator pass.

Actor batch:
- `input_ids`         = padded `x_t`
- `attention_mask`    = ones over `x_t`, pad → 0
- `labels`            = padded `y_t`, pad → -100
- `reward`, `group_id`

Critic batch (built lazily by `_train_critic`, see below):
- `input_ids`         = padded `concat(x_t, y_t)`     **[v4 fix #5]**
- `attention_mask`    = `ones(|x_t|) + ones(|y_t|)`, pad → 0
- `labels`            = discretised reward class (int 0..7)

The collator produces both forms from the same record; the trainer picks the
view it needs.

`group_id` is always present in batches. PPO ignores it; GRPO reads it.

---

## Layer 5 — Mutator (orchestrator)

```python
class Mutator:
    masking: Masking
    model:   LLMModel
    rollout: RolloutBuffer
    trainer: BaseTrainer | None        # None in pure-mutation mode

    # AFL hooks
    def init(seed)
    def queue_get(filename)            # bumps counter, sets _finetune_pending
    def fuzz_count(buf)                # drains _finetune_pending; one batched generate
    def fuzz(buf, add_buf, max_size)   # pops queued sample, sets _has_pending_sample
    def post_run()                     # gated on _has_pending_sample
    def queue_new_entry(new, orig)     # commit_last if collection == "interesting"
    def deinit()                       # flush, save model
```

`Mutator(cfg, collection: "interesting" | "all", trainer: "covrl" | "rllm")`.

**One generate per `fuzz_count` cycle.** All masks sampled first; one
`batch_generate(xs, n_samples=group_size, max_new_tokens=max(budgets))`
covers PPO (group_size=1) and GRPO (group_size>1).

**`post_run` is gated on `_has_pending_sample`** so calibration / dry-run /
trim runs don't corrupt `_last`.

---

## Layer 6 — Training

### BaseTrainer

```python
class BaseTrainer:
    cfg:         Config
    model:       LLMModel              # the actor
    masking:     Masking
    policy_loss: Callable              # ppo.loss | grpo.loss
    corpus:      CorpusSampler
    prev_model:  PreTrainedModel | None  # π_prev; snapshotted on cadence
    _cycle:      int

    def finetune(self, rollout: RolloutBuffer): raise NotImplementedError
    def _snapshot_prev()               # honours cfg.ref_update_every
    def _train_mutator(records, reward_fn)
        # HF Trainer with custom compute_loss:
        #   forward θ + π_prev → log_probs
        #   ratio = exp(log_p_θ - log_p_prev)
        #   R̂ = reward_fn(batch)
        #   L = policy_loss(ratios, R̂, group_id, cfg) + CE(θ, labels)
```

### CovRLTrainer — **[v4 fix #4]** corrected dataset construction

```python
class CovRLTrainer(BaseTrainer):
    def __init__(self, cfg, model, masking, policy_loss):
        super().__init__(cfg, model, masking, policy_loss)
        # [v3 fix #3] critic from random T5Config(); persistence via keeping
        # self.critic alive across finetune() calls.
        # [v4 fix #7] returns SequenceClassifierOutput.
        self.critic = self._init_critic_from_config()
        # [v3 fix #2] D_T accumulates monotonically across cycles.
        self._rollout_history: list[dict] = []

    def finetune(self, rollout):
        # [v3 fix #2] grow the history, don't drain.
        self._rollout_history.extend(rollout.flush())

        # [v4 fix #4] build the SAME dataset once, feed to both training calls.
        # Matches finetuner.py: self.dataset is read by both train_critic
        # and finetune_actor.
        n = max(1, len(self._rollout_history))
        n_corpus = min(4 * n, self.corpus.size)
        corpus_records = self.corpus.sample(n_corpus)

        if self._cycle == 0:
            # CovRL cycle-0: self.dataset = corpus only (mutations are held
            # back in mutation_dataset and only flow into self.dataset on
            # cycle 1+). See finetuner.py:271-281.
            shared_records = corpus_records
        else:
            shared_records = self._rollout_history + corpus_records

        ds = RolloutDataset(shared_records)

        # Critic trains every cycle, including cycle 0.
        self._train_critic(ds)

        # [v3 fix #1] Actor is skipped on cycle 0.
        if self._cycle > 0:
            self._snapshot_prev()
            self._train_mutator(ds, reward_fn=self._critic_reward)

        self._cycle += 1

    def _train_critic(self, ds: RolloutDataset):
        # HF Trainer over the critic view of the batch:
        #   input_ids = concat(x_t, y_t)              [v4 fix #5]
        #   labels    = discretiser.reward_to_class(record["reward"])
        # critic.forward returns SequenceClassifierOutput.   [v4 fix #7]

    def _critic_reward(self, batch) -> torch.Tensor:
        # finetuner.py:154-159 — predicted class → discretised score.
        with torch.no_grad():
            logits = self.critic(
                input_ids      = batch["critic_input_ids"],
                attention_mask = batch["critic_attention_mask"],
            ).logits
        cls = logits.argmax(dim=-1)
        return self.discretiser.class_to_reward(cls)
```

### RLLMTrainer

```python
class RLLMTrainer(BaseTrainer):
    # RLLM/GRPO: cycle-local rollouts (no _rollout_history).
    # Standard on-policy RL; matches design doc's Finetune body.
    def finetune(self, rollout):
        records  = rollout.flush()
        n_corpus = min(4 * max(1, len(records)), self.corpus.size)
        ds = RolloutDataset(records + self.corpus.sample(n_corpus))
        self._snapshot_prev()
        self._train_mutator(ds, reward_fn=lambda batch: batch["reward"])
        self._cycle += 1
```

**Why two trainer classes** (not one parameterised class): they differ in
three orthogonal axes (history vs cycle-local, critic vs direct reward,
cycle-0 gate). A single class would need three flags whose interactions
aren't obvious. Two subclasses are ~30 lines each; the cross-axis combos
that matter (RLLM-GRPO, KL-anchored RLLM) all live in
`{cfg.policy_loss, cfg.ref_update_every}`.

---

## CorpusSampler

```python
class CorpusSampler:
    """Loads + tokenises corpus once at construction. Re-masks each
    sample() call so the model sees fresh mask placements per cycle.

    reward_kind: how `reward` is computed for corpus records.
      "proxy_tfidf"     — sigmoid(program-token TF-IDF) under a frozen IDF
                          built once at startup. DEVIATION from CovRL.
      "cached_coverage" — real coverage TF-IDF computed once via afl-showmap,
                          frozen thereafter. Matches CovRL's "frozen IDF"
                          semantics without per-cycle execution cost.
      "live_coverage"   — real coverage TF-IDF re-computed per cycle via
                          afl-showmap. Matches CovRL exactly. Expensive.
    """
    cfg:         Config
    masking:     Masking
    reward_kind: str
    executor:    ExecutionAdapter | None   # required for cached/live coverage
    size:        int

    def sample(self, n: int) -> list[dict]   # without replacement; group_id=-1
```

`cfg.corpus_dir` must be non-empty — raise at construction. Silent fallback
to rollout-only mixing would invisibly degrade behaviour.

The three reward variants share one class; the swap points are how IDF is
built (once vs per-cycle) and whether `afl-showmap` runs.

---

## Rewarding

Isolated from data plumbing.

```python
class CovRL8ClassDiscretiser:
    """Hardcoded non-uniform bins from base_utils.score_to_label.
        class 0:  r < -0.5         (syntax)
        class 1: -0.5 ≤ r < 0      (semantic)
        class 2:  0  ≤ r ≤ 0.5     (valid, low TF-IDF — wide bin)
        classes 3-7: 0.1-wide bins over (0.5, 1.0]
    """
    @staticmethod
    def reward_to_class(r: float | torch.Tensor) -> int | torch.Tensor
    @staticmethod
    def class_to_reward(c: int | torch.Tensor) -> float | torch.Tensor

class Rewarder:
    """CovRL Eq. 3-6: binary TF (unique-coverage), IDF = log(N/(1+DF))/sqrt(M),
    momentum α-blended at cycle boundaries.

    compute(obs: ExecutionObservation) -> float — single live-path entry.
    observe_seed(bitmap)               -> None  — accumulates DF for IDF.
    update_cycle()                     -> None  — blends new IDF with α=0.6.
    snapshot_bitmap()                  -> np.ndarray — stub; raises until SHM wired.
    """

class ValidityOracle:
    """Maps execution result → {valid, syntax_error, semantic_error}.
    Stub today; engine-specific wiring deferred."""
```

`ExecutionObservation` is the dataclass passed to `Rewarder.compute`. Same
shape from live SHM (online) and from `ExecutionAdapter.execute` (faithful).

---

## policy_gradient_algorithms

```python
# ppo.py
def loss(ratios, R_hat, group_ids, cfg) -> torch.Tensor:
    eps = cfg.clip_epsilon                     # default 0.2  [v4 fix #8]
    clipped = torch.clamp(ratios, 1 - eps, 1 + eps)
    return -torch.min(ratios * R_hat, clipped * R_hat).mean()

# grpo.py
def loss(ratios, R_hat, group_ids, cfg) -> torch.Tensor:
    # A = (R̂ - μ_g) / (σ_g + eps); corpus (gid=-1) gets A=0.
    # Same clipped objective.
```

Both signatures take `group_ids` so the orchestrator can pass one shape
everywhere; PPO ignores it.

---

## Variant matrix

| Variant | entry-point | collection | trainer | policy_loss | corpus_reward | ref_update_every |
|---|---|---|---|---|---|---|
| **CovRL (faithful)** | `covrl.py` | `"interesting"` | `"covrl_faithful"` | `ppo.loss` | `"live_coverage"` | `1` |
| CovRL (our default, proxy) | `covrl.py` | `"interesting"` | `"covrl"` | `ppo.loss` | `"proxy_tfidf"` | `1` |
| CovRL (cached coverage) | `covrl.py` | `"interesting"` | `"covrl"` | `ppo.loss` | `"cached_coverage"` | `1` |
| CovRL-All | `covrl.py` | `"all"` | `"covrl"` | `ppo.loss` | `"proxy_tfidf"` | `1` |
| RLLM-PPO | `rllm.py` | `"all"` | `"rllm"` | `ppo.loss` | `"proxy_tfidf"` | `1` |
| RLLM-GRPO | `rllm.py` | `"all"` | `"rllm"` | `grpo.loss` | (see open issues) | `1` |
| KL-anchored RLLM | `rllm.py` | `"all"` | `"rllm"` | any | any | `N > 1` |

The four paper-primary variants are rows 1, 4, 5, 6. The rest fall out for free.

---

## Faithfulness scorecard (v4)

| Behaviour | CovRL code | This plan |
|---|---|---|
| Critic trains every cycle | `inferencer.py:71` | `_train_critic` always called |
| Actor skipped on cycle 0 | `inferencer.py:72-75` | **[v3 fix #1]** `if self._cycle > 0:` |
| D_T accumulates across cycles | `finetuner.py:283-285` | **[v3 fix #2]** `_rollout_history.extend(...)` |
| Critic + mutator share one dataset | `finetuner.py:117` reads `self.dataset` | **[v4 fix #4]** `ds` built once, used by both |
| Cycle 0 dataset = corpus only | `finetuner.py:276-281` | **[v4 fix #4]** `shared_records = corpus_records` on cycle 0 |
| 8-class non-uniform bins | `base_utils.score_to_label` | `CovRL8ClassDiscretiser` exact match |
| IDF momentum α = 0.6 | `update_idf(alpha=cfg.alpha)` | `Rewarder.update_cycle` blends with α |
| Corpus ratio 4·\|D_T\|, no replacement | `train_dataset.sample(min(4n, \|C\|))` | `min(4 * max(1, n), corpus.size)` |
| Persistent critic | `self.critic` lives on `FineTuner` | `CovRLTrainer.critic` persists |
| Critic init | `T5Config()`, random weights | **[v3 fix #3]** `_init_critic_from_config()` |
| Critic input shape | `concat(masked_input, mask_targets)` | **[v4 fix #5]** `concat(x_t, y_t)` |
| Critic head | CLS hidden state `[:, 0, :]` | **[v4 fix #6]** CLS, not pooled |
| Critic forward output | `(loss, logits)` tuple (HF-incompatible) | **[v4 fix #7]** `SequenceClassifierOutput` |
| PPO clip ε | `clamp(ratio, 0.8, 1.2)` → ε=0.2 | **[v4 fix #8]** `cfg.clip_epsilon = 0.2` |
| PPO objective = policy + CE | `ppo_loss + cur_outputs.loss.mean()` | `policy_loss + CE` in `_train_mutator` |
| Corpus reward source | real coverage via afl-showmap | **deviation**: proxy TF-IDF; flagged; upgradeable via `executor=` |
| `_rollout_history` source | full AFL queue via `load_files()` | **deviation**: LLM rollouts only; upgradeable via Phase 2 `QueueReader` |
| Critic backbone | `T5Config()` default (d_model=512) | matches code; deviates from paper text |

Three explicit deviations remain, all isolated:
- corpus reward → `CorpusSampler(executor=)`
- rollout source → `CovRLTrainer._rollout_history` extension point
- critic backbone → `_init_critic_from_config` returns `T5Config()`; replace with `T5Config.from_pretrained("Salesforce/codet5p-220m")` if paper-text parity is wanted

---

## Open issues (kept deliberately open)

### Validity collapse under GRPO + CE
CE doesn't directionally enforce validity. Models drift to invalid mutations,
groups become all-invalid, advantage = 0, gradient signal vanishes. Filtering
zero-variance groups hides rather than fixes it.

The interface leaves room: `_train_mutator` is a single composed-loss call,
so adding validity-weighted CE / discriminator / reward shaping is local.

### GRPO + corpus mixing
Spec says corpus gets `A=0` so policy contribution is zero — but both
interleaved and separated empirical setups suffered validity collapse.

Interface keeps the option open:
- `GroupedBatchSampler` could treat `group_id < 0` as fillable slots
- `RLLMTrainer.finetune` could call `_train_mutator` twice and accumulate grads

Neither is wired today. **Defer; raise loudly if config requests both.**

---

## File-by-file responsibilities

| File | Layer | Owns |
|---|---|---|
| `config.py` | — | `Config`, JSON load/save, validators, `clip_epsilon=0.2`, `idf_alpha=0.6`, `ref_update_every=1` |
| `masking.py` | 1, 2 | `MaskedSpan`, `MaskedProgram`, `Masking`, `LLMModel` |
| `rollout.py` | 3, 4 | `RolloutBuffer`, `RolloutDataset`, `RolloutCollator`, `GroupedBatchSampler` |
| `mutator.py` | orch. | `Mutator`, lazy `_build_trainer` |
| `rewarding.py` | training | `CovRL8ClassDiscretiser`, `Rewarder`, `ValidityOracle`, `ExecutionObservation` |
| `corpus.py` | training | `CorpusSampler` (proxy / cached / live) |
| `base_trainer.py` | training | `BaseTrainer`, `CovRLTrainer`, `RLLMTrainer`, critic-init helper |
| `policy_gradient_algorithms/ppo.py` | training | `loss(ratios, R_hat, group_ids, cfg)` |
| `policy_gradient_algorithms/grpo.py` | training | `loss(ratios, R_hat, group_ids, cfg)` |
| `covrl.py` | entry | AFL hooks, wires `Mutator(cfg, "interesting", "covrl")` |
| `rllm.py` | entry | AFL hooks, wires `Mutator(cfg, "all", "rllm")` |

Import direction enforced: entry-points → mutator → {masking, rollout,
base_trainer}; base_trainer → {rollout, masking, rewarding, corpus,
policy_gradient_algorithms}. No upward imports.

---

## Decision checkpoints

1. **`utils.py`** → gut. Nothing outside the directory imports it.
2. **`Rewarder.snapshot_bitmap()`** stub → raise `NotImplementedError`.
3. **Critic backbone** → `T5Config()` random init. **[v3 fix #3]** No
   `cfg.critic_path`; CovRL never loaded one.
4. **Critic forward** → `SequenceClassifierOutput`. **[v4 fix #7]**
5. **GRPO + corpus mixing** → defer; raise on config.
6. **HF Trainer** → keep, with lazy `_MutatorTrainer(Trainer)` subclass
   for `compute_loss`.

---

## Implementation order

1. `policy_gradient_algorithms/ppo.py` + `grpo.py` (no deps, unit-testable;
   `cfg.clip_epsilon=0.2`)
2. `rewarding.py` — `Rewarder`, `CovRL8ClassDiscretiser`,
   `ExecutionObservation`; SHM reader as `NotImplementedError` stub
3. `corpus.py` — `CorpusSampler` with proxy reward; `executor=None` hook
4. `base_trainer.py` — `BaseTrainer`, `_init_critic_from_config`
   returning `SequenceClassifierOutput`, `CovRLTrainer` with
   `_rollout_history` + shared-dataset + cycle-0 gate, `RLLMTrainer`
5. `mutator.py` — lazy `_build_trainer`; pass `masking` + `model` through
6. `utils.py` — remove `mix_corpus`, `calc_reward`, `_COV_TFIDF` globals
7. `config.py` — new fields + validators
8. End-to-end smoke test script (separate from the pure-mutation smoke
   test we already have)

---

## Phase 2 — Faithful CovRL (later, additive)

| File | Adds |
|---|---|
| `execution.py` *(new)* | `ExecutionAdapter(showmap_path, target_path)` — `execute(bytes) -> ExecutionObservation`. Stateless. |
| `queue_reader.py` *(new)* | `QueueReader(afl_out_dir)` — yields `(seed_id, bytes)` from `out/queue/`, mtime-watched. |
| `samplers.py` *(new)* | `LLMMutationSampler(masking, model, executor)` — mask + generate + execute → records. Reuses fuzz-time machinery. |
| `base_trainer.py` | New subclass `CovRLFaithfulTrainer(BaseTrainer)`. No edits to existing trainers. |

`CovRLFaithfulTrainer.finetune` body — same shape as `CovRLTrainer.finetune`,
ignores the live `rollout`, pulls programs from `QueueReader.snapshot()`,
runs them through `LLMMutationSampler`, otherwise identical. The
shared-dataset cycle-0 gate and `_rollout_history` accumulation are reused
verbatim.

`online_collection: bool = True` config flag — `Mutator.post_run` skips
`rollout.add` when `False`, so faithful runs pay no in-memory rollout cost.

---

## Why this shape is defensible against the simpler alternatives

- **One mega-`Trainer` parameterised by flags** instead of two subclasses —
  CovRL and RLLM differ on three orthogonal axes (history vs cycle-local,
  critic vs direct, cycle-0 gate). Three booleans with non-obvious
  interactions is worse than two ~30-line subclasses.
- **Single library with one entry-point** instead of `covrl.py` + `rllm.py`
  shims — the shims are 10 lines each; they pin the variant matrix at the
  AFL boundary so the user picks the run by command name, not config.
- **One pipeline class that does everything** instead of layers — would
  fuse generation, data, and training. Already tried during v1; collapsed
  under the weight of corpus + GRPO + critic interactions.

The test of whether this interface is right: any new variant lands as one
new row in the matrix, no edits to existing layers. Anything that needs
more is a sign the abstraction has leaked.
