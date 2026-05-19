# LM-Fuzz Interface

A pluggable interface for language-model-driven AFL++ mutation. Decouples
**collection** (mutator + masking + rollouts) from **training** (whatever
RL/SFT algorithm consumes the rollouts). Lets us implement CovRL-Fuzz [^covrl]
on top of the same collection layer that a pure-mutation baseline or an
RLLM-PPO/GRPO trainer uses.

The mutator/rollout layer is inspired by the implementation in
`rlm_mutator` [^rlm], which adapts the CovRL training loop to AFL++; we
generalise that further by lifting the trainer-specific assumptions out of
the AFL hook layer.

---

## 1. Why a separate interface

Existing LM-based fuzzers (CovRL-Fuzz [^covrl], rlm_mutator [^rlm]) bind a
specific training algorithm into their AFL hook layer:

- the mutator stores fields the trainer happens to need
  (`log_prob`, `ref_log_prob`, `coverage_reward`, `exit_code`, ...)
- the rollout buffer's commit semantics are entangled with the training
  cadence
- changing the trainer means editing the mutator

We split these concerns so that:

- **collection** runs without ever importing trainer code
- the **rollout dataset** is the integration boundary — a
  `torch.utils.data.Dataset` of dicts
- **trainers** plug in below that boundary, each free to choose where to
  compute log-probs, where to compute rewards, whether to re-execute, etc.

A practical consequence: pure LM mutation (no training at all) is the
trivial case — wire a no-op trainer.

---

## 2. Architecture

```
                                ┌─────────────────────────────┐
   AFL hooks  ─────────►  mutator.py  ──────►  masking.py + LLMModel
                                │                   (T5 span masking,
                                ▼                    HF seq2seq inference)
                          rollout.py
                          ┌──────────────────────┐
                          │  RolloutBuffer       │ ◄── CollectInteresting
                          │   stage / commit     │     vs CollectAll
                          │                      │     decided in mutator
                          │  flush() → records   │     hooks; buffer doesn't
                          └──────────────────────┘     care
                                │
                                ▼
                          RolloutDataset
                          RolloutCollator        ◄── + CorpusSampler (optional)
                          GroupedBatchSampler        merge / interleave at
                                │                    dataset construction
                                ▼
   ┌─────────────────────────────────────────────────────────┐
   │       base_trainer.py — pluggable trainer interface     │
   │                                                         │
   │       BaseTrainer.finetune(rollout: RolloutBuffer)      │
   │            ├── CovRLTrainer (critic + corpus + PPO)     │
   │            ├── RLLMTrainer  (direct R, PPO or GRPO)     │
   │            ├── NoOpTrainer  (pure mutation, dump only)  │
   │            └── …            (re-execution-based, SFT, …)│
   └─────────────────────────────────────────────────────────┘
```

The black line between RolloutBuffer and the trainer is the contract.

---

## 3. The collection layer (trainer-agnostic)

### Masking (`masking.py`)

T5-style span corruption + reconstruction. Mirrors CodeT5's pre-training
objective [^codet5] and matches `rlm_mutator/masking.py`:

- `Masking.mask(tokens)` → `MaskedProgram(original_ids, input_ids, spans)`
- `Masking.target_ids(mp)` → T5 supervised target: `<s_0> span_0 … <s_N>`
- `Masking.generation_budget(mp, max_new_per_span)` → exact decoder budget
- `Masking.reconstruct(mp, y)` / `batch_decode(mps, ys)` → executable bytes

`LLMModel.batch_generate(xs, n_samples, max_new_tokens)` runs **one**
seq2seq `generate()` per `fuzz_count` cycle. Same call serves both
ungrouped (PPO, `n_samples=1`) and grouped (GRPO, `n_samples=group_size`)
modes via `num_return_sequences`. Outputs are cleaned (decoder-start
stripped, truncated at first EOS) so `y_t` is in the canonical T5 target
shape and ready for the supervised collator.

### Mutator (`mutator.py`)

Implements the AFL hooks. Stateless from AFL's perspective — all state is
on the Mutator instance.

| AFL hook | Owns |
|---|---|
| `init` | Construct masking, model, rollout, trainer |
| `queue_get` | Finetune trigger (set `_finetune_pending`) |
| `fuzz_count` | Tokenise; sample masks; ONE batched `generate()`; push to pending queue. Drain `_finetune_pending` here so seed selection isn't blocked. |
| `fuzz` | Pop one pending; validate vs `max_size`; mark `_has_pending_sample`; return bytes |
| `post_run` | Gated on `_has_pending_sample` (AFL fires post_run for calibration/trim runs too); compute reward stub; `rollout.add(...)`; commit immediately if `collection == "all"` |
| `queue_new_entry` | Commit if `collection == "interesting"` |
| `deinit` | Flush rollout; save model |

The collection strategy lives purely in the choice of *when* to call
`rollout.commit_last()` — no other code branches on it.

### Rollout (`rollout.py`)

`RolloutBuffer` is two-stage: `add(x, y, R, group_id, **extra)` stages
(overwrites any prior staged entry); `commit_last()` promotes the staged
entry into the committed list. CollectInteresting calls `commit_last` in
`queue_new_entry`; CollectAll calls it in `post_run`. This is the only
place the two strategies diverge.

Records are plain dicts so trainers can attach extra fields via `**extra`
without API churn:

```python
rollout.add(x_t, y_t, R, group_id, log_prob=lp, ref_log_prob=rlp)
```

`flush()` returns `list[dict]` and clears. `RolloutDataset` wraps it for
HF Trainer. `RolloutCollator` pads `x_t` / `y_t`, emits `input_ids`,
`attention_mask`, `labels` (-100 padded), `reward`, `group_id`.
`GroupedBatchSampler` yields batches of complete GRPO groups (raises on
partial groups — surface bugs, no silent filtering).

---

## 4. The contract (interface boundary)

A trainer is anything that implements:

```python
class BaseTrainer:
    def finetune(self, rollout: RolloutBuffer) -> None: ...
```

It receives the buffer and is free to:

- `records = rollout.flush()` — drain
- build any `Dataset` / `Sampler` / `Collator` it wants
- mix in a `CorpusSampler` (4·|D_T| corpus records by CovRL convention,
  or any other ratio)
- compute log-probs / rewards / advantages on its own schedule
- update its own model (the trainer also owns it; the Mutator only sees
  it for inference)

What the trainer **must not** do:
- mutate `RolloutBuffer` schema in a way other trainers wouldn't expect
- assume any field beyond `{x_t, y_t, reward, group_id}` is present
  (use `record.get("log_prob")` for optional fields)

---

## 5. Reward and log-prob: when to compute them

There are two valid timings for each, with different cost/correctness
trade-offs:

| Quantity | At fuzz time (in `post_run` / `fuzz`) | At training time |
|---|---|---|
| **Reward** | Cheaper if coverage is read directly from AFL SHM | Requires re-execution (CovRL-faithful path) |
| **log-prob π_old(y\|x)** | Cheap incremental from `generate()` output, but `outputs.scores` are top-k/top-p filtered [^rlm-recompute] — recompute via raw forward to avoid clipping every gradient | Exact, but pays one extra forward pass per sample at train time |

The interface supports either choice:

- **Fuzz-time capture**: pass through `rollout.add(..., log_prob=lp)`.
  RolloutCollator can be extended to surface it as `old_log_prob` in the
  batch.
- **Train-time computation**: leave the field off; trainer forwards
  `prev_model(x, y)` once per batch and gathers log-probs from raw
  logits.

CovRL-Fuzz [^covrl] takes the latter route for log-probs (re-runs through
π_prev at training time) and computes rewards via afl-showmap re-execution.
rlm_mutator [^rlm] captures both at fuzz time, then recomputes the
log-prob via a raw forward pass to avoid the top-k/top-p inflation bug.
Our interface lets each trainer pick.

---

## 6. Recipes — implementing a trainer

### NoOpTrainer (pure-mutation baseline)

```python
class NoOpTrainer(BaseTrainer):
    """Drain the rollout for inspection; never updates the model."""
    def finetune(self, rollout):
        records = rollout.flush()
        _dump_jsonl(self.cfg.rollout_path, records)
```

### RLLM-PPO (direct reward, on-policy)

```python
class RLLMTrainer(BaseTrainer):
    def finetune(self, rollout):
        records = rollout.flush()
        corpus  = self.corpus.sample(n=min(4 * len(records), self.corpus.size))
        ds      = RolloutDataset(records + corpus)
        self._snapshot_prev()
        self._train_mutator(ds, reward_fn=lambda b: b["reward"])
        self._cycle += 1
```

### CovRL (learned critic + PPO + corpus)

```python
class CovRLTrainer(BaseTrainer):
    def finetune(self, rollout):
        records = rollout.flush()
        self._train_critic(RolloutDataset(records))     # 8-class CE, rollouts only

        corpus = self.corpus.sample(n=min(4 * len(records), self.corpus.size))
        ds     = RolloutDataset(records + corpus)
        self._snapshot_prev()
        self._train_mutator(ds, reward_fn=self._critic_reward)
        self._cycle += 1
```

`_critic_reward(batch)` runs the 8-class critic over `(x_t, y_t)`,
maps `argmax → bin midpoint`, returns the scalar tensor used as R̂.

### CovRL-faithful (re-execution at training time)

A future trainer can implement CovRL's exact behaviour by storing only the
executed program `out_bytes` and re-running it under `afl-showmap` at
training time. The collection layer needs no change — only `add(..., out=out)`
to carry the bytes, and a new dataset class that re-executes on
`__getitem__`. This is the version where corpus reward is real coverage
(not the proxy described in §7).

---

## 7. Deliberate deviations from CovRL-Fuzz

| Aspect | CovRL-Fuzz | Ours |
|---|---|---|
| Rollout schema | `(T, R)` per queue entry | `(x_t, y_t, R, group_id, **extra)` per sample |
| Mask used at training | freshly sampled at train time | same mask used at fuzz time |
| Reward source for corpus | real coverage via afl-showmap re-execution | sigmoid of program-token TF-IDF (PROXY) |
| Re-execution cost per cycle | 4·\|D_T\| target runs (corpus) + 1 per saved seed | 0 |
| log-prob π_old | computed at train time on π_prev | TODO: deferred; design ready to add at fuzz time |
| Reward / IDF model | bitmap-edge TF-IDF, momentum-blended snapshots | TODO: stub; will mirror CovRL Eq. 4-6 when wired |

The corpus-reward deviation is encapsulated in `CorpusSampler._score`,
so an upgrade path to either cached-coverage (pre-execute once at
startup) or full re-execution (CovRL-faithful) is a single-method change.

---

## 8. CollectInteresting vs CollectAll

Both strategies are supported by the same collection layer, differing only
in commit timing:

| Strategy | Commit fires in | What lands in D_T |
|---|---|---|
| CollectInteresting | `queue_new_entry` | Only mutations AFL flagged as new-coverage |
| CollectAll | `post_run` | Every mutation |

Original CovRL-Fuzz [^covrl] uses CollectInteresting. RLLM variants and
GRPO use CollectAll so within-group statistics are well-defined. The
collection layer doesn't care which is chosen; the entry-point file
(`covrl.py` vs `rllm.py`) wires it in at Mutator construction.

GRPO under CollectInteresting produces partial groups (only new-coverage
samples commit) — `GroupedBatchSampler` raises on those. Trainers that
want this combination must use a non-grouped sampler.

---

## 9. Extension points

Hooks where future work plugs in without changing the interface:

- `RolloutBuffer.add(**extra)` — attach `log_prob`, `ref_log_prob`,
  `value_pred`, `out_bytes`, etc.
- `RolloutCollator` — extend the output dict; HF Trainer with
  `remove_unused_columns=False` passes them to `compute_loss`
- `CorpusSampler._score` — swap proxy TF-IDF for cached or live coverage
- `GroupedBatchSampler` — alternative semantics (e.g. mix corpus into
  groups with `group_id=-1` and zero advantage)
- Reference-policy snapshot for KL — new field on the trainer; no
  collection-layer change

---

## References

[^covrl]: **CovRL-Fuzz** — *Coverage-Guided Reinforcement Learning for
    Language Model-Based Fuzzing*. The reference algorithm for combining a
    coverage-guided AFL loop with an LM mutator trained by RL with a
    coverage-derived TF-IDF reward and an 8-class learned critic. See
    `docs/design.md` for the algorithmic specification we follow.

[^rlm]: **rlm_mutator** — `AFLplusplus/custom_mutators/rlm_mutator/` in
    this repository. AFL++ custom mutator implementing a CodeT5-based
    span-prediction mutator with PPO/GRPO training over rollouts captured
    at fuzz time. Used here as the implementation reference for the
    masking, rollout, and trainer wiring; we generalise its design by
    lifting the trainer-specific schema and commit logic into a pluggable
    interface.

[^codet5]: **CodeT5 / CodeT5+** — *Identifier-aware Unified Pre-trained
    Encoder-Decoder Models for Code Understanding and Generation*. The
    masked span prediction objective used by the `Masking` module is the
    T5 / CodeT5 pre-training task: replace contiguous spans with ordered
    sentinels in the encoder input, decoder predicts the missing spans.

[^rlm-recompute]: rlm_mutator recomputes `old_log_prob` via a raw forward
    pass instead of reading `outputs.scores` from `generate()`, because
    HF's `scores` are top-k/top-p filtered — concentrated probability mass
    inflates the stored log-prob relative to the raw distribution
    `compute_loss` sees, making `ratio = exp(new - old) ≪ 1.0` at step 0
    and clipping every gradient. See
    `AFLplusplus/custom_mutators/rlm_mutator/mutator.py:222-232`.