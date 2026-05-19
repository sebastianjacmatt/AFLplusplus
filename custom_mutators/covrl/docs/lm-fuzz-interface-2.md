# LM-Fuzz Interface (v2)

A common interface for LM-driven AFL custom mutators in which **CovRL-Fuzz is one
mapping into the interface, and other variants (RLLM-PPO, RLLM-GRPO, CovRL-All,
future approaches) are different mappings selected by configuration, not by
code paths.**

Earlier design notes (`design.md`, `design_principles.md`) described what each
component does. This document specifies the **contracts between components** so
that variants are reachable by changing config, and so deviations from CovRL
have an obvious, named home in the code.

---

## Layered architecture

```
┌──────────────────────────────────────────────────────────┐
│ AFL entry-points   covrl.py        rllm.py               │  thin shims
├──────────────────────────────────────────────────────────┤
│ Orchestrator       Mutator                               │  AFL hook impl
├──────────────────────────────────────────────────────────┤
│ Generation         Masking      LLMModel                 │  language model
├──────────────────────────────────────────────────────────┤
│ Data               RolloutBuffer (D_T)                   │
│                    RolloutDataset / Collator / Sampler   │  HF wiring
├──────────────────────────────────────────────────────────┤
│ Training           BaseTrainer  CovRLTrainer  RLLMTrainer│
│                    CorpusSampler                         │
│                    rewarding/  policy_gradient_algorithms/│
└──────────────────────────────────────────────────────────┘
```

Each layer below has **one narrow contract**. Variants change the wiring, not
the inside of any layer.

---

## Layer 1 — Masking

Owns tokenisation, T5 span corruption, T5 target construction, generation
budget calculation, and reconstruction.

```python
class Masking:
    tokenizer: PreTrainedTokenizer        # shared with LLMModel
    eos_token_id, pad_token_id: int

    def tokenize(self, buf: bytes) -> list[int]
    def mask(self, tokens: list[int]) -> MaskedProgram
    def target_ids(self, mp: MaskedProgram) -> list[int]
    def generation_budget(self, mp, max_new_tokens_per_span) -> int
    def reconstruct(self, mp, generated_ids) -> list[int]
    def decode(self, mp, generated_ids) -> bytes
    def batch_decode(self, mps, generated_ids_list) -> list[bytes]
```

`MaskedProgram` carries `original_ids`, `input_ids` (encoder input with
sentinels), and `spans` (each with `start`, `end`, `sentinel_id`).

**Invariants the rest of the system relies on:**
- `target_ids(mp)` produces the T5 supervised target for the *same* mask `mp`
  — corpus mixing and critic-target construction both use this.
- `generation_budget(mp, k)` returns an upper bound on the decoder length
  needed to emit all spans plus terminal sentinel + EOS.
- `reconstruct` accepts cleaned decoder output (no decoder_start, no
  post-EOS pads); `LLMModel` is responsible for cleaning.

**Extension points (named, deferred):**
- whole-word masking (CodeT5 §3.2) — a flag on `Masking`, off by default
- identifier-aware corruption — future variant, would subclass

---

## Layer 2 — LLMModel

Owns the HF seq2seq model and one batched-generate primitive.

```python
class LLMModel:
    tokenizer: PreTrainedTokenizer        # passed in by Mutator from Masking
    model: PreTrainedModel
    device: str

    def batch_generate(self, xs, n_samples, max_new_tokens) -> list[list[int]]
    def save(self, path)
```

`batch_generate` runs **one** `model.generate()` call across all of `xs`,
with `num_return_sequences=n_samples`. Returns a flat
`len(xs) * n_samples` list of **cleaned** token-id lists (decoder_start
stripped, truncated at first EOS). Order is HF-standard
`[xs[0]·n_samples, xs[1]·n_samples, ...]`.

**Why cleaned, not raw:** if `y_t` were stored raw, training labels would
contain decoder_start tokens and post-EOS pads as supervised targets,
training the model to emit them. Cleaning happens here, once, at the
generation boundary.

**The tokenizer is shared with `Masking`** (passed in from the Mutator)
to keep sentinel / pad / eos ids consistent.

---

## Layer 3 — RolloutBuffer (= D_T)

Holds per-sample records. The collection strategy lives entirely in **when**
the orchestrator calls `commit_last`, never inside this class.

```python
class RolloutBuffer:
    def add(self, x_t, y_t, reward, group_id, **extra) -> None    # stages
    def commit_last(self) -> None                                  # promotes
    def flush(self) -> list[dict]                                  # drains
    def __len__(self) / __iter__(self)                            # over committed
```

Records are dicts so the schema is extensible without API churn:

```python
{
    "x_t":      list[int],
    "y_t":      list[int],            # cleaned T5 target form (no decoder_start)
    "reward":   float,
    "group_id": int,                  # -1 for corpus samples
    # extras:                         # populated only when relevant — drop-in fields
    # "log_prob":     float,          # π_old (recoverable later, deferred per CovRL)
    # "ref_log_prob": float,          # π_ref (KL anchor, deferred)
    # "executed_program": bytes,      # debugging artefact
}
```

**Add and commit are deliberately decoupled.** This is the *only* mechanism
encoding CollectInteresting vs CollectAll:

| Collection | When `commit_last` runs |
|---|---|
| `"all"` | immediately after `add()` in `post_run` |
| `"interesting"` | in `queue_new_entry` — AFL fires it only when the run is interesting; uncommitted entries are discarded by the next `add()` |

CovRL itself stores **only** the executed program `T`, re-masks and
re-executes at training time. We store the full tuple. The collection
semantics are identical — both modes are queue-as-dataset. See the
**Deviations** section.

---

## Layer 4 — Dataset / Collator / Sampler

Bridges `RolloutBuffer.flush()` records into HF Trainer-compatible batches.

```python
class RolloutDataset(torch.utils.data.Dataset):
    # wraps list[dict] returned by flush(); __getitem__ → dict

class RolloutCollator:
    # pads x_t / y_t, builds input_ids / attention_mask / labels(-100) / reward / group_id

class GroupedBatchSampler(torch.utils.data.Sampler):
    # batches of *complete* GRPO groups
    # validates batch_size % group_size == 0; raises on partial groups
```

The collator emits an identical batch shape regardless of whether records came
from the rollout buffer or the corpus sampler — both are `{x_t, y_t, reward,
group_id}` dicts.

`group_id` is always present in records and in batches; PPO `compute_loss`
simply doesn't read it. Avoids "is the field there?" checks.

---

## Orchestrator — Mutator

```python
class Mutator:
    masking: Masking
    model:   LLMModel
    rollout: RolloutBuffer
    trainer: BaseTrainer | None        # None in pure-mutation mode

    # AFL hooks (delegated from covrl.py / rllm.py shims)
    def init(seed)
    def queue_get(filename)            # increments counter; sets _finetune_pending
    def fuzz_count(buf)                # drains _finetune_pending; samples masks; batched generate
    def fuzz(buf, add_buf, max_size)   # pops from pending; sets _has_pending_sample
    def post_run()                     # gated on _has_pending_sample; add; maybe commit
    def queue_new_entry(new, orig)     # commit_last if collection=="interesting"
    def deinit()                       # flush, save model
```

Construction:

```python
Mutator(cfg, collection: "interesting" | "all", trainer: "covrl" | "rllm")
```

**One generate per `fuzz_count` cycle.** All masks for the cycle are sampled
first; `xs` is the list of `num_groups` distinct encoder inputs; one
`batch_generate(xs, n_samples=group_size, max_new_tokens=max(budgets))`
covers PPO (group_size=1) and GRPO (group_size>1) with the same code path.
HF `num_return_sequences` handles intra-group sample diversity; the
mega-batch handles cross-group throughput.

**`post_run` is gated on `_has_pending_sample`.** AFL fires `post_run` for
calibration / dry-run / trim runs that didn't call `fuzz()`; the gate
prevents those non-fuzz executions from corrupting `_last` rewards.

---

## Training — BaseTrainer + subclasses

Shared scaffolding lives on `BaseTrainer`. CovRL and RLLM are minimal
subclasses that differ in `finetune` only.

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
    def _maybe_snapshot_prev()         # honours cfg.ref_update_every
    def _train_mutator(records, reward_fn)
        # HF Trainer with custom compute_loss:
        #   forward θ + prev → log_probs
        #   ratio = exp(log_p_θ - log_p_prev)
        #   R̂ = reward_fn(batch)
        #   L  = policy_loss(ratios, R̂, group_id, cfg) + CE(θ, labels)

class CovRLTrainer(BaseTrainer):
    critic: PreTrainedModel            # 8-class classifier, same arch as actor, persistent

    def finetune(self, rollout):
        records = rollout.flush()
        self._train_critic(records)                       # rollouts only
        self._maybe_snapshot_prev()
        n_corpus = min(4 * len(records), self.corpus.size)
        mix = records + self.corpus.sample(n_corpus)
        self._train_mutator(mix, reward_fn=self._learned_reward)
        self._cycle += 1

class RLLMTrainer(BaseTrainer):
    def finetune(self, rollout):
        records = rollout.flush()
        self._maybe_snapshot_prev()
        n_corpus = min(4 * len(records), self.corpus.size)
        mix = records + self.corpus.sample(n_corpus)
        self._train_mutator(mix, reward_fn=lambda b: b["reward"])
        self._cycle += 1
```

**Why this is the right shape:**

- One `RolloutDataset` + one `RolloutCollator` everywhere — including for corpus
  samples (`group_id=-1`).
- Two trainers differ in *exactly two lines*: presence/absence of critic
  training, and which `reward_fn` is passed.
- `_train_mutator` is the only place that knows about `policy_loss`,
  `prev_model`, CE auxiliary loss — and it sees the same `reward_fn`
  abstraction regardless of trainer.

---

## CorpusSampler

The single place that knows where corpus reward comes from.

```python
class CorpusSampler:
    """Loads + tokenises corpus once at construction. Re-masks each sample()
    call so the model sees fresh mask placements per cycle.

    reward_kind: how `reward` is computed for corpus records.
        "proxy_tfidf"      — sigmoid of program-token TF-IDF under a frozen IDF
                             built once at startup. DEVIATION FROM CovRL.
        "cached_coverage" — real coverage TF-IDF computed once at startup via
                             afl-showmap, frozen thereafter. Matches CovRL's
                             "frozen IDF" semantics without per-cycle cost.
        "live_coverage"   — real coverage TF-IDF re-computed per cycle via
                             afl-showmap. Matches CovRL exactly. Expensive.
    """
    cfg:         Config
    masking:     Masking
    reward_kind: str
    size:        int                   # |C|

    def sample(self, n: int) -> list[dict]
```

**`cfg.corpus_dir` must be non-empty** — raise at construction. Silent
fallback to rollout-only mixing would invisibly degrade RLLM/CovRL behaviour.

**Re-masking each cycle** is cheap (in-memory tokenisation; corpus tokens are
cached). It gives the model wider mask-distribution exposure across cycles.

The three `reward_kind` variants share a single class — the **only** swap
points are how the corpus IDF is built (once vs per-cycle) and whether
`afl-showmap` is invoked. This is the explicit extension point for going from
β (our default) to γ to α.

---

## Rewarding — discretiser + TF-IDF + validity

Lives in `rewarding.py`, isolated from the data plumbing.

```python
class CovRL8ClassDiscretiser:
    """Hardcoded non-uniform bins from CovRL.
        class 0:  r < -0.5         (syntax)
        class 1: -0.5 ≤ r < 0      (semantic)
        class 2:  0  ≤ r ≤ 0.5     (valid, low TF-IDF — wide bin)
        classes 3-7: 0.1-wide bins over (0.5, 1.0]
    """
    @staticmethod
    def reward_to_class(r: float) -> int
    @staticmethod
    def class_to_reward(c: int) -> float    # bin midpoint

class CoverageTFIDF:
    """CovRL Eq. 4-6: IDF = log(N / (1 + DF_cov[i])) / sqrt(M), blended at
    cycle boundaries with momentum α."""

class ValidityOracle:
    """Maps execution stderr / exit / showmap output → {valid, syntax, semantic}.
    Stub today (returns 'valid'); real wiring depends on engine harness."""
```

**Extension points:**
- Swap `CovRL8ClassDiscretiser` for a uniform / quantile discretiser via the
  trainer config (`cfg.discretiser`).
- Swap `ValidityOracle` for engine-specific implementations (Jerry stderr,
  V8 stderr, afl-showmap-based).

---

## Variant matrix

The whole point of this interface. Every variant maps to **configuration**,
not code changes inside the layers.

| Variant | entry-point | collection | trainer | policy_loss | corpus_reward | ref_update_every |
|---|---|---|---|---|---|---|
| **CovRL (faithful)** | `covrl.py` | `"interesting"` | `"covrl"` | `ppo.loss` | `"live_coverage"` | `1` |
| CovRL (our default, proxy) | `covrl.py` | `"interesting"` | `"covrl"` | `ppo.loss` | `"proxy_tfidf"` | `1` |
| CovRL (cached coverage) | `covrl.py` | `"interesting"` | `"covrl"` | `ppo.loss` | `"cached_coverage"` | `1` |
| CovRL-All | `covrl.py` | `"all"` | `"covrl"` | `ppo.loss` | `"proxy_tfidf"` | `1` |
| RLLM-PPO | `rllm.py` | `"all"` | `"rllm"` | `ppo.loss` | `"proxy_tfidf"` | `1` |
| RLLM-GRPO | `rllm.py` | `"all"` | `"rllm"` | `grpo.loss` | (see open issues) | `1` |
| KL-anchored | `rllm.py` | `"all"` | `"rllm"` | any | any | `N > 1` |

The four "primary" variants from the paper are rows 1, 4, 5, 6. The rows
between them are toggles that fall out of the interface for free.

---

## Snapshot cadence

`cfg.ref_update_every: int` (default `1`).

- `1` → snapshot `prev_model` before every mutator update (CovRL behaviour).
- `N > 1` → hold `prev_model` fixed for N cycles. Useful when validity
  collapses — a slower-moving reference pulls KL / policy loss toward a less
  drifted policy. Strictly an extension of CovRL.

Snapshotting copies `model.model` (the underlying HF model), freezes it,
puts it in eval mode. ~`|θ|` of extra VRAM. For CodeT5-base (220M) that's
~880MB — fine on a single 24GB GPU; reducible to adapter-delta snapshots
later via LoRA (deferred per principle #2).

---

## Deviations from CovRL — named and isolated

| Deviation | Where it lives | How to undo |
|---|---|---|
| **Proxy corpus reward (no re-execution)** | `CorpusSampler.reward_kind = "proxy_tfidf"` | Set to `"live_coverage"` (or implement `"cached_coverage"` for the middle ground) |
| Store full `(x_t, y_t, R, group_id)` per sample rather than re-deriving from T at train time | `RolloutBuffer.add()` shape; `Mutator.post_run` storing the cleaned `y_t` from generation | Trainer would need to receive raw `T` instead and re-mask. Currently no path to undo; can be added by populating the `executed_program` extra field and ignoring `x_t`/`y_t` at training. |
| `read_coverage()` / `validity()` are stubs returning defaults | `rewarding.ValidityOracle` and `rewarding.CoverageTFIDF` | Wire real SHM attach + stderr classification |

These are the only places. Every other CovRL behaviour (persistent critic,
hardcoded 8-class bins, no replacement in corpus sampling, cycle-0 mutator
training, mixed CE + policy loss, 4·|D_T| corpus ratio) is matched.

---

## Open issues (deliberately not closed by this interface)

### Validity collapse under GRPO + CE
**Observation:** in CovRL + RLLM-GRPO experiments, the CE component of the
loss does not directionally enforce program validity. Models drift toward
producing invalid mutations, GRPO groups become all-invalid, advantage = 0
on every group, gradient signal disappears. Filtering zero-variance groups
*hides* this rather than fixing it.

**The interface doesn't resolve this** — it's a loss-design problem. But it
leaves room: `_train_mutator` reads `reward_fn(batch)` and computes a single
composed loss `policy_loss + CE`. Adding new loss terms (validity-weighted
CE, separate discriminator head, validity-aware reward shaping) is a local
change to `_train_mutator`'s loss composition, not a cross-cutting refactor.

Candidate avenues for follow-up work:
- Validity-weighted CE: scale CE per-sample by stored / predicted validity
- Validity-aware reward shaping: floor on valid samples (`validity_bonus`)
  applied before GRPO group normalisation
- Auxiliary discriminator: train a "valid vs invalid" classifier on
  rollouts; add its log-prob of "valid" to the loss as a regulariser

### GRPO + corpus mixing
**Observation:** spec says corpus carries `A=0` so policy loss contribution is
zero — but empirically, both interleaved (`(ii)`: corpus in the same batch as
rollout groups) and separated (`(iii)`: separate batches combined at loss
level) suffered from the validity-collapse failure mode. Filtering didn't
help.

**The interface keeps the option open:**
- `GroupedBatchSampler` could be extended to treat `group_id < 0` as
  fillable / unconstrained slots
- Or `RLLMTrainer.finetune` could call `_train_mutator` twice (once on
  rollouts, once on corpus) and accumulate gradients before stepping

Neither is wired today. Likely path forward: address validity collapse first,
re-run RLLM-GRPO+corpus, see if the issue persists.

---

## File-by-file responsibilities

| File | Layer | Owns |
|---|---|---|
| `config.py` | — | `Config` dataclass, JSON load/save, `resolve_device` |
| `masking.py` | 1, 2 | `MaskedSpan`, `MaskedProgram`, `Masking`, `LLMModel` |
| `rollout.py` | 3, 4 | `RolloutBuffer`, `RolloutDataset`, `RolloutCollator`, `GroupedBatchSampler` |
| `mutator.py` | orch. | `Mutator` |
| `base_trainer.py` | training | `BaseTrainer`, `CovRLTrainer`, `RLLMTrainer`, `CorpusSampler` |
| `rewarding.py` | training | `CovRL8ClassDiscretiser`, `CoverageTFIDF`, `ValidityOracle`, reward helpers |
| `policy_gradient_algorithms/ppo.py` | training | `ppo.loss(ratios, R_hat, group_id, cfg) → scalar` |
| `policy_gradient_algorithms/grpo.py` | training | `grpo.loss(ratios, R_hat, group_id, cfg) → scalar` |
| `covrl.py` | entry | AFL hooks, wires `Mutator(cfg, "interesting", "covrl")` |
| `rllm.py` | entry | AFL hooks, wires `Mutator(cfg, "all", "rllm")` |

**Module boundaries are enforced by import direction:** entry-points → mutator
→ {masking, rollout, base_trainer}. base_trainer → {rollout, masking,
rewarding, policy_gradient_algorithms, CorpusSampler}. No upward imports.
This means a developer can read the stack top-down without ever needing to
descend into training to understand mutation.

---

## Where to start

For each variant the implementation path is:

1. **Pick a row from the variant matrix.**
2. **Set the relevant config fields** (`collection`, `trainer_kind`,
   `policy_loss`, `corpus_reward`, `ref_update_every`).
3. **Run.** The Mutator constructs the right trainer; the trainer reads its
   `reward_fn` from the policy loss choice; the corpus sampler picks the
   right reward path. No code changes per variant.

For a new variant (something the matrix doesn't cover yet) the path is:

1. **Add the new mechanic at its layer** — a new policy_loss in
   `policy_gradient_algorithms/`, a new corpus reward strategy in
   `CorpusSampler`, a new discretiser in `rewarding/`, a new collection
   strategy in `Mutator.post_run`/`queue_new_entry`.
2. **Add a config flag.** No other layer needs to know it exists.
3. **Add a row to the matrix.**

This is the test of whether the interface is right: anything that takes more
than that is a sign the abstraction has leaked.