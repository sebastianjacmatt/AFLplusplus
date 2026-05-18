# rllm — Design

This document describes the architecture of `rllm`, a Python custom mutator
for AFL++ that learns a masked-span generation policy via reinforcement
learning. It is design-only: no implementation choices below should be read
as code. Where a principle is cited, it is named in **bold** with its
canonical source so the reader can verify the trade-off independently.

---

## 1. Scope and constraints

`rllm` is an **AFL++ Python custom mutator**. The entry-point contract is
fixed by AFL++ and documented in [docs/custom_mutators.md](../../../docs/custom_mutators.md):
the module must expose a known set of top-level functions (`init`,
`fuzz_count`, `fuzz`, `post_run`, `queue_get`, `deinit`, …) and AFL++ calls
them on a hot path between target executions. Two consequences fall out of
this and shape every decision below:

1. **The AFL++ ABI is non-negotiable.** It is a procedural, module-level
   interface — we cannot change it, only adapt to it.
2. **The fuzz loop is performance-critical.** [docs/best_practices.md §Improving speed](../../../docs/best_practices.md#improving-speed)
   treats every per-iteration cost as a first-class concern. Anything done
   inside `fuzz()` runs once per target execution and must be cheap;
   training must therefore be amortised across many fuzz calls.

These two constraints — fixed procedural boundary, hot inner loop — are
what motivate the architecture below.

---

## 2. Architectural style

### 2.1 Hexagonal architecture (Ports & Adapters)

Source: Alistair Cockburn, *"Hexagonal Architecture"* (2005). The pattern
splits a system into a **domain core** that knows nothing about its
runtime, surrounded by **adapters** that translate between the core and
external systems.

Mapping to `rllm`:

- **Primary adapter** — [rllm.py](../rllm.py). This is the only file that
  knows about the AFL++ Python ABI. It implements the AFL++ entrypoints
  and translates them into calls on domain objects. Nothing else in the
  package imports or references AFL++ symbols.
- **Domain core** — `Mutator`, `Rewarder`, `Rollout`, `BaseTrainer`, the
  `policy_gradient_algorithms/` package. These reason about masked-span
  generation, rewards, trajectories, and gradients. They are framework-
  agnostic and unit-testable without AFL++.
- **Secondary adapters** — `Rewarder` sub-components (`TFIDFCoverageRewarder`,
  `StderrValidityRewarder`) that read coverage bitmaps or stderr files.
  They sit at the *outbound* edge of the domain, isolating side-effecting
  I/O from policy/training logic.

Why this matters here: the AFL++ contract is a hard external surface we do
not own. Hexagonal architecture is the standard pattern for that exact
situation, and it is what lets the rest of the package be tested without
spinning up `afl-fuzz`.

### 2.2 Single Responsibility Principle (SRP)

Source: Robert C. Martin, *"Agile Software Development: Principles,
Patterns, and Practices"* (2002) — the **S** of SOLID. *"A module should
have one, and only one, reason to change."*

Each top-level module owns exactly one axis of change:

| Module | One reason to change |
|---|---|
| [config.py](../config.py) | Hyperparameters and their validation rules. |
| [mutator.py](../mutator.py) | How masked spans are sampled and decoded into bytes. |
| [rewarder.py](../rewarder.py) | How a target execution is scored. |
| [rollout.py](../rollout.py) | How trajectories are stored and shaped for the trainer. |
| [base_trainer.py](../base_trainer.py) | The shared training loop on top of HF `Trainer`. |
| [policy_gradient_algorithms/*.py](../policy_gradient_algorithms) | The loss for one specific RL algorithm. |
| [rllm.py](../rllm.py) | The AFL++ ABI binding. |

The empirical test for SRP compliance: if a change request touches more
than one of these files, the boundary is wrong.

### 2.3 Dependency Inversion Principle (DIP) + Dependency Injection

Source: Martin, ibid. — the **D** of SOLID. *"High-level modules should
not depend on low-level modules. Both should depend on abstractions."*

`rllm.py` wires concrete components together in `init()` and passes them
to each other by constructor argument. Nothing constructs its own
collaborators. Concretely: the `Mutator` does not import a specific
`Rewarder`; it accepts one. The `BaseTrainer` does not import a specific
`PolicyGradientAlgorithm`; it is given one.

This is enforced *in the design*, not by a DI container — Python doesn't
need one. The discipline is: **only [rllm.py](../rllm.py) does composition.**

### 2.4 Strategy pattern for policy gradient algorithms

Source: Gamma, Helm, Johnson, Vlissides, *"Design Patterns: Elements of
Reusable Object-Oriented Software"* (1994), the **Strategy** pattern.
*"Define a family of algorithms, encapsulate each one, and make them
interchangeable."*

PPO and GRPO have different losses but consume the same inputs (a batch
of rollout samples with log-probs, advantages, and a reference policy
snapshot) and produce the same output (a scalar loss tensor). They are
the textbook case for Strategy: one interface, swap implementations.

- `policy_gradient_algorithms/ppo.py` and `…/grpo.py` each define a class
  implementing a `PolicyGradientAlgorithm` protocol.
- `BaseTrainer` holds a reference to one such strategy and calls it
  inside `compute_loss`.
- [config.py](../config.py) selects which strategy is instantiated based
  on `policy_gradient_algorithm`, and attaches the matching sub-config
  (`PPOConfig` / `GRPOConfig`) — this is the **conditional sub-config**
  pattern already implemented in [config.py:121-150](../config.py#L121-L150).

This is also why the **Open/Closed Principle** (Bertrand Meyer, *Object-
Oriented Software Construction*, 1988; the **O** of SOLID) is satisfied
here: adding a third algorithm (e.g. REINFORCE-with-baseline, RLOO) means
adding one file under `policy_gradient_algorithms/` and one sub-config
dataclass. No existing file is modified.

### 2.5 Template Method for `BaseTrainer`

Source: GoF, ibid., the **Template Method** pattern. *"Define the
skeleton of an algorithm in an operation, deferring some steps to
subclasses."*

HuggingFace `transformers.Trainer` already *is* a template method
implementation — `training_step`, `compute_loss`, `evaluate`, `save_model`
are hook points. `BaseTrainer` extends `Trainer` and overrides only the
hooks it must (principally `compute_loss`, which delegates to the
injected `PolicyGradientAlgorithm` strategy, and the data-collation path
that consumes the rollout).

The composition is therefore: **Template Method (HF Trainer) +
Strategy (PolicyGradientAlgorithm)** — Template provides the loop, Strategy
provides the loss. This is a common GoF combination and is the reason
`base_trainer.py` should stay small: most of the loop is HF's, and the
algorithm-specific work lives in the strategy.

### 2.6 Composition over inheritance

Source: GoF, ibid., stated as a recurring principle. *"Favor object
composition over class inheritance."*

`Mutator`, `Rewarder`, `Rollout`, and `BaseTrainer` are composed by
`rllm.py`; they are not arranged in a deep inheritance hierarchy. The
only inheritance in the design is where it earns its keep:

- `BaseTrainer` extends `transformers.Trainer` — required by HF's API.
- `PPOAlgorithm` / `GRPOAlgorithm` implement a `PolicyGradientAlgorithm`
  protocol — a structural interface in the `typing.Protocol` sense, not
  a base class. (Source: PEP 544 — *Protocols: Structural subtyping*.)

`Rewarder` composes a coverage rewarder and a validity rewarder; it is
not subclassed for each combination. This is the same shape as the call
site already present in [rllm.py:27-36](../rllm.py#L27-L36).

### 2.7 Twelve-Factor configuration

Source: Adam Wiggins, *The Twelve-Factor App* (2011), Factor III:
*"Store config in the environment."*

`rllm` follows this in two ways already encoded in [config.py](../config.py):

- The config object is **declarative** (a dataclass of values), separate
  from the code that uses it.
- Resolution order is **explicit arg → env var (`RLLM_CONFIG`) →
  defaults**. The env-var path is the Twelve-Factor compliance.

The design rule that follows: **no module reads the environment directly
except [rllm.py](../rllm.py)**. Reading env vars from inside the domain
core would re-couple it to the runtime and break §2.1.

---

## 3. Component responsibilities

### 3.1 `config.py` — declarative configuration

- Single unified `Config` dataclass + algorithm sub-configs (`PPOConfig`,
  `GRPOConfig`).
- Validation lives in `load_config` (e.g. `train_batch_size %
  grpo.group_size == 0`). Validation is a **boundary concern** — the
  rest of the system trusts a loaded `Config`.
- No torch/transformers objects are constructed here. No env vars read
  beyond `RLLM_CONFIG`.

### 3.2 `mutator.py` — the `Mutator`

Responsibility: take a seed buffer, produce mutated buffers via masked-
span generation, and remember enough state per call so that the
post-execution reward can be attributed to the right rollout sample.

Surface (sketch, not implementation):

- `mask(buf: bytes) -> None` — called once per AFL++ `fuzz_count`,
  prepares the masked input context shared by the next batch of
  `fuzz()` calls. This matches the AFL++ lifecycle: `fuzz_count` runs
  once per queue entry, `fuzz` runs `n` times after.
- `mutate(max_size: int) -> bytes` — called once per AFL++ `fuzz()` call,
  samples one completion from the policy and records the trajectory.
- `collect(rewarder: Rewarder) -> None` — called from `post_run`, asks
  the rewarder for the reward of the *just-executed* trajectory and
  appends it to the rollout.

The Mutator owns the model+tokenizer at runtime, but it does **not** own
the rollout — it appends to a `RolloutBuffer` it was injected with.
This keeps the storage policy out of the mutator (§2.2 SRP).

### 3.3 `rewarder.py` — the `Rewarder`

Responsibility: turn one target execution into one scalar reward.

The composite shape already shown in [rllm.py:27-36](../rllm.py#L27-L36)
is the right one: a top-level `Rewarder` composes a coverage rewarder
(`TFIDFCoverageRewarder`) and a validity rewarder
(`StderrValidityRewarder`). Each sub-rewarder is independently testable
and independently swappable.

The combination rule (e.g. `reward = validity_bonus + (1 - validity_bonus)
* coverage_reward`) lives in the top-level `Rewarder`, not in either
sub-rewarder. That keeps the sub-rewarders free of policy.

### 3.4 `rollout.py` — trajectory storage

Responsibility: hold the trajectories generated since the last training
cycle, in a shape the trainer can consume.

Two collaborating types:

- `RolloutBuffer` — append-only, mutable, owned by the `Mutator`. One
  entry per `fuzz()` call: tokens, log-probs at sample time, the masked
  context, and (once `post_run` fires) the reward.
- `RolloutDataset` — read-only view built from the buffer, conforming
  to `torch.utils.data.Dataset`. This is what `BaseTrainer` consumes.

The split exists because the buffer's append API is wrong for training
(no shuffling, no grouping) and the dataset's read API is wrong for the
mutator (immutable). Two types, one for each role: SRP again (§2.2).

### 3.5 `base_trainer.py` — `BaseTrainer(transformers.Trainer)`

Responsibility: own the training loop, accept an injected policy-gradient
strategy, run one finetune cycle per call.

- Constructor takes `(model, ref_model, cfg, algorithm)` where
  `algorithm: PolicyGradientAlgorithm` is the Strategy (§2.4).
- Overrides `compute_loss` to delegate to `algorithm.loss(batch,
  model_out, ref_model_out, cfg)`. Nothing else algorithm-specific
  appears in this file.
- Exposes one domain method, `finetune(dataset: RolloutDataset)`, that
  wraps HF's `train()` for one cycle.

The reference-policy snapshot (`pi_ref`) and KL term are HF/loop
concerns and live here. The KL *weighting* is configured via
`Config.kl_coef` (declarative, §3.1); the KL *computation* is invoked
inside the strategy because both PPO and GRPO need access to it inside
their loss formula. This is a deliberate split: the loop owns when, the
strategy owns how.

### 3.6 `policy_gradient_algorithms/` — Strategy implementations

Each file defines one algorithm conforming to a `PolicyGradientAlgorithm`
Protocol (PEP 544). The protocol method is roughly:

    loss(batch, model_out, ref_model_out, cfg) -> Tensor

- `ppo.py` — actor-critic loss, uses `cfg.ppo.{value_coef, entropy_coef,
  gae_lambda}` and `cfg.clip_epsilon`.
- `grpo.py` — group-relative loss, uses `cfg.grpo.{group_size,
  norm_epsilon, advantage_clip}` and `cfg.clip_epsilon`.

Adding `rloo.py` later is a pure addition: implement the protocol, add a
sub-config dataclass in `config.py`, extend the `if/elif` ladder in
`load_config`. No existing algorithm is touched. (Open/Closed, §2.4.)

### 3.7 `rllm.py` — the AFL++ adapter

Responsibility: implement the AFL++ Python ABI (the functions listed in
[docs/custom_mutators.md §2 APIs](../../../docs/custom_mutators.md#2-apis))
and translate them into calls on the domain core.

The file is intentionally thin. Every entrypoint maps to a single domain
call:

| AFL++ entrypoint | Translation |
|---|---|
| `init(seed)` | Build `Config`, `RolloutBuffer`, `Mutator`, `Rewarder`, `BaseTrainer`. Hold them in module globals. |
| `fuzz_count(buf)` | `MUTATOR.mask(buf); return CFG.fuzz_count` |
| `fuzz(buf, add_buf, max_size)` | `return MUTATOR.mutate(max_size)` |
| `post_run()` | `MUTATOR.collect(REWARDER)` |
| `queue_get(filename)` | Increment counter; if multiple of `CFG.finetune_every`, call `TRAINER.finetune(RolloutDataset(BUFFER))`. |
| `deinit()` | Flush rollout, save model checkpoint. |

Two design rules for this file:

- **No algorithm logic.** If `rllm.py` ever contains an `if cfg.policy_gradient_algorithm == ...`, the abstraction has leaked back to the
  adapter. The Strategy (§2.4) is supposed to absorb that branch.
- **No env vars except inside `init`.** Anything that reads env state
  reads it once at startup and passes the resolved value forward, so the
  rest of the loop is deterministic — which is what
  [docs/best_practices.md §Improving stability](../../../docs/best_practices.md#improving-stability)
  asks for.

---

## 4. Lifecycle and data flow

The interaction below is one AFL++ "queue entry" — the unit of work
AFL++ schedules. `n = CFG.fuzz_count` is the number of fuzz calls per
entry; `n` must be a multiple of `cfg.grpo.group_size` under GRPO
(enforced in [config.py:147-150](../config.py#L147-L150)).

    AFL++                          rllm.py            domain core
    -----                          -------            -----------
    startup ───────────────►  init(seed) ──────► Config / RolloutBuffer
                                                  / Mutator / Rewarder
                                                  / BaseTrainer

    select queue entry ────►  queue_get(f) ────► increment counter
                                                  if counter % finetune_every == 0:
                                                      TRAINER.finetune(
                                                          RolloutDataset(BUFFER))

    schedule fuzzing ──────►  fuzz_count(buf) ─► MUTATOR.mask(buf)
                                                  (one masked context
                                                   for the next n calls)

    for i in 1..n:
        ───────────────────►  fuzz(buf,…)  ────► MUTATOR.mutate()
                                                  (sample, record sample
                                                   into BUFFER)
        execute target
        ───────────────────►  post_run()  ─────► MUTATOR.collect(REWARDER)
                                                  (REWARDER.score() →
                                                   attach reward to the
                                                   last BUFFER entry)

    shutdown ──────────────►  deinit() ───────► flush + checkpoint

A few invariants follow directly from this flow:

- **One mask per group.** `mask()` runs once per queue entry; the next
  `n` `mutate()` calls share the masked context. Under GRPO this is the
  group; under PPO it is just a batch of independent samples.
- **Reward attribution is positional.** AFL++ guarantees that
  `post_run()` is the next mutator call after the `fuzz()` it scores.
  `MUTATOR.collect()` therefore attaches the reward to the *last*
  buffer entry — no IDs needed.
- **Training is amortised.** `finetune_every` controls how many queue
  entries pass between training cycles, so the cost of a gradient step
  is paid once per many fuzz iterations — the amortisation
  [best_practices.md §Improving speed](../../../docs/best_practices.md#improving-speed)
  is asking for.

---

## 5. Why these boundaries and not others

A handful of alternatives were considered and rejected; recording them
here so future edits don't relitigate them.

- **Putting the `Rewarder` factory in `config.py`.** Rejected: the
  rewarder depends on runtime state (`RLM_STDERR_FILE`) read at startup,
  and constructing torch/IO objects in the config breaks §2.7
  (declarative config) and §2.1 (domain core is import-safe without a
  runtime).

- **Making `BaseTrainer` own the rollout.** Rejected: the rollout is
  produced by the mutator and only *consumed* by the trainer. Giving the
  trainer ownership would force the mutator to import the trainer,
  reversing the composition direction set by [rllm.py:14-21](../rllm.py#L14-L21)
  and entangling two SRPs.

- **One class per (mutator × algorithm) pair (`PPOMutator`,
  `GRPOMutator`).** Rejected: this is the textbook anti-pattern that
  Strategy exists to solve (GoF, §2.4). It multiplies classes
  combinatorially and forces algorithm changes to touch the mutator.

- **Free functions instead of `Mutator` / `Rewarder` classes.** Rejected:
  both objects hold non-trivial state across AFL++ calls (model,
  tokenizer, bitmap, IDF EMA). Free functions would push that state into
  module globals — the exact coupling Hexagonal Architecture is meant to
  prevent.

---

## 6. Extension: per-algorithm data ordering (`GRPOSampler`)

This section was added after the initial design met implementation: the
HF training loop's default sampler silently breaks GRPO's group
statistics. The decision below is derived from the principles in §2–§3;
nothing in those sections needed to change.

### 6.1 The constraint

`GRPOAlgorithm.loss` normalises advantages *within* groups by reshaping
the per-sample reward tensor `(B,) → (B/G, G)` and computing
`(r − mean_g) / (std_g + ε)`. The reshape only recovers actual groups if
the training batch contains *consecutive* same-group samples.
HuggingFace `Trainer`'s default sampler is `RandomSampler` (shuffled),
which scrambles that ordering — under the default sampler, GRPO's group
statistics collapse to noise indistinguishable from PPO's batch-mean
baseline. A custom sampler that preserves intra-group contiguity is
therefore required. Call it `GRPOSampler`.

### 6.2 Where it lives

Four candidates, evaluated against the principles already in §2–§3.

1. **Inside `BaseTrainer` directly** — override `_get_train_sampler` and
   pick the sampler there. Rejected by §3.5: *"Nothing else algorithm-
   specific appears in this file."* An `if isinstance(self.algorithm,
   GRPOAlgorithm)` here is exactly the leak §3.7 names — *"If `rllm.py`
   ever contains an `if cfg.policy_gradient_algorithm == ...`, the
   abstraction has leaked back to the adapter"* — the rule applies one
   floor up too.

2. **Inside `rollout.py`** — co-locate with `RolloutDataset` /
   `RolloutCollator`. Rejected by §3.4: rollout's *one reason to change*
   is how trajectories are stored and shaped. *Why* the trainer wants a
   specific ordering is an algorithm concern (it exists because GRPO's
   loss requires it), not a storage concern. Different axis, different
   file — SRP (§2.2).

3. **A new `samplers.py` module.** Rejected as premature abstraction.
   There is one sampler in the system; a module for one class violates
   the project-wide guidance against pre-emptive abstraction. Reconsider
   when a second algorithm needs grouped sampling.

4. **Inside `policy_gradient_algorithms/grpo.py`, alongside
   `GRPOAlgorithm`.** *Chosen.* SRP (§2.2): both the loss and the
   sampler are facets of the same axis of change — "how GRPO computes
   its update." Open/Closed (§2.4): adding GRPO already entailed
   creating a file under `policy_gradient_algorithms/`; the sampler
   joins that same addition. The §7 summary table gets one new row; no
   existing entry changes.

### 6.3 Mechanism — widen the Strategy protocol

`BaseTrainer` must select the correct sampler *without knowing which
algorithm is active.* Extend `PolicyGradientAlgorithm` with an optional
hook:

```python
class PolicyGradientAlgorithm(Protocol):
    def loss(...) -> torch.Tensor: ...
    def make_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        cfg: Config,
    ) -> torch.utils.data.Sampler | None:
        return None          # default: defer to HF Trainer
```

`BaseTrainer` overrides `_get_train_sampler` and delegates:

```python
def _get_train_sampler(self):
    custom = self.algorithm.make_sampler(self.train_dataset, self.cfg)
    return custom if custom is not None else super()._get_train_sampler()
```

Why this shape:

- **Template Method + Strategy continued (§2.5).** The HF training loop
  is still the template; the Strategy now controls *two* hooks — `loss`
  and `make_sampler` — instead of one. Same composition, slightly wider
  Strategy surface.
- **Liskov Substitution (§2.4 LSP).** The default returns `None`, so
  `PPOAlgorithm` (which does not override `make_sampler`) remains a
  drop-in substitute.
- **Dependency Inversion (§2.3 DIP).** `BaseTrainer` depends on the
  abstract `make_sampler` hook, never on the concrete `GRPOSampler`.
- **No leak into the adapter (§3.7).** `rllm.py` is unaffected; it
  neither imports nor references the sampler.

### 6.4 What `GRPOSampler` does

Given a dataset of length `N` and group size `G = cfg.grpo.group_size`,
yield `N` indices in an order that keeps every contiguous block of `G`
together. The order *of groups* may be shuffled each epoch (so SGD
still sees variation); the order *within* a group is fixed.

`N % G == 0` is already guaranteed upstream by `config.load_config`
(`train_batch_size % group_size == 0` and
`fuzz_count % group_size == 0`), so the sampler can assume clean
partitioning. The rollout buffer's insertion order — `fuzz_count`
consecutive samples per `mask()` call (§4) — already places same-group
samples adjacent in the dataset, so `GRPOSampler`'s job is to
*preserve* contiguity, not to *recover* it.

---

## 7. Summary — principles applied

| Principle | Source | Where it shows up |
|---|---|---|
| Hexagonal Architecture | Cockburn 2005 | `rllm.py` as the only AFL++-aware module |
| Single Responsibility | Martin 2002 (SOLID-S) | One axis of change per file (§3) |
| Open/Closed | Meyer 1988 (SOLID-O) | New algorithm = new file under `policy_gradient_algorithms/` |
| Liskov Substitution | Liskov 1987 (SOLID-L) | PPO / GRPO interchangeable behind `PolicyGradientAlgorithm` |
| Interface Segregation | Martin 2002 (SOLID-I) | Small protocols (`PolicyGradientAlgorithm`, sub-rewarder protocols) instead of one fat base class |
| Dependency Inversion | Martin 2002 (SOLID-D) | Components receive collaborators; only `rllm.py` composes |
| Strategy | GoF 1994 | `policy_gradient_algorithms/{ppo,grpo}.py` |
| Template Method | GoF 1994 | `BaseTrainer` extending HF `Trainer` |
| Composition over Inheritance | GoF 1994 | `Rewarder` composes sub-rewarders; `Mutator` composes `RolloutBuffer` |
| Structural Protocols | PEP 544 | `PolicyGradientAlgorithm` as `typing.Protocol`, not ABC |
| Twelve-Factor Config | Wiggins 2011, Factor III | `RLLM_CONFIG` env-var + declarative dataclass |
| AFL++ ABI conformance | `docs/custom_mutators.md` | All ABI entrypoints live in `rllm.py` |
| Speed amortisation | `docs/best_practices.md` §Improving speed | `finetune_every` separates training cost from fuzz cost |
| Determinism / stability | `docs/best_practices.md` §Improving stability | Env vars read only in `init`; no I/O in the inner loop beyond what the rewarder requires |
| Per-algorithm data ordering | This doc, §6 | `GRPOSampler` lives in `policy_gradient_algorithms/grpo.py`; `make_sampler` hook on the Strategy protocol; `BaseTrainer._get_train_sampler` delegates |
