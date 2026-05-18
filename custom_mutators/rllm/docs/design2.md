# rllm — Design v2: Critic-Guided Policy Gradient

This document extends [docs/design.md](design.md) (hereafter "v1"). All
principles, boundaries, and module responsibilities from v1 remain in force.
This document records one addition: an optional discriminative reward model
(the "critic") that can be composed with any existing policy-gradient
algorithm, and the design decisions that determine its shape.

The motivation is the CovRL-Fuzz paper (Eom, Jeong & Kwon, ISSTA 2024, §3.3).
Their key finding is that training a separate encoder to *predict* which reward
bucket a `(context, generation)` pair will achieve, then training the actor on
those predictions rather than raw environment rewards, improves coverage and
reduces error rates. rllm already implements the TF-IDF coverage reward (CovRL
Eqs. 3–6). This document adds the critic component while preserving GRPO-based
variance reduction — a property CovRL's PPO-only design does not have.

---

## 1. What changes in the total composition

### 1.1 The raw reward changes roles

v1 data flow:

    Rewarder.score() → raw_reward → inputs["rewards"] → advantage → policy gradient

v2 data flow (when `use_critic = true`):

    Rewarder.score() → raw_reward ─┬─► bucket_label ─► critic training target
                                    │
                                    └─► (optional blend, weight 1-w)

    critic.forward(ctx ∥ gen) ────────► predicted_reward ─► blended_reward
                                                                    │
                                                                    ▼
                                                          advantage → policy gradient

The `Rewarder` is unchanged. It produces the TF-IDF coverage signal that rllm
already implements. The critic learns to approximate that signal from token
sequences alone, generalising reward prediction to unseen completions. The
`reward_weight` parameter (§3.5) controls how much of the blended signal comes
from the critic versus the environment; this knob has no equivalent in CovRL,
which uses the critic exclusively.

### 1.2 A second trainable model enters the domain core

v1's domain core has one `nn.Module`: the actor, owned by `Mutator`. Adding
the critic introduces a second with three concrete consequences:

1. **`deinit()`** in `rllm.py` must checkpoint the critic alongside the actor.
2. **`finetune()`** in `BaseTrainer` gains a pre-phase: critic training
   (supervised classification) precedes actor training (policy gradient).
   Both phases are amortised behind `finetune_every`; the AFL++ hot path is
   unaffected, but per-cycle wall time increases.
3. **Memory** increases by the critic backbone size. §3.1 addresses how to
   bound this.

None of these changes touch the AFL++ adapter (`rllm.py`) beyond composition
and checkpointing; the hexagonal boundary from v1 §2.1 is unchanged.

### 1.3 The rollout serves dual purpose — and the existing structure handles it

v1's rollout entries already contain `(masked_input, gen_tokens, logprobs,
reward)`. Critic training needs exactly these fields:
`cat(masked_input, gen_tokens)` as input, `score_to_label(reward)` as target.
No new fields are added to `RolloutBuffer`. The rollout's *storage* interface
is unchanged; a second consumer is added (the critic trainer in `pre_finetune`).
This is consistent with v1 §3.4: the rollout stores trajectories; who reads
them is not its concern.

### 1.4 GRPO's group structure and the critic compound cleanly — double variance reduction

CovRL uses PPO with a batch-mean baseline. rllm uses GRPO with group-relative
normalisation. With the critic:

- The critic is trained on the rollout, which already places same-seed
  mutations in consecutive groups (v1 §4).
- During actor training, `GRPOSampler` still orders samples in group-contiguous
  blocks. GRPO normalises the *critic-blended* rewards within those groups.
- The two reductions operate independently:
  - The critic reduces variance by generalising noisy environment rewards to
    smooth predicted reward signals.
  - GRPO reduces variance by within-group normalisation.

Combined GRPO + critic outperforms either alone. CovRL (PPO + critic) gets
only the first reduction. rllm without critic gets only the second. The design
target is both simultaneously.

---

## 2. Why the Decorator pattern, not a third algorithm literal

v1 §2.4 names the correct extension point for new algorithms: add one file
under `policy_gradient_algorithms/`. That rule applies to algorithms whose
**loss formula** differs. The critic does not change the loss formula of PPO
or GRPO — it changes the **reward input** to that formula. CovRL's `L_CovRL`
(Eq. 7) is the same PPO clipped surrogate; only `reward` is substituted.

A third literal `"actor_critic"` would require choosing one base algorithm
to embed. Pairing with PPO sacrifices GRPO's group normalisation. Pairing with
GRPO leaves PPO users without a critic-compatible option. Supporting both means
four classes (`PPO`, `GRPO`, `PPO+Critic`, `GRPO+Critic`) — the combinatorial
explosion v1 §5 explicitly rejects.

Source: Gamma, Helm, Johnson, Vlissides, *"Design Patterns"* (1994), the
**Decorator** pattern. *"Attach additional responsibilities to an object
dynamically. Decorators provide a flexible alternative to subclassing for
extending functionality."*

Mapping to rllm:

- `CriticDecorator` **is** a `PolicyGradientAlgorithm` (implements the protocol).
- It holds a reference to another `PolicyGradientAlgorithm` — the wrapped
  algorithm, either `PPOAlgorithm` or `GRPOAlgorithm`.
- It adds critic training in `pre_finetune` and reward substitution in `loss`.
- It **delegates** `make_sampler` to the wrapped algorithm — so
  `CriticDecorator(GRPOAlgorithm()).make_sampler(...)` returns a `GRPOSampler`
  without the decorator knowing anything about GRPO.

The `policy_gradient_algorithm` config field stays `Literal["ppo", "grpo"]`,
unchanged. A new `use_critic: bool` flag controls whether the selected
algorithm is wrapped. Composition in `rllm.py init()`:

```python
algorithm = _build_algorithm(cfg)          # PPOAlgorithm or GRPOAlgorithm — unchanged
if cfg.use_critic:
    algorithm = CriticDecorator(algorithm, Critic(cfg.critic_cfg), cfg)
TRAINER = BaseTrainer(cfg, MUTATOR, algorithm)
```

One composition guard in `rllm.py`. Zero changes to `ppo.py` or `grpo.py`
beyond a no-op `pre_finetune` method (§3.3).

---

## 3. Component responsibilities

### 3.1 `critic.py` — the discriminative reward model

Responsibility: "how reward prediction is parameterised." Single axis of
change: swap the backbone, change the number of buckets, adjust the head.
Nothing else in the package imports this module except `rllm.py` (wiring)
and `critic_decorator.py` (consumer).

Two types plus two utility functions:

**`Critic(nn.Module)`**

```
backbone   : T5EncoderModel
classifier : nn.Sequential(Dropout(dropout_rate), Linear(d_model, num_labels))
loss_fct   : CrossEntropyLoss

forward(input_ids, attention_mask, labels=None)
    hidden = backbone(input_ids, attention_mask).last_hidden_state[:, 0]  # CLS pool
    logits = classifier(hidden)                                             # (B, num_labels)
    if labels:
        return loss_fct(logits, labels), logits
    return logits
```

**`score_to_label(reward: float, thresholds: list[float]) → int`**
Maps a float reward to a bucket index by finding the first threshold it
does not exceed. Configurable thresholds; the default reproduces CovRL's
`base_utils.score_to_label` exactly:

    thresholds : [-0.5,  0.0,  0.5,  0.6,  0.7,  0.8,  0.9]   (7 → 8 classes)
    values     : [-1.0, -0.5,  0.5,  0.6,  0.7,  0.8,  0.9, 1.0]

**`build_label_to_score(values: list[float]) → dict[int, float]`**
Constructs the inverse mapping used at actor training time. Called once
during `CriticDecorator.__init__`.

**Backbone initialisation.** CovRL's `CriticModel` creates `T5EncoderModel(T5Config())`
— a randomly initialised T5-small (~60 M params) with `d_model=512`, ignoring
the `model_path` argument entirely. For rllm the better default is
`T5EncoderModel.from_pretrained(cfg.critic_cfg.model_name)`, loading the
*encoder half* of the same CodeT5p family model the actor uses. This gives
the critic a warm start on code-understanding representations and keeps
`d_model` consistent across actor and critic. `CriticConfig.model_name`
controls this (default `"Salesforce/codet5p-220m"`). If memory is the
binding constraint, `"Salesforce/codet5p-110m"` or `""` (random T5-small,
matching CovRL) are valid overrides.

### 3.2 `policy_gradient_algorithms/critic_decorator.py` — the Decorator

Responsibility: "how the critic is composed with an existing policy-gradient
algorithm." One axis of change: the critic training procedure and the reward
blending rule. The underlying loss formula lives in the wrapped algorithm and
is not repeated here.

**Constructor**

```python
CriticDecorator(
    wrapped: PolicyGradientAlgorithm,
    critic:  Critic,
    cfg:     Config,
)
```

Builds `label_to_score` from `cfg.critic_cfg.bucket_values`, instantiates
`AdamW(critic.parameters(), lr=cfg.critic_cfg.critic_lr)`. The critic
optimiser is owned by the decorator — it is not managed by HF `Trainer`,
which only sees the actor. The two training phases are genuinely separate
and require no shared optimiser.

**`pre_finetune(dataset: RolloutDataset, cfg: Config) → None`**

Phase 1 of the finetune cycle. Trains the critic as a supervised classifier
on the rollout data collected since the last cycle.

```
for batch in DataLoader(dataset, batch_size=cfg.train_batch_size):
    gen_ids      = inputs["labels"].clamp(min=0)          # strip -100 padding
    critic_ids   = cat([inputs["input_ids"],   gen_ids],              dim=1)
    critic_mask  = cat([inputs["attention_mask"], ones_like(gen_ids)], dim=1)
    targets      = tensor([score_to_label(r, thresholds) for r in inputs["rewards"]])
    loss, _      = critic(critic_ids, critic_mask, labels=targets)
    loss.backward()
    critic_opt.step()
    critic_opt.zero_grad()

self.wrapped.pre_finetune(dataset, cfg)   # delegate (no-op for PPO / GRPO)
```

Runs for `cfg.critic_cfg.critic_epochs` epochs. The critic is frozen during
actor training (§3.2 `loss` below uses `torch.no_grad`).

**`loss(inputs, model_out, ref_out, cfg) → Tensor`**

Phase 2 of the finetune cycle, called inside `BaseTrainer.compute_loss`.

```
# 1. Greedy-decoded current generation
cur_tokens   = model_out.logits.argmax(dim=-1)                   # (B, T_labels)

# 2. Critic evaluation — frozen during actor step
critic_ids   = cat([inputs["input_ids"],   cur_tokens],              dim=1)
critic_mask  = cat([inputs["attention_mask"], ones_like(cur_tokens)], dim=1)
with no_grad:
    critic_logits = self.critic(critic_ids, critic_mask)         # (B, num_labels)
pred_buckets = critic_logits.argmax(dim=-1)
critic_rew   = tensor([label_to_score[b.item()] for b in pred_buckets]).to(device)

# 3. Blend with raw environment reward
w          = cfg.critic_cfg.reward_weight                        # 1.0 → pure critic
blended    = w * critic_rew + (1.0 - w) * inputs["rewards"]

# 4. Delegate to wrapped algorithm with substituted rewards
modified   = {**inputs, "rewards": blended}
base_loss  = self.wrapped.loss(modified, model_out, ref_out, cfg)

# 5. Optional MLM term (existing cfg.mlm_coef field)
return base_loss + cfg.mlm_coef * model_out.loss
```

Remark on step 4: when the wrapped algorithm is `GRPOAlgorithm`, `blended`
is the tensor that enters `_group_relative_advantages`. GRPO normalises
within groups regardless of whether the rewards are raw or critic-blended,
so group ordering invariants from v1 §6 are unaffected.

**`make_sampler(dataset, cfg) → Sampler | None`**

```python
return self.wrapped.make_sampler(dataset, cfg)
```

This single delegation line is why the Decorator is the right pattern. The
decorator does not know what `GRPOSampler` is; it simply forwards. GRPO
group ordering is preserved automatically.

### 3.3 Modified: `policy_gradient_algorithms/base.py`

Add `pre_finetune` as a third hook on the `PolicyGradientAlgorithm` protocol:

```python
class PolicyGradientAlgorithm(Protocol):
    def loss(...) -> torch.Tensor: ...
    def make_sampler(...) -> Optional[Sampler]: ...
    def pre_finetune(                             # NEW
        self,
        dataset: "RolloutDataset",
        cfg: "Config",
    ) -> None: ...
```

**`ppo.py` and `grpo.py`** each gain one method:

```python
def pre_finetune(self, dataset, cfg) -> None:
    return None
```

Same pattern as their existing `make_sampler`. Neither file changes beyond
this single method addition. Liskov Substitution (v1 §2.4) is preserved:
both algorithms remain drop-in substitutes behind the protocol.

### 3.4 Modified: `base_trainer.py`

One line added to `finetune()`:

```python
def finetune(self, dataset: RolloutDataset) -> None:
    if len(dataset) == 0:
        return
    self.algorithm.pre_finetune(dataset, self.cfg)  # NEW — no-op without critic
    self.train_dataset = dataset
    self.model.train()
    self.train()
    self.model.eval()
    ...
```

`BaseTrainer` remains algorithm-agnostic. It calls the abstract hook; whether
a critic trains is the decorator's concern. No import of `Critic` or
`CriticDecorator` appears here.

### 3.5 Modified: `config.py`

```python
@dataclass
class CriticConfig:
    model_name:         str         = "Salesforce/codet5p-220m"
    critic_lr:          float       = 1e-4
    critic_epochs:      int         = 1
    num_labels:         int         = 8
    reward_weight:      float       = 1.0    # 1.0 = CovRL pure critic; 0.0 = no critic signal
    # len(bucket_thresholds) must equal num_labels - 1
    bucket_thresholds:  list[float] = field(default_factory=lambda: [-0.5, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9])
    # len(bucket_values) must equal num_labels
    bucket_values:      list[float] = field(default_factory=lambda: [-1.0, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

@dataclass
class Config:
    ...
    use_critic:  bool                    = False   # NEW
    critic_cfg:  Optional[CriticConfig]  = None    # NEW — populated by load_config
```

`load_config` gains one block at validation time:

```python
if cfg.use_critic:
    cfg.critic_cfg = CriticConfig(**_sub_kwargs(raw, CriticConfig))
    if len(cfg.critic_cfg.bucket_thresholds) != cfg.critic_cfg.num_labels - 1:
        raise ValueError("len(bucket_thresholds) must be num_labels - 1")
    if len(cfg.critic_cfg.bucket_values) != cfg.critic_cfg.num_labels:
        raise ValueError("len(bucket_values) must be num_labels")
```

`policy_gradient_algorithm` stays `Literal["ppo", "grpo"]`. No change to
existing algorithm selection logic.

### 3.6 Modified: `rllm.py`

Composition (the only place new imports appear, v1 §3.7):

```python
# init():
algorithm = _build_algorithm(cfg)
if cfg.use_critic:
    from critic import Critic
    from policy_gradient_algorithms.critic_decorator import CriticDecorator
    critic    = Critic(cfg.critic_cfg)
    algorithm = CriticDecorator(algorithm, critic, cfg)
TRAINER = BaseTrainer(cfg, MUTATOR, algorithm)

# deinit():
if cfg.use_critic:
    ALGORITHM.save_critic(output_dir)
```

Two conditional blocks, both in the adapter. Nothing leaks to the domain core.

---

## 4. Updated lifecycle

The `queue_get → finetune_every → finetune()` section from v1 §4 gains one
sub-step inside the `finetune()` gate:

```
queue_get(filename)
│
├─ increment counter
│
└─ if counter % finetune_every == 0:
       dataset = RolloutDataset(BUFFER)

       ┌─ Phase 1: critic training (only if use_critic=True) ─────────────────┐
       │  for epoch in critic_epochs:                                          │
       │      for batch in DataLoader(dataset):                                │
       │          critic_input = cat(masked_input, gen_tokens)                 │
       │          target       = score_to_label(raw_reward)                    │
       │          loss, _      = critic(critic_input, labels=target)           │
       │          loss.backward(); critic_opt.step()                           │
       └──────────────────────────────────────────────────────────────────────┘

       ┌─ Phase 2: actor training (always) ───────────────────────────────────┐
       │  BaseTrainer.train():                                                 │
       │      for batch in DataLoader(dataset, sampler=make_sampler()):        │
       │          actor_out = actor(input_ids, labels)                         │
       │          ref_out   = ref_model(input_ids, labels)                     │
       │          [if use_critic]:                                             │
       │              critic_rew = critic(cat(input_ids, actor_out.argmax))    │
       │              blended    = w*critic_rew + (1-w)*raw_reward             │
       │          algorithm.loss(blended_inputs, actor_out, ref_out, cfg)      │
       │              → GRPO/PPO clipped surrogate on blended rewards          │
       │              + cfg.mlm_coef * actor_out.loss                          │
       └──────────────────────────────────────────────────────────────────────┘

       BUFFER.clear()
```

The invariants from v1 §4 are preserved without qualification:

- **Reward attribution is positional.** `post_run` still attaches environment
  rewards to the last buffer entry via `set_last_reward`. The critic is a
  consumer of those rewards at training time, not a replacement for the
  attribution mechanism.
- **Training is amortised.** Both phases run once per `finetune_every` entries.
  The per-cycle cost is higher, but the AFL++ hot path is unaffected.

---

## 5. Alternatives considered

**`ActorCriticAlgorithm` as a third `policy_gradient_algorithm` literal.**
Rejected (§2): embedding a specific base algorithm inside the actor-critic
class sacrifices GRPO's group normalisation and leads to combinatorial class
explosion.

**Shared actor–critic backbone (single model, two heads).** Rejected: LoRA
fine-tunes the actor's encoder for masked-span generation; adding a
classification head on the same encoder entangles two gradient signals with
different objectives. Separate models, separate optimisers, clean separation.

**Bucketing at `Rewarder.score()` time.** Rejected by SRP: the bucket
scheme is the critic's concern (thresholds live in `CriticConfig`). A change
to bucket boundaries should touch only `critic.py` and `config.py`, not
`rewarder.py` or `rollout.py`. Computing bucket labels from raw rewards at
training time (inside `pre_finetune`) keeps storage independent of the
classification scheme.

**Using raw rewards as advantages and critic rewards as an auxiliary term.**
Rejected: the critic's purpose is to replace the noisy environment signal with
a generalised, smoother estimate. An auxiliary-only role retains the noisy
signal in the advantage, gaining nothing from the critic's generalisation.

---

## 6. Deviations from CovRL-Fuzz — justified

| CovRL practice | rllm v2 choice | Reason |
|---|---|---|
| PPO as the base algorithm | GRPO (configurable; default) | GRPO's group normalisation compounds with the critic for double variance reduction (§1.4) |
| Critic trained on `cat(ctx, ground_truth_tokens)` | Critic trained on `cat(ctx, gen_tokens)` — actual rollout | Online setting: generated tokens with observed rewards are available. Training on the same distribution the critic evaluates is strictly more correct than ground-truth tokens, which are not available at inference time |
| `reward_weight` = 1.0 implicit | Explicit configurable weight in `CriticConfig` | Handles miscalibrated critic early in training; enables ablation; gradual adoption |
| Critic backbone: `T5Config()` (random T5-small, d_model=512) | `T5EncoderModel.from_pretrained(model_name)` (warm start, d_model matches actor) | Warm-start on code representations is better for reward prediction over code tokens |
| `mlm_coef` not separated from base loss | `CriticConfig.mlm_coef` alongside existing `cfg.mlm_coef` | Allows tuning the MLM term independently for actor-with-critic experiments versus baseline runs |
| Always active once running | `use_critic=False` by default | Opt-in; preserves v1 behaviour as the baseline; does not force users into higher memory/time cost |

---

## 7. Summary — principles applied (additions to v1 §7)

| Principle | Source | New addition |
|---|---|---|
| Decorator | GoF 1994 | `CriticDecorator` wraps any `PolicyGradientAlgorithm`; adds critic training + reward substitution without modifying the wrapped algorithm |
| Open/Closed | Meyer 1988 (SOLID-O) | `CriticDecorator` and `Critic` are pure additions; `PPOAlgorithm`, `GRPOAlgorithm`, `BaseTrainer`, `Rewarder`, `Rollout` are all unchanged except one no-op method per algorithm |
| Liskov Substitution | Liskov 1987 (SOLID-L) | `CriticDecorator` satisfies `PolicyGradientAlgorithm`; `PPOAlgorithm` / `GRPOAlgorithm` gain an explicit no-op `pre_finetune` and remain drop-in substitutes |
| Dependency Inversion | Martin 2002 (SOLID-D) | `CriticDecorator` receives `Critic` by injection; `BaseTrainer` calls `pre_finetune` via the abstract protocol, never importing `Critic` or `CriticDecorator` |
| SRP | Martin 2002 (SOLID-S) | `critic.py` owns only reward-prediction parameterisation and bucket mapping; `critic_decorator.py` owns only the composition of critic with any base algorithm |
| Hexagonal Architecture | Cockburn 2005 | `Critic` is a domain component; the AFL++ adapter (`rllm.py`) only wires it; no AFL++ symbol enters `critic.py` or `critic_decorator.py` |
| Reward generalisation | CovRL §3.3 | Critic trained on observed `(context, generation, reward)` triples; actor trained on critic-predicted rewards; generalises reward signal to unseen completions |
| Double variance reduction | This doc §1.4 | Critic smooths the reward signal; GRPO group normalisation reduces advantage variance; effects compound and are independent |
