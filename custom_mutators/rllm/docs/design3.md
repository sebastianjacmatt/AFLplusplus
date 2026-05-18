# rllm — Design v3: Loss Functions and Actor-Critic Clarification

This document is a mathematical reference for all four training variants in
rllm. It complements [design.md](design.md) (architecture) and
[design2.md](design2.md) (Decorator design). It answers the question: *what
exactly is computed in each case, and why is GRPO + Critic not classical
Actor-Critic?*

---

## 1. Notation

| Symbol | Meaning | Config field |
|---|---|---|
| `x` | Masked input (context with sentinel tokens) | — |
| `y` | Generated token sequence (completion) | — |
| `y_t` | Token at decoder step `t` | — |
| `B` | Training batch size | `train_batch_size` |
| `T` | Sequence length (decoder steps) | — |
| `mask_{i,t}` | 1 if `labels[i,t] ≠ -100`, else 0 | — |
| `π_θ(y_t\|x, y_{<t})` | Current policy (actor) | — |
| `π_old(y_t\|x, y_{<t})` | Policy at rollout time (frozen) | — |
| `π_ref(y_t\|x, y_{<t})` | Reference policy snapshot (frozen) | — |
| `ρ_{i,t}` | Per-token importance ratio (see §2) | — |
| `r_i` | Raw environment reward for sample i | — |
| `ε` | PPO clipping range | `clip_epsilon` = 0.2 |
| `β` | KL penalty coefficient | `kl_coef` = 0.0 |
| `γ` | Entropy coefficient (PPO only) | `ppo.entropy_coef` = 0.01 |
| `G` | GRPO group size | `grpo.group_size` = 8 |
| `ε_g` | GRPO normalisation floor | `grpo.norm_epsilon` = 1e-8 |
| `c_g` | GRPO advantage clip (0 = off) | `grpo.advantage_clip` = 0.0 |
| `f_φ` | Critic network | — |
| `B(r)` | Bucket index for reward r | — |
| `l2r(k)` | Representative scalar for bucket k | — |
| `w` | Critic reward blend weight | `reward_weight` = 1.0 |
| `λ` | MLM loss coefficient | `mlm_coef` = 0.0 |

---

## 2. Shared building blocks

These functions live in `policy_gradient_algorithms/base.py` and are used by
all four variants.

**Per-token log-probabilities**

    log π(y_t | x, y_{<t})

Computed by gathering from `log_softmax(logits)` at the label token position.
Padding positions (`labels = -100`) return 0 and are excluded by `mask`.

**Masked mean** over non-padding token positions:

    masked_mean(z, mask) = Σ_{i,t} mask_{i,t} · z_{i,t}  /  Σ_{i,t} mask_{i,t}

**Importance ratio** (per token):

    ρ_{i,t} = exp(log π_θ(y_{i,t}|...) − log π_old(y_{i,t}|...))

`log π_old` comes from the rollout buffer (`old_logprobs`), recorded at
`mutate()` time. It is the policy that generated the token, not the reference.

**Clipped surrogate** (per-token PPO lower bound):

    clip(ρ_{i,t}, Â_i) = min(ρ_{i,t} · Â_i,  clamp(ρ_{i,t}, 1−ε, 1+ε) · Â_i)

Note: `Â` is a per-sample scalar broadcast to the token dimension. All tokens
in sample i share the same advantage.

**KL penalty** (per-token, evaluated at rollout tokens):

    KL_i = masked_mean(log π_θ(y_t|x) − log π_ref(y_t|x), mask)

This is an on-policy sample estimate of `KL(π_θ || π_ref)`. When `kl_coef = 0`
(the default) it vanishes.

---

## 3. The environment reward function

Shared by all variants. Implemented in `rewarder.py`.

    r(W*) =  −1.0                         if W* produces a syntax error
             −0.5                         if W* produces a semantic error
             b + (1−b) · R_cov(W*)        if W* executes without errors

where `b = validity_bonus` (default 0.0) and the coverage reward is (CovRL
Eqs. 3–6):

    R_cov = σ( log( Σ_i  tf_i · idf_{i,t−1} ) )

    tf_i     = 1[ bitmap[i] > 0 ]                          (binary TF, Eq. 3)

    idf_{i,t} = α · idf_{i,t−1}
              + (1−α) · (1/√M) · log( N_t / (1 + df_i) )  (EMA IDF, Eq. 6)

`σ` is the sigmoid function. `M` is the bitmap size. `N_t` is the number of
executions seen so far. `df_i` is the document frequency of edge i. `α` is
`idf_alpha` (default 0.6).

This reward is a continuous scalar in `{−1.0, −0.5} ∪ (0.5, 1.0]`. The
0.5 floor comes from `TFIDFCoverageRewarder.FLOOR` for cold-start / zero
coverage executions.

---

## 4. PPO — batch-mean REINFORCE baseline

**File:** `policy_gradient_algorithms/ppo.py`
**Config:** `policy_gradient_algorithm = "ppo"`, `use_critic = false`

### 4.1 Advantage

A batch-mean baseline is subtracted from each sample's reward:

    Â_i^PPO = r_i − (1/B) Σ_j r_j

This is equivalent to REINFORCE with baseline where the baseline is the
batch mean. No value function is learned.

### 4.2 Loss

    L_PPO(θ) = masked_mean( −clip(ρ, Â^PPO), mask )
             + β · masked_mean( log π_θ − log π_ref, mask )
             − γ · masked_mean( −log π_θ, mask )

Written out in full:

    L_PPO(θ) = − (1/|mask|) Σ_{i,t} mask_{i,t} · min( ρ_{i,t} · Â_i,
                                                        clamp(ρ_{i,t}, 1−ε, 1+ε) · Â_i )
             + β · (1/|mask|) Σ_{i,t} mask_{i,t} · ( log π_θ(y_{i,t}) − log π_ref(y_{i,t}) )
             + γ · (1/|mask|) Σ_{i,t} mask_{i,t} · log π_θ(y_{i,t})

The third term is `−γ · H̃(π_θ)` where `H̃` is a per-token entropy estimate
using the sampled tokens. When `γ > 0` this penalises low-entropy (peaked)
policies, encouraging exploration.

**Defaults:** `kl_coef = 0.0` (KL term off), `entropy_coef = 0.01`.

---

## 5. GRPO — group-relative normalisation

**File:** `policy_gradient_algorithms/grpo.py`
**Config:** `policy_gradient_algorithm = "grpo"`, `use_critic = false`

### 5.1 Group structure

Each mask call produces `fuzz_count` completions from the same masked context.
These form GRPO groups of size `G`. The rollout buffer preserves insertion
order, so group g contains samples `{(g−1)G+1, ..., gG}` — samples sharing
the same masked context `x_g`.

### 5.2 Advantage

Within each group, rewards are normalised by the group's own mean and standard
deviation:

    μ_g = (1/G) Σ_{j=1}^{G} r_{g,j}

    σ_g = sqrt( (1/G) Σ_{j=1}^{G} (r_{g,j} − μ_g)² )

    Â_{g,j}^GRPO = (r_{g,j} − μ_g) / (σ_g + ε_g)

If `advantage_clip > 0`: `Â = clamp(Â, −c_g, c_g)`.

The group mean `μ_g` acts as the baseline. Unlike PPO's batch-mean baseline,
GRPO's baseline is specific to the masked context that produced the group,
not averaged over different contexts in the batch. This reduces variance for
a group of samples that are informatively comparable (same prompt).

### 5.3 Loss

    L_GRPO(θ) = masked_mean( −clip(ρ, Â^GRPO), mask )
              + β · masked_mean( log π_θ − log π_ref, mask )

No entropy term. No value head.

**Defaults:** `kl_coef = 0.0`, `group_size = 8`, `norm_epsilon = 1e-8`.

---

## 6. PPO + Critic — CovRL-style

**Files:** `policy_gradient_algorithms/ppo.py` + `critic_decorator.py`
**Config:** `policy_gradient_algorithm = "ppo"`, `use_critic = true`

This is the closest to the published CovRL-Fuzz method.

### 6.1 Critic training (Phase 1, per finetune cycle)

The critic `f_φ` is a T5 encoder with a linear classification head (`critic.py`).
It is trained as a supervised multi-class classifier on the rollout data.

**Input:** `cat(x_i, y_i)` — the masked context concatenated with the
recorded generated tokens from the rollout. Padding in `y_i` is masked out
in the attention mask.

**Target:** `B(r_i) = score_to_label(r_i)` — the bucket index for the
observed raw reward.

The default 8-bucket scheme (reproducing CovRL's `base_utils.py`):

| Label k | Reward range       | `l2r(k)` |
|---------|--------------------|----------|
| 0       | r ≤ −0.5           | −1.0     |
| 1       | −0.5 < r ≤ 0.0     | −0.5     |
| 2       | 0.0 < r ≤ 0.5      |  0.5     |
| 3       | 0.5 < r ≤ 0.6      |  0.6     |
| 4       | 0.6 < r ≤ 0.7      |  0.7     |
| 5       | 0.7 < r ≤ 0.8      |  0.8     |
| 6       | 0.8 < r ≤ 0.9      |  0.9     |
| 7       | 0.9 < r             |  1.0     |

**Critic loss** (CrossEntropy, one epoch per cycle):

    L_critic(φ) = − (1/N) Σ_{i=1}^{N} log p_φ( B(r_i) | cat(x_i, y_i) )

where `p_φ(k | z) = softmax(f_φ(z))_k`.

### 6.2 Critic-predicted reward (Phase 2)

During actor training the frozen critic evaluates the actor's *current greedy
predictions*, not the rollout tokens. The greedy tokens are obtained from the
teacher-forced logits in a single forward pass — no autoregressive decode:

    ŷ_{i,t} = argmax_v  π_θ(v | x_i, y_{i,<t})^{teacher-forced}
             = model_out.logits[i, t, :].argmax()

This is the argmax at each decoder step given the ground-truth prefix from the
rollout (teacher forcing). It approximates the true greedy output but avoids
a second autoregressive forward pass.

The critic-predicted reward:

    k̂_i   = argmax_k  f_φ( cat(x_i, ŷ_i) )_k
    r̂_i   = l2r(k̂_i)

The blended reward:

    r̃_i   = w · r̂_i  +  (1−w) · r_i

where `w = reward_weight`. At `w = 1.0` (default): pure critic signal,
matching CovRL. At `w = 0.0`: raw environment reward only.

### 6.3 Actor loss

Batch-mean advantage on the blended rewards:

    Â_i^PPO+C = r̃_i − (1/B) Σ_j r̃_j

    L_PPO+Critic(θ) = masked_mean( −clip(ρ, Â^PPO+C), mask )
                    + β · masked_mean( log π_θ − log π_ref, mask )
                    − γ · masked_mean( −log π_θ, mask )
                    + λ · L_MLM(θ)

The MLM term:

    L_MLM(θ) = − (1/|mask|) Σ_{i,t} mask_{i,t} · log π_θ(y_{i,t} | x_i, y_{i,<t})

This is the standard teacher-forcing reconstruction loss (the seq-to-seq CE
loss returned by T5ForConditionalGeneration when `labels` are passed). It
prevents catastrophic forgetting of the model's masked-span prediction
capability. **Default `mlm_coef = 0.0`** so it is off unless explicitly set.

---

## 7. GRPO + Critic — rllm v2

**Files:** `policy_gradient_algorithms/grpo.py` + `critic_decorator.py`
**Config:** `policy_gradient_algorithm = "grpo"`, `use_critic = true`

This is the novel combination described in [design2.md](design2.md) §1.4.

### 7.1 Critic training (Phase 1)

Identical to §6.1. The critic sees the same rollout data regardless of which
base algorithm is selected.

### 7.2 Critic-predicted reward (Phase 2)

Identical to §6.2: teacher-forced greedy argmax → critic bucket prediction →
`l2r` → blend.

    r̃_{g,j} = w · r̂_{g,j}  +  (1−w) · r_{g,j}

### 7.3 Actor loss

Group-relative normalisation is applied to the *blended* rewards, not the raw
rewards:

    μ̃_g = (1/G) Σ_{j=1}^{G} r̃_{g,j}

    σ̃_g = sqrt( (1/G) Σ_{j=1}^{G} (r̃_{g,j} − μ̃_g)² )

    Â_{g,j}^GRPO+C = (r̃_{g,j} − μ̃_g) / (σ̃_g + ε_g)

    L_GRPO+Critic(θ) = masked_mean( −clip(ρ, Â^GRPO+C), mask )
                     + β · masked_mean( log π_θ − log π_ref, mask )
                     + λ · L_MLM(θ)

The two variance-reduction mechanisms are independent and sequential:

1. **Critic** converts noisy raw rewards `r_{g,j}` into smooth bucket-predicted
   scalars `r̂_{g,j}`. Predictions for similar completions are close together;
   outliers are reduced.
2. **GRPO** normalises the blended rewards `r̃_{g,j}` within each group,
   centering around the group mean and scaling by the group std. This removes
   between-group reward scale differences and focuses gradients on within-group
   relative ordering.

---

## 8. What this is not: classical Actor-Critic

A classical Actor-Critic (e.g. A2C, A3C) has:

- **Critic:** learns `V(s)` — the expected *future cumulative return* from
  state s under the current policy.
- **Advantage:** `A(s, a) = r + γ·V(s') − V(s)` — the TD error.

None of these variants implement that. The critic in rllm (following CovRL)
is a **discriminative reward model**: given `(context, completion)`, it
predicts which reward class *this specific completion achieves immediately*. It
does not model future returns, trajectories, or a value function. There is no
Bellman backup, no temporal difference, no bootstrapping.

| Property | Classical Actor-Critic | rllm Critic |
|---|---|---|
| What critic learns | `V(s)` — expected future return | `B(r)` — immediate reward class |
| Advantage formula | `r + γV(s') − V(s)` (TD error) | `r̂ − mean(r̂)` or `(r̂ − μ_g)/σ_g` |
| Critic output | Scalar V-value | 8-class distribution |
| Critic training | TD or MC targets | Supervised CE on observed rewards |
| Temporal structure | Bootstraps across timesteps | Per-execution, no bootstrapping |

The rllm setup is closer to a **reward model** (as in RLHF) than to a value
function: a separate model trained to predict reward from a token sequence,
whose output replaces or augments the raw environment signal.

The "Actor-Critic" terminology in CovRL-Fuzz refers to the two-model
structure (one generates, one evaluates), borrowing the name from the RL
literature for the general pattern of having a separate evaluator guide the
policy.

---

## 9. Variance reduction — quantitative comparison

Let the raw reward variance within a group be `Var_g(r)`.

| Variant | Advantage | Residual variance |
|---|---|---|
| PPO | `r_i − mean_batch(r)` | `Var_batch(r)` minus between-context signal removed |
| GRPO | `(r_{g,j} − μ_g) / σ_g` | Always unit variance per group by construction |
| PPO + Critic | `r̂_i − mean_batch(r̂)` | `Var_batch(r̂)` ≤ `Var_batch(r)` since critic is a smooth function of tokens |
| **GRPO + Critic** | `(r̃_{g,j} − μ̃_g) / σ̃_g` | **Unit variance per group** on the already-smooth `r̃` |

GRPO always produces unit within-group advantage variance by construction, but
the *signal-to-noise ratio* of the normalised advantages depends on how much
variation the rewards show within each group. If raw rewards within a group
are nearly identical (all syntax errors, or all similar coverage), `σ_g ≈ 0`
and advantages collapse near zero — the group contributes no learning signal.

The critic addresses this: by predicting smooth bucket scores that vary based
on token content rather than the binary outcomes of execution, it introduces
variation within groups that the raw reward might not. After critic blending,
`σ̃_g` is more reliably non-zero.

**The risk:** if the critic is miscalibrated early in training and predicts
the same bucket for all samples in a group, `σ̃_g` still collapses. The
`reward_weight` knob handles this: start with `w < 1.0` to retain raw-reward
variation, increase as the critic converges.

---

## 10. Configuration reference

All four variants are selected by two config fields. Everything else is common.

```json
{
  "policy_gradient_algorithm": "grpo",
  "use_critic": false
}
```

| Variant | `policy_gradient_algorithm` | `use_critic` |
|---|---|---|
| PPO (§4) | `"ppo"` | `false` |
| GRPO (§5) | `"grpo"` | `false` |
| PPO + Critic / CovRL-style (§6) | `"ppo"` | `true` |
| GRPO + Critic / rllm v2 (§7) | `"grpo"` | `true` |

### Relevant hyperparameters when `use_critic = true`

```json
{
  "use_critic":      true,
  "model_name":      "Salesforce/codet5p-220m",
  "critic_lr":       1e-4,
  "critic_epochs":   1,
  "num_labels":      8,
  "reward_weight":   1.0,
  "mlm_coef":        0.0,
  "bucket_thresholds": [-0.5, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
  "bucket_values":     [-1.0, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

Set `model_name = ""` for a randomly initialised T5-small critic (faster
startup, matches CovRL's original implementation). Set `reward_weight = 0.0`
to reduce to the non-critic variant with zero overhead (the critic is trained
but its predictions are not used in the actor loss).

The `mlm_coef` field controls the MLM regularisation term added by the
decorator. CovRL uses `mlm_coef = 1.0`. rllm defaults to `0.0` to preserve
the baseline behaviour.
