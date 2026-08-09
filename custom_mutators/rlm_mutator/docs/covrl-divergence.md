# Divergences from CovRL-Fuzz

This document describes where our implementation aligns with CovRL-Fuzz (Eom et al., ISSTA 2024) and where it deliberately diverges, with reasoning for each divergence. Written to support the methodology section and respond to reviewers.

---

## Aligned with CovRL-Fuzz

- CodeT5+ 220m as the policy model
- TF-IDF coverage reward (Eq. 4-5), IDF momentum α=0.6
- Reward signal: syntax=−1.0, semantic=−0.5, valid=R_cov
- top_k=32, 1 epoch per finetune cycle
- Span/mask mutation by fill-in-the-middle (mutation by mask, Figure 1b)
- RL finetuning loop integrated with AFL

---

## Divergences

### 1. Algorithm: PPO → GRPO (intentional)

CovRL's loss is IS-reweighted CE: `−E[R · log(π_new/π_old)]`. Ours is clipped group-relative advantage with within-group z-score normalization (GRPO). GRPO removes the need for a separate critic. We also apply PPO-style ratio clipping, which Eq. 7 in the paper does not show explicitly but is implied by their reference to the PPO algorithm.

### 2. No separate rewarder model

CovRL trains two models in tandem: a mutator (actor) and a learned rewarder (critic) trained with contrastive search to predict the reward signal. We compute rewards directly from the coverage bitmap and exit code. No second model, no contrastive training.

### 3. Training dataset: AFL-interesting seeds → all rollouts

CovRL's finetuning dataset D_T is curated by AFL's `IsInteresting()` filter — only mutations that discovered new coverage enter training (Algorithm 1, lines 10-15). We train on all rollouts, filtering only incomplete GRPO groups and zero-variance groups (where all rewards within a group are identical, producing zero advantage for every member).

We never read AFL's interesting flag. The implication: GRPO's within-group advantage normalization is our selection mechanism. It produces gradient signal from relative variance within a group even when no sample would pass AFL's IsInteresting test, as long as rewards differ across the group. This is a feature of GRPO, not a limitation — group-relative normalization replaces survivorship-biased curation with within-group contrast as the source of gradient signal.

### 4. Auxiliary objective composition: unified loss vs. decoupled terms

**The precise difference:**

CovRL mixes corpus examples into D_T alongside rollout samples and processes them through the same PPO objective. The implicit assumption is that corpus examples either (a) receive reward 0 and contribute only through KL, or (b) receive a real reward via execution — the paper is ambiguous. Either way, the unified-objective philosophy is: *all training data passes through the same policy gradient loss; the reward structure differentiates examples.*

Our corpus MLM loss is a separate term added to the GRPO loss: `loss = actor_loss + kl_coef * kl + mlm_coef * mlm_loss`. Corpus examples receive standard CE, with no IS weighting, no KL penalty, and no reward modulation. The decoupled-objective philosophy is: *different signals for different purposes, summed at the loss level.*

**Arguments for the unified approach (CovRL, InstructGPT, CodeRL):**

- Conceptual elegance. One loss, one optimizer step, one set of hyperparameters.
- Reward-modulated regularization. Corpus samples participate in the reward-and-KL system, so they get pulled harder when the policy drifts further. This is what InstructGPT does and it is mathematically clean.
- No objective conflict. RL and supervised signals cannot be in tension because there is only one signal.

**Arguments for the decoupled approach (DeepSeekMath, R1, RLTF, this work):**

- **Structural compatibility with GRPO** (load-bearing). GRPO computes advantages within groups of rollouts that share a masked context. Corpus examples do not have groups in this sense — they are individual MLM samples with no peer rollouts to z-score against. Mixing them into the rollout batch would either break the advantage computation or require an ad-hoc treatment of corpus examples as singleton groups, which is conceptually muddled. CovRL did not face this problem because PPO uses per-sample advantages from a value network. GRPO removed the value network and replaced it with group normalization — which makes corpus mixing in the unified style structurally infeasible.
- Independent signal monitoring. `train/mlm_loss` and `train/actor_loss` are observable separately. If validity collapses, it is diagnosable whether the RL signal drifted the policy off-manifold or the MLM signal failed to anchor. With a unified loss, these signals are entangled.
- Reward-distribution independence. The MLM signal has constant magnitude regardless of reward shaping. In fuzzing, rewards skew heavily negative early (most outputs are invalid), so reward-modulated approaches give very weak signal on the rare valid samples in those early cycles. The separate CE term provides the same gradient pressure toward valid JS distribution whether rewards are mostly −1.0 or mostly +0.5.
- Hyperparameter modularity. `mlm_coef` is independent of `kl_coef`, `clip_epsilon`, and reward shaping. With unified mixing, all four quantities interact, making ablations harder to interpret.
- Aligns with the modern lineage. Every GRPO-based system since DeepSeekMath uses separate auxiliary loss terms. R1, RLTF, and others follow the same structure. This is not a coincidence — it is because the group structure makes the unified approach impractical and the decoupled approach makes debugging tractable.

**Arguments against the decoupled approach:**

- Risk of explicit objective conflict. RL and MLM can pull in different directions. Without reward modulation, this conflict does not auto-resolve based on how well training is going. `mlm_coef` must be tuned empirically.
- No principled connection between corpus signal and policy state. In the unified objective, corpus regularization scales with policy drift via the KL term. In the decoupled approach, it is a constant pull regardless of how far the policy has moved.

**Should we align with CovRL here? No.**

The structural argument is decisive: GRPO + within-group advantages + corpus mixing in the same loss is not a coherent design. You would have to either drop GRPO's group structure (defeating the point of using GRPO) or invent an ad-hoc treatment of corpus examples as their own "groups," which breaks the advantage computation. The separate CE term is the natural design for a GRPO-based system. CovRL did not face this problem because they used PPO.

This is also exactly the philosophical shift DeepSeekMath argued for when introducing GRPO: remove the value network, use group statistics, treat auxiliary objectives as separate terms. Aligning with CovRL here would mean adopting GRPO's algorithm but the pre-GRPO objective structure, which is incoherent.

### 5. KL structure: reward-folded vs. loss-added

CovRL folds KL directly into the reward (Eq. 8: `R(x,y) = r(W*) + log(π_t/π_{t-1})`), so KL appears inside the IS-weighted objective — effectively scaling the KL penalty by the reward magnitude. Invalid mutations (large negative reward) get stronger KL pull back toward the reference. This is conceptually similar to reward-shaping rather than a separate regulariser.

Our KL is a fixed-magnitude separate loss term: `actor_loss + kl_coef * k3_kl.mean()`. KL penalty is constant regardless of reward magnitude. Both are valid; ours matches modern RLHF practice (TRL, DeepSpeed-Chat, all GRPO papers). Our k3 estimator (`(ratio−1) − log_ratio ≥ 0`) also guarantees non-negativity, which CovRL's k1 log-ratio formulation does not.

### 6. Masking: 3 strategies → span only

CovRL uses Insert (add [MASK] tokens), Overwrite (replace existing tokens with [MASK]), and Splice (replace a segment with a segment from another seed, formatted as [MASK]). We use only span masking — the CodeT5+ pre-training distribution (contiguous span corruption). No cross-seed splice.

### 7. Pre-RL SFT warmup (ablation in progress)

CovRL has no explicit SFT phase. They rely on CodeT5+'s span-infilling pretraining plus continuous corpus mixing during RL to maintain valid-output behavior from the start.

We optionally run a pre-RL SFT phase (`scripts/sft_warmup.py`) producing a LoRA adapter that becomes both the initial policy and the frozen KL reference (`pi_ref`). This addresses cold-start validity collapse observed in our GRPO+LoRA setup — zero-init LoRA on CodeT5+ starts with valid-rate ~16% and collapses toward ~1% within 6 finetune cycles without anchoring. CovRL's PPO+full-FT configuration did not exhibit this because full fine-tuning preserves the pretrained distribution more conservatively than zero-init LoRA adapters.

The current ablation (`grpo_v4_no_warmup.json` vs `grpo_v5.json`) tests whether MLM aux loss alone (CovRL-style, no warmup) is sufficient to hold validity, or whether the pre-RL SFT warmup is necessary. The two configs differ only in `sft_adapter_path`.

---

## Summary table

| Aspect | CovRL-Fuzz | This work |
|---|---|---|
| RL algorithm | PPO (actor-critic) | GRPO (group-relative, critic-free) |
| Rewarder | Separate learned model (contrastive) | Direct bitmap + exit code |
| Training data | AFL-interesting seeds only | All rollouts (zero-variance groups dropped) |
| Corpus signal | Unified: mixed into PPO loss | Decoupled: separate CE term |
| KL structure | Reward-folded (k1) | Loss-added (k3, always ≥ 0) |
| Masking | Insert + Overwrite + Splice | Span masking only |
| Pre-RL warmup | None | Optional SFT adapter (ablation pending) |
| Fine-tuning mode | Full FT | LoRA (r=8, q+v) |
