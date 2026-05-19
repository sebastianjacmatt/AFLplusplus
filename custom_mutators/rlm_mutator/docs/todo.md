# CovRL-Fuzz GRPO Baseline — TODO

Goal: a working GRPO baseline where validity rate is stable across finetune cycles
and coverage reward trends upward.  Items are ordered by priority; 1-3 are
non-negotiable for a trustworthy baseline.

---

## 1. Fix and verify KL computation [DONE — grpo_v4]

**Status:** Fixed in `grpo.py`.

Switched from k1 estimator (`log_ratio = log_prob - ref_log_prob`, can be negative)
to the k3 estimator (`kl = (ratio - 1) - log_ratio`, always ≥ 0 by Jensen's inequality).

k1 caused `train/kl_mean` to drop to -2.856 in v2_5, making the KL term act as a
KL bonus (reinforcing drift rather than penalizing it).

Also fixed: `_ref_model.eval()` is now called in `snapshot_ref()` so LoRA dropout is
disabled in the reference forward, making `ref_log_prob` deterministic.

**Acceptance check:** `train/kl_min ≥ 0` at every logged step. If `kl_min` goes negative,
the estimator is broken. Watch `train/kl_mean` vs `train/actor_loss` ratio in tensorboard.

---

## 2. MLM/MSP SFT warmup before RL [DONE — base_trainer, config, rlm.py]

**Status:** Implemented in `BaseTrainer.sft_warmup()`.

Run 2k–5k steps of pure span-MLM SFT on the UglifyJS corpus using the exact masking
distribution used at fuzz time (mask_probability=0.15, spans 1–5, mean 3, whole_word_masking).
The SFT checkpoint is saved and used as both the initial policy AND the frozen reference for GRPO.

**Config fields:**
- `sft_corpus_path`: directory of valid JS files
- `sft_warmup_steps`: number of optimizer steps (recommend 2000–5000)

**How to activate:** set both fields in `grpo_v4.json` and point `sft_corpus_path` at the
AFL seed corpus directory. The SFT warmup runs once at init before any RL cycle.

**Why it matters:** The single biggest stability lever. A random-init LoRA adapter generates
almost entirely invalid JS. SFT pre-conditions the adapter to the masking reconstruction task
so the RL phase starts with a non-trivial validity prior.

---

## 3. Auxiliary MLM/MSP loss in GRPO objective [DONE — grpo.py, config]

**Status:** Implemented. New config field `mlm_coef` (default 0.0, recommend 0.1).

`L_total = L_GRPO + β * L_KL + λ * L_MLM`

`L_MLM = -mean(log_prob | reward ≥ 0)` — CE on valid rollout samples only.

This is a no-extra-forward-pass implementation: `log_prob` is already computed in
`compute_loss`. The CE term gives the model a direct supervised signal toward valid-program
generation, independent of within-group reward variance. Fires even when most rollout groups
are all-invalid, as long as some valid samples survive pre-filtering into the batch.

**Watch:** `train/mlm_loss` in tensorboard. Should decrease across finetune cycles as the
model produces more valid programs.

---

## 4. Add advantage clipping [DONE — grpo.py, config]

**Status:** Implemented. New config field `GRPOConfig.advantage_clip` (default 0.0).
Set to 3.0 in `grpo_v4.json`.

Clips |A| ≤ advantage_clip before computing the policy loss. Prevents one-off lucky
valid samples in mostly-invalid groups from dominating gradients with advantages of
magnitude 4–5+.

---

## 5. Instrument the training loop [DONE — grpo.py]

**Status:** Added to `_log_train_scalars`:
- `train/kl_min` — detects negative KL (broken estimator)
- `train/validity_rate_in_batch` — fraction of batch samples with reward ≥ 0
- `train/mlm_loss` — auxiliary CE loss value (0.0 when disabled)

Already logged: `train/actor_loss`, `train/kl_mean`, `train/clip_fraction`,
`train/advantage_mean/std/min/max`, `train/ratio_mean/std/min/max`,
`train/reward_mean_in_batch`, `train/group_count_in_batch`.

Also logged at rollout time: `rollout/valid_rate`, `rollout/all_invalid_group_rate`,
`rollout/degenerate_group_rate`, `rollout/reward_mean/std`.

---

## 6. Raise kl_coef if needed — but only after KL is verified [PENDING]

`kl_coef = 0.05` is the starting point in `grpo_v4.json`.

Once k3 KL is confirmed non-negative, watch `train/kl_mean` trajectory:
- If KL crosses ~0.5–1.0 nats within the first few updates AND validity drops: raise β to 0.2.
- If KL stays near 0 and coverage gradient is being suppressed: reduce to 0.02.

Do not tune β blind — wait for a confirmed trustworthy KL readout.

---

## 7. What is already correct — do not change [REFERENCE]

- LoRA config: `r=8, alpha=16, target=q,v`
- `learning_rate = 2e-5`
- `clip_epsilon = 0.2`
- `group_size = 16`
- Pre-filter for zero-variance groups before batching
- Batch-alignment trim so dataset is divisible by batch_size
- Static reference policy with `ref_update_every = 3`
- `validity_bonus = 0.3` (3-level reward floor: syntax=-1 / semantic=-0.5 / valid≥0.3)
- `bf16 = true`

---

## 8. Acceptance criteria before declaring the baseline working [PENDING]

A run passes the baseline bar if ALL of the following hold:

1. `train/kl_min ≥ 0` at every step. If kl_min < 0, stop — the KL estimator is broken.
2. Validity rate stays within ~10 percentage points of the SFT checkpoint's validity rate
   across at least 5 finetune cycles. Collapse from 15% to 1% over 3 cycles = failure.
3. Coverage reward (`rollout/coverage_reward_mean`) trends upward relative to a
   non-RL MSP baseline (same masking, no finetuning). If coverage is flat or decreasing
   while validity is stable, the reward shaping may be misaligned.
4. `train/mlm_loss` decreases across finetune cycles (model reproduces valid outputs
   more reliably). Flat or increasing MLM loss while validity collapses = the RL objective
   is destroying the SFT initialization faster than the CE term restores it — increase
   `mlm_coef` or reduce `learning_rate`.
