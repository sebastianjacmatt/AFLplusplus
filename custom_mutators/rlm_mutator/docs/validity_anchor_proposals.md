# Validity Anchor Proposals

**Context:** grpo_v3_1 run confirmed that `ref_update_every=3` anchors KL direction correctly
(17.5% positive steps vs 2.9% in v2_5). However `kl_coef=0.005` is ~20× too weak to compete
with the actor loss, so the improved anchor contributes near-zero gradient. Valid rate still
collapses monotonically: ft0=10.5% → ft1=16.5% → ft6=1.0%.

**Root cause (from grpo_v2_critique.md):** GRPO has no gradient toward validity when rollout
groups are all-invalid. CovRL-Fuzz avoids this by mixing 4× valid corpus with CE loss into
every training batch. We have no equivalent. The KL term is the only indirect validity anchor,
and it is too weak and too fast-moving to provide real pull.

**Goal:** Make the language model trend toward higher validity across finetune iterations rather
than collapsing after ft1. Coverage optimization is the primary long-term objective but requires
validity to be stable first.

---

## Scale 1 — Strengthen the existing KL anchor (config only)

**Change:** `kl_coef: 0.005 → 0.05`, `learning_rate: 1e-4 → 2e-5`

**Why it helps:** With `ref_update_every=3`, the frozen reference now genuinely represents a
less-drifted (more valid) policy. But at kl_coef=0.005 the KL term contributes ~0 to the loss
relative to actor loss. At 0.05 it competes: during the ft1→ft3 window, the reference is
anchored to the ft0 snapshot (10.5% valid rate), and the KL penalty actively resists moving away
from it.

`learning_rate: 2e-5` addresses the 41% clip fraction. High LR means learned valid patterns from
ft1 get overwritten in the next large update. Reducing LR does not add validity gradient but
slows the rate of destruction.

**Risk:** If coverage signal is sparse (5-15% valid rate), a dominant KL term may suppress the
coverage gradient. Watch `train/kl_mean` vs `train/actor_loss` ratio in tensorboard. If KL
dominates and valid_rate is already recovering, reduce back to 0.02.

**Config:** `grpo_v3_2.json` — `kl_coef=0.05`, `learning_rate=2e-5`, `ref_update_every=3`.

---

## Scale 2 — CE regularization on valid rollout samples (grpo.py, ~15 lines)

**Change:** Add CE loss term in `compute_loss` over samples where `reward >= 0.0`.

**Mechanism:** Every training batch that survives pre-filtering and makes it to `compute_loss`
contains some valid programs (groups are non-degenerate because they have mixed validity, meaning
at least one valid sample per group). Those valid programs' output sequences are already in
`labels`. A CE term on those positions gives the model a direct supervised signal: "reproduce
what you generated that was valid." This is dense (all tokens of all valid sequences) and does
not depend on within-group variance.

```python
# grpo.py — compute_loss, after computing log_prob and advantages
ce_coef = self.training_cfg.ce_coef  # new field, default 0.0
if ce_coef > 0.0:
    valid_mask = (reward >= 0.0).float()
    if valid_mask.sum() > 0:
        ce_loss = -(log_prob * valid_mask).sum() / valid_mask.sum()
        loss = actor_loss + kl_coef * kl.mean() + ce_coef * ce_loss
```

**Why it works even at low valid_rate:** A group of mixed validity (e.g., 2 valid + 14 invalid)
survives pre-filtering and reaches compute_loss. Those 2 valid samples get both:
- GRPO advantage signal (they outperformed their 14 invalid peers → positive advantage)
- CE reinforcement signal (reproduce this valid sequence)

The CE term gradient is independent of group structure. It fires regardless of how rare valid
samples are, as long as any survive into the batch.

**New config field:** `ce_coef: float = 0.0` in `TrainingConfig`. Start at 0.1.

**Risk:** CE reinforces the current valid distribution, which may overfit to the current AFL
corpus before it shifts. If the corpus shifts significantly between cycles, the CE-reinforced
patterns may become incorrect for the new seeds. Counterbalanced by the short rollout cycle
(finetune_every=16).

**Status:** Not yet implemented.

---

## Scale 3 — Corpus injection (CovRL architecture, new component)

**Change:** Load a static valid JS corpus at `init()`. Each `maybe_finetune`, sample N corpus
entries, inject into training batch with CE loss only. No GRPO objective on corpus samples.

**Why it is the structural fix:** Even when 95% of rollout groups are degenerate (all-invalid),
corpus samples in the batch still push the model toward valid JS generation. With a 1:4
rollout:corpus ratio (mirroring CovRL), 80% of the batch gradient is always supervised valid JS.
The model cannot fully escape its validity prior regardless of what the rollout produces.

**Implementation sketch:**
- New config: `corpus_path` (directory or JSONL file of valid JS), `corpus_mix_ratio: 0.2`
- At `init()`: load corpus, tokenize, store as a fixed pool
- In `maybe_finetune`: sample `len(rollout_records) * corpus_mix_ratio` corpus entries,
  create `RolloutDataset` entries with `is_corpus=True`, `reward=None`, `group_id=-1`
- In `compute_loss`: detect `is_corpus` flag, apply CE on those samples, GRPO on rollout only
- In `GroupedBatchSampler`: corpus samples slot into batches without group constraint

**Corpus source options:**
- AFL's initial seed corpus (directory specified via config or env var) — hand-selected valid JS
- External JS corpus (e.g., subset of the-algorithms/javascript or similar)
- AFL's queue after the first hour of fuzzing (seeds promoted to queue are valid by AFL's standard)

**VRAM impact:** Corpus samples are additional forward passes. At `corpus_mix_ratio=0.2` and
rollout batch 32, effective batch grows to ~38. bf16 keeps this manageable on 24GB. Verify before
committing to higher ratios.

**Status:** Not yet implemented. Requires corpus source decision before implementation.

---

## Scale 4 — Two-phase training per cycle (separate BC + GRPO passes)

**Change:** Before the GRPO update, run a separate behavior cloning pass on valid rollout samples
only. Then run the standard GRPO pass on all non-degenerate groups.

**Mechanism:** Phase 1 (BC): filter rollout to valid-only records, run supervised epoch. This
pre-conditions the model to reinforce valid outputs before coverage-differentiated GRPO fires.
Phase 2 (GRPO): standard group-relative update on all surviving non-degenerate groups.

**Why separate phases:** Combining CE + GRPO in one loss (Scale 2) means the CE gradient on
valid samples may be partly cancelled by the GRPO gradient on those same samples (valid sample
may have positive or negative advantage depending on group composition). Separate phases give
each objective a clean update step.

**Cost:** Doubles training time per finetune cycle. On a 3090, typical cycle is 3–21 seconds
already — doubling is acceptable.

**Status:** Not yet implemented. Lower priority than Scale 2 since Scale 2 achieves similar
benefits at lower implementation cost and no additional training time.

---

## Recommended Sequence

1. **Now:** Run Scale 1 (`kl_coef=0.05`, `learning_rate=2e-5`) — zero code risk, validates
   whether the KL anchor strengthened sufficiently to slow collapse.

2. **If Scale 1 slows but does not prevent collapse:** Implement Scale 2 (CE on valid rollout).
   This adds real validity gradient without requiring a corpus.

3. **If Scale 2 stabilizes validity but not coverage:** The two-objective tension (validity CE vs
   coverage GRPO) may require Scale 4 (two-phase) to cleanly separate the optimization targets.

4. **If corpus is available:** Scale 3 is the long-term structurally correct fix, especially once
   the AFL queue has accumulated 500+ valid seeds. Revisit after ~1 hour of fuzzing.
