# The coverage-learning signal in the GRPO fuzzer

How much does our GRPO infiller actually *learn coverage* (vs. just validity), why
it's mostly inert today, a rigorous definition of the signal, what moves it, and the
measurements behind all of it. Companion to [methodology.md](methodology.md) and
[aligning_with_covrl.md](aligning_with_covrl.md).

## The problem in one line

Our reward is `r = −1` (syntax) / `−0.5` (semantic) / `b + (1−b)·R_cov` (valid), and
GRPO normalizes **within a mask group** (the `G` infills of one masked program `x`).
But for a fixed `x`, **coverage barely varies with the infill `y`** — at single-token
masks the within-group `R_cov` std is ~0.001 vs ~0.05 between masks. A reward component
that is ~constant in the action contributes **exactly zero** to the policy gradient *in
expectation* (`E_y[R_cov(x)·∇log π(y|x)] = R_cov(x)·∇1 = 0`) — and this holds for *any*
baseline, because the score function `∇log π` integrates to zero. The baseline only sets the
gradient's *variance* (GRPO's within-group mean removes the constant part **exactly**; a PPO
critic removes the same conditional mean; no baseline leaves it as mean-zero noise) — **never
its learnability**. So no critic/PPO swap recovers coverage; it stays inert until `R_cov` is
made to vary with `y`. The model learns validity (which *does* vary with `y`), and the AFL
**scheduler** carries coverage by selection. This doc is about how much coverage signal exists and how
to grow it.

## 1. Rigorous definition

GRPO's per-group gradient is governed by the **within-group reward variance**
`s_g² = Var_g(r)` (the advantage is `A_i = (r_i − r̄_g)/s_g`). Decompose it by validity
class `κ ∈ {valid, syntax, semantic}` via the law of total variance:

```
Var_g(r) = Var_κ(E[r|κ])  +  E_κ[Var(r|κ)]
         =   V_validity(g) +   V_coverage(g)
```

- **`V_validity`** = between-class spread (the −1 / −0.5 / +(b..1) gaps) → the *validity*
  gradient.
- **`V_coverage`** = within-class spread. Invalid classes are constant (Var = 0), so only
  the valid class contributes:

```
V_coverage(g) = p_valid(g) · (1−b)² · Var( R_cov | valid, g )
```

This is, precisely, a **product of a validity term and a coverage term**. Define:

- **Magnitude** — the per-cycle coverage signal:
  ```
  S_cov = E_g[ p_valid(g) · (1−b)² · Var(R_cov | valid, g) ]
  ```
- **Purity** — the fraction of the gradient that is coverage-directed:
  ```
  η = E_g[ V_coverage(g) / Var_g(r) ] ∈ [0,1]
  ```
  After z-scoring, a group's coverage gradient scales like **√η_g**, and **η = 1 exactly
  in all-valid groups** (no validity variance to swamp it). This is why all-valid groups
  are the ones where coverage can be learned at all.

`b` is `cfg.validity_bonus` (`config.py`); `R_cov = σ(log Σ_e tf_e·idf_e)` lives in
`data/rewarding.py`; the advantage/grouping in `training/grpo.py` + `data/rollout.py`.

The operational metric used in the sweeps below — *% of groups with ≥2 valid infills and
coverage Jaccard-distance > 0.05* — is a **thresholded estimator of the support of
`S_cov`**. Replacing the threshold with `Var(R_cov|valid)` recovers `S_cov` itself.

## 2. Empirical — the validity–coverage hump

Offline sweep (base CodeT5p-220m, 10 valid seeds from the queue, `M=3` masks × `G=16`
infills, validity from `jerry` stderr via `data/validity.py`, per-infill edge set via
`afl-showmap`, coverage diversity = mean pairwise **Jaccard distance** of valid infills'
edge sets per group). `SIGNAL` = % of all groups with ≥2 valids and Jaccard > 0.05.

| config (masking / max_new_tokens) | valid % | meanJacc (cov-diversity) | SIGNAL |
|---|---|---|---|
| `tok ×1-3 / 24` (current) | 45 | **0.006** (flat) | 3.3% |
| `span~1.5 / 20` | 47 | 0.004 (flat) | 0.0% |
| `span~2 / 24` | 32 | 0.016 | 6.7% |
| **`span~2.5 / 28`** | 35 | 0.017 | **10.0%** ← peak |
| `span~3 / 32` | 34 | 0.018 | 6.7% |

Findings:
1. **Single-token (and span<2) is coverage-flat** (`meanJacc ≈ 0.006`): valid infills of a
   mask hit ~identical edges. The inert-coverage claim, measured directly.
2. **Coverage diversity switches on at span ≥ 2** (~3× jump, 0.006 → 0.017) — bigger valid
   infills land on different branches.
3. **Validity is the price**: it drops ~45% → ~33% at the same threshold, then plateaus.
4. **Peak is `span~2.5 / mnt 28`** — best *validity-adjusted* signal (10% of groups carry a
   coverage gradient, 3× the single-token baseline). Beyond it, diversity barely grows but
   validity doesn't recover, so the signal falls back.

Sweet-spot config: `max_masks=0` (T5-span path), `mean_span_length≈2.5`,
`corruption_rate≈0.07`, `max_span_length≈5`, `max_new_tokens≈28`.

**The R_cov transform is a bottleneck.** At span~2.5, `meanJacc=0.017` (1.7% edge
difference) collapses through `σ(log Σtf·idf)` to a R_cov difference of only ~0.004 →
`Var(R_cov) ≈ 1.6e-5`. The sigmoid saturates and **throws away most of the diversity the
masking produces** — the masking creates edge differences that the reward then compresses
to near-zero variance. (Refined by §3's probe: the **dominant** compressor is the shared
~1300-edge interpreter *floor* — variance GRPO cancels between seeds — not the sigmoid;
**delta-vs-parent recovers ~800×** of it, de-saturation alone almost none.)

## 3. Levers — read straight off `S_cov = p_valid · (1−b)² · Var(R_cov|valid)`

| factor | lever | cost / tradeoff |
|---|---|---|
| `Var(R_cov\|valid)` ↑ | **(a) masking** that makes valids hit different edges: `span~2.5` (found) or **control-flow-aware** masking (mask where a *valid* token flips a branch) | span lowers `p_valid`; control-flow masking does **not** (needs a lexer/heuristic) |
| `Var(R_cov\|valid)` ↑ | **(b) sharpen `R_cov`**: drop the saturating sigmoid, or use **coverage *delta* (new edges vs parent)** | cheap, touches neither validity nor masking — likely best signal-per-effort |
| `(1−b)²` ↑ | **lower `b`** (`validity_bonus`): 0.5→0.25 ≈ 2.25×, →0 = 4× | shrinks `V_validity` too → weaker validity anchor; only safe **with the KL anchor** |
| `p_valid` ↑ | **stability** (KL-to-snapshot anchor; optional seed-validity cull) → more all-valid groups (η=1) | the anchor is needed anyway; the cull trades exploration |

**The tension:** ↑span raises `Var(R_cov)` but lowers `p_valid` — they fight (hence the
hump, not a monotone climb). **Levers that beat it** (raise `Var(R_cov)` at constant
`p_valid`): **sharper `R_cov`** (b) and **control-flow masking** (a). The others trade off.

No baseline/critic change can help the infiller here — coverage is ~constant in `y` at
small masks, and a constant-in-`y` term has zero gradient regardless of baseline. The
lever is always *what `y` can affect* (masking/representation) or *what we credit*
(`R_cov` transform / b), never the normalization.

### CovRL's answer, from their source: decouple the *train* action from the *fuzz* action

The reward transform is **byte-for-byte ours** (`σ(log(bitmap·idf))`, CovRL
`rewarding.py:212-227`), so the difference is structural — and it is the **action size at
training time**:

| | CovRL | us (current) |
|---|---|---|
| fuzz-time mask | 1–3 single `MASK_TOKEN`s (`afl-fuzz.c`, `MASK_COUNT=3`) | 1–3 single-token spans (same) |
| **train-time action** | **re-mask 15% / mean-3 spans** (`actor_dataset.py`: `mask_probability=0.15`, `λ=3`) | **the same 1–3 single tokens it fuzzes with** |
| reward granularity | whole test case | whole test case (same) |
| objective | reward-weighted reconstruction, **no baseline**, **+ CE recon loss** (`finetuner.py:169-172`) | GRPO z-score within the (seed, mask) group |
| reward in the gradient | a learned **critic reward-model** scores `input ++ prediction` (`finetuner.py:140-159`) | the measured `R_cov` of the infill |

The load-bearing row is **train-time action**: CovRL regenerates ~15% of the program in
training, so its reconstructions genuinely hit different edges (`Var(R_cov)` large → coverage
learnable); we train on the *same* single-token masks we fuzz with (`Var(R_cov)≈0` → inert).
Its other two choices (no baseline, so nothing is *exactly* zeroed; a CE anchor that carries
validity separately) reinforce but do not cause the gap. **R1-Fuzz** reaches the same place
the orthogonal way: action = the whole input, reward = a dense distance-to-target — maximal
action–coverage coupling *and* no saturating floor.

### The levers as three tiers (ordered by whether they *create* signal)

`S_cov` grows only if `Var_y(R_cov)` grows:

- **(A) make the action move coverage** — the *only* tier that creates learnable signal:
  - **A1** bigger spans (`span~2.5`; CovRL 15%/mean-3) — costs validity;
  - **A2** *decouple* like CovRL — fuzz cheap small masks, but build the **GRPO groups from
    bigger-span reconstructions** of the executed seeds (attacks our root cause: train mask =
    fuzz mask);
  - **A3** control-flow-aware masking — raises `Var(R_cov)` at ~constant validity;
  - **A4** learned masker — the *masker's* action moves coverage even at single-token
    granularity (the baseline theorem is *why* it's motivated; collapsed in practice).
- **(B) amplify the variance that already exists** — cheap, no validity cost, do regardless:
  - **B1** coverage **delta vs parent** — new idf-weighted edges vs the *parent seed* (a
    **stationary** per-context baseline: the parent ∈ the context, so `R(context,action)` is
    fixed — *not* the global coverage frontier, whose monotone growth would be non-stationary
    and is the scheduler's job). Removes the constant floor `R_cov(x)`. **Measured the dominant
    lever, ~800× (below). Implemented + verified live:** `cfg.delta_coverage` → `data/rewarding.py`.
    Each queue entry's edge set is cached from the AFL bitmap when AFL *creates* the entry
    (`cache_parent`/`queue_new_entry`, reusing the SHM read `score` already did) and looked up when
    it's fuzzed (`set_parent`/`fuzz_count`) — pure SHM read, no subprocess; `score` zeroes the
    parent's edges; **no-new-coverage → 0** so any new edge outranks finding none. (A first cut
    snapshotted the SHM in `fuzz_count`, but in the mutations-on-mutations regime AFL doesn't re-run
    the parent, so that grabbed the *previous* seed's trace and degraded delta to absolute — the
    per-entry cache is the fix.) On a live jerry run: cycle-1 valid infills went **~80% at `r_cov`=0**
    + a spread above (vs 0% / flat-0.66 for absolute), and **38–60% of all-valid groups became live**
    — magnitude lever confirmed end-to-end.
  - **B2** drop the saturating `σ(log)` (raw `Σtf·idf` / sharper map). **Measured negligible
    alone** — the floor, not the sigmoid, is the killer; a minor help only on top of B1.
- **(C) baseline / objective structure** — **secondary; changes variance & stability, never
  learnability**:
  - **C1** reward-weighted reconstruction + CE anchor (CovRL's objective) — only helps *with*
    A/B; on a constant-in-action reward it just injects noise. Swapping GRPO→PPO/critic does
    **not** create coverage signal (the theorem in [§1](#the-problem-in-one-line)).

**Do-first:** B1 (delta-vs-parent — the ~800× lever, free), then A2 or A3 (the real structural
fix). B2 alone isn't worth it; a critic is not a coverage fix.

### Measured (offline probe — `eval/signal_probe.py`)

8 seeds × 6 groups × `G`=12 = 1160 real jerry executions (edges via `afl-showmap`, `b`=0.5),
current single-token masks vs `span~2.5`, comparing the **absolute** reward `σ(log Σ_all idf)`
to **B1 delta-vs-parent** `σ(log Σ_new idf)` (`new` = edges ∉ the parent *seed*):

| config | reward | `Var(R_cov\|valid)` | `S_cov` | `η` | `%grp_cov` | `max\|resid\|` |
|---|---|---|---|---|---|---|
| tok ×1-3 (p_valid 0.54) | cur `σ(logΣ_all)` | 0.000006 | 0.000001 | 0.071 | 29% | 3e-16 |
| | **B1 `σ(logΣ_new)`** | **0.005331** | **0.000856** | 0.075 | 29% | 2e-16 |
| span~2.5 (p_valid 0.33) | cur `σ(logΣ_all)` | 0.000009 | 0.000002 | 0.111 | 35% | 2e-16 |
| | **B1 `σ(logΣ_new)`** | **0.006694** | **0.000866** | 0.116 | 35% | 2e-16 |

1. **The decomposition identity holds on real data — `S_cov` verified, not just derived.**
   `max|resid| = |Var_g(r) − (V_validity + V_coverage)| ≈ 2–3e-16` for every config and reward.
   The §1 formula is an exact identity, confirmed on live jerry coverage — it **generalizes**.
2. **B1 (delta-vs-parent) ≈ 800× the signal; B2 (de-saturate) ≈ 0 — correcting the lever table.**
   `Var(R_cov|valid)` and `S_cov` jump ~750–900× cur→B1. Measured mechanism: **jerry hits ~1300
   shared interpreter-startup edges on *every* run**, so absolute coverage is floor-dominated —
   its variance lives *between seeds* (pool `std(logΣ_all)≈0.4`, within-group ≈0, i.e.
   `Var(R_cov|valid)=6e-6`), exactly where GRPO's within-group baseline **cancels** it. Delta
   subtracts the parent's edges *before* the saturating `σ(log)`, moving the variance *inside* the
   group (`std(logΣ_new)≈0.88`). The sigmoid is a minor compressor; **the constant floor is the
   killer** ⇒ B1 ≫ B2 (and B1B2 ≈ B1). [Refines §2's "the `σ(log)` transform is a bottleneck."]
3. **Magnitude (`S_cov`) and share (`η`) are orthogonal.** B1 raises the signal *magnitude* ~800×
   but `η` (coverage's *share* of the group gradient) barely moves (0.07→0.075). `η` is
   bottlenecked by the **all-valid-group rate** (validity composition), not the reward — in a mixed
   group the −1/−0.5 gaps dominate `Var(r)` however sharp `R_cov` is; only all-valid groups give
   `η→1`. So the levers don't substitute: **B1 makes the coverage gradient inside all-valid groups
   *real instead of noise* (6e-6→5e-3); only stability — more all-valid groups (KL anchor / lower
   `b` / the span–validity tradeoff) — raises how *many* groups carry it. You need both.**
4. **Both levers are now wired (delta live-verified).** Magnitude = `cfg.delta_coverage` (B1).
   Share = `cfg.kl_ref_coef`, a **KL-to-frozen-reference anchor** (`training/grpo.py`): a *separate*
   k3 KL vs a frozen snapshot (the base model's valid prior) — distinct from the inert on-policy
   `kappa` KL — that resists the unanchored random-walk behind validity collapse (`ref_update_every`
   advances the snapshot; 0 = anchor to base forever). The delta payoff that `grp_std_med` hides is
   surfaced directly as **`pct_allvalid_live`** in `rllm_train.tsv` (`data/rollout.py`): the % of
   all-valid groups carrying a coverage gradient (38–60% on the live run). `kl_ref_coef` is the
   stability⇄adaptation knob — too high pins the policy to base (validity stable, coverage frozen),
   too low lets it collapse; start ~0.1 and tune on the `valid%` / `pct_allvalid_live` trend.

## 4. Caveats and open questions

- **The signal is weak even at the peak** (10% of groups, `meanJacc=0.017`). Bigger spans
  make coverage a *real but secondary* gradient; validity still dominates each group and
  the **AFL scheduler remains the main coverage engine** ([methodology.md](methodology.md)).
  Don't expect the model alone to "solve" coverage.
- **Deploying `span~2.5` needs the stability layer.** It runs at ~35% vs ~45% validity;
  lower validity → faster corpus rot (see [queue_validity_collapse.md](queue_validity_collapse.md))
  → earlier collapse. Masking sweet spot and stability (KL anchor) are complementary, not
  alternatives.
- **CovRL avoids all this** by decoupling: it *trains* on 15% span-masked reconstruction
  (coverage varies with `y`), with a per-seed reward + critic + CE/KL anchor, while
  *fuzzing* with small masks. We train and fuzz with the same small masks → inert.
- Open / to measure next:
  1. Re-instrument the sweep to report **`S_cov`, `η`, all-valid-group rate** directly
     (the grounded metric) instead of the Jaccard proxy.
  2. Test **R_cov sharpening** (raw `Σtf·idf`, `delta-edges`) — how much signal is the
     `σ(log)` eating?
  3. **Control-flow-aware masking** — the one lever that raises `Var(R_cov)` without paying
     `p_valid`.
  4. **`b` ablation** under the KL anchor.

## 5. Prior art — am I first?

Short answer: **not on the ingredients, plausibly on the synthesis.** The two strands — the
*fuzzing* validity-vs-coverage tradeoff and the *GRPO* inert-coverage observation — grade very
differently against the literature.

| claim | status | closest prior art |
|---|---|---|
| validity-vs-diversity tradeoff in (LLM) fuzzing | **established** | [CovRL](https://arxiv.org/abs/2402.12222), [LLM-fuzzing vision paper §C3](https://arxiv.org/abs/2503.00795), [BeDivFuzz](https://arxiv.org/abs/2202.13114) |
| bias-variance *framing* of it (structure=bias, diversity=variance) | **a lens, not a finding** | no exact match; a re-description of the above |
| GRPO zero-variance / dead-group collapse | **established, named** | [DAPO Dynamic Sampling](https://arxiv.org/abs/2503.14476), [NGRPO](https://arxiv.org/abs/2509.18851) |
| coverage constant-in-action ⇒ zero gradient for any baseline | **textbook** | policy-gradient baseline theorem; [Mirage of Action-Dependent Baselines](https://arxiv.org/abs/1802.10031) |
| **the synthesis** (below) | **no match found** | even [R1-Fuzz](https://arxiv.org/abs/2509.20384) (GRPO-for-fuzzing) never names it |

**Not ours — three established pieces.**
- *The fuzzing tradeoff is explicit prior art.* The vision paper lists it as challenge **C3 "Seed
  Quality Trade-offs"** (Fuzz4All: +36.8% coverage, ~56% lower validity) and quotes CovRL:
  "LLM-based mutations tend to conduct grammar-level mutations … but also limit variability."
  BeDivFuzz is built on biasing between structure-preserving (validity) and structure-changing
  (diversity) mutations. The **bias-variance relabeling** is useful exposition but too generic to
  carry novelty on its own.
- *The GRPO dead-group problem is named and solved.* DAPO §3.2: "if all outputs of a particular
  prompt are correct and receive the same reward 1, the resulting advantage for this group is
  zero" → vanishing gradient; Dynamic Sampling over-samples and filters accuracy-0/1 groups. Our
  `pct_dead`/`pct_all_inval` ([data/rollout.py](../data/rollout.py)) measure exactly this.
- *Coverage cancelling in the gradient is the baseline theorem.* An action-independent reward term
  behaves like a state-only baseline and contributes exactly zero to the policy gradient — that is
  *why* baselines are admissible (Williams 1992 / Sutton & Barto). Our
  `E_y[R_cov(x)·∇log π(y|x)] = R_cov(x)·∇1 = 0` is one instance; a textbook result re-derived in a
  new setting.

**Ours — the specific synthesis + its boundary condition.** The claim that *coverage-guided
group-relative RL with **span-infill actions on a fixed seed*** turns coverage into an
action-independent term that GRPO's within-group baseline structurally cancels — leaving only
validity learnable, with `V_coverage = p_valid·(1−b)²·Var(R_cov|valid)` — does not appear in the
literature. The sharpest evidence is **R1-Fuzz** (Sep 2025), the *one* other paper that does GRPO +
coverage reward for fuzzing: it sidesteps the pathology **by construction** and never names it,
because (1) its action is the **whole input** (coverage *does* vary with the action → no
cancellation), and (2) its reward is a **dense distance-to-target** (longest-common-prefix of
execution traces), not a saturating indicator. That is exactly our boundary condition: **the
inert-coverage pathology = small span-infill action × same-seed group normalization.** Make the
action the whole input, or the reward dense-and-relative (R1-Fuzz's distance, or our deferred
coverage-*delta*, [§3](#3-levers--read-straight-off-s_cov--p_valid--1b--varr_covvalid)), and it
disappears.

**Positioning.** Do not claim the validity-coverage tradeoff or the GRPO zero-variance problem —
both are citable prior art. The defensible contribution is narrow and mechanistic:

> Coverage-guided **group-relative** RL for **span-infill** fuzzing has a structural
> gradient-cancellation that whole-sequence RL (R1-Fuzz) and per-seed-critic RL (CovRL) avoid: at
> infill granularity the coverage reward is action-independent within a mask group and is
> annihilated by the GRPO baseline, so the policy learns validity while the scheduler carries
> coverage. We characterize the surviving signal as `S_cov = p_valid·(1−b)²·Var(R_cov|valid)`.

**Verification (done 2026-06-04).** Both open items are now closed; neither overturns the verdict,
and the NGRPO check sharpens it.

- *NGRPO read in full* ([2509.18851](https://arxiv.org/abs/2509.18851)) frames the problem as
  **whole-reward** uniformity, not a term cancellation: "when a group is homogeneous, containing
  either all correct or all incorrect responses … the variance of rewards within such a group is
  zero, which results in normalized advantages of zero." The mechanism is **std-normalization going
  to 0/0**; the fix is a **virtual maximum-reward sample** forcing `std > 0`. This confirms the crux
  distinction — the inert-coverage case is *not* the dead-group case:
  - DAPO/NGRPO treat the **dead group** (`Var(r)=0`, advantage undefined, *no* gradient) — what our
    `pct_dead`/`pct_all_inval` track; their fixes (filter / virtual sample) apply here.
  - Ours is a **live** group: `Var(r)=V_validity>0`, so it looks healthy (DAPO would *keep* it; the
    virtual sample is moot), yet `V_coverage≈0` so the gradient is *entirely validity-directed*.
    **Neither DAPO nor NGRPO touches this** — they lift the denominator off zero; they do nothing
    about an action-independent reward *term* inside a nonzero-variance group. That is the
    policy-gradient baseline theorem, not the `std=0` degeneracy.
- *Grammar-aware / neural RL fuzzing surveyed* for a validity-vs-coverage reward **variance**
  decomposition — none found. The recurring pattern is reward **composition** (additively summing
  validity + coverage + novelty terms): [Deep Reinforcement Fuzzing](https://arxiv.org/abs/1801.04589)
  (MDP + Q-learning, edge-coverage reward; path-depth "indirectly rewards validity"),
  [Learn&Fuzz](https://patricegodefroid.github.io/public_psfiles/ase2017.pdf) (closest *prose
  articulation* of the tension — "generate diverse well-formed inputs … while still injecting enough
  ill-formed input parts," but not RL and no decomposition), [Montage](https://arxiv.org/abs/2001.04107),
  and surveys that list "coverage, validity, novelty" as standard reward terms. **Composition (add
  the terms) is ubiquitous; decomposition (show one term's within-group variance is ~0, so its
  gradient cancels under group normalization) appears nowhere.** That variance analysis — not the use
  of a coverage+validity reward — is the part that is ours.

## Bottom line

The coverage signal is **`S_cov = p_valid · (1−b)² · Var(R_cov|valid)`**, and it is small
today for three compounding reasons: single-token masks give `Var(R_cov)≈0`, the shared
~1300-edge interpreter floor pins absolute coverage between-seed (cancelled), and `b=0.5`
quarters what survives. The cheapest fix that doesn't trade away validity is **delta-vs-parent
coverage** — measured at **~800×** the within-group variance (`eval/signal_probe.py`); de-saturating
`σ(log)` alone is negligible (the floor, not the sigmoid, is the killer). `span~2.5` and **lower
`b`** help but cost validity, so they require the **KL anchor** to be safe. The one *structural*
fix is CovRL's **decoupling** — train on bigger-span reconstructions while fuzzing small; and per
the baseline theorem, **no critic/PPO swap creates coverage signal — only a coverage-bearing action
(A) or a sharper reward (B) does.** The `S_cov` decomposition itself is **empirically verified** on
real jerry coverage — the law-of-total-variance identity holds to ~1e-16 (`eval/signal_probe.py`).
