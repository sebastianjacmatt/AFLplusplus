# Queue validity collapse

The LLM mutator's per-call validity rate drifts downward over a fuzzing
run even though the model and its decoding strategy are static. We
observed it collapse from ~36% at t≈50s to ~8% at t≈12min on JerryScript
with `configs/conservative.json`. This document explains why the drop
is **not a model regression** — it is a fixed point of AFL's
coverage-only selection feedback loop, and no masking knob or decoding
change inside the mutator can prevent it. Only a validity-aware
selection or training signal can.

## The phenomenon

Measured directly off `<out>/debug.log` from a fresh run (un-tuned
CodeT5+ 220M, contrastive search, conservative masking — `corruption_rate
=0.03`, `mean_span_length=1`):

| t       | seeds | muts  | runs  | runs/muts | valid | syntax | semantic |
|---------|-------|-------|-------|-----------|-------|--------|----------|
| 9.7s    | 3     | 48    | 32    | 0.67      | 28.1% | 71.9%  | 0.0%     |
| 47.1s   | 20    | 320   | 304   | 0.95      | 36.2% | 56.9%  | 6.9%     |
| 117.5s  | 64    | 1024  | 993   | 0.97      | 17.4% | 75.2%  | 7.4%     |
| 580.5s  | 337   | 5392  | 5361  | 0.99      | 9.0%  | 84.4%  | 6.6%     |
| 719.1s  | 414   | 6624  | 6595  | 0.996     | 8.3%  | 85.1%  | 6.6%     |

`runs ≈ muts` confirms the post-fix `_last_run_was_mutation` gate in
`mutator.py:167` correctly excludes calibration replays — the rate
being reported is the LLM's per-mutation validity, nothing else. Yet
the rate still falls by ~4× over the run, then plateaus.

Two corpus snapshots taken at the end of that same run pin down the
mechanism:

| Corpus                                        | Valid | Syntax | Semantic |
|-----------------------------------------------|-------|--------|----------|
| Initial seed dataset (33 `.js` files)         | **100%** | 0%     | 0%       |
| AFL queue after 12 min (1242 entries)         | **19%** | 59%    | 22%      |

The seed corpus is hand-curated and 100% valid on JerryScript. The
queue AFL accumulated over twelve minutes is 81% invalid. The LLM's
per-call validity tracks the validity of the corpus it is being asked
to mutate — which AFL controls, not us.

## Mechanism

The collapse is the predictable outcome of three independently
reasonable design choices interacting:

**1. AFL saves any new-coverage input, including invalid JS.**
JerryScript's parser is itself instrumented, so a SyntaxError-causing
input can still trip fresh edges in the parser's error-handling paths.
`save_if_interesting` (`afl-fuzz-bitmap.c:544`) only inspects the trace
bitmap; it has no notion of "valid input" for the target language.
A single-token mutation that turns `===` into `=]=` produces a
SyntaxError but legitimately exercises a different parser branch — AFL
keeps it.

**2. AFL preferentially fuzzes recent favored finds.** Once an entry
is added to the queue with `favorite=1` and a high `perf_score` (we
observe `perf_score=600, weight=2` on fresh finds in the debug log),
AFL's selection schedule spends most of its time on those new entries
rather than re-fuzzing the original seeds. After a few minutes the
selection mass is concentrated on entries that — per (1) — are mostly
invalid.

**3. The LLM has no validity reward.** `Salesforce/codet5p-220m`
straight from the hub is a masked-span infiller pretrained on
mixed-quality code. When AFL hands it an already-broken JerryScript
program, the model fills the masked spans with locally-plausible code
that does not repair the surrounding syntax error. The mutation
inherits the parent's invalidity ≈ deterministically. There is no
gradient pulling the model back toward valid JS — it just continues
whatever it was given.

The result is a positive feedback loop:

```
  valid seed
     │ LLM mutate (≈40% per-mutation P(valid))
     ▼
  mostly-invalid mutations
     │ AFL coverage filter (keeps any new-edge input,
     │                      regardless of validity)
     ▼
  invalid entry saved to queue with favorite=1, perf_score=600
     │ AFL selection (favors fresh, high-score entries)
     ▼
  LLM mutates an invalid parent
     │ LLM mutate (≈0% per-mutation P(valid) — base is broken)
     ▼
  invalid mutation → AFL → invalid queue entry → ...
```

Steady-state validity is the LLM's per-call P(valid | parent is valid)
weighted by P(parent is valid in the queue) — and AFL's selection
makes the latter shrink monotonically.

## Why it cannot be fixed inside the mutator

The temptation is to tighten masking (smaller `corruption_rate`, fewer
spans) so each mutation is less likely to break the parent. That
helps the **first** factor (LLM produces fewer broken outputs per mask
fill), but it does not break the feedback loop:

- AFL will still save any new-coverage input. Even a 1% chance of
  emitting an invalid-but-coverage-finding mutation means hundreds of
  invalid entries enter the queue per hour.
- Once an invalid entry is in the queue, no amount of conservative
  masking lets the LLM repair it. The model is doing local span
  infill, not global validity correction.

The collapse is independent of:

- Masking parameters (`corruption_rate`, `mean_span_length`,
  `whole_word_masking`).
- Sampling strategy (`contrastive` vs `nucleus`, `top_k`, `top_p`).
- Model size or pretraining checkpoint.

It depends only on AFL's coverage-only objective + the model having
no validity signal. Both choices have to be revisited together.

## Why CovRL-Fuzz does not exhibit this (as strongly)

CovRL-Fuzz reports ~21.5% **steady-state** validity on JerryScript
(Table 7, "LLM w/o CovRL" — the pure-SFT baseline before their RL
phase). That is several times higher than our 8% asymptote despite
both starting from a similar pretrained backbone. Three differences
account for it:

1. **Smaller per-mutation mask footprint.** CovRL applies 1–3
   single-token masks per mutation at fuzz time (`afl-fuzz.c:5190-5221`,
   `MASK_COUNT=3`); we apply T5 span corruption with ≈ 5 spans of
   ≈ 3 tokens each. Per-mutation P(valid) is structurally higher for
   them. See `docs/design.md` for the full comparison.

2. **The published 21.5% is *with* their AFL fork's queue policy**, not
   a controlled measurement against a fixed queue. They report the
   number that emerges from the same feedback loop, just with a less
   destructive mask. A smaller mask → more valid mutations saved →
   a less invalid queue → higher steady state. Same dynamics, better
   asymptote.

3. **GRPO closes the rest.** Their full system (with reward) lifts
   the validity to 60%+ because the reward function explicitly
   penalises invalid outputs (-1.0 for syntax, -0.5 for semantic).
   That is the gradient that breaks the feedback loop: even when the
   parent is invalid, the model is *trained* to emit something less
   invalid than the parent. The reward is the validity signal that
   AFL's coverage objective does not provide.

## What this means for the project

The measured ~8% asymptote is **the right number to report for an
un-tuned, coverage-only baseline**. It is not a bug, and tightening
the masking config will only move it by a few points.

Three avenues actually break the feedback loop:

1. **Validity-gated `queue_get`.** Reject queue entries the target
   would not parse — return `False` from `queue_get` for entries
   whose parent fails the language's syntax check. AFL skips them
   and moves on. Expected effect: validity stabilises around the
   per-mutation rate (~30-40%) because the LLM only ever sees valid
   parents. Cost: one target invocation per `queue_get`. Risk: AFL's
   selection may keep returning to skipped entries; need to monitor
   `cur_skipped_items`.

2. **Validity-gated `fuzz_count` / `post_process`.** Less invasive
   than (1): the LLM still mutates invalid parents but we drop
   mutations that themselves fail to parse (`fuzz_count` returns 0,
   or `post_process` returns 0 to suppress the execution). Keeps the
   queue intact but starves AFL of invalid finds from this mutator.

3. **GRPO with a validity reward.** The project's existing roadmap.
   Adds the gradient that pulls the model toward valid output even
   when starting from a broken parent. This is what lets CovRL-Fuzz
   sustain 60%+ validity in steady state. The 8% baseline established
   here is the warm-start point for that training run — the reward
   landscape is strongest where the policy is worst, which is useful
   for bootstrapping.

The right ordering is (1) or (2) for a quick empirical confirmation of
this analysis, then (3) for the actual project deliverable.
