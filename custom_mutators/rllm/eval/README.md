# rllm/eval — coverage/validity/exec_rate over time

A small evaluation harness that runs the fuzzer for a fixed wall-clock
duration and produces both a JSON summary of headline metrics and a
4-panel figure tracking the run over time. Designed to match the
metrics CovRL-Fuzz reports in their ISSTA '24 paper (specifically the
RQ3 ablation in Table 7), at a fraction of their runtime.

The fuzzer itself does the data collection — it now appends to
`rllm_history.tsv` (full flush history) and `rllm_seeds.tsv` (seed
selection events) inside the run's output directory. AFL writes
`plot_data` and `fuzzer_stats` automatically. The two scripts here just
drive the run and turn those files into a chart and a summary.

---

## What this measures

| Metric | Source | What it shows |
|---|---|---|
| `exec_rate` | `plot_data.execs_per_sec` | Target executions per second (mutation throughput). |
| `validity_rate` | `rllm_history.tsv.valid_pct` | Fraction of LLM-mutation runs whose stderr was empty (no syntax/semantic error). Cumulative + windowed-N. |
| `coverage` | `plot_data.edges_found` | AFL edge count, total (across all queue entries). |
| `seed_selection` | `rllm_seeds.tsv.filename` | Which queue entry was active at each `queue_get`. Used as the bottom strip of the plot. |

What we deliberately do **not** measure here:
- Valid-only coverage (edges hit only by valid runs) — would require
  re-running `afl-showmap` on each queue entry, filtered by validity.
  Defer to a follow-up.
- Bug counts beyond `saved_crashes` (no deduplication).
- Statistical significance vs other fuzzers (CovRL ran 5 replicates +
  Mann-Whitney; we run once or a few times).

---

## Quickstart

5-minute smoke (use this first to confirm everything renders):

```bash
cd custom_mutators/rllm
python -m eval.runner --name eval_smoke --duration 300
python -m eval.plot \
    --out-dir ~/Documents/data_store/out/eval_smoke/default \
    --output ~/Documents/data_store/out/eval_smoke/default/eval_plot.png
```

1-hour standard run:

```bash
python -m eval.runner --name eval_1h --duration 3600
python -m eval.plot --out-dir ~/Documents/data_store/out/eval_1h/default
```

Variance check (3 independent 1-hour runs):

```bash
python -m eval.runner --name eval_var --duration 3600 --runs 3
# Plot each individually:
for i in 1 2 3; do
    python -m eval.plot --out-dir ~/Documents/data_store/out/eval_var-$i/default
done
```

CovRL RQ3-match (5 hours, 1 core — matches their ablation):

```bash
python -m eval.runner --name eval_5h --duration 18000
```

---

## Files produced

Per run:

| Path | Written by | What's in it |
|---|---|---|
| `<out>/default/plot_data` | AFL | Standard AFL plot data (CSV). |
| `<out>/default/fuzzer_stats` | AFL | Standard AFL key:value stats. |
| `<out>/default/rllm_stats.txt` | rllm | Latest one-line snapshot for `tail -f`. |
| `<out>/default/rllm_history.tsv` | rllm | **NEW.** Append-only flush history. Header + N rows. |
| `<out>/default/rllm_seeds.tsv` | rllm | **NEW.** One row per `queue_get` call. |

### `rllm_history.tsv` column key

The header is on the first line; here's what each column means:

| Column | Meaning |
|---|---|
| `epoch_s` | Unix wall-clock timestamp from `time.time()`. |
| `elapsed_s` | Seconds since `Mutator.__init__`. |
| `seeds` | `fuzz_count` calls. One per queue cycle on a given entry. |
| `muts` | Total mutations the LLM generated. |
| `finds` | `queue_new_entry` calls — entries AFL accepted into its queue. |
| `run_total` | Mutations AFL actually executed and `post_run` classified. |
| `run_valid` | Of those, runs with empty stderr (valid JS). |
| `run_syntax` | Of those, runs with a `SyntaxError`. |
| `run_semantic` | Of those, runs with other JS errors (Reference/Type/Range/URI/Eval). |
| `valid_pct` | `run_valid / run_total × 100` — cumulative validity %. |
| `syntax_pct` | `run_syntax / run_total × 100`. |
| `semantic_pct` | `run_semantic / run_total × 100`. |
| `gen_ms_avg` | Avg ms per `fuzz_count` batch = total `model.generate` time / `seeds`. |
| `muts_per_s` | `muts / elapsed_s` throughput. |

### Watching the live mutation with diff highlighting

While a run is in progress, `<out>/default/.cur_input` holds the
current mutation's decoded JS source. To see *where* the LLM mutated
relative to the parent queue entry, the mutator also writes a sidecar
`<out>/default/.cur_input.diff` — same source, but with ANSI-yellow
highlights wrapping any region whose tokens differ from the parent.

```bash
watch -c -n 2 cat <out>/default/.cur_input.diff
```

`-c` interprets ANSI color codes (bold yellow for changed/inserted
regions, dim red `[--]` markers where the mutation removed tokens
without replacement). The diff is computed in `post_process` via
`difflib.SequenceMatcher` on the token sequences (parent vs mutation),
costs ~1–3 ms per mutation, and is only written when a fresh LLM
mutation is active — calibration / trim post_process calls leave the
sidecar untouched.

### Watching the history with column headers pinned

`rllm_history.tsv` is a wide TSV. To monitor it live with the header
visible at the top:

```bash
( head -1 <out>/default/rllm_history.tsv;
  tail -f -n 0 <out>/default/rllm_history.tsv ) \
  | column -t -s $'\t'
```

Or read the latest snapshot in human-readable form via:

```bash
cat <out>/default/rllm_stats.txt
```

(that file is truncate-and-write, single line — has labels like
`valid=22.9% syntax=64.2% semantic=12.9% muts/s=4.4`.)

Per evaluation campaign (across runs):

| Path | Written by | What's in it |
|---|---|---|
| `<out>/<name>_summary.json` | `eval.runner` | Aggregate of headline metrics, plus the CovRL Table 7 reference for comparison. |
| `<out>/default/eval_plot.png` (and `.pdf`) | `eval.plot` | The 4-panel figure. |

---

## How to read the plot

The figure has four panels stacked on a shared time axis (minutes since
the mutator's `init()` was called):

**Row 1 — `execs/sec`** (blue). AFL's mutation throughput. Drops here
usually mean the model generated a slow-to-execute mutation (e.g.,
hit AFL's timeout limit), or AFL spent time calibrating a batch of new
finds. Steady mid-run rate is the headline metric.

**Row 2 — `valid_pct` (%)** (green). Solid is cumulative since `init`;
dashed is the validity rate over the last ~100 mutations. The dashed
line is what to watch — sharp dips correspond to time windows where the
LLM was emitting invalid JS. The dotted grey horizontal line is the
CovRL Table 7 reference (21.5% on Jerry).

**Row 3 — `edges_found`** (red). AFL's edge counter. Always
monotonically nondecreasing. Plateaus mean the current seed isn't
producing new coverage. The dotted grey reference is the CovRL Table 7
total coverage (14068 on Jerry, but that's a 5h number).

**Row 4 — active seed strip** (purple dots). Each point is a `queue_get`
call. Y position = seed index (0, 1, 2, …, in order of first appearance).
Top-K most-fuzzed seed *filenames* are labeled on the right edge for
quick identification.

**Vertical dashed grey lines** (cross-panel) mark moments where the
windowed validity dropped more than 5 percentage points below the
cumulative mean. Use them to spot "this seed kills validity" patterns.

**Top-right text box** shows the comparison vs CovRL Table 7 in the
form `ours / CovRL = ratio`. For a 1h run on Jerry you should expect:
- `valid_pct` ratio ≈ 1.0–1.7× (validity is per-mutation, not very
  time-dependent; we typically beat the SFT-free baseline)
- `edges_found` ratio ≈ 0.25–0.40× (CovRL had 5× our wall-clock plus
  100 seeds vs our 33)

---

## Comparison to CovRL Table 7

The reference is "LLM w/o CovRL" on JerryScript, 5h on 1 core, 100
valid seeds, 5 runs averaged. This is the *closest* CovRL-published
configuration to what we're running — SFT-free CodeT5+ mask-fill over
a token-level queue. We deliberately match the target and the
LLM-mask-fill setup; what differs is duration, seeds, replicates.

| Metric | CovRL (5h, 1 core) | ours (1h, 1 core) | Expected ratio |
|---|---|---|---|
| Error rate | 78.48% | ~60–70% | < 1.0 (we have a better mask shape and fewer broken seeds) |
| Valid coverage | 12833 | n/a (no valid-only counter yet) | — |
| Total coverage | 14068 | ~3000–6000 | 0.25–0.4× (we run 1/5 the time) |
| `valid_pct` | 21.52% | ~30–40% | > 1.0× |

If our 1h numbers fall well outside these ranges, something's wrong —
either with the run config (wrong target, wrong masking) or with the
plot script (parsing). Use the figure's annotation box for the quick
sanity check.

---

## Why validity declines over the run — expected behavior

You will see `valid_pct` drift downward over the first 20–40 minutes of
any 1h run, often falling below CovRL's published 21.5% SFT-free
baseline before stabilizing. This is *not a regression*; it is the
queue-validity-collapse dynamic described in `docs/queue_validity_collapse.md`:

1. AFL keeps any input that hits new edges, including invalid ones
   (the parser's error-handling paths are also instrumented). Most
   new finds are syntactically invalid.
2. AFL preferentially fuzzes recent favored finds. Most of
   `fuzz_count`'s budget goes to invalid parents.
3. The LLM does local span infill, not global syntax repair — it can't
   un-break a parent. Mutations from invalid parents inherit invalidity
   ~deterministically.
4. `valid_pct` (cumulative) drifts toward the steady-state validity of
   the queue's own contents, which is determined by AFL's selection
   policy, not by the model.

CovRL's full system (`LLM w/ CovRL` in Table 7) lifts this asymptote
to ~41% on Jerry via PPO training with `reward = -1.0` (syntax),
`-0.5` (semantic), `+R_cov` (valid). Without that gradient our model
has no signal to fight the queue drift — landing near 20% at the end
of a 1h run is the *correct* outcome for the SFT-free baseline and the
intended verification target of this evaluation.

If you want to see the per-mutation validity *holding* at ~30-40% (the
pre-collapse rate the model is actually capable of), watch the
*windowed* line in the plot (green dashed) during the early minutes
before the queue grows large. The cumulative line is dragged down by
the lengthening tail of invalid-parent mutations.

## Caveats

- **Single machine.** The CovRL paper used a 64-core 2× Xeon Gold 6134
  + 3× RTX 3090. We measure on whatever the developer has handy. Don't
  compare raw throughput numbers across hardware.
- **Few seeds.** Our `final-dataset-dec22` is 33 entries; CovRL used
  100. With fewer seeds the first cycles concentrate fuzzing on each
  seed longer, which exaggerates per-seed validity effects in the
  plot. That's actually useful for the "see where validity suffers
  by seed" question.
- **No statistical test.** Single runs (or a handful) only. To make
  load-bearing claims about coverage, use `--runs 3+` and look at the
  spread.
- **Windowed validity is approximate.** The window in `eval.plot` is
  measured in *mutations completed*, not wall-clock; per-row deltas
  in `rllm_history.tsv` are bucketed by the mutator's 2-second flush
  cadence, so the window aligns to mutation-count boundaries only
  approximately.
- **No valid-only coverage.** Reproducing CovRL's "valid coverage"
  column requires post-hoc `afl-showmap` runs filtered by per-input
  validity. Not in this harness; track separately.

---

## When something looks wrong

If the figure looks empty or wrong, check in this order:

1. Did the run actually fuzz? `cat <out>/default/rllm_stats.txt` should
   show non-zero `muts=`. If not, the AFL/Python setup didn't engage
   the mutator — look at `<out>/default/error.txt`.
2. Does `rllm_history.tsv` have rows beyond the header? If only one
   row, the run terminated before the first flush (≤ 2 seconds).
   Increase `--duration`.
3. Does `rllm_seeds.tsv` have rows beyond the header? If empty, AFL
   isn't calling our `queue_get`. Likely `AFL_CUSTOM_MUTATOR_ONLY` got
   unset.
4. `plot_data` exists but is empty? AFL hadn't reached its first plot
   write — same fix as (2), increase duration.
