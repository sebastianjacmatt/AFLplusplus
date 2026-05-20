# What CovRL-Fuzz gets right

This document audits how CovRL-Fuzz [Eom et al., ISSTA '24] avoids the
validity collapse we identified in [queue_validity_collapse.md](queue_validity_collapse.md).
Claims framed at the level of *what the system does* come from the paper;
claims about *how the implementation actually achieves it* come from
`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c` and `~/Documents/CovRL-Fuzz/covrl/`,
which differ from the paper in several load-bearing ways. The numbers we
care about (Table 3 / Table 7):

| Configuration                | JerryScript error rate | JerryScript valid % |
|------------------------------|-----------------------:|--------------------:|
| Token-Level AFL (baseline)   | 80.50%                 | 19.50%              |
| **LLM w/o CovRL** (SFT only) | **78.48%**             | **21.52%**          |
| LLM w/CovRL (full)           | 58.59%                 | 41.41%              |

So the SFT-only LLM mutator — no RL, no reward — is at 21.5% steady state
on JerryScript. RL training takes it to 41.4%. Our current asymptote of
~8% is below even the SFT-only baseline, and the gap is structural — see
the breakdown below.

## TL;DR — what CovRL changes vs. baseline AFL + LLM mutator

Three things that matter, in order of impact on steady-state validity:

1. **Token-level queue and mutation** (not byte-level). Queue entries are
   stored as `u16` token arrays; mutations are token inserts/overwrites;
   only the final exec call decodes tokens → bytes via the tokenizer's
   round-trip-safe `decode()`. This eliminates the entire class of
   byte-level UTF-8 corruption that produces our `�`-littered queue entries.
2. **Tiny per-mutation mask footprint.** 1–3 single-token `MASK_TOKEN`
   positions per mutation (`MASK_COUNT=3` in `afl-fuzz.c:338`). One,
   maybe two, maybe three holes — vs. our T5 span corruption with ≈ 5
   spans of ≈ 3 tokens each. Per-hole compounding hurts us.
3. **PPO/GRPO training with a validity-explicit reward.** Reward is
   `-1.0` (syntax), `-0.5` (semantic), or `+R_cov` (TF-IDF coverage,
   sigmoid-scaled into [0, 1]) — applied to every mutation that gets
   reward-evaluated. This is the gradient that pulls the model back
   toward valid output even when AFL is feeding it broken parents. It
   is the *only* mechanism in the entire pipeline that pushes back on
   the queue's coverage-greedy drift toward invalidity.

What CovRL does **not** change vs. baseline AFL:

- `save_if_interesting` is byte-for-byte identical to upstream AFL on the
  validity question (`afl-fuzz.c:3179-3233`). It gates only on
  `has_new_bits(virgin_bits)`. Any coverage-finding input is saved
  regardless of whether the JerryScript parser accepted it.
- Seed selection in `fuzz_one` uses the standard favored-queue heuristic
  with `SKIP_TO_NEW_PROB`/`SKIP_NFAV_NEW_PROB`/`SKIP_NFAV_OLD_PROB`
  (`afl-fuzz.c:5027-5054`). No validity check, no skip-if-broken.
- `cull_queue` is unchanged (`afl-fuzz.c:1305-1355`) — same
  speed-and-size favorability. Broken-but-fast inputs still win slots.
- The `queue/` directory contains broken outputs all the same.

The validity gradient lives entirely in the loss function. The queue
remains coverage-greedy; the model is what changes.

## How the mutation actually flows (code-grounded)

The fork is tightly coupled: the AFL binary speaks a 3-verb protocol to a
Python process over a TCP socket bound to `127.0.0.1:$PORT` (afl-fuzz.c:6873).

```
        ┌────────────────────────┐                ┌──────────────────────┐
        │  AFL (C, modified)     │ "predict" (7B) │  Python inferencer   │
        │  - queue: u16 tokens   │ ───────────►   │  (covrl/models/      │
        │  - mask insert/over    │                │   inferencer.py)     │
        │  - decode via socket   │ ◄─── len (2B)  │   - HF T5 generate   │
        │  - exec target         │                │   - contrastive      │
        └────────────────────────┘                └──────────────────────┘
                ▲                                          │
                │                                          │ writes to
                │ reads from <out>/MLM_pred (u16 tokens)   │ <out>/MLM_pred
                └──────────────────────────────────────────┘
```

Three verbs, all length-prefixed via a 2-byte reply:

| AFL sends   | What it means                                          | Where                          |
|-------------|--------------------------------------------------------|--------------------------------|
| `"predict"` | "I wrote a masked u16 token array to `<out>/MLM_pred`; fill the masks and write back to the same file." | `afl-fuzz.c:5228` |
| `"decode"`  | "I wrote a u16 token array to `<out>/MLM_decoded`; decode to UTF-8 bytes and write back." | `afl-fuzz.c:2536` |
| `"finetune"`| "Run a training cycle on whatever's accumulated in the queue dir." | `afl-fuzz.c:5485` (called from `sync_fuzzers`) |

Communication is via file (the model is too slow to round-trip through a
socket payload-wise) plus a sync barrier on the socket. The data path is
disk; the socket is a semaphore.

### Mutation in detail (afl-fuzz.c:5183-5259)

For each iteration of the havoc stage (`stage_max` iterations per queue
entry, perf-score-scaled):

```c
u32 mutation_mode = UR(2);   // 0=insert, 1=overwrite

if (mutation_mode == RANDOM_INSERT) {
    u32 insert_count = UR(MASK_COUNT) + 1;            // 1..3
    for (cnt = 0; cnt < insert_count; cnt++) {
        u32 index = UR(temp_len / 2);
        // insert MASK_TOKEN (=4) at `index`, expanding the buffer
    }
}
else if (mutation_mode == RANDOM_OVERWRITE) {
    u32 overwrite_count = UR(MASK_COUNT) + 1;         // 1..3
    for (cnt = 0; cnt < overwrite_count; cnt++) {
        u32 index = UR(temp_len / 2 - 1);
        memcpy(ex_tmp + index, &(u16[]){MASK_TOKEN}, 2); // overwrite
    }
}

write_to_MLM(MLM_pred, ex_tmp, temp_len);
send(sock, "predict", 7, 0);                          // wait for model
recv(sock, buf, 2, 0);                                // <- reply length
// re-read MLM_pred, now containing model output
decode(decoded_tokens, ex_tmp, temp_len);             // tokens -> UTF-8
common_fuzz_stuff(...);                                // run target
memcpy(out_buf, in_buf, len);                          // reset to parent
```

Key properties:

- **The parent is reset between iterations** (line 5267). Each mutation
  starts from the same queue entry. No stacked havoc-style accumulation.
  Per the paper, this is "mutation by mask" and "preserves overall
  structure while allowing specific tokens to be mutated" (§3.1).
- **At most 3 masks per mutation.** Splice (`afl-fuzz.c:5404`) places
  exactly 2 masks (`u16 a[2] = {MASK_TOKEN, MASK_TOKEN}`) around a
  spliced statement. Total: 1–3 single-position masks per mutation.
- **`MASK_TOKEN = 4`** is a magic value; the Python side
  (`inferencer.py:_mask_unknowns`, lines 89-98) walks the input and
  converts every occurrence of `MASK_TOKEN` (or `UNKNOWN_TOKEN`) into a
  distinct T5 sentinel (`vocab_size - i` for the i-th mask). So one
  mask byte in the buffer ≙ one sentinel for the model.

### Decoding: tokens → bytes is a tokenizer call, not a buffer cast

`decode()` in `afl-fuzz.c:2526-2554` does not look at bytes. It sends
the u16 token array over the socket and waits for the Python side to do
`tokenizer.decode(token_ids)`. The result is round-trip-safe UTF-8
because the HF tokenizer's vocabulary is constructed from valid UTF-8
substrings. **This is structurally why CovRL queue entries don't have
the `�` artifacts we see.**

In `rllm`, our queue entries go through `Tokenizer.reconstruct` which
returns `bytes` directly; if the model emits a token whose byte form is
a UTF-8 fragment (which byte-level BPE absolutely will), the resulting
buffer fails strict UTF-8 decoding and we `decode(errors="replace")` in
`post_process` — converting the fragment into `�`. CovRL never
hits this path because it operates on tokens through the tokenizer, not
on bytes through `bytes()`.

### Periodic finetuning, gated by `sync_fuzzers`

`sync_fuzzers` (`afl-fuzz.c:5465-5612`) is repurposed: the *first*
thing it does on entry is `send(sock, "finetune", 8, 0)` and block on
the reply. So every `SYNC_INTERVAL` (a `config.h` constant — typically
5–8 minutes wall clock) the LLM is taken offline and trained on
whatever has accumulated in the queue since the last cycle. The paper
says "every 2.5 hours" but the constant in the code is per-`fuzz_one`
calls, so the wall-clock depends on per-call cost; the per-cycle
training time the paper quotes is ~10 minutes.

After `"finetune"` returns, the rest of `sync_fuzzers` does AFL's normal
work (reading from other sync_id queues, calling `save_if_interesting`
on each — `afl-fuzz.c:5591`). Without `-S/-M` sync IDs there are no
other queues; the finetune happens but the sync loop is empty.

### Reward computation (`covrl/models/rewarding.py`)

For every mutation marked `is_orig=False`:

1. Run JerryScript with `afl-showmap` to get stderr/stdout and a coverage
   map (lines 75-152).
2. Classify the run with `map_target_error()` into syntax / semantic /
   passed.
3. Assign reward:
   - `SYNTAX_ERROR` → `data["reward"] = -1.0` (line 192)
   - Any other classified error → `data["reward"] = -0.5` (line 195)
   - Pass → reward is set to **sigmoid(log(TF-IDF·bitmap))** in
     `get_reward()` (lines 217-231), a small positive in [0, 1].

This is the **r(W*)** of paper Eq. 2. Implementation matches the paper.

A wrinkle worth noting: the positive reward is small (sigmoid of a log
score is dominated by mass near 0.5), while the negative reward is
large (-1.0). The loss landscape is therefore much steeper on the
"don't be invalid" side than on the "find rare coverage" side. This
biases the policy gradient toward producing parseable output before it
optimises coverage diversity.

### The actual PPO loss (`covrl/models/finetuner.py:compute_actor_loss`)

The paper writes the loss as standard PPO with the policy ratio
`π_θ_t / π_θ_{t-1}` weighted by the reward and KL-regularised. The
implementation is rougher:

```python
ratio = torch.exp(cur_log_probs - prev_log_probs)            # token-level
clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
ppo_loss = -torch.min(ratio * reward, clipped_ratio * reward).mean()
final_loss = ppo_loss + cur_outputs.loss.mean()
```

`cur_log_probs` and `prev_log_probs` are full vocab log-probs, *not
gathered to the actual next-token labels*; the elementwise product with
a scalar reward is summed by `mean()` over every (batch, seq, vocab)
slot. Effectively the PPO term is averaged across the full softmax
output, which dilutes the signal compared to a proper per-token PPO. It
still works because the negative reward signal is large and the SFT
auxiliary loss (`cur_outputs.loss.mean()`) keeps the model anchored.

This is one of several places where the *behaviour* the paper describes
is achieved by an implementation that does not quite match the equations.
**For our reproduction, be willing to deviate from the paper's formulation
where the released code already does.**

## Side-by-side: CovRL vs. rllm on the choices that matter

| Choice                                  | CovRL                                                                            | rllm (current)                                                                | Effect on validity                          |
|-----------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------|
| Queue entry storage                     | u16 token arrays (`afl-fuzz.c:5070`)                                            | Byte buffers (AFL++ standard)                                                | UTF-8 corruption avoided ↑                  |
| Mutation level                          | Token inserts/overwrites on u16 array                                            | T5 span corruption on tokenizer-produced IDs                                  | Smaller footprint ↑                          |
| Masks per mutation                      | 1–3 single-token positions                                                       | ≈5 spans × ≈3 tokens (≈15 token holes total)                                  | Lower compounding ↑                          |
| Token → byte path                       | `tokenizer.decode()` via socket                                                  | `bytes(tokenizer.reconstruct(...))` + `errors="replace"`                      | No UTF-8 fallbacks ↑                         |
| `save_if_interesting` validity check    | None (same as upstream)                                                          | None (same as upstream)                                                       | =                                            |
| Seed selection                          | Standard favored-queue                                                           | Standard favored-queue                                                        | =                                            |
| Validity reward in loss                 | -1.0 / -0.5 / +R_cov via PPO                                                     | None (SFT-only, frozen weights)                                               | Their main lever ↑↑↑                          |
| Coverage reward shape                   | Sigmoid(log(TF-IDF · bitmap))                                                    | n/a                                                                            | Diversity ↑                                  |
| Training cadence                        | Every `SYNC_INTERVAL` cycles                                                     | n/a (no training)                                                              | Adapts to drift ↑                            |
| Tokenizer                               | CodeT5+ 220M (paper §4)                                                         | CodeT5+ 220M (`configs/default.json`)                                         | =                                            |
| Sampling                                | Contrastive (penalty_alpha=0.6, top_k=32) — `inferencer.py:177-187`              | Contrastive (penalty_alpha=0.6, top_k=4 after OOM fix)                        | They use larger top_k                        |
| `do_sample`                             | `True` with `penalty_alpha` set (HF treats this as contrastive search anyway)    | `False` with `penalty_alpha` set (our reading of HF's "contrastive trigger")  | Worth verifying we agree on what HF runs    |

## What this means for the rllm roadmap

To match CovRL's **SFT-only baseline** (~21.5% steady state) without
training:

1. **Smaller mask footprint.** `corruption_rate=0.03`, `mean_span_length=1`,
   `max_span_length=1` already does this in `configs/conservative.json`.
   Initial measurements show this lifts the *peak* validity from ~28% to
   ~36% — but the queue collapse still pulls it down.

2. **Token-level token-aware queue handling.** This is the deeper change.
   Two ways to approach it without forking AFL++:
   - **In `post_process`**, instead of returning byte-level
     `errors="replace"`, decode the tokenizer's output through
     `tokenizer.decode(skip_special_tokens=True).encode("utf-8")` so the
     mutation is always emitted as well-formed UTF-8 derived from
     tokens. This matches CovRL's `decode` step.
   - **Mutation should always start from a tokenize-then-detokenize
     round-trip of the parent.** If the parent is a stored byte buffer
     whose UTF-8 we've already corrupted, re-tokenising it and emitting
     the re-detokenised bytes washes out the corruption (at the cost of
     some information). This is implicit in CovRL because the queue
     stores tokens not bytes.

3. **No need to filter the queue.** CovRL explicitly does *not* filter
   the queue on validity (verified above). Our [queue_validity_collapse.md]
   proposal to skip invalid entries in `queue_get` is more aggressive
   than CovRL's approach and may starve AFL of coverage-finding parents.
   Test it as a debug feature, but recognise it's not how CovRL solves
   the problem.

To match CovRL's **full RL pipeline** (~41.4% steady state):

4. **Wire up the validity reward.** -1.0 / -0.5 / +R_cov, computed by
   running the mutation through JerryScript+afl-showmap (we already
   have the validity classification working via `exit_hook.so` — the
   stderr signal is the same one CovRL extracts via `map_target_error`).

5. **PPO/GRPO with a per-mutation reward.** The CovRL implementation
   uses an unusually loose PPO formulation (full-vocab log-prob
   product, not per-token gathered). A proper GRPO implementation
   should be tighter and may close the gap faster.

6. **TF-IDF Coverage-Weighted Rewarding.** The IDF weight map is small
   (one float per bitmap edge, ~64KB total) and the math is cheap;
   reproducing it is mostly bookkeeping.

## Anomalies and things worth checking

- **CovRL `do_sample=True` + `penalty_alpha`.** HF's contrastive search
  trigger is `do_sample=False` with `penalty_alpha` set. With
  `do_sample=True` and `penalty_alpha` set, HF's behaviour is
  technically `sample_contrastive` (a sampling-with-degeneration-penalty
  variant), not pure contrastive search. We use `do_sample=False`; the
  reference does not. Worth verifying which mode each one is actually
  running and whether it matters.

- **PPO loss not gathered to labels.** As above. The released
  implementation may underperform the paper's claimed reward gradient.
  Our GRPO implementation should *not* copy this verbatim; gather to
  labels and average per token.

- **Reward distribution is asymmetric.** -1.0 vs sigmoid(...) ∈ [0, 1]
  means the loss pulls harder away from invalid than toward rare
  coverage. This is probably intentional (validity is the hard
  constraint, coverage is the soft objective), but it's worth
  documenting as a design choice.

- **`u16` token IDs cap the vocabulary at 65k.** Fine for CodeT5+
  (~32k vocab) but rules out larger tokenizers. We've inherited
  unbounded vocab support; check whether reference implementations
  we'd want to swap in stay under 65k.

- **`SYNC_INTERVAL` is per fuzz_one call, not wall-clock.** With slow
  exec rates the "every 2.5 hours" cadence the paper quotes is
  approximate at best. For a fair reproduction we should fix the
  retraining cadence to wall-clock (e.g., a timestamp check inside
  `_maybe_finetune`).

- **The single seed dataset in the paper (52K JS files from V8/JSC/Chakra/Jerry/Test262/jsvuln)
  is curated.** Our seed corpus (33 files, hand-checked 100% valid)
  is much smaller. Diversity matters for the training distribution
  but not for the validity-collapse question we measured here.

- **`save_if_interesting` writes the *encoded* (u16) buffer to the
  queue, not the decoded bytes** (`afl-fuzz.c:3229`). When the entry is
  later reread by `fuzz_one`, it's read as `u16*` again
  (`afl-fuzz.c:5070`). This is the load-bearing piece that keeps the
  queue token-clean across cycles.

## What does this not tell us?

- Whether 41% is the steady state of CovRL's reward gradient or just
  where it sits at 24h. Their plots (Fig. 4) show coverage still rising
  at 24h, so the system isn't at equilibrium. Error rate plots
  (Fig. 5) are a single snapshot — we can't infer dynamics.

- Whether the validity gradient is doing the work or whether the
  TF-IDF coverage weighting is. Their ablation (Table 7) shows
  `LLM w/o RL` and `LLM w/CR` give similar coverage but different
  error rates; `LLM w/CRR` and `LLM w/CWR` differ from each other on
  coverage but track similarly on error rate. Suggests the validity
  reward dominates the error-rate improvement; CWR mostly affects
  coverage. Useful when budgeting our implementation effort.
