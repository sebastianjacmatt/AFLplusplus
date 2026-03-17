# CovRL AFL++ Port — Implementation Plan

## Ground truth: modified afl-fuzz.c behaviour

Before planning anything, these facts must be taken as fixed constraints derived by reading the code.

### All classical AFL deterministic stages are removed

Between `doing_det = 1` (line 5136) and `havoc_stage:` (line 5143) there is no code. The section header is literally labelled `RANDOM HAVOC with Mask Mutation`. Bit-flip, arithmetic, interesting-value, extras — all gone. The `passed_det` / `was_fuzzed` fields survive from AFL 2.52b data structures, but `mark_as_det_done()` is never called during normal operation.

The only consequence of `doing_det` is selecting `stage_max`:

```c
stage_max = (doing_det ? HAVOC_CYCLES_INIT : HAVOC_CYCLES) * perf_score / havoc_div / 100;
if (stage_max < HAVOC_MIN) stage_max = HAVOC_MIN;
```

Where (from `config.h`, read directly — do not assume stock AFL values):

| Constant | Value | Source |
|----------|-------|--------|
| `HAVOC_CYCLES_INIT` | 1024 | config.h |
| `HAVOC_CYCLES` | 256 | config.h |
| `HAVOC_MIN` | 16 | config.h |
| `HAVOC_MAX_MULT` | 16 | config.h |
| `SPLICE_CYCLES` | 15 | config.h |
| `SPLICE_HAVOC` | 32 | config.h |
| `SYNC_INTERVAL` | 100 | config.h |
| `MASK_TOKEN` | 4 | afl-fuzz.c line 337 |
| `MASK_COUNT` | 3 | afl-fuzz.c line 338 |

`havoc_div` defaults to 2 at startup and adjusts dynamically:

```c
if (avg_us > 50000) havoc_div = 10;   // < 20 execs/sec
else if (avg_us > 20000) havoc_div = 5; // 20-49 execs/sec
else if (avg_us > 10000) havoc_div = 2; // 50-100 execs/sec
// else stays at 2
```

`perf_score` is computed per-seed by `calculate_score()` and ranges from 10 to `HAVOC_MAX_MULT * 100 = 1600`.

### stage_max dynamic doubling — not reproducible

Inside the havoc loop, when a new path is found:

```c
if (queued_paths != havoc_queued) {
    if (perf_score <= HAVOC_MAX_MULT * 100) {
        stage_max  *= 2;
        perf_score *= 2;
    }
    havoc_queued = queued_paths;
}
```

`stage_max` can double on each new discovery, capped at HAVOC_MAX_MULT * 100 = 1600. This is an online feedback loop that AFL++ `fuzz_count()` cannot replicate because `fuzz_count()` must return N before any mutations run.

### Finetune is gated on sync mode (`-S`)

`sync_fuzzers()` (which contains the `send("finetune")` call) is only reached when `sync_id` is non-null, which requires the `-S` flag. The counter:

```c
if (!stop_soon && sync_id && !skipped_fuzz) {
    if (!(sync_interval_cnt++ % SYNC_INTERVAL))
        sync_fuzzers(sock, use_argv);
}
```

`sync_interval_cnt` increments once per non-skipped `fuzz_one()` call (per seed, not per havoc iteration). Finetune fires every 100 non-skipped seed fuzzing operations.

### Startup environment variables

The modified AFL requires before `perform_dry_run()`:
- `PORT` env var: TCP port to `do_covrl.py`
- `VOCAB_SIZE` env var: tokenizer vocabulary size

### Splice stage

Splice is only entered when `use_splicing` is true (enabled after a full queue cycle with no new finds) and only for up to `SPLICE_CYCLES = 15` cycles. Each splice cycle resets `perf_score = orig_perf` and uses `stage_max = SPLICE_HAVOC * perf_score / havoc_div / 100`. The splice logic:

1. Pick a random queue entry as target.
2. Identify statement boundaries (semicolons) in both buffers via `get_splice_list()` — requires ≥ 2 boundaries each.
3. Pick a boundary range from target (new_index_f, new_index_l) and one from source (in_index_f, in_index_l).
4. Overwrite the target range with tokens from the source range.
5. Place two `MASK_TOKEN` sentinels at the boundary positions of the spliced region.
6. Jump back to `havoc_stage:` with `in_buf = new_buf`.

After splicing, the havoc loop runs identically (mask + predict + decode + execute).

---

## AFL++ vs CovRL 2.52b: differences that affect the plan

This section documents every divergence between `AFLplusplus/config.h` + `AFLplusplus/src/afl-fuzz.c` and their CovRL counterparts. Each entry states the difference, explains why it matters, and records the decision the port must make.

### Config constant comparison

| Constant | CovRL AFL 2.52b | AFL++ 4.36a | Same? | Port decision |
|----------|-----------------|-------------|-------|---------------|
| `HAVOC_CYCLES` | 256 | 256 | ✓ | Use 256 |
| `HAVOC_CYCLES_INIT` | 1024 | 1024 | ✓ | Use 1024 |
| `SPLICE_CYCLES` | 15 | 15 | ✓ | Use 15 |
| `SPLICE_HAVOC` | 32 | 32 | ✓ | Use 32 |
| `HAVOC_MIN` | **16** | 12 | ✗ | Use **16** — CovRL value governs the floor on N_havoc and N_splice calculations |
| `HAVOC_MAX_MULT` | **16** | 64 | ✗ | Use **16** — CovRL's cap on perf_score is what shaped the original N ceiling |
| `SYNC_INTERVAL` | **100** | **8** | ✗ | **Completely different concepts** — see section below |
| `SYNC_TIME` | N/A | 20 min | — | Irrelevant — not used in the port |

`HAVOC_MIN=16` is the correct floor for all budget calculations. `HAVOC_MAX_MULT=16` is the cap that limited the original perf_score doubling to 1600. Although dynamic doubling is not implementable in `fuzz_count()`, the cap is still the reference for reasoning about the original's maximum N. Neither AFL++'s `HAVOC_MIN=12` nor `HAVOC_MAX_MULT=64` should be used.

### SYNC_INTERVAL — entirely different semantics

**CovRL 2.52b**: `SYNC_INTERVAL=100` is the interval between calls to `sync_fuzzers()`, which in the CovRL patch is the mechanism that sends `"finetune"` over TCP. It counts non-skipped `fuzz_one()` calls (one per queue entry). This is the finetune trigger.

**AFL++ 4.36a**: `SYNC_INTERVAL=8` is used inside `maybe_sync_fuzzers()` to count calls between fuzzer-to-fuzzer queue syncs (sharing test cases across parallel AFL++ instances). It is additionally time-gated: sync only fires if `cur_time > last_sync_time + SYNC_TIME` (20 min). This has nothing to do with LLM finetuning.

**Port decision**: AFL++'s `SYNC_INTERVAL` and `maybe_sync_fuzzers()` are completely irrelevant to the finetune trigger. The finetune trigger is implemented independently in `queue_get()` using the CovRL constant `SYNC_INTERVAL=100` (named `FINETUNE_INTERVAL` in the port to avoid confusion with AFL++'s own symbol). The plan already implements this in section 1.4.

### AFL++ auto-sync vs CovRL's explicit `-S` requirement

**CovRL 2.52b**: `sync_fuzzers()` (and therefore the finetune trigger) is only reached when `sync_id` is non-null. `sync_id` is only set when the user passes `-S <name>`. Running without `-S` means finetune never fires. This was an intentional design choice in CovRL: finetune is part of the multi-instance sync protocol.

**AFL++ 4.36a**: AFL++ auto-configures a default sync identity even without `-S`. When no sync_id is provided, AFL++ sets `afl->sync_id = ck_strdup("default")` and `afl->is_secondary_node = 1` (src/afl-fuzz.c line ~1721). The condition that guarded CovRL's finetune trigger (`if (sync_id && ...)`) always evaluates true in AFL++.

**Port decision**: The port does not replicate the `-S` gate. `queue_get()` always increments the counter and always triggers finetune at the FINETUNE_INTERVAL boundary. This is strictly more correct than the original: the finetune trigger is independent of whether the user runs multi-instance fuzzing. No flag or environment variable is needed to enable finetuning.

### AFL_CUSTOM_MUTATOR_ONLY=1 and fuzz_count() as sole N controller

When `AFL_CUSTOM_MUTATOR_ONLY=1` is set, AFL++ suppresses all of its own internal mutation stages (`bit_flip`, `arith`, `interest`, `havoc`, `splice`). AFL++ still calls `fuzz_count()` to determine how many times to invoke `fuzz()`, but AFL++ no longer computes its own `stage_max` via `calculate_score()` and applies no independent mutation budget.

This means the N returned by `fuzz_count()` is the total number of mutations AFL++ will perform for that seed selection — there is no internal AFL++ multiplier or override on top of it. The custom `N_total = N_havoc + N_splice` formula is the complete budget with no interference from AFL++ internals.

Without `AFL_CUSTOM_MUTATOR_ONLY=1`, AFL++ would run its own mutation stages in addition to the custom mutator, producing an unpredictable mix of token-level and byte-level mutations. This flag is mandatory.

---

## Stage 1 — Fuzzing scheme (independent of CovRL ML)

This stage is planned entirely without reference to the model or finetuner. All model calls are treated as opaque functions with known signatures.

### 1.1 Hook mapping

| Modified AFL concept | AFL++ custom mutator hook |
|---------------------|--------------------------|
| `fuzz_one()` seed load + decode + calibration | `fuzz_count(buf)` — tokenise once |
| Per-havoc-iteration mask → predict → decode → execute | `fuzz()` — one mutation per call |
| `save_if_interesting()` callback | handled internally by AFL++ |
| `sync_fuzzers()` / SYNC_INTERVAL counter | `queue_get(filename)` — count non-skipped calls |
| `queue_new_entry` equivalent | `queue_new_entry(filename_new, filename_orig)` |
| Splice stage re-entry into havoc | Subset of `fuzz()` calls use `add_buf` |

### 1.2 fuzz_count(buf) — tokenisation and N

**This is the only place tokenisation happens.** `fuzz()` must not re-tokenise.

Steps:
1. Decode `buf` bytes to UTF-8 JavaScript source text.
2. Tokenise using `TOKENIZER(text)` → `current_seed_token_ids` (list of ints).
3. Cache `current_seed_token_ids` in a module-level global.
4. Look up `current_seed_filename` (set by the immediately preceding `queue_get()` call) in `_seed_fuzz_state: dict[str, bool]`. If absent or False → `doing_det = True`, mark entry as seen.
5. Compute N:

```
if doing_det:
    N = HAVOC_CYCLES_INIT * DEFAULT_PERF_SCORE // HAVOC_DIV_DEFAULT // 100
      = 1024 * 100 // 2 // 100
      = 512
else:
    N = HAVOC_CYCLES * DEFAULT_PERF_SCORE // HAVOC_DIV_DEFAULT // 100
      = 256 * 100 // 2 // 100
      = 128
N = max(N, HAVOC_MIN)   # floor at 16
```

**Why fixed perf_score / havoc_div**: AFL++ does not expose the per-seed performance score to the custom mutator. `havoc_div` is a runtime exec-speed adjustment. Using the defaults (perf_score=100, havoc_div=2) is the only faithful approximation available. Dynamic doubling when new paths are found is **not implementable** in `fuzz_count()` because N must be returned before any mutations execute. This is a known, accepted deviation from the original.

**Splice budget within N**: The last `SPLICE_CYCLES * SPLICE_HAVOC // HAVOC_CYCLES = 15 * 32 // 256 ≈ 2` calls out of N (when N = 128) can be reserved for splice-mode mutations (using `add_buf`). This is a conservative approximation of the original's separate splice stage. Exact proportionality is computed from the same config.h constants.

6. Return N.

### 1.3 fuzz(buf, add_buf, max_size) — per mutation

`buf` is ignored here because the token representation was already derived in `fuzz_count()`. `current_seed_token_ids` is the working representation.

Steps:
1. `base = list(current_seed_token_ids)` — fresh copy per call.
2. Determine whether to use splice mode based on call index vs N (see 1.2 splice budget and section 1.5).
3. **Normal mode (RANDOM_INSERT or RANDOM_OVERWRITE, equal probability)**:
   - `RANDOM_INSERT`: pick `insert_count = random.randint(1, MASK_COUNT)` positions; for each, insert `MASK_TOKEN` at a uniformly random index in `base`.
   - `RANDOM_OVERWRITE`: pick `overwrite_count = random.randint(1, MASK_COUNT)` positions (capped at `len(base) - 1`); for each, set `base[index] = MASK_TOKEN`.
4. **Splice mode** (when add_buf is non-empty and splice mode selected):
   - Decode and tokenise `add_buf` → `splice_token_ids`.
   - Find semicolon token positions in both `base` and `splice_token_ids` (token value for `;` determined from TOKENIZER vocabulary at init time).
   - If either has fewer than 2 semicolons, fall back to normal mode.
   - Pick a contiguous statement range from `splice_token_ids` (between two semicolon positions).
   - Pick a statement range in `base` to replace.
   - Replace that range in `base` with the tokens from `splice_token_ids`.
   - Place `MASK_TOKEN` at the two boundary positions of the inserted region.
5. Guard: if `len(masked) <= 3`, return `buf` unchanged (mirrors `if (temp_len <= 6) break;`).
6. Call `ACTOR.inference(masked)` → `infilled_ids` (opaque model call).
7. Decode: `js_text = TOKENIZER.decode(infilled_ids, skip_special_tokens=True)`.
8. Return `js_text.encode("utf-8")`.

The `post_process()` hook is not needed. Decoding to JS text happens inside `fuzz()` which returns UTF-8 bytes directly to AFL++.

### 1.4 queue_get(filename) — skip logic and finetune counter

AFL++ handles favored/was_fuzzed skip logic internally. `queue_get()` is used for:
1. Store `_current_seed_filename = filename` so `fuzz_count()` can look up seed state.
2. Increment `_fuzz_one_counter`. Every `SYNC_INTERVAL = 100` non-skipped calls, set a `_finetune_pending = True` flag.
3. Return `True` always (do not add a second skip layer on top of AFL++'s own).

Finetune is then triggered at the start of the next `fuzz_count()` call when `_finetune_pending` is True. Triggering inside `fuzz_count()` (before tokenisation) mirrors the original's placement: `sync_fuzzers()` runs after the previous `fuzz_one()` completes and before the next seed is processed.

### 1.5 Splice mode tracking

A module-level `_call_index: int` is incremented inside `fuzz()` and reset to 0 in `fuzz_count()`. Splice mode activates when `_call_index >= (N - splice_budget)` AND `add_buf` is non-empty. This concentrates splice mutations at the tail of the N-call window, matching the original's splice stage placement (after the main havoc loop completes).

`splice_budget = max(HAVOC_MIN, SPLICE_CYCLES * SPLICE_HAVOC * DEFAULT_PERF_SCORE // HAVOC_DIV_DEFAULT // 100) = max(16, 15 * 32 * 100 // 2 // 100) = max(16, 240) = 240`.

Wait — this is a problem. 240 out of N=512 (first pass) or 240 out of N=128 (subsequent) means splice dominates on repeat passes. The original ran splice in a separate loop AFTER the main havoc, not as a fraction of it. The splice budget cannot simply be additive within N without inflating N.

**Resolution**: For N calculation in 1.2, separate the splice count from the main havoc count:

```
N_havoc = max(HAVOC_MIN, (HAVOC_CYCLES_INIT if doing_det else HAVOC_CYCLES)
              * DEFAULT_PERF_SCORE // HAVOC_DIV_DEFAULT // 100)
N_splice = SPLICE_CYCLES * max(HAVOC_MIN, SPLICE_HAVOC * DEFAULT_PERF_SCORE
                                          // HAVOC_DIV_DEFAULT // 100)
         = 15 * max(16, 32 * 100 // 2 // 100)
         = 15 * 16   # since 32*100//2//100 = 16
         = 240

N_total = N_havoc + N_splice
```

Splice mode activates when `_call_index >= N_havoc`. This faithfully mirrors the original's two-phase structure. `N_total` is what `fuzz_count()` returns.

Default values:
- First pass: N_total = 512 + 240 = 752
- Subsequent: N_total = 128 + 240 = 368

### 1.6 Vocabulary size

`VOCAB_SIZE` in the original is read from the environment and used by the havoc loop to bound token insertion. In the port it is `TOKENIZER.vocab_size` (available after tokenizer is loaded in `init()`). No environment variable is needed for this.

### 1.7 Module-level state summary

```python
# Set once in init()
CONFIG          # loaded from config file
ACTOR           # Inferencer instance (opaque in stage 1)
TOKENIZER       # loaded tokenizer

# Updated in queue_get() / fuzz_count()
_current_seed_filename: str
_seed_fuzz_state: dict[str, bool]   # filename -> was_already_fuzzed_once
_current_seed_token_ids: list[int]
_call_index: int                     # fuzz() call index within current seed's N
_N_havoc: int                        # stored so fuzz() knows when splice starts
_fuzz_one_counter: int               # counts queue_get() calls
_finetune_pending: bool
```

---

## Stage 2 — CovRL ML integration

Stage 2 is planned after Stage 1 is complete. It replaces the opaque model calls from Stage 1 with concrete implementations drawing from the CovRL Python codebase. Key open questions to be resolved in Stage 2:

1. **Config loading**: How `CONFIG` is passed to the mutator (env var path, hardcoded path, or AFL++ `AFL_PYTHON_MODULE_XTRA` extra argument).
2. **Inferencer init**: The `Inferencer` constructor loads `T5ForConditionalGeneration` and a `FineTuner`. Both require GPU or CPU fallback. Init must happen in `init()`, not in `fuzz()` or `fuzz_count()`.
3. **Finetune dataset path**: `_finetune()` calls `finetuner.preprocess(corpus_dir)`. The corpus_dir must be the AFL++ queue directory (obtained from `queue_new_entry` filename paths or from an env var equivalent to AFL's `out_dir`).
4. **first-cycle critic-only rule**: `finetune_actor` is skipped on the first finetune call. Stage 2 must track `_finetune_cycle_index` and gate actor training on `_finetune_cycle_index > 0`.
5. **afl-showmap path**: `rewarding.py` hardcodes `./AFL/afl-showmap`. In the AFL++ port this must point to AFLplusplus' `afl-showmap` binary. Resolved via config or env var.
6. **Dataset mixing**: `FineTuner.preprocess()` samples from `train_dataset` at 4× the size of `mutation_dataset`. The `train_dataset_path` in config must point to the pre-processed training corpus.
