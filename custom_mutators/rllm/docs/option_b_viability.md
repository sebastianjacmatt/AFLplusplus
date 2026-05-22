# Option B viability — can we match CovRL-Fuzz's afl-fuzz.c on stock AFL++?

## Verdict

**Yes.** Option B (u16 binary token-ID queue + `post_process` decode) is
viable on stock AFL++, with no source modifications. The load-bearing
mechanism is the documented env var
`AFL_POST_PROCESS_KEEP_ORIGINAL=1`. With it set, the bytes our
`fuzz()` returns are exactly what AFL writes to the queue, and only
the post-processed decode goes to `.cur_input` for the target. This is
the CovRL-Fuzz queue-as-tokens architecture, achieved through an
extension point AFL++ already provides (originally for
libprotobuf-mutator's mutator-format-vs-target-format split). The doc
[`aligning_with_covrl.md`](aligning_with_covrl.md) §3.2 was conservative
about Option B's viability; that conservatism is no longer warranted.

This doc replaces the viability-question side of that section. The
*recommendation* (default A or default B) is a separate decision the
broader doc still owns; this writeup only resolves the can-we-do-it
question.

---

## 1. Why we used to think Option B was risky

The conservative reading in `aligning_with_covrl.md` §3.2 listed three
cons that *sounded* like they made Option B fragile:

- AFL++ would write decoded JS to the queue (so the u16 form wouldn't
  persist).
- Tools like `afl-cmin`/`afl-tmin`/`afl-showmap` would break.
- A `post_process` step would need to be written from scratch.

Looking at the AFL++ source, only the second bullet is real. The first
is wrong — AFL++ has an explicit env var for exactly this case. The
third is a one-page Python function, not a major piece of plumbing.

---

## 2. The mechanism — `AFL_POST_PROCESS_KEEP_ORIGINAL=1`

All line references below are to the current
`~/Documents/AFLplusplus/` tree (verified at session time, 2026-05-21).

### 2.1 What `write_to_testcase` does

`src/afl-fuzz-run.c:134` defines:
```c
u32 write_to_testcase(afl_state_t *afl, void **mem, u32 len, u32 fix)
```

`*mem` is the buffer the caller (typically `common_fuzz_stuff`) passed
in — for the custom-mutator path that's the buffer `fuzz()` just
returned. The function then walks the custom-mutator list and applies
each `post_process` in turn (`src/afl-fuzz-run.c:145–167`).

When `post_process` produces a different buffer, the function makes a
fresh `new_buf` from it (`:194–198`) and conditionally stashes the
original pointer in a local `new_mem` if KEEP_ORIGINAL is set
(`:202–206`). It then writes the **post-processed** bytes to the
forkserver's input via `afl_fsrv_write_to_testcase(&afl->fsrv, *mem, new_size)`
at `:237`.

After that write, lines `:241–251`:

- **Default** (`AFL_POST_PROCESS_KEEP_ORIGINAL` unset or 0):
  `len = new_size;` — `*mem` is left pointing at the post-processed
  buffer. The caller sees post-processed.
- **`AFL_POST_PROCESS_KEEP_ORIGINAL=1`**:
  `*mem = new_mem;` — `*mem` is restored to the original buffer the
  caller passed in. The caller sees the original.

### 2.2 What `save_if_interesting` writes to the queue

`src/afl-fuzz-bitmap.c:752`:
```c
ck_write(fd, mem, len, queue_fn);
```

`mem` here is the same buffer the caller of `save_if_interesting`
passed in — which is the same `*mem` `common_fuzz_stuff` is holding
after `write_to_testcase` returned. With KEEP_ORIGINAL on, that's the
original pre-post-process buffer. So the **queue file contains exactly
what `fuzz()` returned** — for us, u16 binary token IDs.

That is the property CovRL's modified `afl-fuzz.c` achieves at lines
`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:3207, 3229` by storing the
`encoded_buf` (u16) directly. AFL++ reaches the same end state via the
env var.

### 2.3 Calibration is already pinned to KEEP_ORIGINAL semantics

`src/afl-fuzz-run.c:528–530`:
```c
u8 saved_afl_post_process_keep_original =
    afl->afl_env.afl_post_process_keep_original;
afl->afl_env.afl_post_process_keep_original = 1;
```

`calibrate_case` forcibly enables KEEP_ORIGINAL for its lifetime and
restores at `:713–714`. So even when the env var is not set globally,
the calibration path — the path that handles new finds' first
executions and adds them to the queue — already preserves the
mutator's original buffer. This is the AFL++ maintainers explicitly
ensuring the "mutator format ≠ target format" pattern works on the
critical "first execution of a new queue entry" path.

Setting the env var globally just extends that behavior to **every**
post-`fuzz()` write_to_testcase. We want it on for the whole run.

### 2.4 No byte-level mutation sneaks in between `fuzz()` and queue save

`AFL_CUSTOM_MUTATOR_ONLY=1` (already set in `run_rllm.sh`) makes
`src/afl-fuzz-one.c:2115–2123` skip the havoc stage:
```c
if (afl->custom_only) goto abandon_entry;
```

`AFL_DISABLE_TRIM=1` (already set) disables `trim_case`. The
deterministic byte-level stages (interesting-8/16/32, arith, bit
flips) operate on `out_buf` loaded at the start of `fuzz_one` (from
`queue_testcase_get`) and only when `custom_only` is unset. With our
env vars, the chain from `fuzz()` to `save_if_interesting` is
unmodified.

### 2.5 Initial seeds load as opaque bytes

`src/afl-fuzz-init.c:read_testcases` adds each `-i` file to the queue
without inspecting content; `add_to_queue`
(`src/afl-fuzz-queue.c:746–747`) records `fname` and `len` only.
Loading is lazy via `queue_testcase_get` and returns raw bytes. So a
pre-processed `-i` directory of u16 binary files works without any
AFL++ change.

---

## 3. CovRL property → AFL++ mechanism

Every load-bearing property of CovRL's modified AFL 2.52b has a
corresponding AFL++ extension-point. None require source modification.

| CovRL property | CovRL code | AFL++ mechanism |
|---|---|---|
| Queue stores u16 tokens | `ck_write(fd, encoded_buf, encoded_len, fn)` — `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:3229` | `fuzz()` returns u16 binary; `AFL_POST_PROCESS_KEEP_ORIGINAL=1` causes `save_if_interesting` to ck_write the original via `src/afl-fuzz-bitmap.c:752` |
| Queue reads u16 tokens | `u16 *in_buf = mmap(...)` — `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5070` | AFL++ reads opaque bytes via `queue_testcase_get`; we cast to `uint16_t*` inside `fuzz_count`/`fuzz` |
| Decode once per execution | `decode(decoded_tokens, ex_tmp, temp_len)` — `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5247–5252` | `post_process(buf)` invoked from `write_to_testcase` at `src/afl-fuzz-run.c:147–167` |
| No re-tokenization in steady state | Decode is one-way to a tmp file | Same — post_process output goes only to `.cur_input`; queue keeps the original (with KEEP_ORIGINAL=1) |
| Calibration: tokens → decode → target | `decode(decoded_tokens, in_buf, len)` before `calibrate_case` — `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5095–5099` | `calibrate_case` calls `write_to_testcase(fix=1)` at `src/afl-fuzz-run.c:583`; KEEP_ORIGINAL forcibly on per `:528–530` |
| Mutation on u16 buffer | `u16 *out_buf, *ex_tmp` — `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5008` | Our `fuzz()` receives `buf` as `u8*` and casts to `uint16_t*` |
| MASK_TOKEN sentinel | `#define MASK_TOKEN 4` — `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:337` | Pick a reserved CodeT5+ vocab ID (e.g., one of `<extra_id_N>`) and use it identically |
| Sentinel mapping for the LLM | `~/Documents/CovRL-Fuzz/covrl/models/inferencer.py:_mask_unknowns` lines 89–98 (`vocab_size - mask_idx`) | Done inside our Python `Mutator` — already implemented analogously in `tokenizer.py:_resolve_sentinel_id` |
| Mutation operations: INSERT, OVERWRITE, SPLICE | `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5193–5221`, `:5305+` | Implemented in our Python `Mutator` operating on token lists; same shapes, no extra plumbing |

Every row is a one-to-one mapping. There is no row where CovRL's
behavior needs a code path AFL++ doesn't already expose.

---

## 4. Required configuration

To run Option B faithfully, the env vars and hook choices are:

- **`AFL_POST_PROCESS_KEEP_ORIGINAL=1`** — load-bearing. Without it,
  queue stores decoded JS and Option B collapses to a worse Option A
  (decode shim cost without invariant gain).
- `AFL_CUSTOM_MUTATOR_ONLY=1` — already set. Disables havoc and
  deterministic stages.
- `AFL_DISABLE_TRIM=1` — already set. Disables `trim_case`.
- `AFL_NO_FASTRESUME=1` — already set. Forces fresh queue ingestion
  on restart.
- `splice_optout()` hook present — already in `rllm.py`. Disables AFL's
  byte-level splice. (We'd implement our own SPLICE on tokens later if
  we want it; CovRL's SPLICE is two MASK sentinels around a pasted
  statement.)
- `AFL_PYTHON_MODULE=rllm` — already set.

**Not** required: no AFL++ source modifications, no `fuzz_send`
override, no new custom-mutator hooks beyond what `rllm.py` already
exposes.

---

## 5. Shape of `fuzz_count` / `fuzz` / `post_process` under Option B

Pseudocode for illustration only — no implementation in this doc:

```python
def fuzz_count(buf):
    # buf is u16 binary read from a queue entry.
    self._tokens = [t for (t,) in struct.iter_unpack("<H", bytes(buf))]
    return self.cfg.fuzz_count

def fuzz(buf, add_buf, max_size):
    # Run masking + LLM fill on the cached token list.
    masked = self.masking.mask(self._tokens)
    out_tokens = self.model.generate_one(masked)
    return struct.pack(f"<{len(out_tokens)}H", *out_tokens)

def post_process(buf):
    # Decode u16 -> JS source bytes for the target.
    tokens = [t for (t,) in struct.iter_unpack("<H", bytes(buf))]
    src = self.model.tokenizer._hf.decode(tokens, skip_special_tokens=True)
    return src.encode("utf-8")
```

Two observations:

- The in-process `filename → token_ids` cache pattern (the central
  trick of Option A from autotokens) is **not needed** under Option B.
  `buf` is the canonical token sequence already — no cache to keep in
  sync.
- The logit-mask LLM constraint of Option A is also not needed. The
  full BPE vocab is legal here; partial-multibyte tokens are fine
  because the only byte conversion is the one-shot `post_process`
  whose output goes to `.cur_input` and is *never* re-tokenized.

These two simplifications are exactly what made CovRL's implementation
manageable in the first place. Option B inherits them.

---

## 6. Caveats — real but small

Listed honestly. None of these are deal-breakers.

### `afl-cmin` / `afl-tmin` / `afl-showmap` need a decode shim

These tools run the target directly on a queue file. With Option B,
the file is u16 binary and the target won't parse it. Workarounds:

- **Decode wrapper script**: `decode_u16.py file_in file_out` calls
  the tokenizer once; feed the decoded file to the AFL++ tool. Trivial.
- **Direct bitmap reads**: `covrl/models/rewarding.py:75–152` already
  shows the pattern — decode to a tmp dir, run `afl-showmap` over the
  decoded files. Same pattern works for us when we wire up the
  reward subsystem.
- **Avoidance**: cmin/tmin are not in our critical path. We can defer
  using them on Option B queues until needed.

### Crash files in `out/crashes/` are u16 binary

To reproduce a crash, decode first. Same decode shim. Document this in
the run output so future-us doesn't try to `cat` a crash file and get
confused.

### Queues are not portable across tokenizer revisions

If we upgrade CodeT5+ or change the tokenizer entirely, existing queue
files become unreadable. This is the same caveat CovRL has. Mitigation:
write a `tokenizer_id` marker into the run's `out/` dir and refuse to
resume a queue under a different tokenizer.

### Seeds must be pre-processed to u16 binary

A one-time `data/preprocess.py` writes `<u16-binary>` files into the
`-i` directory, mirroring `~/Documents/CovRL-Fuzz/covrl/utils/preprocess.py`.
This step is already on our roadmap regardless of A vs B (the doc's
phase 3).

### `fuzz()` must return `bytearray`, not `bytes`

AFL++'s `py_bytes()` in `src/afl-fuzz-python.c` claims to accept "bytearray
or bytes" (the FATAL message at line 129 even says so), but the bytes path
crashes in `memcpy()` at line 138 with a corrupted source pointer (verified
via gdb at session time: `rsi` ends up holding the QWORD at offset 8 of the
payload rather than its address). The official example mutator
(`custom_mutators/examples/example.py`) returns `bytearray`, which works.
Our `Mutator.fuzz()` returns `bytearray(self._pending_outputs.pop(0))`
accordingly. Worth filing upstream — but not blocking.

### MASK_TOKEN ID choice

CovRL uses `MASK_TOKEN = 4` (a low number). For us this is unsafe
because token ID 4 in CodeT5+ has a real meaning (`</s>` or similar
special token, depending on the tokenizer revision). The correct
choice is one of the `<extra_id_N>` sentinels at the top of the vocab,
which is also where CovRL's Python side maps MASK_TOKEN to
(`~/Documents/CovRL-Fuzz/covrl/models/inferencer.py:_mask_unknowns`
remaps `4` to `vocab_size - mask_idx`). We can either:

- Use `<extra_id_99>` (or any reserved sentinel) directly as our
  MASK_TOKEN — cleaner.
- Mirror CovRL's MASK_TOKEN=4 then remap inside the mutator —
  faithful to CovRL's wire format but adds remapping.

The first option is what we'd recommend if we make Option B the
default.

---

## 7. Comparison to Option A on the invariant axis

Re-stated against the now-verified mechanism for B:

- **Option A** (autotokens-inspired): the queue-as-tokens invariant
  rests on a **contract** — the logit-mask must filter every partial
  multibyte BPE token, on every generation path, forever. If a future
  refactor adds a generation path that skips the filter (e.g., a
  research branch that uses a different sampling helper), drift
  returns silently. The invariant is contractual.
- **Option B** (verified viable): the queue-as-tokens invariant is
  **structural** — AFL never tokenizes the queue bytes because tokens
  *are* what the bytes already are. There is no round-trip. No future
  refactor can re-introduce the drift class because there is no
  decoding step that could be misconfigured.

This was the reason `aligning_with_covrl.md` §3.2 hedged ("Direct
invariant: no round-trip contract to maintain") — but the doc still
defaulted to A because the cost side looked higher than it is. With
the AFL++ extension point now identified explicitly, Option B's cost
side is:

- 1 env var (`AFL_POST_PROCESS_KEEP_ORIGINAL=1`).
- 1 `post_process` function (the ~5-line decode pseudocode in §5).
- 1 pre-processing script for seeds (in scope regardless).
- Decode wrapper for `afl-*` tools if/when we need them.

That is small. The doc's recommendation of A as default was made under
a more pessimistic reading; that reading is no longer accurate.

---

## 8. Conclusion

Option B is viable. The mechanism is documented AFL++ behavior the
maintainers designed for exactly this use case. CovRL-Fuzz's
queue-as-tokens architecture maps cleanly onto AFL++ via
`AFL_POST_PROCESS_KEEP_ORIGINAL=1` plus the standard custom-mutator
hooks. There are no AFL++ source modifications required and no
behavior we'd need to fake; every load-bearing property of the
reference implementation has an AFL++ extension point.

The remaining choice between Option A and Option B is now a real
trade-off (tooling friendliness vs invariant strength), not a
viability gate. The choice itself belongs in
[`aligning_with_covrl.md`](aligning_with_covrl.md) §3.3, which can be
updated based on this finding when the user decides.

Implementation planning — when, in what order, against which seeds,
with what MASK_TOKEN choice — is deliberately out of scope here. This
doc only answers "can it be done." Yes.

---

## References

- `~/Documents/AFLplusplus/src/afl-fuzz-run.c:134, 145–167, 194–211,
  237, 241–251, 528–530, 583, 713–714`.
- `~/Documents/AFLplusplus/src/afl-fuzz-bitmap.c:752`.
- `~/Documents/AFLplusplus/src/afl-fuzz-one.c:2115–2123` (custom_only
  abandon).
- `~/Documents/AFLplusplus/src/afl-fuzz-init.c:read_testcases` and
  `src/afl-fuzz-queue.c:746–747` (opaque seed ingestion).
- `~/Documents/AFLplusplus/src/afl-fuzz-state.c:515–519` and
  `src/afl-fuzz.c:453` (env var parsing and help text for
  `AFL_POST_PROCESS_KEEP_ORIGINAL`).
- `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:337, 3229, 5008, 5070,
  5095–5099, 5193–5221, 5247–5252, 5305+`.
- `~/Documents/CovRL-Fuzz/covrl/models/inferencer.py` —
  `_mask_unknowns` sentinel remap.
- Sibling docs: [`aligning_with_covrl.md`](aligning_with_covrl.md)
  (broader architecture), [`autotokens_losslessness.md`](autotokens_losslessness.md)
  (why Option A's invariant is contractual rather than structural).
