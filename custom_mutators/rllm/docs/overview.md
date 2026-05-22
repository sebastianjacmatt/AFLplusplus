# rllm — Token-Level LLM Mutation Fuzzing (overview)

A Python custom mutator for AFL++ that fuzzes JavaScript interpreters with
LLM-driven mask-fill mutation over a Token-Level-AFL-style queue. The
design follows CovRL-Fuzz's data + model architecture but **omits the RL
training loop** — the model runs SFT-free (pretrained CodeT5+, no
fine-tuning). This doc covers the parts that are implemented and working;
the future training subsystem (`data/rewarding.py` + `data/rollout.py`,
both currently empty placeholders) is intentionally out of scope here.
Design rationale lives in [`aligning_with_covrl.md`](aligning_with_covrl.md);
this doc is the operational overview.

## What's in this doc

- The big-picture data flow.
- The queue-as-tokens invariant and how it is achieved on stock AFL++.
- The `data/` layer (seed preprocessing, decode inspection helper).
- The `model/` layer (CodeT5+ tokenizer + sentinel masking + Seq2seq
  generate).
- The masking strategy at fuzz time (CovRL's `RANDOM_OVERWRITE` shape).
- How it all wires into AFL++ via custom-mutator hooks.
- Configuration knobs and where they live.

What's **not** here: RL training (reward shape, PPO loss, training
cadence) — see future plans.

---

## 1. Big picture

```
                                ┌─────────────────────────────────────┐
                                │  seed JS files (.js)                │
                                │  ~/Documents/data_store/dataset/    │
                                │     final-dataset-dec22/            │
                                └──────────────┬──────────────────────┘
                                               │
                                               ▼   (one-shot)
                                ┌─────────────────────────────────────┐
                                │  data/preprocess.py                 │
                                │  uglifyjs -m -b    ──►   tokenize   │
                                │  tokenize     ──►   encode as u16   │
                                └──────────────┬──────────────────────┘
                                               ▼
                                ┌─────────────────────────────────────┐
                                │  u16 binary seeds (-i dir)          │
                                │  data_store/dataset/                │
                                │     final-dataset-dec22-u16/        │
                                └──────────────┬──────────────────────┘
                                               │
              ┌────────────────────────────────┴───────────────────────────────┐
              │                                                                │
              │             ┌────────────────────────────────────┐             │
              │             │             AFL++                  │             │
              │             │                                    │             │
              │  ┌──────────▼──────────┐         ┌───────────────▼──────────┐  │
              │  │ queue/id_*  (u16)   │         │  .cur_input  (JS source) │  │
              │  └──────────┬──────────┘         └───────────────▲──────────┘  │
              │             │                                    │             │
              │             │ queue_get / fuzz_count             │ post_process│
              │             ▼                                    │ (decode)    │
              │  ┌──────────────────────────────────────────────┴──────────┐  │
              │  │                       rllm (Python)                     │  │
              │  │                                                         │  │
              │  │   ┌───────────────────────┐    ┌──────────────────────┐ │  │
              │  │   │  parse_u16 ──► tokens │    │  tokens ──► detok    │ │  │
              │  │   └───────────┬───────────┘    └──────────▲───────────┘ │  │
              │  │               │                           │             │  │
              │  │               ▼                           │             │  │
              │  │   ┌───────────────────────┐               │             │  │
              │  │   │  data/masking.Masking │               │             │  │
              │  │   │  (sentinel insertion) │               │             │  │
              │  │   └───────────┬───────────┘               │             │  │
              │  │               ▼                           │             │  │
              │  │   ┌───────────────────────┐               │             │  │
              │  │   │  model/llm.Model      │               │             │  │
              │  │   │  CodeT5+ batched      │               │             │  │
              │  │   │  generate (T5 MSP)    │               │             │  │
              │  │   └───────────┬───────────┘               │             │  │
              │  │               ▼                           │             │  │
              │  │   ┌───────────────────────┐               │             │  │
              │  │   │  reconstruct_tokens   │               │             │  │
              │  │   │  ──► encode_u16       │               │             │  │
              │  │   └───────────┬───────────┘               │             │  │
              │  │               │                           │             │  │
              │  │     ┌─────────▼─────────────┐             │             │  │
              │  │     │ fuzz() returns u16 bin│─────────────┘             │  │
              │  │     │ (bytearray)           │                           │  │
              │  │     └───────────────────────┘                           │  │
              │  └────────────────────────────────────────────────────────-┘  │
              │                                                                │
              │  AFL_POST_PROCESS_KEEP_ORIGINAL=1 ⇒ queue keeps u16; only      │
              │  .cur_input gets decoded JS                                    │
              │                                                                │
              └────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────────────┐
                                │  target (jerryscript) runs JS       │
                                │  AFL captures coverage              │
                                │  exit_hook.so writes stderr file    │
                                └──────────────┬──────────────────────┘
                                               │
                                               ▼
                                ┌─────────────────────────────────────┐
                                │  rllm post_run                      │
                                │  classify(stderr) → valid/syn/sem   │
                                │  update _Stats                      │
                                └─────────────────────────────────────┘
```

---

## 2. The queue-as-tokens invariant

The single most important architectural property:

> The persistent fuzzing queue holds tokens (u16 binary), never source
> bytes. Tokens are decoded to JS source only once per execution, when
> AFL writes `.cur_input` for the target.

This is the property TLAFL and CovRL both maintain by storing u16 token
arrays directly to disk. On AFL++ we get the same property through a
documented extension point — `AFL_POST_PROCESS_KEEP_ORIGINAL=1` — which
makes `save_if_interesting` preserve the mutator's `fuzz()` output (our
u16 binary) instead of the decoded `post_process` output. See
`src/afl-fuzz-run.c:241-251` and the full mechanism breakdown in
[`option_b_viability.md`](option_b_viability.md).

Why this matters: without the invariant, AFL's queue files would be
decoded JS, and the next iteration would re-tokenize bytes with the BPE,
introducing drift at byte-token boundaries. With the invariant, the LLM
operates on the exact token sequence it (or a prior call) produced — no
re-tokenization drift, no U+FFFD class of artifacts.

---

## 3. The `data/` layer

Three files; only two have code today. The third (`rewarding.py`) is a
placeholder for the future training subsystem.

### `data/preprocess.py` — seed pipeline

Run once per seed corpus. Converts JS files into u16 binary files that
AFL ingests as the `-i` directory.

Per-file pipeline:
1. Read the raw JS source.
2. Pipe through `uglifyjs --mangle --beautify` (`-m -b`). Mangling
   renames identifiers to short consistent names; beautify normalizes
   whitespace. Mirrors CovRL paper §4.
3. Tokenize the uglified source with `Tokenizer.tokenize` (the standard
   BPE encode).
4. Truncate to `--max-tokens` (default 1024, matches CodeT5+'s
   pretraining source-sequence ceiling).
5. Write the token-IDs as little-endian uint16 to
   `<output_dir>/<basename>`.

A marker file is written **alongside** (not inside) the output dir as
`<output_dir>.tokenizer.json`. It records `tokenizer`, `vocab_size`,
`max_tokens`, and per-run stats. `run_rllm.sh` checks this marker to
detect "already preprocessed" and skip the rebuild. The marker is
deliberately *outside* the dir because AFL ingests every file under `-i`
as a seed — leaving the marker inside fed garbage to `fuzz_count`.

### `data/decode.py` — inspection shim

CLI helper that does the reverse of preprocess for one file:

```
python -m data.decode <u16_file>              # JS source to stdout
python -m data.decode <u16_file> -o <out>     # JS source to a file
```

Used when:
- Inspecting AFL queue / crash / hang files (which are u16 binary).
- Wrapping `afl-cmin`/`afl-tmin`/`afl-showmap` for queues that are u16.
- Debugging mutations post-hoc.

### `data/masking.py` — span / single-token masking

The masker is shared between fuzz time and (future) training time; only
its configuration differs.

`Masking.mask(token_ids) -> MaskedProgram` produces:
- `original_ids`: the input tokens.
- `input_ids`: the sentinelized encoder input (masked spans replaced by
  sentinel IDs like `<extra_id_0>`, `<extra_id_1>`, ...).
- `spans`: list of `MaskedSpan(start, end, sentinel_id)` so the
  reconstruct step can splice the model's output back into the right
  positions.

The masker is tokenizer-agnostic. The sentinel IDs are injected at
construction time (`model.tokenizer.sentinel_ids`); the masker never
imports or calls the HF tokenizer directly. It is also generator-agnostic
— it just decorates a `list[int]` with span metadata.

Two parameter presets sit in `configs/`:

| Preset | `corruption_rate` | `min/max/mean_span` | Effect |
|---|---|---|---|
| `default.json` (fuzz-time) | 0.03 | 1 / 1 / 1.0 | ≈1–3 single-token sentinels per mutation. Functionally identical to CovRL's `RANDOM_OVERWRITE` (`afl-fuzz.c:5211-5221`). |
| (future) training preset | 0.15 | 1 / 5 / 3.0 | T5-style span corruption. Matches CodeT5+ MSP pretraining and CovRL's `actor_dataset.py`. |

Both presets run through the *same* code path in `Masking.mask`. The
single-token shape is just degenerate-parameter span corruption; we
didn't write two maskers. See [`aligning_with_covrl.md`](aligning_with_covrl.md)
§4 for why this is one machinery, two knobs.

---

## 4. The `model/` layer

Two files: `tokenizer.py` wraps the HF tokenizer; `llm.py` wraps the
Seq2seq model with batched generation.

### `model/tokenizer.py`

Encapsulates everything tokenizer-specific so the rest of the code can
operate on `list[int]` without touching HF.

Methods used by the hot path:

| Method | Direction | Notes |
|---|---|---|
| `parse_u16(buf) -> list[int]` | u16 binary → token IDs | Hot path: queue file → token list (no string roundtrip). |
| `encode_u16(token_ids) -> bytes` | token IDs → u16 binary | Hot path: model output → queue-ready bytes. |
| `reconstruct_tokens(masked_program, generated_ids) -> list[int]` | sentinel splice | Splices the model's `<extra_id_N>`-delimited target back into the encoder input, returning the spliced token list (no detokenize). |
| `detokenize(token_ids) -> bytes` | token IDs → UTF-8 source | Called once per execution in `post_process`. Strips any U+FFFD bytes that HF's `decode(errors="replace")` may introduce at BPE byte boundaries (no longer load-bearing — the queue is u16, so any U+FFFD bytes affect only `.cur_input`, not future iterations). |
| `tokenize(buf) -> list[int]` | UTF-8 bytes → token IDs | Used by `data/preprocess.py` only. Not on the hot path. |

Sentinel IDs (`<extra_id_0>` ... `<extra_id_99>`) are resolved at
construction and exposed via `sentinel_ids`. The masker uses the first N
for the N spans it produces; the last sentinel is reserved as the T5
target-sequence terminator (consumed inside `reconstruct_tokens` via
`_parse_generated_spans`).

The vocab fits in u16 with headroom: CodeT5+'s 32100 tokens + 100
sentinels + special tokens (EOS, PAD, BOS, UNK) is well under 65535.

### `model/llm.py`

Thin wrapper around an HF `AutoModelForSeq2SeqLM`. One method matters:

```python
@torch.no_grad()
def batch_generate(self, input_ids_list, n_samples=1, max_new_tokens=None) -> list[list[int]]
```

It:
1. Right-pads `input_ids_list` to a single tensor.
2. Builds the attention mask.
3. Calls `model.generate(...)` once for the whole batch.
4. Returns a list of `len(input_ids_list) * n_samples` token sequences,
   each including HF's leading decoder-start token (which
   `reconstruct_tokens` strips by skipping `pos 0`).

Generation kwargs are passed in at construction time, chosen by
`rllm._build_mutator` based on `cfg.sampling_method`:

- `contrastive` (default, matches CovRL §4): `do_sample=False`,
  `penalty_alpha=0.6`, `top_k=4`, `no_repeat_ngram_size=3`.
- `nucleus`: `do_sample=True`, `temperature=1.0`, `top_p=0.95`,
  `top_k=50`.

The model is loaded once at `init()` and stays in memory for the
lifetime of the run. CodeT5+ 220M (`Salesforce/codet5p-220m`) by
default; configurable via `configs/*.json`.

`batch_generate` is the only place the model touches token IDs; the
encoder consumes `input_ids` (sentinelized), the decoder emits a
T5-style span-delimited target. Everything else in `model/` is plumbing.

---

## 5. Masking strategy at fuzz time

We use CovRL's `RANDOM_OVERWRITE` shape: 1–3 single-token sentinels per
mutation, no multi-token deletions. Concretely, with
`corruption_rate=0.03` and `min=max=mean_span_length=1`, a 100-token
program gets ~3 single-token sentinels placed at random positions; the
model then fills each independently.

Why this shape:
- Higher per-mutation P(valid). Compared to multi-token span corruption,
  filling one token at a time preserves the bracket/comma scaffolding
  around the mask.
- Matches CovRL's published configuration at `afl-fuzz.c:5211-5221`
  (their `MASK_COUNT=3`).
- Same masking code as the training-time T5-span variant; only the
  parameters differ.

`whole_word_masking` is **off** at fuzz time. With BPE, "whole-word"
groups multi-token identifiers and dotted access chains
(`obj.method().chain`) into a single mask unit, which is too coarse for
the 1–3 mask budget. Token-level masking is the right granularity.

INSERT (sentinel placement without removing a token, growing the buffer)
and SPLICE (statement-level cross-input mixing) are valid CovRL
operations that we have not yet added; both are clean extensions of the
same masker. See `aligning_with_covrl.md` §4.1.

---

## 6. The mutator: AFL integration

The AFL++ Python custom-mutator hooks all live in `rllm.py` as a thin
shim that delegates to a single `Mutator` instance. The `Mutator` class
(in `mutator.py`) owns the hot-path logic.

| Hook | Calls | What it does |
|---|---|---|
| `init(seed)` | `_build_mutator` | Loads config, constructs Tokenizer/Model/Masking, creates the `Mutator`. |
| `queue_get(filename)` | `Mutator.queue_get` | Returns True. Tracks the `finetune_every` cadence (for future training); under Option B we don't need an invalid-filter — the queue can't drift. |
| `fuzz_count(buf)` | `Mutator.fuzz_count` | `buf` is u16 binary. Parse to tokens; create `cfg.fuzz_count` masked programs; batched `model.batch_generate`; per output, `reconstruct_tokens → encode_u16`; cache in `_pending_outputs`; return its length. |
| `fuzz(buf, add_buf, max_size)` | `Mutator.fuzz` | Pop and return the next u16 binary from `_pending_outputs`. **Returns `bytearray`** (see §8). |
| `post_process(buf)` | `Mutator.post_process` | `parse_u16(buf) → detokenize` — produces JS source bytes for `.cur_input`. Runs once per execution. |
| `queue_new_entry(new, orig)` | `Mutator.queue_new_entry` | Increment finds counter for stats. |
| `post_run()` | `Mutator.post_run` | Reads the stderr file `exit_hook.so` wrote, classifies as syntax / semantic / valid via `_classify_stderr`, updates `_Stats`. Only counted on fresh LLM mutations (`_last_run_was_mutation` gate excludes calibration/trim replays). |
| `splice_optout()` | (presence only) | Disables AFL's byte-level splice. We will implement our own SPLICE later. |
| `deinit()` | `Mutator.deinit` | Flushes stats; saves model checkpoint (no-op until training lands). |

The `_pending_outputs` cache holds u16 `bytes` objects between
`fuzz_count` and the subsequent `fuzz()` calls — one entry per mutation.
`fuzz()` pops one per call.

The `_Stats` object writes a one-line snapshot to `<out>/rllm_stats.txt`
every ~2s, plus to stderr when `AFL_NO_UI=1`. Format: time, seeds
fuzzed, mutations produced, queue finds, per-mutation validity rates,
mean generation time. Useful for live monitoring (`tail -f <out>/rllm_stats.txt`).

---

## 7. AFL environment configuration

The fuzzer is invoked via `run_rllm.sh`. The load-bearing env vars:

| Env var | Value | Purpose |
|---|---|---|
| `AFL_PYTHON_MODULE` | `rllm` | Selects our Python mutator. |
| `AFL_CUSTOM_MUTATOR_ONLY` | `1` | Disables AFL's havoc + deterministic stages. Only our `fuzz()` produces mutations. |
| `AFL_DISABLE_TRIM` | `1` | Skips `trim_case` (which would mangle u16 buffers byte-wise). |
| `AFL_NO_FASTRESUME` | `1` | Forces fresh queue ingestion on restart (avoids mixing pre/post-fix entries). |
| `AFL_POST_PROCESS_KEEP_ORIGINAL` | `1` | **Load-bearing for the queue invariant.** Tells AFL to keep `fuzz()`'s output in the queue rather than overwriting it with `post_process`'s decoded bytes. Without it, the queue would store JS source and the architecture collapses. |
| `AFL_FRAMESHIFT_DISABLE` | `1` | Skips the FrameShift `fs_sanitize` pass (which is a byte-level transform that doesn't understand u16). |
| `AFL_PRELOAD` | `exit_hook.so` | LD_PRELOAD into the target. Redirects target stderr to `RLM_STDERR_FILE` so `post_run` can classify runs. |
| `RLM_STDERR_FILE` | `<out>/rllm_stderr.txt` | Where `exit_hook.so` writes the target's stderr each run. |
| `RLLM_CONFIG` | `configs/default.json` | Selects the mutator config. |

`run_rllm.sh` also runs the preprocessing bootstrap once before invoking
`afl-fuzz`: it checks for the `.tokenizer.json` marker next to the
preprocessed dir, and if missing, runs `python -m data.preprocess` to
build the `-u16` seed dir from the raw JS corpus.

---

## 8. Two small gotchas worth knowing

### Marker file lives outside the seed dir

AFL ingests every file under `-i` (including dotfiles). The
`<output_dir>.tokenizer.json` marker is therefore written *alongside*
the output dir, not inside it. Putting it inside makes AFL pick it up
as a seed; `fuzz_count` would parse JSON-as-u16-tokens and produce
garbage.

### `fuzz()` must return `bytearray`, not `bytes`

AFL++'s `py_bytes` helper at `src/afl-fuzz-python.c:32-59` advertises
support for both `bytes` and `bytearray`, but the `bytes` path is
broken on the current tree: the C-side `bytes` pointer ends up
corrupted after `PyBytes_AsString` and the subsequent `memcpy()` at
`afl-fuzz-python.c:138` segfaults. We confirmed this under gdb; the
official example mutator at `custom_mutators/examples/example.py` also
returns `bytearray`. Our `Mutator.fuzz()` wraps its output in
`bytearray(...)` accordingly. The comment in `mutator.py:fuzz()`
flags this. Worth reporting upstream, not blocking.

---

## 9. What's deliberately not here

This doc covers the SFT-free baseline — what CovRL Table 7 calls "LLM
w/o CovRL." The training subsystem that takes it from that baseline to
CovRL's full result (their ~41% Jerry validity) is not implemented yet:

- `data/rewarding.py` is an empty stub. The plan is to mirror
  `~/Documents/CovRL-Fuzz/covrl/models/rewarding.py`: read AFL coverage
  via `afl-showmap` on the decoded output, classify validity from
  `RLM_STDERR_FILE`, compute TF-IDF weighted rewards with α=0.6 IDF
  momentum, output `-1.0 / -0.5 / sigmoid(TF-IDF · bitmap)`.
- `data/rollout.py` is an empty stub. The plan is the PPO loop mirroring
  `~/Documents/CovRL-Fuzz/covrl/models/finetuner.py`: two model copies
  (current actor + frozen previous), per-token PPO with clip
  `[0.8, 1.2]`, KL regularization, hot-swap of the trained checkpoint
  back into the live mutator.

Those land in a separate plan. The data + model layers described above
are designed so the training subsystem plugs in as a `trainer` argument
to `Mutator(...)` — the `_maybe_finetune` cadence and the model load
points are already there, just connected to `None`.

---

## 10. Where to look next

- [`aligning_with_covrl.md`](aligning_with_covrl.md) — design rationale,
  why Option B over Option A, why CovRL's masking split.
- [`option_b_viability.md`](option_b_viability.md) — the AFL++
  mechanism (`AFL_POST_PROCESS_KEEP_ORIGINAL`) that makes the u16 queue
  work on stock AFL++.
- [`autotokens_losslessness.md`](autotokens_losslessness.md) — adjacent
  comparison with AFL++'s built-in `autotokens` custom mutator and what
  "lossless" actually means in this context.
- `~/Documents/CovRL-Fuzz/` — the reference implementation. Their
  `AFL/afl-fuzz.c` is the original token-level architecture (forked
  from AFL 2.52b); their `covrl/models/` is the training pipeline we
  haven't built yet.
- *Token-Level Fuzzing*, Salls et al., USENIX Security '21 — the
  underlying TLAFL idea.
- *Fuzzing JavaScript Interpreters with Coverage-Guided Reinforcement
  Learning for LLM-Based Mutation*, Eom et al., ISSTA '24 — CovRL-Fuzz.
