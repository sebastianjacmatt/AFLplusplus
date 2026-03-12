# CovRL-Fuzz: AFL++ Port Design

Port of CovRL-Fuzz (AFL 2.52b) to the AFL++ custom mutator API.

---

## How the original CovRL works (AFL 2.52b)

Understanding the original is essential before porting.

### Data representation

The original CovRL stores **token IDs, not bytes**, in the queue.
Every corpus entry on disk is a `u16[]` — a sequence of vocabulary token IDs.
Source bytes only appear transiently, decoded from tokens immediately before
the target is executed, and discarded after.

```
Queue on disk:   [ tok_id, tok_id, tok_id, ... ]  ← u16[] stored in file
                        ↓  decode()
Target receives: [ raw bytes ]                     ← never persisted
```

When an interesting input is found, `encoded_buf` (the token sequence) is saved,
not `decoded_tokens`. The corpus is always token-valid by construction.

### Mutation (fuzz_one)

The mutation itself is extremely simple — just mask insertion:

- **RANDOM_INSERT**: pick 1–3 random positions, insert `MASK_TOKEN=4` at each,
  lengthening the sequence.
- **RANDOM_OVERWRITE**: pick 1–3 random positions, set them to `MASK_TOKEN=4`,
  same length.

After masking, the masked token sequence is sent to the model server, which
fills the masks and returns the completed token sequence. No AFL-style bitflips
or arithmetic are used.

### Model server (socket architecture)

The fuzzer and model are **separate processes** communicating over TCP.
The C fuzzer is a client; the Python model server is a persistent daemon.

Protocol (all commands are raw strings, responses are `u16` length values):

| Command      | Fuzzer writes                   | Server does                        | Server replies  |
|--------------|---------------------------------|------------------------------------|-----------------|
| `"predict"`  | masked token seq → `MLM_pred`   | fills masks, writes result back    | new length u16  |
| `"decode"`   | token seq → `MLM_decoded`       | decodes tokens → bytes, writes back| byte length u16 |
| `"finetune"` | nothing                         | trains on new corpus entries       | ack u16         |

The server maintains the model in memory across all calls, enabling RL state
to accumulate across the entire fuzzing session.

### Reinforcement learning loop

At the start of every `sync_fuzzers()` call (every `SYNC_INTERVAL` iterations),
the fuzzer sends `"finetune"`. The server fine-tunes the model on corpus entries
that found new coverage. Coverage = reward signal; new queue entries = training
data. This is the entire RL loop.

---

## The fundamental porting problem

AFL++ always stores queue entries as **bytes**. CovRL's design relies on the
queue storing token IDs. This must be resolved before anything else.

There are two valid approaches:

### Option A — Token IDs as the stored bytes (faithful to original)

The AFL++ queue stores packed `u16` token IDs as the "bytes" of each entry.
`fuzz()` receives and returns token IDs (as raw bytes).
`post_process()` decodes token IDs → real source bytes immediately before
execution — exactly what `decode()` did inside `common_fuzz_stuff()` in the
original.

```
AFL++ queue:   [ u16 tok, u16 tok, ... ]   ← what fuzz() works on
                        ↓  post_process()
Target input:  [ raw source bytes ]        ← decoded only at execution time
```

Advantages:
- All corpus entries remain token-valid by construction (same as original)
- `fuzz()` works directly on token sequences; no tokenization needed per call
- `init_trim()` / `trim()` can remove tokens naturally (no boundary problem)

Disadvantages:
- Initial seed corpus must be pre-tokenized before first run
- AFL++ tools (afl-tmin, afl-cmin) will see token byte arrays, not source

### Option B — Bytes in queue, tokenize per call

The queue stores real source bytes. `fuzz()` receives bytes, tokenizes them,
masks, sends to server, detokenizes, returns bytes. `post_process()` is not
needed.

Advantages:
- Seeds can be any source files; no pre-tokenization step
- AFL++ tools work normally on the corpus

Disadvantages:
- Tokenization happens on every `fuzz()` call (minor overhead in Python)
- Corpus entries are not guaranteed to be token-valid after AFL++'s own
  mutations run alongside this mutator (if `AFL_CUSTOM_MUTATOR_ONLY` is not set)

**Recommendation**: Option A is more faithful to the original and enables
correct token-level trimming. Option B is simpler to get running initially.
Start with B, migrate to A once the server and loop are validated.

---

## AFL++ API mapping

| CovRL (2.52b)                    | AFL++ custom mutator equivalent        |
|----------------------------------|----------------------------------------|
| Startup socket connection        | `init(seed)` — connect to server here  |
| `decode()` before execution      | `post_process(buf)` — decode tok→bytes |
| Mask insertion in `fuzz_one()`   | `fuzz(buf, add_buf, max_size)`         |
| `"predict"` socket call          | inside `fuzz()` or `fuzz_count()`      |
| `sync_fuzzers()` → `"finetune"`  | `queue_new_entry(new_fn, orig_fn)`     |
| Token-level trim (not in orig.)  | `init_trim()` / `trim()` / `post_trim()`|
| Corpus stored as u16 token IDs   | Option A: `post_process()` decodes     |

---

## Component-by-component implementation plan

### `init(seed)`

- Seed the RNG
- Load env vars (`COVRL_PORT`, `COVRL_MODEL`, `COVRL_BATCH_SIZE`, etc.)
- Connect to the model server via TCP socket
- Send a handshake and verify the server is ready
- Store the socket in a global for use by all other functions

The model server must already be running before AFL++ starts.

### `post_process(buf)` — only needed for Option A

Receives a `u16[]` token sequence (as raw bytes).
Sends `"decode"` to the server.
Returns the decoded source bytes that the target will actually receive.
This function must be fast — it is called on every single execution.

For Option B this function is not needed.

### `fuzz_count(buf)` — optional but useful

Called once per queue entry before `fuzz()` is called.
Use this to run a single `"predict"` call that generates `BATCH_SIZE` masked
variants, caching all results. Return `BATCH_SIZE` (or however many were
successfully generated). `fuzz()` then drains the cache one result at a time.

This amortises the socket round-trip cost across multiple `fuzz()` calls.
If not implemented, each `fuzz()` call makes its own `"predict"` request.

See `custom_mutators/symcc/symcc.c` lines 239–310 for the canonical example of
this pattern: `fuzz_count` scans pre-generated files and returns the count;
`fuzz` reads them one at a time.

### `fuzz(buf, add_buf, max_size)`

Core mutation. Two strategies matching the original:

**RANDOM_OVERWRITE** (equivalent to original's `RANDOM_OVERWRITE`):
- Tokenize `buf` (Option B) or treat `buf` as token bytes (Option A)
- Pick 1–`MASK_COUNT` random token positions
- Set each to `MASK_TOKEN`
- Send `"predict"` to server with the masked sequence
- Receive the completed token sequence
- Detokenize and return bytes (Option B), or return token bytes (Option A)

**RANDOM_INSERT** (equivalent to original's `RANDOM_INSERT`):
- Same as above but insert `MASK_TOKEN` at random positions rather than
  overwriting, increasing sequence length

If using `fuzz_count()` for batching, this function just pops from the cache.

### `queue_new_entry(filename_new_queue, filename_orig_queue)`

Called every time a new coverage-finding input is added to the queue.
This is where the RL feedback loop lives.

Two reasonable strategies:
1. **Immediate**: send `"finetune"` to the server on every new entry
2. **Deferred**: increment a counter; send `"finetune"` every N new entries

The original sends `"finetune"` at the start of every sync interval, which
is roughly every few hundred executions. Strategy 2 with a configurable N
(`COVRL_FINETUNE_INTERVAL`, default ~50) is a reasonable match.

The server receives `"finetune"` and trains on the new entry (or a batch of
recent entries). The training data is the token sequences of coverage-finding
inputs; the reward signal is implicit (they found coverage, so they're good).

### `init_trim(buf)` / `trim()` / `post_trim()`

AFL++'s default trimmer bisects at arbitrary byte offsets. For Option A
(token IDs in queue), a byte-level bisection will split a `u16` token in half,
corrupting all subsequent tokens. Custom trimming is **required** for Option A.

For Option B it is optional but beneficial — byte-level trimming of source
code will produce syntactically broken inputs that the model can't meaningfully
learn from.

**Token-level trim algorithm**:
- `init_trim(buf)`: tokenize `buf`, store token list globally, return `len(tokens)`
- `trim()`: reconstruct the sequence with token at `_trim_index` removed,
  detokenize, return the result bytes
- `post_trim(success)`:
  - If `success=True`: the token is permanently removed; shrink `_trim_tokens`,
    do NOT advance index (next token has slid into this position)
  - If `success=False`: advance `_trim_index` past this token and try next
  - Return `_trim_index`; return `len(_trim_tokens)` to signal done

---

## Model server contract

The model server is a separate Python process. It must implement:

| Command      | Input (file)          | Output (file)         | Reply       |
|--------------|-----------------------|-----------------------|-------------|
| `"predict"`  | `MLM_pred` (u16[])    | `MLM_pred` (u16[])    | new_len u16 |
| `"decode"`   | `MLM_decoded` (u16[]) | `MLM_decoded` (bytes) | byte_len u16|
| `"finetune"` | —                     | —                     | ack u16     |

The server must handle requests sequentially (the fuzzer is single-threaded
per instance). For multi-instance parallel fuzzing, run one server per fuzzer
instance on separate ports.

For CodeT5 specifically:
- `"predict"`: read masked token IDs, run `model.generate()` with
  `do_sample=True`, extract fill tokens between `<extra_id_0>` and
  `<extra_id_1>`, reconstruct full sequence, write back
- `"decode"`: call `tokenizer.decode(token_ids, skip_special_tokens=True)`,
  encode result as UTF-8, write back
- `"finetune"`: load the new corpus entry, run a fine-tuning step with the
  coverage signal as implicit reward

---

## Environment variables

Follow the `autotokens` convention of making all tunable parameters
overridable via env vars without modifying the source:

| Variable                  | Default                      | Purpose                              |
|---------------------------|------------------------------|--------------------------------------|
| `COVRL_PORT`              | `9999`                       | Model server TCP port                |
| `COVRL_MODEL`             | `Salesforce/codet5-base`     | HuggingFace model name or local path |
| `COVRL_BATCH_SIZE`        | `8`                          | Mutations to generate per fuzz_count |
| `COVRL_MAX_TOKENS`        | `512`                        | Max input token length               |
| `COVRL_MAX_NEW`           | `64`                         | Max new tokens from model.generate() |
| `COVRL_MASK_MIN`          | `0.10`                       | Min fraction of tokens to mask       |
| `COVRL_MASK_MAX`          | `0.30`                       | Max fraction of tokens to mask       |
| `COVRL_MASK_COUNT`        | `3`                          | Max number of masked spans (orig: 3) |
| `COVRL_TEMP`              | `0.8`                        | Sampling temperature                 |
| `COVRL_FINETUNE_INTERVAL` | `50`                         | New entries between finetune calls   |
| `COVRL_QUEUE_REPR`        | `bytes`                      | `bytes` (Option B) or `tokens` (A)   |

---

## Running

```bash
# 1. Start the model server (must be running before afl-fuzz)
python covrl_server.py --port 9999 --model Salesforce/codet5-base

# 2. Run AFL++ with the mutator
AFL_CUSTOM_MUTATOR_LIBRARY=covrl_mutator.py \
AFL_CUSTOM_MUTATOR_ONLY=1 \
COVRL_PORT=9999 \
afl-fuzz -i seeds/ -o out/ -- ./target @@
```

`AFL_CUSTOM_MUTATOR_ONLY=1` disables AFL++'s own havoc mutations so the LLM
is the sole mutation source, matching the original CovRL design. Remove it
to run LLM mutations alongside AFL++'s standard mutations.

---

## Key differences from the original

| Aspect               | CovRL 2.52b                          | This AFL++ port                         |
|----------------------|--------------------------------------|-----------------------------------------|
| Queue representation | Token IDs (`u16[]`) natively         | Option A (token bytes) or B (src bytes) |
| Decode timing        | Inside `common_fuzz_stuff()`         | `post_process()` (Option A only)        |
| RL trigger           | Every `sync_fuzzers()` call          | `queue_new_entry()` every N entries     |
| Batching             | No (one predict per havoc iteration) | Optional via `fuzz_count()` cache       |
| Trim                 | AFL default (byte bisect)            | Token-level custom trim                 |
| Model process        | Separate server (socket)             | Same: separate server (socket)          |
