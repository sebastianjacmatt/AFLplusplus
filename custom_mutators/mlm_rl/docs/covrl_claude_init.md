# CovRL-Fuzz & AFL++ Port — Project Reference

## Repository Layout

```
remote_master/
├── CovRL-Fuzz/          # original CovRL — reference only, do NOT modify
│   ├── AFL/afl-fuzz.c   # modified AFL 2.52b with token-level fuzzing
│   ├── do_covrl.py      # TCP server: predict / decode / finetune
│   ├── covrl/
│   │   ├── models/
│   │   │   ├── inferencer.py    # mask-infill wrapper around T5 + owns FineTuner
│   │   │   ├── finetuner.py     # actor-critic PPO-style training loop
│   │   │   ├── rewarding.py     # subprocess coverage + TF-IDF reward calculation
│   │   │   ├── critic.py        # T5Encoder + linear head, 8-class classifier
│   │   │   ├── actor_dataset.py # T5 span-masking dataset for actor training
│   │   │   └── critic_dataset.py
│   │   └── utils/
│   │       ├── base_utils.py        # hex↔token encoding, score↔label maps
│   │       └── map_target_error.py  # per-engine error string → ErrorType
│   └── config/sample_config.json
└── AFLplusplus/
    └── custom_mutators/covrl/covrl.py  # TARGET: our AFL++ port (in-progress)
```

**Critical rule**: `CovRL-Fuzz/AFL/` is read-only reference. All implementation goes into `AFLplusplus/`.

---

## Part 1 — Modified AFL 2.52b (`CovRL-Fuzz/AFL/afl-fuzz.c`)

### Token-level representation

CovRL replaces AFL's byte buffers with **binary u16 token sequences**. Every seed file and every intermediate mutation buffer is a flat array of little-endian `uint16_t` values — the CodeT5+ token IDs of a JavaScript source file. The byte length of a seed is therefore always even (`len % 2 == 0`).

Relevant constants (`config.h` / top of `afl-fuzz.c`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MASK_TOKEN` | 4 | sentinel inserted where LLM should infill |
| `MASK_COUNT` | 3 | max masks inserted per mutation step |
| `MAX_TOKEN` | 65535 | max legal token ID |
| `MAX_DECODED_TOKENS_SIZE` | 35 000 000 | scratch buffer for decoded JS text |
| `MAP_SIZE` | 65 536 (2¹⁶) | AFL coverage bitmap size |
| `SYNC_INTERVAL` | 100 | havoc iterations between `sync_fuzzers()` calls |

### TCP socket

A single global `int sock` connects to `do_covrl.py`. AFL is the TCP client; `do_covrl.py` is the server. Three message types are used:

| Message | Direction | Purpose |
|---------|-----------|---------|
| `"predict"` (7 bytes) | AFL → server | request mask infilling |
| `"decode"` (6 bytes) | AFL → server | request token→JS text decoding |
| `"finetune"` (8 bytes) | AFL → server | request a finetuning cycle |

All replies are a single `uint16_t` (little-endian) giving the byte length of the result. AFL always reads `2` bytes back.

### Shared file protocol

Communication uses three files inside `out_dir/`:

| File | Written by | Read by | Content |
|------|-----------|---------|---------|
| `MLM_pred` | AFL (`write_to_MLM`) then do_covrl | AFL after predict | binary u16 token IDs |
| `MLM_decoded` | AFL (`decode()`) then do_covrl | AFL after decode | binary u16 (in) → UTF-8 JS text (out) |
| `MLM_Record` | do_covrl | do_covrl | path to current model checkpoint |

### Deterministic stages — completely removed

All classical AFL deterministic stages (bit-flip, arithmetic, interesting-value substitution, extras) have been **entirely removed** from CovRL. After computing the performance score and setting `doing_det = 1`, `fuzz_one()` jumps directly to `havoc_stage:` with no intervening code. The label comment explicitly reads `RANDOM HAVOC with Mask Mutation`. `passed_det` and `was_fuzzed` flags remain in the code (carried over from AFL 2.52b structs) but `mark_as_det_done()` is never called during normal operation. The `doing_det` flag's only effect is selecting `HAVOC_CYCLES_INIT` (1024) over `HAVOC_CYCLES` (256) for `stage_max` on the first fuzz pass of a seed.

### Havoc stage — mask mutation

The havoc stage is **the key CovRL contribution**. All normal AFL byte-level havoc mutations are removed; the loop body does the following on every iteration:

```
1. copy current token buffer into ex_tmp
2. choose mutation mode at random:
   RANDOM_INSERT  (mode 0): insert 1–3 MASK_TOKEN (value 4) at random u16 positions
   RANDOM_OVERWRITE (mode 1): overwrite 1–3 random u16 positions with MASK_TOKEN
3. write_to_MLM(MLM_pred, ex_tmp, temp_len)
4. send("predict") → recv u16 new_len
5. read new_len bytes from MLM_pred (now infilled token IDs)
6. call decode(decoded_tokens, ex_tmp, temp_len)
     write_to_MLM(MLM_decoded, ex_tmp, temp_len)   ← token IDs
     send("decode") → recv u16 text_len
     read text_len bytes from MLM_decoded           ← JS source text
7. common_fuzz_stuff(argv, decoded_tokens, decoded_len, ex_tmp, temp_len)
```

Each havoc iteration is therefore a synchronous round-trip over the TCP socket.

### Splice stage

The splice stage is also adapted for tokens. It picks statement boundaries (`;` token positions) via `get_splice_list()`, splices two token sequences at those positions, then proceeds into a second pass through the (modified) havoc loop. The stage name is `"token splicing"`.

### Finetuning trigger — `sync_fuzzers()`

Fine-tuning is triggered inside `sync_fuzzers()`, which CovRL repurposes:

```c
// first thing sync_fuzzers() does:
send(sock, "finetune", 8, 0);
recv(sock, buf, 2, 0);  // block until finetune completes
// then proceeds with normal inter-fuzzer queue sync
```

`sync_fuzzers()` is called from the main loop at:

```c
if (!(sync_interval_cnt++ % SYNC_INTERVAL))
    sync_fuzzers(sock, use_argv);
```

So a finetune request is sent every `SYNC_INTERVAL = 100` calls to `fuzz_one()`. Whether the server actually trains on that call depends on `MAX_FINETUNE_CNT` in `do_covrl.py`.

---

## Part 2 — Python Server (`CovRL-Fuzz/do_covrl.py`)

`do_covrl.py` is a blocking single-connection TCP server. After the handshake it spins in a `while True` loop:

```python
"predict"  → mask_mutation(conf, model, PREDICTION_PATH)
               reads  MLM_pred as binary u16 tokens (hex_to_dec)
               runs   model.inference(data, num_samples=n_samples)
               writes result back to MLM_pred as binary u16 (dec_to_hex)
               returns byte length (len(ret) * 2)

"decode"   → decode_data(model, PREDICTION_PATH)
               reads  MLM_decoded as binary u16 tokens (hex_to_dec)
               calls  tokenizer.decode(data, skip_special_tokens=True)
               writes UTF-8 JS source text back to MLM_decoded
               returns len(decoded_data) in bytes

"finetune" → finetune(model, PREDICTION_PATH)  (conditional on MAX_FINETUNE_CNT)
               reads  MLM_Record to get current model path
               clears MLM_Record
               calls  model.finetune(prediction_path)
               writes new model path to MLM_Record
```

`hex_to_dec` / `dec_to_hex` use `struct.iter_unpack("<H", ...)` / `struct.pack("<H", id)` — little-endian u16.

`PREDICTION_PATH` is the AFL `out_dir/` subdirectory where the shared files live.

---

## Part 3 — Inferencer (`covrl/models/inferencer.py`)

`Inferencer` wraps T5ForConditionalGeneration (CodeT5+ 220M) and owns a `FineTuner` instance.

### `inference(input_ids, num_samples)`

The core method called on every `"predict"` message:

1. **Split**: divide token sequence into chunks of `split_length = model_max_length - 3 = 765`.
2. **Mask**: for each chunk, find tokens equal to `MASK_TOKEN` (4) or `UNKNOWN_TOKEN`. Replace each with a unique sentinel from the top of the vocabulary (`vocab_size - 1`, `vocab_size - 2`, …). Record `mask_dict: {sentinel_id → original_position}`.
3. **Generate**: call `model.generate()` with contrastive search (`penalty_alpha=0.6`, `top_k=num_samples=32`, `no_repeat_ngram_size=3`, `max_length=round(model_max_length * mask_probability)`). The decoder autoregressively produces sentinel IDs interleaved with predicted tokens.
4. **Reconstruct**: parse the generated sequence. Each sentinel ID signals the start of replacement tokens for the masked position. Splice predicted tokens back at the masked positions.
5. Concatenate results across chunks.

### `finetune(predict_path)`

Called on every `"finetune"` message after the conditional check:

```python
is_first = finetuner.preprocess(predict_path)   # build/update dataset
finetuner.train_critic(epochs=critic_epochs)     # always update critic
if not is_first:
    model_path = finetuner.finetune_actor(epochs=actor_epochs)
self.load_model(self.model_path)                 # hot-reload actor
```

On the **first** call only the critic is trained; the actor is not fine-tuned until the second call, giving the critic a head-start at labelling quality.

---

## Part 4 — FineTuner (`covrl/models/finetuner.py`)

### Dataset construction — `preprocess(dir_path)`

Loads new testcases from the AFL queue directory. Files are sorted by queue ID (parsed from AFL's `id:NNNNNN,...` filename convention). Only files not yet seen by `mutation_dataset` (by `file_id`) are processed.

Each new file is read as binary u16, converted to token IDs via `hex_to_dec`, then decoded to JS text via `tokenizer.decode()`.

After collecting new data, calls `Rewarding.update(mutation_dataset, is_update_idf=True)` which:
- Runs coverage and validity checks (see Part 5)
- Updates the IDF embedding
- Assigns reward scalars to every program

The final training dataset mixes mutation corpus with pre-collected training data at roughly 4:1 (train:mutation) via `train_dataset.sample(len(mutation_dataset) * 4, ...)`.

### Critic training — `train_critic(epochs)`

- Dataset: `CriticDataset` — tokenizes each JS program, assigns label via `score_to_label(reward)`.
- Model: `CriticModel` — T5EncoderModel + dropout + linear head (8 output classes).
- Loss: cross-entropy over 8 labels.
- Labels (reward → label): `< -0.5 → 0`, `< 0 → 1`, `≤ 0.5 → 2`, `≤ 0.6 → 3`, `≤ 0.7 → 4`, `≤ 0.8 → 5`, `≤ 0.9 → 6`, `≤ 1.0 → 7`.
- Uses Hugging Face `Trainer` with `save_strategy="epoch"`, `save_total_limit=1`.

### Actor training — `finetune_actor(epochs)`

PPO-like update with the critic as reward signal.

```
For each batch:
  1. current_actor(masked_input)  → cur_logits, cur_log_probs
  2. prev_actor(masked_input)     → prev_log_probs  (frozen)
  3. critic([input; argmax(cur_logits)])  → 8-class logits → pred_label → reward scalar
  4. ratio = exp(cur_log_probs - prev_log_probs)
  5. clipped_ratio = clamp(ratio, 0.8, 1.2)
  6. ppo_loss = -mean(min(ratio * reward, clipped_ratio * reward))
  7. final_loss = ppo_loss + cur_outputs.loss  (CE language model loss)
```

`label_to_score` maps class → reward: `{0: -1.0, 1: -0.5, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8, 6: 0.9, 7: 1.0}`.

Dataset: `ActorDataset` — T5 span-masking using Poisson-length noise spans (`poisson_lambda=3.0`, `mask_probability=0.15`). Targets are the masked-out spans in sentinel format (standard T5 denoising objective).

Checkpoint saved via `model.save_pretrained(save_dir/actor_final)`.

---

## Part 5 — Rewarding (`covrl/models/rewarding.py`)

### Coverage measurement — `check_validity()`

For each testcase, a subprocess runs:

```bash
afl-showmap -o <savepath> -m none -t 5000 -- <interpreter_path> <testcase.js>
```

`afl-showmap` instruments the target binary and writes a text file of `edge_id:hit_count\n` lines — one line per executed edge in the coverage bitmap. A bitmap array of `bitmap_size = 131 072` integers is built by parsing this output.

Error classification from `stderr`/`stdout`:
- `"SEGV"` in stderr or `"assertion"` in stdout → `error = False` (crash, not a validity error)
- Otherwise: match against engine-specific error strings (`map_target_error.py`):
  - `SyntaxError` → `ErrorType.SYNTAX_ERROR` → reward = **-1.0**
  - Any other JS error → reward = **-0.5**
  - No error → reward deferred to TF-IDF step

Subprocess delegation uses `multiprocessing.Pool` with `pool.imap_unordered` for parallelism (currently `core_count = 1` in the codebase but the abstraction is there).

### IDF embedding — `update_idf(dataset, alpha=0.6)`

After coverage is computed for all new testcases, the IDF vector is updated:

```python
df[i] = number of programs (among is_orig entries) that hit edge i
total_docs = len(dataset)
new_idf[i] = (log(total_docs / (1 + df[i])) / map_size_pow2) * (1 - alpha)
idf = alpha * idf_prev + new_idf
```

This is an exponential moving average (`alpha = 0.6`) of a log-IDF score normalised by `sqrt(bitmap_size)`. Edges hit by few programs accumulate higher IDF. The IDF vector is persisted to `idf_embedding.bin` via pickle after each update.

### Reward scalar — `get_reward(dataset)`

For programs with `reward == 0` (i.e., valid, no error, not yet scored):

```python
score    = dot(bitmap, idf)          # weighted sum over hit edges
log_score = log(score) if score > 0 else 0
reward   = sigmoid(log_score)        # squashed to (0, 1)
reward   = round(reward, 2)
```

Programs that activate rarer edges (high IDF weight) receive scores closer to 1.0. Programs that execute only common edges receive scores close to 0.5 (sigmoid of a small positive number). Programs with errors receive fixed negative rewards (-0.5 or -1.0) and are excluded from the TF-IDF calculation.

---

## Part 6 — AFL++ Port Design (`AFLplusplus/custom_mutators/covrl/covrl.py`)

The port replaces the original CovRL TCP inter-process architecture with an AFL++ Python custom mutator loaded via `AFL_PYTHON_MODULE`. Key design decisions already encoded in the file:

| Original CovRL | AFL++ Port |
|----------------|-----------|
| TCP "predict" per havoc iter | `fuzz()` hook: mask + infill in-process |
| TCP "decode" per exec | `post_process()` or `fuzz()` decodes tokens → JS text before exec |
| TCP "finetune" at SYNC_INTERVAL | `queue_get()` counter triggers `_maybe_finetune()` |
| `sync_fuzzers()` side-effect | `queue_new_entry()` accumulates new files |
| Seeds: binary u16 LE token files | Seeds/queue: AFL++ native byte arrays (8-bit `bytearray`) |
| `AFL_CUSTOM_MUTATOR_ONLY=1` | All byte-level AFL++ stages suppressed |

**Queue format difference — critical**: The original CovRL stores seeds and queue entries as binary little-endian u16 token sequences because AFL 2.52b has no abstraction boundary between the fuzzer and its queue files. The AFL++ port does **not** replicate this. AFL++ owns the queue and passes inputs to the custom mutator as standard `bytearray` (8-bit values). Tokenization (bytes → token IDs) and re-encoding (token IDs → bytes) happen entirely inside the mutator boundary, invisible to AFL++. The `hex_to_dec` / `dec_to_hex` functions from `base_utils.py` are relevant only for understanding the original CovRL wire format; they are not the encoding contract for the port.

The `post_process()` hook is commented out in the current stub pending a decision on whether decoding (token IDs → JS UTF-8 text) should happen there or inside `fuzz()`. Both are valid placements since AFL++ native buffers carry whatever bytes the mutator writes.

---

## Configuration (`config/sample_config.json`)

| Key | Value | Meaning |
|-----|-------|---------|
| `load_path` | `Salesforce/codet5p-220m` | tokenizer + model base |
| `model_max_length` | 768 | max input tokens |
| `mask_probability` | 0.15 | fraction of tokens to mask |
| `n_samples` | 32 | contrastive search top-k |
| `alpha` | 0.6 | IDF EMA momentum |
| `critic_epochs` | 2 | |
| `actor_epochs` | 1 | |
| `train_batch_size` | 8 | |
| `target_interpreter` | `"jerry"` | maps to JERRY_ERROR dict |
| `interpreter_path` | path to `jerry` binary | |

---

## Key Invariants to Preserve in the Port

1. **Original CovRL uses u16 LE; AFL++ port uses native bytearrays**: The original CovRL queue files are binary little-endian u16 token sequences. The AFL++ port does not replicate this. AFL++ passes inputs as standard `bytearray`. All token ↔ byte conversion is an internal concern of the custom mutator.
2. **MASK_TOKEN = 4**: the sentinel value inserted during mutation. The infiller (`Inferencer._mask_unknowns`) also treats `UNKNOWN_TOKEN` as a mask position.
3. **Critic trained before actor, actor skipped on first cycle**: `finetune_actor` is only called when `not is_first`. This is load-bearing — the critic needs at least one training pass before its scores are used as PPO rewards.
4. **IDF is a running average, not recomputed from scratch**: each finetuning call updates `idf` in-place via EMA. The embedding is persisted to disk between calls.
5. **TF-IDF reward applies only to programs with reward == 0**: error cases (-1.0, -0.5) are assigned at fit time and excluded from the dot-product step.
6. **afl-showmap, not the AFL coverage bitmap**: reward computation uses a fresh `afl-showmap` subprocess, not the online AFL coverage bitmap. The two bitmaps measure the same edges but are independent.
