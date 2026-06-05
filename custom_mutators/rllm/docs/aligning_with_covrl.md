# Aligning rllm with CovRL-Fuzz on AFL++

This doc supersedes the earlier design notes — the historical `design.md` and
`what_covrl_gets_right.md` have since been removed, and `queue_validity_collapse.md`
is kept only for the validity-collapse / corpus-rot dynamic — as well as the
parallel work in `custom_mutators/covrl/`, `mlm_rl/`, `rlm_mutator/`. Those
artifacts are historical. The intent moving forward is to implement
CovRL-Fuzz on AFL++ via the data → model → training subsystems already
stubbed in `rllm/`, and the architectural choices that follow.

The reasoning in this doc came out of reading the Token-Level Fuzzing paper
(Salls et al., USENIX '21) and its TLAFL fork, the CovRL-Fuzz paper (Eom et
al., ISSTA '24) and its reference repo (`~/Documents/CovRL-Fuzz/`), the
AFL++ `custom_mutators/autotokens/` port of TLAFL, and the current
`rllm/` implementation. Code references are file:line where useful.

---

## 1. The architectural invariant

> **The persistent fuzzing queue must hold tokens, not source bytes.**
> Either by storing tokens directly on disk, or by storing bytes that
> round-trip losslessly through the tokenizer.

Every working system in the TLAFL/CovRL/autotokens lineage maintains this
property. rllm currently does not, and that is the load-bearing reason for
the validity drift we've been observing.

| System | Queue file | Invariant mechanism |
|---|---|---|
| TLAFL (Salls '21) | u16 binary token IDs | Direct token storage; only decoded at execution |
| CovRL-Fuzz | u16 binary token IDs | Direct token storage; decoded once per execution via Python IPC |
| autotokens (AFL++) | bytes (canonical concatenation of token strings) | Deterministic round-trip: `lex(emit(tokens)) == tokens` |
| **rllm (current)** | **bytes (BPE-detokenized text, with a U+FFFD strip)** | **None — best-effort lossy round-trip** |

Source-of-truth references for the table:

- CovRL writes u16 tokens to the queue:
  `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:3207, 3229`
  (`add_to_queue(fn, encoded_len, 0)` and
  `ck_write(fd, encoded_buf, encoded_len, fn)` inside `save_if_interesting`).
- CovRL reads u16 from the queue:
  `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5008, 5070`
  (`u16 *in_buf, *out_buf` declarations and `mmap(0, len, ..., fd, 0)`
  cast as `u16*`).
- CovRL decodes once per execution:
  `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5247–5252`
  (`decode(decoded_tokens, ex_tmp, temp_len);
  common_fuzz_stuff(argv, decoded_tokens, decoded_len, ex_tmp, temp_len);`).
- autotokens builds output canonically from the token map:
  `~/Documents/AFLplusplus/custom_mutators/autotokens/autotokens.cpp:417–448`
  (the loop appending `id_to_token[m[i]]` strings to `output`).
- autotokens caches tokens per filename:
  `~/Documents/AFLplusplus/custom_mutators/autotokens/autotokens.cpp:707–966`
  (`file_mapping` lookup, lazy lex on first sight).

Every downstream design choice in this doc follows from how we maintain
this invariant.

---

## 2. autotokens is a valid AFL++ adaptation of TLAFL — with bounded loss

An earlier reading of this material dismissed autotokens as a strawman.
That was wrong: autotokens is a real AFL++ adaptation of TLAFL's
*strategic idea* (mutate at the token level with a closed vocabulary,
decode for execution). The headline I used in the first draft of this
doc — "lossless implementation via deterministic byte round-trip" — is
too strong, though. autotokens has documented one-shot losses on
initial ingestion. They are bounded, but they exist.

See [autotokens_losslessness.md](autotokens_losslessness.md) for the
full preservation/loss breakdown with line refs and worked examples.
The summary:

- **Preserved**: closed vocabulary (`autotokens.cpp:299, 446`), in-process
  cache canonical between mutations (`autotokens.cpp:707, 966`),
  whitespace runs byte-preserved within their token
  (`autotokens.cpp:810-822`), convergent byte round-trip after one
  ingestion (`emit(lex(emit(t))) == emit(t)`).
- **Lost** (one-shot, on initial ingestion):
  - `/* ... */` comments stripped by regex
    (`autotokens.cpp:758-766`) — fully unrecoverable.
  - Whitespace tokens injected between adjacent multi-char tokens at
    emit time (`autotokens.cpp:430-439`) — the cache *grows* on the next
    re-lex; the byte sequence on disk does not match the cache that
    produced it before re-lex.
  - Identifier class includes `.` and `/`
    (`autotokens.cpp:826-827`); not byte loss but structurally coarser
    than a JS lexer.
  - Non-ASCII seeds disable the module entirely
    (`autotokens.cpp:181-217`) — scope limitation rather than per-byte
    loss.

In short: autotokens implements TLAFL's *strategic idea* (mutate on
tokens) but not its *architectural property* (tokens are the only
state). CovRL/TLAFL sidestep round-trip questions by storing tokens
directly on disk; autotokens substitutes a different property
("closed-vocab convergent byte round-trip with bounded one-shot
ingestion loss") that is genuinely useful but not the same thing.

For the rest of this doc, "autotokens-style" or "byte queue with
canonical round-trip" should be read with this refinement in mind.

### What we take from autotokens

- **In-process `filename → token sequence` cache pattern**, populated lazily
  on first `queue_get`, consumed for the lifetime of the run. This is how
  we get O(1) per-iteration token access without modifying AFL.
- **Canonical-byte-encoding discipline**: the bytes we write must be the
  exact bytes the tokenizer would emit for those tokens. autotokens
  achieves this by concatenating fixed token strings; we will achieve it by
  ensuring the BPE round-trip is the identity on our output domain (see
  §3, Option A).
- **Closed-vocab bookkeeping with budgeted growth**: autotokens caps its
  token vocabulary at "tokens observed in seeds plus dictionary entries"
  (`autotokens.cpp:600–660`). We won't reuse the regex lexer, but the
  pattern of *restricting the mutator's universe to a chosen vocab subset*
  is exactly what we want when we logit-mask the LLM.

### Where we will differ from autotokens

- **Vocabulary**: CodeT5+ byte-level BPE, not regex-derived JS-shape tokens.
- **Mutator**: LLM mask-fill, not random replace/insert/erase.
- **In-process state shape**: Python `list[int]` rather than `vector<u32>`.
  The u16-vs-u32 question is moot under the byte-queue architecture; see
  §3.4.

### Why we don't compose with autotokens via AFL++'s multi-mutator support

AFL++ allows multiple mutators via
`AFL_CUSTOM_MUTATOR_LIBRARY="m1.so;m2.so"` plus a Python module from
`AFL_PYTHON_MODULE`. Confirmed in
`~/Documents/AFLplusplus/src/afl-fuzz-one.c:1957`:

```c
LIST_FOREACH(&afl->custom_mutator_list, struct custom_mutator, {
  if (el->afl_custom_fuzz) { /* per-mutator stage */ }
});
```

Each mutator gets its own stage with its own `stage_max` iterations. The
outputs are independent — mutator N does not see the output of mutator
N-1. So co-installing rllm and autotokens gives us two parallel fuzzing
stages with disjoint token spaces (CodeT5+ BPE vs regex-lexed JS). That
isn't an integration; it's two unrelated fuzzers sharing a queue. Not
useful.

The right way to take inspiration from autotokens is to **steal its
pattern** (in-process token cache + canonical encoding) inside rllm's
existing Python custom mutator, not to run it alongside.

---

## 3. The two architectural options that actually maintain the invariant

Both options below maintain the queue-as-tokens invariant; both are
implementable as a Python custom mutator on stock AFL++. The differences
are in how the queue file is shaped and what constraints we need to enforce.

### 3.1 Option A — byte queue with canonical round-trip (autotokens-inspired)

- Queue files: bytes (standard AFL++).
- Mutator emits bytes that are the canonical detokenization of some BPE
  token sequence, guaranteed by a logit constraint on the LLM.
- In-process `dict[filename, list[int]]` cache in `Mutator`. Populated in
  `queue_new_entry(new, orig)` when the just-emitted tokens are paired
  with the filename AFL assigns; consumed in `queue_get` / `fuzz_count`
  for subsequent mutations of that entry.
- On cache miss (restart, initial seeds, AFL `sync_fuzzers` imports): we
  re-tokenize from bytes. Lossless because the bytes are guaranteed to be
  canonical for our output domain.

#### The constraint we have to enforce

> The set of token IDs the LLM may emit must be exactly the IDs whose
> standalone byte decoding is well-formed UTF-8 *and* whose
> `encode(decode(id)) == [id]`. Filter the model's logits accordingly at
> generation time.

In CodeT5+'s byte-level BPE this means dropping the "partial multi-byte"
tokens — entries whose decoded bytes are not a complete UTF-8 codepoint on
their own (e.g., a token mapping to just `\xef`). These are the tokens
that produce today's U+FFFD artifacts when they land adjacent to other
tokens at decode time.

Implementation: a HF `LogitsProcessor` (or the simpler `bad_words_ids`
list) constructed once at startup by scanning the full vocab:

```python
# pseudocode
bad = []
for tid in range(vocab_size):
    text = tok.decode([tid])
    if "�" in text:                  # partial-byte token
        bad.append(tid); continue
    if tok.encode(text, add_special_tokens=False) != [tid]:  # not round-trip safe
        bad.append(tid); continue
```

Optional tightening: further restrict to ASCII-decoding tokens. Tighter
round-trip guarantee at the cost of mutation diversity on seeds with
legitimate non-ASCII string content.

#### Pros

- AFL++ tooling works directly on the queue: `afl-cmin`, `afl-tmin`,
  `afl-showmap`, manual inspection, anything that expects source bytes.
- Queue files are human-readable JS.
- No seed pre-processing required.
- No custom `post_process` required.
- The shape of the solution is what AFL++ itself uses for autotokens.

#### Cons

- The round-trip guarantee is a contract we have to maintain. Forget the
  logit mask and drift returns.
- The cache is in-process: cold start (or `sync_fuzzers` import) re-lexes
  from bytes. Lossless by construction under the constraint, but more code
  than direct token storage.

### 3.2 Option B — u16 binary queue via custom mutator (CovRL-exact)

- Queue files: u16 binary token-ID arrays. AFL treats them as opaque bytes.
- `fuzz()` returns u16 binary.
- `post_process(buf)` decodes u16 → JS source bytes for the target.
- Initial seeds pre-processed once into u16 binary in the `-i` directory.

#### Pros

- Exactly matches CovRL's behavior including on-disk format.
- No logit constraint required — the full BPE vocab is legal.
- Direct invariant: no round-trip contract to maintain.

#### Cons

- AFL++ tooling on the queue breaks (cmin/tmin/showmap see u16 binary).
  Either avoid those tools or wrap them in decode shims.
- Queue files are not human-readable.
- Custom `post_process` required.
- Seed pre-processing step required.
- Queues are not portable across tokenizer revisions.

### 3.3 Recommendation: default Option B

**Option B is the default**, given the priority ordering: (1) reproduce
CovRL-Fuzz > (2) align with AFL++ docs/practices > (3) retain rllm's
structure. Option B is what CovRL-Fuzz actually does, and
[`option_b_viability.md`](option_b_viability.md) verified that it is
fully achievable on stock AFL++ via the documented
`AFL_POST_PROCESS_KEEP_ORIGINAL=1` env var — no AFL++ source changes,
no contortions. The earlier preference for Option A was made under
the assumption that Option B required custom plumbing or AFL++
modifications; that assumption is no longer accurate.

Why Option B wins under the current priorities:

- **CovRL reproduction (priority 1)**: Option B *is* CovRL's
  queue-as-tokens architecture. The queue holds u16 token IDs;
  `post_process` decodes once per execution. Identical to
  `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c` end-to-end.
- **AFL++ alignment (priority 2)**: `AFL_POST_PROCESS_KEEP_ORIGINAL`
  is documented AFL++ behavior, used by libprotobuf-mutator and
  similar custom mutators in the tree. We're using the AFL++ API as
  the maintainers intended for the "mutator format ≠ target format"
  pattern.
- **rllm structure (priority 3)**: only `Mutator` internals and the
  preprocessing pipeline change; `data/`, `model/`, the future
  `training/` and the AFL hook shim (`rllm.py`) stay as they are.

Option A is documented above as a viable alternative but no longer the
default. It remains the right call only if:

- Standard AFL++ tooling on the queue (`afl-cmin`, `afl-tmin`,
  `afl-showmap` without a decode shim) becomes a hard requirement.
- Reproduction faithfulness to CovRL drops below priority 2 (e.g.,
  for a research direction that explicitly deviates from CovRL).

Neither condition holds today.

### 3.4 u16 vs u32 — rigorously

The earlier framing of this question was a category error. The
representation choice depends on which option you take:

- **Under Option A**, the in-process cache is Python `list[int]`. Python
  ints are arbitrary-width. We never serialize tokens to a fixed-width
  format. **The question does not arise.**
- **Under Option B**, the question is real and the answer is u16:
  - CodeT5+'s vocab is ~32,100 tokens, plus 100 sentinels for spans, plus
    `MASK_TOKEN`/`EOS`/`PAD`. All of this fits in 16 bits (≤65535) with
    headroom.
  - u32 doubles every queue file with no information gain.
  - CovRL chose u16 for the same reason; the choice is faithful and
    storage-efficient.
- **autotokens uses u32** because *its* vocabulary grows dynamically with
  new finds (`autotokens.cpp:619–655` adds previously-unseen JS tokens at
  runtime). That rationale does not apply to a fixed-vocab BPE.

So: under Option A the in-process representation is Python ints (moot);
under Option B u16 is the right default, u32 the fallback if we ever push
past 65k vocab.

---

## 4. Masking is one machinery with two knobs

The earlier drafts of this analysis framed masking as a three-way choice
(M1 = single-token, M2 = T5 span corruption, M3 = CovRL's split). That
framing was overcomplicated. The honest reading is:

> **T5 span corruption is CovRL's RANDOM_OVERWRITE when the span-shape
> parameters are degenerate.** The same code path produces either behavior
> depending on the knob values.

The equivalence, expressed in our existing `data/masking.py` parameters:

| Setting | Behavior | Equivalent to |
|---|---|---|
| `min_span_length=max_span_length=mean_span_length=1`, `corruption_rate` set so `round(n * rate) ≈ 1–3`, `whole_word_masking=False` | k ∈ {1,2,3} single-token sentinels at random positions; no multi-token deletions | CovRL `RANDOM_OVERWRITE` (`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5211–5221`); TLAFL "Random Overwrite" |
| `corruption_rate=0.15`, `mean_span_length=3.0`, `min=1`, `max=5`, `whole_word_masking=False` | ≈5 multi-token spans collapsed into sentinels | CodeT5+ MSP pretraining objective; what `data/masking.py` does today |

The interesting design question is therefore not "which of M1/M2/M3" but:

1. What parameter setting do we default to?
2. Which operations beyond plain overwrite do we add on top?

### 4.1 The operations CovRL has that pure span overwrite does not

- **INSERT** (`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5193–5209`): place a
  fresh MASK token at a random position, expanding the buffer length.
  Lets mutations grow programs.
- **SPLICE** (referenced at
  `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5305+`): copy a statement from
  another queue entry, paste into the current input with MASKs at the join
  boundaries.

Why they matter:

- Without INSERT, buffer length only changes via the model emitting more
  or fewer tokens than the span being filled. Over many cycles the queue
  tends toward similar-length programs with limited structural growth.
- Without SPLICE, exploration is local. Cross-pollination between distinct
  queue entries happens only through AFL's normal queue selection (each
  iteration picks a different parent, but no statement-level material
  transfers).

Both are useful, neither is required for a minimum viable CovRL-aligned
mutator. Both extend the same masking machinery cleanly:

- INSERT = "place a sentinel at a position without removing tokens".
- SPLICE = "place two sentinels around a pasted statement". Reuses the
  same model call.

### 4.2 The recommendation: adopt CovRL's split for reproduction fidelity

Given priority 1 (reproduce CovRL-Fuzz), we adopt CovRL's split exactly:

1. **Fuzz time**: single-token Insert/Overwrite — the first row of the
   table above (`min=max=mean=1`, `corruption_rate≈0.03`,
   `whole_word_masking=false`). Functionally identical to CovRL's
   `RANDOM_OVERWRITE` at
   `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5211–5221`. This is the
   default in `configs/default.json`.

2. **Training time** (when the training subsystem lands): T5 span
   corruption with the broad parameters (the second row in the table).
   Same `data/masking.py`, just a different config consumed by the
   training loop. Matches
   `~/Documents/CovRL-Fuzz/covrl/models/actor_dataset.py:random_spans_noise_mask`
   and CodeT5+'s pretraining objective.

3. **Add INSERT as a second operation mode** before RL. Small extension
   to `data/masking.py` (sentinel placement without removing a token).

4. **Defer SPLICE.** Marginal benefit unclear; needs `add_buf` plumbing.

### 4.3 Why CovRL's split — re-evaluated under priority 1

An earlier draft of this section recommended matching train and fuzz
distributions, arguing every fuzzing observation should be a directly
usable training sample under a unified masking strategy. That is a
design-purity argument and it loses under priority 1.

The reasoning, plainly:

- CovRL achieves its published validity *with* the split. The PPO step
  is the bridge that reconciles the train/fuzz distribution gap; that
  reconciliation is part of what makes their pipeline work.
- Adopting a different masking strategy would be "rllm's improvement
  on CovRL" — a different research question from "reproduce CovRL."
- We still have one masking machinery (one code path, two parameter
  presets). The duplication concern doesn't apply; only the call-site
  choice of preset differs between training and fuzz.

This is a deliberate change from the earlier draft, recorded so the
shift is visible.

### 4.4 Masking and architecture are orthogonal

The masking choice does not depend on whether we go Option A or Option B
for the queue. Both architectures operate on `list[int]` internally;
masking runs on that list regardless.

Where the two architectures interact with masking:

- **Under Option A** the LLM's logit filter is the round-trip-safety
  contract. The masker doesn't care; every token in its input is safe
  (because the cached parent's tokens came either from a prior LLM call,
  which was logit-filtered, or from re-tokenizing canonical bytes), and
  every token in the LLM's output is safe (because the filter enforces
  it). The masking module operates on safe IDs throughout.
- **Under Option B** no logit filter is required; the full vocab is legal.
  Masking operates on u16 queue contents as-is. The only byte conversion
  is `post_process(buf)` once per execution.

For INSERT specifically:

- Under Option A, the inserted sentinel becomes part of the encoder input
  for the LLM. `Tokenizer.reconstruct` splices the LLM's filled tokens
  into the parent at the insertion point and detokenizes the whole thing.
  Output bytes remain canonical because of the logit filter.
- Under Option B, the u16 buffer grows by one slot per inserted MASK;
  post_process decodes the lot for execution.

For SPLICE specifically:

- Under Option A, `add_buf` arrives as bytes; we tokenize it through the
  cache (if it's a known filename) or via a fresh lossless re-tokenization
  (lossless because the canonical-encoding invariant applies to all queue
  entries our mutator wrote).
- Under Option B, `add_buf` arrives as opaque bytes which we cast to u16.
  Simpler.

Net: Option B is marginally simpler for the extensions but Option A still
works; this does not change the §3.3 recommendation.

---

## 5. The data / model / training pipeline matching CovRL

CovRL has three subsystems; rllm already has stubs for two and is missing
the third. This section maps each to a concrete change.

### 5.1 data/

- **New**: `data/preprocess.py`. Mirror
  `~/Documents/CovRL-Fuzz/covrl/utils/preprocess.py`. Pipeline: UglifyJS
  `-m -b` over each seed JS file, then tokenize.
- **Under Option A**: write the preprocessed bytes back as canonicalized
  JS source. This is what AFL reads as seeds. The canonicalization
  ensures re-tokenization in `queue_get` is lossless.
- **Under Option B**: write u16 binary to the `-i` directory. AFL reads
  these directly; `post_process` decodes them for execution.

The UglifyJS step is what CovRL uses (paper §4) to normalize identifiers
and remove most non-ASCII content. This is what makes their BPE round-trip
behave well in practice. Without it our seed corpus carries non-ASCII
strings from regression tests that produce the byte-boundary U+FFFD class
of artifacts we've been seeing.

### 5.2 model/

- Keep `Salesforce/codet5p-220m`. Same model CovRL uses.
- Use `data/masking.py` as-is with the degenerate parameter setting from
  §4. No new masking module; this is a config change in
  `configs/default.json`.
- Add a config-gated INSERT operation mode in `data/masking.py`. Clean
  extension; sentinel placement without removing tokens.
- Defer SPLICE until the basic loop is verified working.
- Under Option A, add the `LogitsProcessor` described in §3.1 to
  `model/llm.py`'s `batch_generate`. Build the bad-words list once at
  `Model.__init__`.

### 5.3 training/

Net-new subsystem. Two pieces, separable into their own follow-up plans.

- **Reward** (mirror `~/Documents/CovRL-Fuzz/covrl/models/rewarding.py`):
  - Per-mutation stderr classification we already have via `exit_hook.so`
    and `_classify_stderr` in `mutator.py:10–18`.
  - AFL coverage bitmap reader. Read `<out>/queue/.state/cov_*` or invoke
    `afl-showmap` on the mutation output, mirroring `rewarding.py:75–152`.
  - Reward assignment: `-1.0` (syntax), `-0.5` (semantic), `+R_cov`
    (TF-IDF weighted, sigmoid-normalized). Exactly the assignment in
    `rewarding.py:190–199`.
  - TF-IDF IDF update with α momentum, exactly per
    `rewarding.py:59–73` (`idf = α·idf_prev + (1−α)·idf_new`).

- **PPO step** (mirror
  `~/Documents/CovRL-Fuzz/covrl/models/finetuner.py:compute_actor_loss`):
  - Two model copies: current actor + previous-cycle actor (frozen).
  - Compute `ratio = exp(log_prob_cur - log_prob_prev)`; clip to
    `[0.8, 1.2]`; PPO loss `-min(ratio·r, clip(ratio)·r).mean()`.
  - Add the SFT auxiliary loss as the CovRL implementation does.
  - **Correction from CovRL's released code**: gather log-probs to the
    actual label tokens; don't mean across the full vocab axis. CovRL's
    released code averages across the entire softmax output, which
    dilutes the per-token signal. This was noted in our prior reading.

---

## 6. Phased implementation order

The intent is for each phase to be a self-contained follow-up plan.

1. **Architecture lock-in (Option A).** Wire up the in-process
   `filename → list[int]` cache in `Mutator`. Populate in
   `queue_new_entry`, consume in `queue_get` and `fuzz_count`. Build the
   `LogitsProcessor` that masks partial-byte BPE tokens. Verify the
   round-trip invariant holds end-to-end on a short fuzzing run by
   reading the queue back through the same tokenizer and confirming token
   identity with the cached IDs.

2. **Masking knob change.** Set
   `min_span_length=max_span_length=mean_span_length=1`, choose
   `corruption_rate` to target ≈1–3 masked tokens for representative
   program lengths, `whole_word_masking=False`. Config-only; no new code.
   Keep the broad-span configuration available as an ablation. Optionally
   add INSERT as a second operation mode in `data/masking.py` (clean
   extension of the same module); defer SPLICE.

3. **Seed pre-processing.** UglifyJS `-m -b` over the seed corpus once,
   write canonicalized JS back into the `-i` directory. Confirm
   per-mutation P(valid) reaches CovRL's reported ~21.5% on JerryScript
   (Table 7, "LLM w/o CovRL" — the SFT-free baseline). Without this our
   numbers are not comparable to the paper.

4. **SFT warm-up.** Brief finetuning of CodeT5+ on the preprocessed corpus
   with the new single-token masking. Confirms the model can fill the
   degenerate-mask objective well before RL begins. This is the
   distribution-shift fix for the "model wasn't pretrained on
   single-token masks" objection.

5. **Reward subsystem.** afl-showmap wiring (or direct bitmap read),
   validity classification (already partly present), TF-IDF IDF map with
   α momentum, reward assignment with `-1.0 / -0.5 / sigmoid(TF-IDF)`.

6. **PPO loop.** Train every N fuzzing cycles. Hot-swap the new
   checkpoint into the running mutator. Cadence and N to be chosen based
   on the per-call cost and the rate at which the queue diversifies.

---

## What this doc explicitly does not pick

- It does not pick between Option A and Option B as a final commitment.
  Option A is the recommended starting point and the rationale is given;
  Option B is a documented fallback with explicit triggers for the switch.
- It does not declare an empirical winner between the masking
  configurations. Single-token overwrite is the default for design
  reasons (train/fuzz alignment + per-mutation validity); broad-span
  remains available as an ablation.
- It does not commit to a specific tokenizer revision or PPO
  hyperparameter set. Those land in follow-up plans alongside the code.

## What it does claim

- The queue-as-tokens invariant is load-bearing for everything downstream.
  rllm's current failure to maintain it is the root cause of validity
  drift.
- autotokens is a valid AFL++ implementation of TLAFL and its pattern
  (in-process token cache + canonical-byte encoding) is the right one to
  reuse inside rllm.
- Option A keeps AFL++ idioms intact and is the practical default; Option
  B is the strict CovRL match.
- Masking is one machinery with parameter knobs; the M1/M2 framing was
  overcomplicated; matching train and fuzz distributions wins over
  reproducing CovRL's documented split.
- The phased implementation order above is the path to landing the system
  CovRL describes, on AFL++, via the data / model / training subsystems
  rllm has already begun.
