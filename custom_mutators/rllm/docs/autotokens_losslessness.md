# Is autotokens a lossless application of Token-Level AFL?

Short answer: **no, not in the strict sense**. autotokens is an
AFL++-idiomatic adaptation of TLAFL's *strategic idea* (mutate at the
token level, with a closed vocabulary, on a byte queue). It does not
preserve the bytes→tokens→bytes round-trip as the identity. It does
maintain a weaker but still useful property — *convergent round-trip
with closed-vocab canonical-byte encoding* — and its loss is *bounded
and one-shot* on initial ingestion.

This doc refines the broader claim made in
[aligning_with_covrl.md](aligning_with_covrl.md) §2 ("valid lossless
implementation"). The headline conclusion there should be read with
this breakdown.

All code references in this doc were verified against the trees in
`~/Documents/AFLplusplus/custom_mutators/autotokens/` and
`~/Documents/CovRL-Fuzz/AFL/` at session time (2026-05-21).

---

## 1. What "lossless" could mean

There are three plausible definitions; only the first one matches the
intuitive headline. The other two are weaker substitute properties.

| Definition | Statement | Holds for autotokens? |
|---|---|---|
| **Strict** (byte-identity) | `bytes_in == emit(lex(bytes_in))` for every input the lexer accepts. | **No** — broken by comment stripping and emit-side whitespace injection. |
| **Weak** (convergent) | `emit(lex(emit(t))) == emit(t)` for any cached token sequence `t` produced by a previous lex/mutate. After one ingestion, further round-trips are stable. | **Yes** — by construction (every emitted byte run is a known token; the lexer is deterministic on the canonical concatenation). |
| **Practical** (closed-vocab) | Every byte sequence written to the queue is a canonical concatenation of tokens drawn from the in-process token map. Mutations operate exclusively on known tokens. | **Yes** — `new_item = rand_below(afl_ptr, current_id)` in `autotokens.cpp:299`; output built only from `id_to_token[m[i]]` in `autotokens.cpp:446`. |

So the answer to "is autotokens lossless TLAFL" depends on which
property the asker means by "lossless." If they mean **strict
byte-identity**, autotokens is not. If they mean either of the weaker
properties, autotokens is. The existing `aligning_with_covrl.md` §2
elided the distinction and overclaimed.

---

## 2. Where autotokens preserves information

### 2.1 Closed vocabulary on output

Mutation picks tokens by index from the in-process map:

```
custom_mutators/autotokens/autotokens.cpp:299
    new_item = rand_below(afl_ptr, current_id);
```

and the output is built only from those tokens' canonical strings:

```
custom_mutators/autotokens/autotokens.cpp:446
    output += id_to_token[m[i]];
```

No mutation can introduce a byte sequence that wasn't part of some
previously-observed token. This is the analog of CovRL/TLAFL's
"mutation operates on token IDs only."

### 2.2 In-process cache is canonical between mutations

The cache is keyed by filename:

```
custom_mutators/autotokens/autotokens.cpp:707
    auto entry = file_mapping.find(fn);
    ...
    if (entry == file_mapping.end()) { /* lex from bytes */ }
```

For the lifetime of the process, a queue entry's token sequence is
authoritative once it has been lexed. Subsequent `queue_get` calls for
the same filename skip the lexer entirely and return the cached
`vector<u32>` (`autotokens.cpp:966`). So mutations never re-tokenize
their own previous output — they mutate the cache directly.

### 2.3 Whitespace runs are byte-preserved within their token

The lexer captures a run of `isspace()` characters as a single token
whose content is the exact byte run:

```
custom_mutators/autotokens/autotokens.cpp:810–822
    if (isspace(*prev)) {
      auto start = prev;
      while (isspace(*prev)) { ++prev; }
      tokens.push_back(std::string(start, prev));  // WHITESPACE
    }
```

So `"  \t\n"` becomes one token whose content is `"  \t\n"`. Emitting
it writes those exact bytes. This is *not* a loss point — listed here
for completeness so the loss list below can't be misread to include
whitespace-run normalization.

### 2.4 Convergent round-trip after one ingestion

Claim: for any cached token sequence `t` that autotokens itself
emitted, `emit(lex(emit(t))) == emit(t)`.

Sketch: `emit(t)` is the canonical concatenation of token strings
(plus the rare repair-whitespace insertions discussed in §3.2). The
lexer is deterministic on its input. By case analysis over the
token-class machinery in `autotokens.cpp:807-895`, the lex of
`emit(t)` returns a token sequence `t'` whose canonical concatenation
is byte-identical to `emit(t)` — even if `t' != t`. So `emit(t')` is
byte-identical to `emit(t)`. The byte stream stabilizes after the
first round.

This is the property the existing doc was gesturing at when it called
autotokens "lossless via deterministic byte round-trip." More
precisely: the *byte stream* stabilizes; the cached *token sequence*
may expand by one round (e.g., the whitespace-insertion case
materializes a new whitespace token).

---

## 3. Where autotokens loses information

Each item below: the property, the line reference, and a worked
example showing the input bytes vs. the bytes that come back after one
lex / cache / emit cycle.

### 3.1 Comment stripping (the largest loss)

```
custom_mutators/autotokens/autotokens.cpp:758-766
    if (regex_comment_custom) {
      input = regex_replace(input, *regex_comment_custom, "$2");
    } else {
      input = regex_replace(input, regex_comment_star, "");
    }
```

The lexer pre-processes input with a `/* ... */` regex (or a
user-supplied one via `AUTOTOKENS_COMMENT`) and replaces matches with
the empty string before tokenization. Anything between the markers,
including line breaks and identifiers, is unrecoverable.

Worked example. Input:

```
var x = 1; /* note: this matters */ var y = 2;
```

After `regex_replace(..., regex_comment_star, "")`:

```
var x = 1;  var y = 2;
```

The cache contains tokens for `var x = 1; var y = 2;` (with whatever
whitespace was around the comment). The mutator can never produce a
comment again, because no token in the vocab corresponds to one. The
bytes on disk after emit do not contain the comment.

### 3.2 Whitespace injection on emit

When two adjacent cached tokens are both > 2 chars and neither is
whitespace, the emit code splices in a whitespace/single-char token:

```
custom_mutators/autotokens/autotokens.cpp:430-439
    if (unlikely(!(prev_size == 1 || was_whitespace ||
                   this_size == 1 || is_whitespace))) {
      output += id_to_token[good_whitespace_or_singleval()];
    }
```

This is a deliberate **repair** to prevent adjacent multi-char
identifier tokens from merging when read back. From the lexer's
perspective the injected token is "free": when the bytes are re-lexed,
the injected whitespace becomes a new whitespace token in the cache.
So the cache grows by one token after the round-trip, and that growth
is real (not an artifact of re-reading the bytes).

Worked example. Suppose mutation produced a cache `["return", "obj"]`
(two multi-char non-whitespace tokens, adjacent). Emit writes:

```
return<WS>obj
```

where `<WS>` is whichever whitespace/single-char token
`good_whitespace_or_singleval()` returned. Re-lex of those bytes:

```
["return", "<WS>", "obj"]
```

Three tokens in the new cache; two in the old. Subsequent emit/lex
cycles are stable on the three-token version, but the original
two-token state is unrecoverable.

This is *bounded* (at most one whitespace token per adjacent-multi-char
pair) and *one-shot* (the second round-trip is identity-on-bytes).
It is still a non-trivial divergence from the strict definition of
losslessness.

### 3.3 Identifier class diverges from a JS lexer

```
custom_mutators/autotokens/autotokens.cpp:823-836
    } else if (isalnum(*prev) || *prev == '$' || *prev == '_') {
      auto start = prev;
      while (isalnum(*prev) || *prev == '$' || *prev == '_' ||
             *prev == '.' || *prev == '/') {
        ++prev;
      }
      tokens.push_back(string(start, prev));  // IDENTIFIER
```

The "identifier" class includes `.` and `/`. So `a.b.c` is one token,
not five. `path/to/file` is one token. `1.5e10` is one token. This
isn't *byte loss* per se — the bytes round-trip cleanly — but it is a
semantic divergence from any real JS lexer:

- Mutation on an "identifier" token can swap a whole dotted-property
  chain. That is sometimes what you want (autotokens treats this as a
  feature) and sometimes a coarser-than-intended edit.
- The pseudo-lexer can never *introduce* a new dot or slash by adjacent
  juxtaposition, because dots and slashes only appear inside
  identifier tokens.

Net: not a loss of bytes, but a loss of the structural fidelity a real
JS lexer would provide.

### 3.4 ASCII gate disables the mutator on non-ASCII seeds

```
custom_mutators/autotokens/autotokens.cpp:181-217
    /* we want at least 99% of text characters ... */
    if (((q->len * AFL_TXT_MIN_PERCENT) / 100) <= valid_chars) { ... }
    ...
    if ((is_ascii * 100) / valid <= 70) { module_disabled = 1; }
```

If the seed corpus is < 70% "ASCII-looking" by autotokens' definition,
the module disables itself and the queue gets only AFL's standard
byte-level mutations. This is a scope limitation rather than a per-byte
loss, but it means autotokens can't be the queue's token-level fidelity
guarantor on seeds with significant non-ASCII content (e.g., JS
regression tests with non-ASCII string literals, JSON with international
keys).

### 3.5 Repeated `dict[empty_key]` registers token-ID 0

```
custom_mutators/autotokens/autotokens.cpp:927-942
    if ((id = token_to_id[tokens[i]]) == 0) {
      // First time we see this token, add it to the list
      token_to_id[tokens[i]] = current_id;
      ...
    }
```

`token_to_id` is `unordered_map<string, u32>`. A miss on `tokens[i]`
returns 0 (the value-initialized default for u32), which is then
checked against zero to detect "unknown token." This conflates "token
not in map" with "token mapped to ID 0." In practice the first slot
(ID 0) is occupied by the first whitespace token initialized in
`afl_custom_init` (autotokens.cpp:1046), and the lexer never produces
the same string for a new token without seeing it before — but this is
a sharp edge worth knowing about. Not a byte-level loss, but a
correctness condition that depends on `id_to_token[0]` being a
sentinel.

---

## 4. CovRL/TLAFL: there is no round-trip to lose information in

The whole question is moot for CovRL/TLAFL because they never have a
bytes→tokens reverse path in steady state. Sketch with refs:

- `save_if_interesting` (`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:3179-3233`)
  writes the **encoded** (u16) buffer to the queue file:

  ```
  add_to_queue(fn, encoded_len, 0);
  ...
  ck_write(fd, encoded_buf, encoded_len, fn);  // line 3229
  ```

  The byte form (`mem` = decoded source) is freed; only `encoded_buf`
  hits disk.

- `fuzz_one` reads queue entries back as u16
  (`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:5008, 5070`):

  ```
  u16 *in_buf, *out_buf, *orig_in, *ex_tmp, *eff_map = 0;
  ...
  orig_in = in_buf = mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ```

- `decode` (`~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:2526-2554`) is the
  *only* tokens-to-bytes conversion, and it is one-way: it writes the
  decoded bytes to a temp file the target consumes. The bytes are
  never re-tokenized into the queue.

So the question "are CovRL's queue entries lossless under
re-tokenization" doesn't arise. The queue *is* tokens. The bytes are
ephemeral and per-execution. Losslessness is trivially true by
absence of the operation.

---

## 5. What TLAFL's paper specifies

USENIX '21 paper, §4.2 ("Implementation") and Figure 2 describe a
*one-way* pre-parser pipeline:

> The pre-processor runs the following steps:
> **Rename** ... **Renumber** ... **Token Analysis** ... **Encoding**

And:

> Mutations are slightly modified to work on an array of 16-bit numbers
> rather than an array of bytes ... **Decoding**: the input is decoded
> immediately before executing the input in the target JavaScript
> interpreter.

The architecture is: source → preprocessor → u16 → mutate → decode →
target. No step in the cycle reads the decoded bytes back as a fresh
source. There is no round-trip to constrain.

So the paper's claim isn't "lossless round-trip" — it's "tokens are
the only persistent state; bytes are a one-shot derived view." That's
the property autotokens does not match (it persists bytes, not tokens).
autotokens substitutes a different property (closed-vocab convergent
byte round-trip) that is genuinely useful but architecturally
different.

---

## 6. Refined claim

> autotokens is an AFL++-idiomatic adaptation of Token-Level AFL's
> strategic idea (mutate at token level with a closed vocabulary,
> decode for execution). It is *not* a lossless implementation in the
> strict bytes→tokens→bytes identity sense: comments are stripped on
> initial ingestion, the emit step injects whitespace between adjacent
> multi-char tokens that may not have been adjacent in the cache, and
> the "identifier" class deliberately blurs dots and slashes into
> identifier runs. What autotokens *does* preserve is **closed
> vocabulary** (every byte written is the canonical encoding of a
> known token) and **convergence after one cycle** (further round
> trips are byte-stable). TLAFL/CovRL avoid the question entirely by
> persisting tokens directly; bytes are an ephemeral per-execution
> view.

The systems are not interchangeable. The earlier framing in
`aligning_with_covrl.md` §2 — "valid lossless implementation" — was
too strong. Read it as "valid TLAFL-style adaptation, with the loss
profile catalogued here."

---

## 7. Implications for rllm

Implementation planning is deliberately out of scope here (the user
asked to defer it). One paragraph of relevance to keep this doc
self-contained:

> If a future rllm follows autotokens' pattern (byte queue +
> in-process token cache + canonical-byte emit), it inherits a
> bounded-loss profile *specific to its tokenizer*. For
> CodeT5+ byte-level BPE that profile is different from autotokens'
> (no comment stripping, no whitespace-injection repair; but partial
> multi-byte tokens at span boundaries become a candidate loss class
> instead). If rllm instead stores u16 tokens directly via
> `fuzz()`/`post_process` (the Option B path in
> `aligning_with_covrl.md` §3.2), the round-trip question is sidestepped
> the same way CovRL/TLAFL sidestep it. The cost/benefit of those two
> paths is in `aligning_with_covrl.md` §3.3. This doc only sharpens
> the *cost side* of the autotokens-pattern option.

---

## 8. References

- `~/Documents/AFLplusplus/custom_mutators/autotokens/autotokens.cpp`
  (lines cited above).
- `~/Documents/CovRL-Fuzz/AFL/afl-fuzz.c:2526-2554`
  (`decode`), `:3179-3233` (`save_if_interesting`), `:5008-5070`
  (u16 in/out buffers in `fuzz_one`).
- Salls et al., *Token-Level Fuzzing*, USENIX Security '21, §4.2 and
  Figure 2.
- Eom et al., *CovRL-Fuzz*, ISSTA '24 — implementation discussion in
  §4 (forks AFL 2.52b, retains the Token-Level AFL queue format).
- Sibling doc: [`aligning_with_covrl.md`](aligning_with_covrl.md) — the
  broader architectural context; this doc refines its §2 claim.
