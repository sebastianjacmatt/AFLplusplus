# Masking strategy: span vs single-token

CovRL-Fuzz reports ~21.5% pre-training validity on JerryScript (Table 7,
"LLM w/o CovRL"). Our equivalent baseline currently sits at ~6-8%. The gap
is structural: CovRL applies a fundamentally more conservative masking
strategy at fuzz time than we do. This document explains exactly what
differs and why it lowers validity.

The non-obvious fact: CovRL uses **two different masking strategies** —
one at training time, one at fuzzing time. The training one matches ours.
The fuzzing one does not, and that is what produces the higher baseline.

## CovRL — training time: T5 span corruption (matches rllm)

`CovRL-Fuzz/covrl/models/actor_dataset.py:random_spans_noise_mask`
(lines 30-61) implements standard T5-style span-corruption masking with
`mask_probability=0.15` and `poisson_lambda=3.0`. This is the objective
CodeT5+ was pretrained on and is structurally identical to our
`data/masking.py:_random_spans_noise_mask` (line 206). No daylight here —
both pretrain/finetune against multi-token contiguous spans collapsed
into single sentinels.

## CovRL — fuzzing time: 1-3 single-token masks per mutation

The paper (§3.1, "Phase 1. Mutation by Mask") describes three operations
— Insert, Overwrite, Splice — but their implementation lives in
`CovRL-Fuzz/AFL/afl-fuzz.c`, not in any Python module. Two constants
dominate (`afl-fuzz.c:337-338`):

```c
#define MASK_TOKEN 4
#define MASK_COUNT 3
```

Per mutation, AFL picks one operation and applies it 1-3 times
(`afl-fuzz.c:5190-5221`):

```c
u32 mutation_mode = UR(2);

if (mutation_mode == RANDOM_INSERT) {
    u32 insert_count = UR(MASK_COUNT) + 1;     // 1..3 inserts
    /* insert a single MASK_TOKEN at each chosen position */
}
else if (mutation_mode == RANDOM_OVERWRITE) {
    u32 overwrite_count = UR(MASK_COUNT) + 1;  // 1..3 overwrites
    /* replace one token with MASK_TOKEN at each chosen position */
}
```

Each `MASK_TOKEN` byte in the buffer maps to a separate T5 sentinel in the
Python inferencer
(`CovRL-Fuzz/covrl/models/inferencer.py:_mask_unknowns`, lines 89-98):

```python
for i, token in enumerate(input_ids):
    if token in {self.MASK_TOKEN, self.UNKNOWN_TOKEN}:
        mask_positions.append(i)
        mask_cnt += 1
        input_ids[i] = self.tokenizer.vocab_size - mask_cnt
```

So a CovRL mutation produces **1, 2, or 3 sentinels**, each marking **one
token position** (or one insertion point). Splice (`afl-fuzz.c:5305+`,
the "last-resort" stage) places exactly two sentinels around a spliced-in
statement — same shape: a small handful of single-position sentinels per
mutation.

## rllm: T5 span corruption at both train and fuzz time

We do not distinguish. `data/masking.py:Masking.mask` (line 123) runs T5
span corruption for every mutation, parameterised by `corruption_rate=0.15`
and `mean_span_length=3.0` (`config.py:45-48`).

For a 100-token program this produces approximately:

- 15 masked tokens total (15% × 100)
- collected into ≈ 5 spans of ≈ 3 tokens each
- ≈ 5 sentinels, each replacing a 1-5 token *span*

Span sampling lives in `data/masking.py:_random_spans_noise_mask` (line
206) and `_random_word_spans_noise_mask` (line 248).
`model/tokenizer.py:reconstruct` (line 62) splices the model's
sentinel-delimited output back into the original positions.

## Why the difference lowers baseline validity

Per mutation, on the same 100-token program:

|                 | sentinels | original tokens deleted | model task                          |
|-----------------|-----------|-------------------------|-------------------------------------|
| CovRL Insert    | 1-3       | 0                       | fill 1-3 single insertion points    |
| CovRL Overwrite | 1-3       | 1-3                     | fill 1-3 single deleted tokens      |
| rllm span       | ≈ 5       | ≈ 15                    | fill ≈ 5 multi-token deletion holes |

Two compounding factors hurt validity:

**1. Many more holes per mutation.** If the model fills any one hole
correctly with probability *p*, the mutation succeeds with *p^k* where
*k* is the number of holes. CovRL has 1-3 holes; we have ~5. Five-vs-one
compounds quickly even with a strong model.

**2. Each hole is structurally larger.** CovRL's Overwrite removes one
leaf token at a time. Our span masking removes contiguous runs of up to
5 tokens, which can swallow entire syntactic constructs — an argument
list, a return expression, the head of a `for` loop. The model has to
regenerate not just the leaves but the brackets, commas, and operators
that held them together, with no internal context to anchor against.

Worked example. `function foo(a, b) { return a + b; }`:

- CovRL Overwrite (2 random single-token positions):
  `function foo([M], b) { return a + [M]; }`
  — bracketing intact, model picks two leaf-like tokens
- rllm span (one 4-token span):
  `function foo([S1]) { return a + b; }`
  — model must emit `a, b` *and* keep the parens/comma structure that
  was destroyed by the span

The first task is nearly trivial for a code-trained LM. The second
requires correct bracket balancing across a multi-token gap with no
internal context. Per-hole validity drops, and that drop is compounded
across multiple holes.

## Implications

This is a deliberate design choice, not an oversight:

1. **rllm's masking matches CodeT5+'s pretraining objective.** Both T5
   and CodeT5+ were pretrained on span-corruption MLM with similar
   parameters. Tellingly, CovRL also uses T5 span corruption during
   their *training* phase (`actor_dataset.py:random_spans_noise_mask`) —
   they only switch to the 1-3 single-token approach at inference time.

2. **The validity/diversity tradeoff is real.** The CovRL paper itself
   notes (§ Problem) that LLM mutators "predict common tokens and
   unintentionally reduce diversity." Their Coverage-Weighted Rewarding
   exists to navigate this tradeoff. A more aggressive base masking
   gives the eventual policy gradient more room to learn.

3. **Training will close most of the gap.** With RL fine-tuning, the
   validity signal (-1.0 syntax / -0.5 semantic) directly penalises
   low-validity output. Starting from 6-8% means the reward landscape
   has strong negative signal early in training, which is useful for
   bootstrapping.

If raising baseline validity becomes a hard constraint — for example to
bootstrap GRPO from a less brittle warm start — the right knob is **how
many positions are masked per mutation**, not the span shape. Dropping
`corruption_rate` from 0.15 to ~0.05 and `mean_span_length` from 3.0 to
~1.5 shrinks our per-mutation mask footprint toward CovRL's 1-3 region
while preserving the T5 span-corruption form that matches pretraining.
