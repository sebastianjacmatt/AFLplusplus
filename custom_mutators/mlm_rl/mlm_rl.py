#!/usr/bin/env python
# encoding: utf-8
"""
AFL++ custom mutator implementing the CovRL staged fuzzing scheme.

This port replaces the original CovRL inter-process TCP architecture with a
single AFL++ Python custom mutator loaded via AFL_PYTHON_MODULE. The tokenized
representation of each selected seed is stored once in fuzz_count() so that
each fuzz() call applies a fresh mask-based mutation without re-tokenizing.

Deviations from original CovRL:
  - TODO:Splice mode omitted: less mutation diversity, deferred for a working baseline.
  - TODO:Adaptive energy scheduling omitted: N must be returned before mutations run,
    so the dynamic stage_max doubling from the original havoc loop cannot be
    replicated in fuzz_count() through python API, can maybe be implemented with c-api bridge
  - Little-endian u16 seed store not used: AFL++ owns the queue and passes inputs
    as standard bytearray. Token <-> byte conversion is internal to the mutator.

Usage::

    # 1. Build AFL++ if not already done
    cd /path/to/AFLplusplus && make

    # 2. Install Python dependencies
    pip install torch transformers

    # 3. Run — PYTHONPATH must point to the directory containing mlm_rl.py,
    #    AFL_PYTHON_MODULE is the module name without .py,
    #    AFL_CUSTOM_MUTATOR_ONLY=1 suppresses all AFL++ byte-level mutations
    #    so only the token-level CovRL mutations run.
    PYTHONPATH=/path/to/AFLplusplus/custom_mutators/covrl \
    AFL_PYTHON_MODULE=covrl \
    AFL_CUSTOM_MUTATOR_ONLY=1 \
    afl-fuzz -i <seeds_dir> -o <output_dir> -- <js_interpreter> @@

    # Example with jerry:
    PYTHONPATH=/path/to/AFLplusplus/custom_mutators/covrl \
    AFL_PYTHON_MODULE=covrl \
    AFL_CUSTOM_MUTATOR_ONLY=1 \
    afl-fuzz -i seeds/ -o out/ -- /path/to/jerry @@

@author:     Sebastian Jacobsen Matthews
@contact:    sebastianjacmatt@gmail.com

@license:
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import random

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from covrl_trainer import PPOTrainer

# ---------------------------------------------------------------------------
# CovRL constants — taken from CovRL AFL 2.52b config.h / afl-fuzz.c.
# ---------------------------------------------------------------------------

MODEL_NAME = "Salesforce/codet5p-220m"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# CovRL SYNC_INTERVAL=100 — seeds between finetune triggers.
# Named FINETUNE_INTERVAL here to avoid confusion with AFL++ SYNC_INTERVAL=8,
# which is an entirely different concept (inter-fuzzer queue sync).
FINETUNE_INTERVAL  = 100

# Fixed mutation budget returned by fuzz_count() for every seed.
# TODO: revisit once adaptive energy scheduling.
FUZZ_COUNT         = 128

MASK_TOKEN         = 4      # sentinel token inserted at mutation positions
MASK_COUNT         = 3      # max masks inserted per mutation step

# TODO: understand contrastive search
# Contrastive search parameters (from CovRL sample_config.json)
MODEL_MAX_LENGTH   = 768    # model_max_length in config
MASK_PROBABILITY   = 0.15   # fraction of tokens masked; sets max_length ceiling
N_SAMPLES          = 32     # top_k for contrastive search
PENALTY_ALPHA      = 0.6    # degeneration penalty for contrastive search

# Directory where actor/critic checkpoints are written after each finetune cycle.
# Override via environment variable or AFL_PYTHON_MODULE_XTRA (TODO: Stage 2).
SAVE_DIR = "./covrl_checkpoints"


# ---------------------------------------------------------------------------
# Module-level state — all initialised in init()
# ---------------------------------------------------------------------------

# Set once in init()
CONFIG        = None   # loaded config object  TODO:
ACTOR         = None   # AutoModelForSeq2SeqLM (loaded in init)
TOKENIZER     = None   # AutoTokenizer         (loaded in init)
UNKNOWN_TOKEN = None   # TOKENIZER.unk_token_id (set in init after tokenizer loads)
TRAINER       = None   # PPOTrainer (created in init, owns _finetune_cycle_index)

# Updated in queue_get() / fuzz_count()
_current_seed_token_ids  = None   # token ids from the most recent fuzz_count()
# TODO: _call_index              = 0      # fuzz() call index within the current seed's window
_finetune_pending        = False  # set by queue_get(), consumed by fuzz_count()

# Finetune bookkeeping
_pending_new_queue_files = []     # accumulated by queue_new_entry()
# NOTE: _finetune_cycle_index is owned by TRAINER, not tracked here.


# ---------------------------------------------------------------------------
# AFL++ custom mutator hooks
# ---------------------------------------------------------------------------

def init(seed):
    """
    Called once when AFLFuzz starts up.

    @type seed: int
    @param seed: A 32-bit random value
    """
    # TODO: remove/make reason about reproducability random.seed(seed)

    global CONFIG, ACTOR, TOKENIZER, UNKNOWN_TOKEN, TRAINER
    global _current_seed_token_ids
    global _finetune_pending
    global _pending_new_queue_files

    TOKENIZER     = AutoTokenizer.from_pretrained(MODEL_NAME)
    ACTOR         = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    UNKNOWN_TOKEN = TOKENIZER.unk_token_id
    ACTOR.eval()

    assert TOKENIZER.mask_token_id == MASK_TOKEN, (
        f"TOKENIZER.mask_token_id={TOKENIZER.mask_token_id} does not match "
        f"MASK_TOKEN={MASK_TOKEN}; update the MASK_TOKEN constant to match"
    )

    # TODO: load CONFIG; derive SAVE_DIR from env / AFL_PYTHON_MODULE_XTRA
    TRAINER = PPOTrainer(
        actor=ACTOR,
        tokenizer=TOKENIZER,
        device=DEVICE,
        save_dir=SAVE_DIR,
    )

    _current_seed_token_ids  = None
    _finetune_pending        = False
    _pending_new_queue_files = []


def deinit():
    """
    Called once before AFLFuzz exits.
    """
    pass


def queue_get(filename):
    """
    Called at the beginning of each fuzz iteration.

    Increments the per-seed counter and marks a finetune as pending every
    FINETUNE_INTERVAL non-skipped queue entries. AFL++ handles favored/was_fuzzed
    skip logic internally; this hook always returns True.

    @type filename: str
    @param filename: File name of the test case in the current queue entry

    @rtype: bool
    @return: Always True
    """
    global _finetune_pending

    # finetun    
    if len(_pending_new_queue_files) % FINETUNE_INTERVAL == 0:
        _finetune_pending = True

    return True


def fuzz_count(buf):
    """
    Called when AFL++ selects a seed. Returns FUZZ_COUNT: the number of times
    fuzz() will be invoked for this seed. 
    TODO: FUZZ_COUNT should be dependent on adaptive seed scheduling

    Triggers any pending finetune cycle before tokenizing (mirrors original
    placement: sync_fuzzers() runs after fuzz_one() completes and before the
    next seed is processed).

    Tokenizes buf once and the result; fuzz() won't re-tokenize.

    @type buf: bytearray
    @param buf: Raw seed bytes from AFL++

    @rtype: int
    @return: Number of fuzz() calls to schedule
    """
    global _current_seed_token_ids, _finetune_pending #, _call_index # TODO: _call_index needed for splice/scheduling

    # Trigger finetune before processing the new seed
    if _finetune_pending:
        _finetune_pending = False
        _finetune(corpus_dir=None)  # TODO pass/define AFL queue dir TODO: we need an efficient storage solution

    # Tokenize once — fuzz() reuses _current_seed_token_ids
    _current_seed_token_ids = _tokenize(buf)
    
    # TODO: _call_index = 0 # _call_index needed for splice/scheduling
    
    # TODO: use *adaptive energy scheduling* through c-bridge custom_mutator api, if exec/sec is slow
    return FUZZ_COUNT


def fuzz(buf, add_buf, max_size):
    """
    Called once per fuzzing iteration. Applies a fresh mask-based mutation to
    the token sequence and returns the re-encoded result.

    buf is not re-tokenized here — the token representation was derived in
    fuzz_count() and is reused across all N calls for this seed.
    TODO: add_buf is reserved for splice mode

    @type buf: bytearray
    @param buf: Current seed bytes (not re-tokenized)

    TODO:
    @type add_buf: bytearray
    @param add_buf: Second seed for splice mode (Stage 1: unused)

    TODO:
    @type max_size: int
    @param max_size: Maximum byte length of the returned buffer

    @rtype: bytearray
    @return: Mutated seed bytes
    """
    # TODO: global _call_index

    if _current_seed_token_ids is None:
        raise RuntimeError("fuzz() called without a seed token sequence")

    # _call_index += 1 # TODO: needed for unimplemented splicing/scheduling

    # Fresh copy per call — _current_seed_token_ids is never mutated in-place
    base = list(_current_seed_token_ids)

    masked = _random_mask(base)

    # skip sequences too short to mask meaningfully (< 3 tokens)
    if len(masked) <= 3:
        return buf

    infilled_ids = _actor_infill(masked)

    out_buf = _encode(infilled_ids)

    if len(out_buf) > max_size:
        out_buf = out_buf[:max_size]

    return bytearray(out_buf)


def queue_new_entry(filename_new_queue, filename_orig_queue):
    """
    Called after AFL++ adds a new test case to the queue.

    Accumulates new queue filenames for the next finetune cycle.

    @type filename_new_queue: str
    @param filename_new_queue: Path to the new queue entry

    @type filename_orig_queue: str
    @param filename_orig_queue: Path to the originating queue entry
    """
    global _pending_new_queue_files

    _pending_new_queue_files.append({
        "new":  filename_new_queue,
        "orig": filename_orig_queue,
    })


# ---------------------------------------------------------------------------
# Internal mutation helpers
# ---------------------------------------------------------------------------

def _random_mask(token_ids):
    """
    Apply one random mask mutation to a token sequence.

    Randomly selects between two modes (equal probability) matching the
    original CovRL havoc loop:
      RANDOM_INSERT   (mode 0): insert 1–MASK_COUNT MASK_TOKENs at random positions
      RANDOM_OVERWRITE (mode 1): replace 1–MASK_COUNT tokens with MASK_TOKEN
    """
    result = list(token_ids)
    mode   = random.randint(0, 1)

    if mode == 0:
        # RANDOM_INSERT: insert fresh mask tokens without removing existing tokens
        count = random.randint(1, MASK_COUNT)
        for _ in range(count):
            pos = random.randint(0, len(result))
            result.insert(pos, MASK_TOKEN)
    else:
        # RANDOM_OVERWRITE: overwrite existing tokens in-place
        if result:
            count     = random.randint(1, min(MASK_COUNT, len(result)))
            positions = random.sample(range(len(result)), count)
            for pos in positions:
                result[pos] = MASK_TOKEN

    return result


def _actor_infill(masked_token_ids, sample_method="greedy"):
    """
    Sentinel conversion → pad → generate → reconstruct for a token sequence.

    Mirrors Inferencer.masking() + _generate_predictions() + reconstruct().

    Sentinel conversion: scan for MASK_TOKEN and UNKNOWN_TOKEN, replace each
    with a unique top-of-vocabulary sentinel (vocab_size-1, vocab_size-2, ...).
    mask_dict mirrors Inferencer.masking(): {vocab_size-i: original_pos, ...}.

    Padding: append one PADDING token with attention_mask=0, matching
    Inferencer._pad_input_ids(). Required for generate() to behave correctly.

    Reconstruction mirrors Inferencer.reconstruct(): collect predicted tokens
    after each sentinel into result_dict. Span terminates on any token with
    id > vocab_size-100 (stray near-top-vocab tokens / unrecognised sentinels)
    or == eos_token_id. Rebuild sequence by interleaving original tokens between
    masked positions with predicted spans; append remaining tokens up to PADDING.

    sample_method="greedy" (default): fast, good for testing.
    sample_method="contrastive": do_sample=True, penalty_alpha=0.6, top_k=32,
        no_repeat_ngram_size=3, min_length=1 — matches original CovRL Inferencer.

    # TODO: implement chunking (split at MODEL_MAX_LENGTH - 3 = 765 tokens,
    # process each chunk independently, concatenate results) to handle seeds
    # longer than the model's context window. See Inferencer.inference() and
    # Inferencer.split_sentences() for the reference implementation.
    # Until then, sequences longer than 765 tokens will error at generate().
    """
    vocab_size = TOKENIZER.vocab_size

    # ----- sentinel conversion -----
    converted      = list(masked_token_ids)
    mask_positions = []
    sentinel_n     = 0

    for i, tok in enumerate(converted):
        if tok == MASK_TOKEN or tok == UNKNOWN_TOKEN:
            mask_positions.append(i)
            sentinel_n  += 1
            converted[i] = vocab_size - sentinel_n

    mask_dict = {vocab_size - i: pos for i, pos in enumerate(mask_positions, 1)}

    if not mask_dict:
        return masked_token_ids

    if len(converted) > MODEL_MAX_LENGTH - 3:
        print(
            f"[covrl] WARNING: token sequence length {len(converted)} exceeds model "
            f"context window {MODEL_MAX_LENGTH - 3}; chunking is not yet implemented"
        )

    # ----- pad + attention mask -----
    padded    = converted + [TOKENIZER.pad_token_id]
    attn_mask = [1] * len(converted) + [0]
    input_ids_t = torch.tensor([padded],    dtype=torch.long, device=DEVICE)
    attn_mask_t = torch.tensor([attn_mask], dtype=torch.long, device=DEVICE)

    # ----- generate -----
    max_pred_len = round(MODEL_MAX_LENGTH * MASK_PROBABILITY)

    with torch.no_grad():
        if sample_method == "contrastive":
            outputs = ACTOR.generate(
                input_ids=input_ids_t,
                attention_mask=attn_mask_t,
                do_sample=True,
                penalty_alpha=PENALTY_ALPHA,
                top_k=N_SAMPLES,
                eos_token_id=TOKENIZER.eos_token_id,
                no_repeat_ngram_size=3,
                min_length=1,
                max_length=max_pred_len,
            )
        else:
            outputs = ACTOR.generate(
                input_ids=input_ids_t,
                attention_mask=attn_mask_t,
                top_k=N_SAMPLES,
                eos_token_id=TOKENIZER.eos_token_id,
                no_repeat_ngram_size=3,
                max_length=max_pred_len,
            )

    predictions = outputs.tolist()[0]

    # ----- reconstruct -----
    result_dict  = {mask: (pos, []) for mask, pos in mask_dict.items()}
    prev_mask_id = None

    for pred in predictions:
        if pred in mask_dict:
            prev_mask_id = pred
        elif pred > (vocab_size - 100) or pred == TOKENIZER.eos_token_id:
            prev_mask_id = None
        elif prev_mask_id is not None:
            result_dict[prev_mask_id][1].append(pred)

    new_inputs = []
    prev_pos   = 0
    for pos, preds in sorted(result_dict.values()):
        new_inputs.extend(padded[prev_pos:pos] + preds)
        prev_pos = pos + 1

    remaining = padded[prev_pos:]
    pad_id    = TOKENIZER.pad_token_id
    trim      = next((i for i, t in enumerate(remaining) if t == pad_id), len(remaining))
    new_inputs.extend(remaining[:trim])

    if not new_inputs:
        raise ValueError("reconstruction produced an empty token sequence")

    return new_inputs


def _tokenize(buf):
    """
    Decode an AFL++ byte buffer to UTF-8 text and tokenize to a list of token IDs.

    This is the only entry point for tokenization. fuzz() must not call this.
    errors="replace" keeps the mutator alive on seeds with non-UTF-8 bytes;
    replacement characters will be treated as unknown tokens by the model.
    """
    text = buf.decode("utf-8", errors="replace")
    # add_special_tokens=False prevents T5's tokenizer from appending EOS.
    # The inferencer receives raw token sequences without special token affixes;
    # padding is added per-chunk inside _infill_chunk via _pad_input_ids logic.
    return TOKENIZER.encode(text, add_special_tokens=False)


def _encode(token_ids):
    """
    Detokenize a token ID sequence back to UTF-8 bytes for AFL++.
    """
    js_text = TOKENIZER.decode(token_ids, skip_special_tokens=True)
    return js_text.encode("utf-8")

# ---------------------------------------------------------------------------
# Finetune cycle
# ---------------------------------------------------------------------------

def _finetune(corpus_dir):
    """
    Delegate one staged CovRL finetuning cycle to TRAINER, then hot-swap ACTOR.

    All data loading, reward computation, and dataset construction are the
    trainer's responsibility.  mlm.py passes only corpus_dir.

    @type  corpus_dir: str or None
    @param corpus_dir: Path to the AFL++ output queue directory.
                       None until Stage 2 corpus loading is implemented.
    """
    global _pending_new_queue_files

    TRAINER.finetune(corpus_dir)
    _reload_actor()

    _pending_new_queue_files = []


def _reload_actor():
    """
    Hot-swap ACTOR with the model returned by TRAINER after a finetune cycle.
    """
    global ACTOR

    ACTOR = TRAINER.get_actor()
    ACTOR.eval()


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

def load_config():
    # TODO (Stage 2): load from env var path or AFL_PYTHON_MODULE_XTRA argument
    return config
