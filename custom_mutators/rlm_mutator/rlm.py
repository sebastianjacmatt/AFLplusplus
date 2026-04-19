#!/usr/bin/env python
# encoding: utf-8
"""
AFL++ custom mutator — online TF-IDF RL rollout recorder.

This module owns the AFL++ integration layer only:

    - AFL++ hook callbacks (init, deinit, queue_get, fuzz_count, fuzz, post_run)
    - Seed tokenisation caching (one tokenize per seed, shared across N fuzz calls)
    - Masking strategy (random insert or overwrite with TRAINER.mask_token)
    - Store logging (x_t, y_t, log_prob) and reward patching (from post_run)
    - Finetune scheduling (trigger every AFL_CFG.finetune_every queue_get calls)

Everything model-related lives in trainer.py (Trainer) and is called via the
TRAINER singleton.  Config lives in config.py.

Hook call order per AFL++ fuzzing cycle::

    init(seed)
    queue_get(filename)          ← increment seed counter; schedule finetune
    fuzz_count(buf)              ← tokenize once; trigger finetune if pending
      fuzz(buf, add_buf, max)    ← mask → TRAINER.infill() → store.log() → return
      post_run()                 ← SHM trace_bits + exit code → reward → store.patch_reward()
      ...                          (repeated AFL_CFG.fuzz_count times)

Usage::

    export RLM_CONFIG=/path/to/rlm_config.json   # optional
    AFL_PYTHON_MODULE=rlm \\
    AFL_CUSTOM_MUTATOR_ONLY=1 \\
    afl-fuzz -i seeds/ -o out/ -- /path/to/jerry @@

@author:  Sebastian Matthews
@contact: sebastianjacmatt@gmail.com
@license: MPL 2.0
"""

import logging
import os
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

from config    import AFLConfig, ModelConfig, TrainingConfig, load_config
from rewarding import OnlineIDF, attach_trace_bits
from store     import RLStore
from trainer   import Trainer

# ---------------------------------------------------------------------------
# Module-level singletons — all set in init()
# ---------------------------------------------------------------------------

TRAINER   : Trainer       | None = None
STORE     : RLStore       | None = None
IDF       : OnlineIDF     | None = None
MODEL_CFG : ModelConfig   | None = None
AFL_CFG   : AFLConfig     | None = None
TRAIN_CFG : TrainingConfig| None = None

_trace_bits_view = None   # numpy uint8 view into AFL++ SHM trace_bits
_exit_code_path  = None   # tmpfile path written by exit_hook.so per execution

# Per-cycle mutable state
_pending_sample_id : str  | None = None  # last sample logged; patched in post_run()
_current_token_ids : list | None = None  # token ids of the current seed
_current_masked    : list | None = None  # masked context shared across one group
group              : int  = 0            # group index within the current seed
sample             : int  = 0            # sample index within the current seed
_queue_get_count   : int  = 0
_finetune_pending  : bool = False


# ---------------------------------------------------------------------------
# AFL++ hooks
# ---------------------------------------------------------------------------

def init(seed: int) -> None:
    """
    Called once when AFL++ starts up.

    Loads config, model, tokenizer, SHM coverage view, IDF state, and store.

    @type  seed: int
    @param seed: 32-bit random value supplied by AFL++.
    """
    global TRAINER, STORE, IDF, MODEL_CFG, AFL_CFG, TRAIN_CFG
    global _trace_bits_view, _exit_code_path
    global _pending_sample_id, _current_token_ids, _current_masked
    global group, sample, _queue_get_count, _finetune_pending

    MODEL_CFG, AFL_CFG, TRAIN_CFG = load_config()

    random.seed(seed)

    TRAINER = Trainer(MODEL_CFG, TRAIN_CFG)

    if TRAIN_CFG.kl_coef > 0.0:
        TRAINER.load_ref_model()

    STORE = RLStore()
    IDF   = OnlineIDF(bitmap_size=AFL_CFG.bitmap_size, alpha=AFL_CFG.idf_alpha)

    _trace_bits_view = attach_trace_bits(AFL_CFG.bitmap_size)

    # Set RLM_EXIT_FILE before the forkserver starts so the target inherits it.
    # exit_hook.so intercepts exit() and writes the code here; post_run() reads it.
    _exit_code_path = f"/tmp/rlm_exit_{os.getpid()}"
    os.environ["RLM_EXIT_FILE"] = _exit_code_path

    _pending_sample_id  = None
    _current_token_ids  = None
    _current_masked     = None
    group              = 0
    sample             = 0
    _queue_get_count   = 0
    _finetune_pending   = False

    log.info("[rlm] init complete — model=%s device=%s bitmap=%d algorithm=%s",
             MODEL_CFG.model_name_or_path,
             MODEL_CFG.resolve_device(),
             AFL_CFG.bitmap_size,
             TRAIN_CFG.algorithm)


def deinit() -> None:
    """Called once before AFL++ exits.  Discards any incomplete rollout group."""
    if STORE is not None:
        remaining = STORE.close_group()
        if remaining:
            log.info("[rlm] deinit — %d incomplete records discarded", len(remaining))


def splice_optout() -> bool:
    """Disable AFL++ splice mode; keeps add_buf unused in fuzz()."""
    return True


def queue_get(filename: str) -> bool:
    """
    Called at the start of each fuzz iteration.

    Sets the current parent seed path and schedules a finetune every
    AFL_CFG.finetune_every seeds.

    @type  filename: str
    @param filename: Absolute path to the current AFL++ queue entry.
    @rtype:  bool
    @return: Always True.
    """
    global _queue_get_count, _finetune_pending

    _queue_get_count += 1

    if _queue_get_count % AFL_CFG.finetune_every == 0:
        _finetune_pending = True

    return True


def fuzz_count(buf: bytearray) -> int:
    """
    Called when AFL++ selects a seed.

    Tokenizes the seed once — result is cached in _current_token_ids and reused
    across all AFL_CFG.fuzz_count fuzz() calls for this seed.

    If a finetune cycle is pending it is triggered here, before tokenising, so
    the new model weights are active for the upcoming mutation batch.

    @type  buf: bytearray
    @param buf: Raw seed bytes from AFL++.
    @rtype:  int
    @return: Total number of fuzz() calls AFL++ will schedule for this seed
             (AFL_CFG.fuzz_count).  For GRPO this is a multiple of group_size;
             for PPO it equals the desired sample count directly.
    """
    global _current_token_ids, _current_masked, _finetune_pending, group, sample

    if _finetune_pending:
        _finetune_pending = False
        _run_finetune()

    _current_token_ids = TRAINER.tokenize(buf)
    _current_masked    = None
    group              = 0
    sample             = 0

    return AFL_CFG.fuzz_count


def fuzz(buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
    """
    Called once per fuzzing iteration.

    Applies one mask-based mutation via TRAINER.infill(), logs the sample
    to STORE (reward=None, filled later by post_run()), and returns the
    mutated bytes to AFL++.

    @type  buf:      bytearray
    @param buf:      Current seed bytes (re-tokenising is skipped; uses cache).
    @type  add_buf:  bytearray
    @param add_buf:  Unused; splice_optout() prevents AFL++ from setting it.
    @type  max_size: int
    @param max_size: Maximum byte length AFL++ will accept.
    @rtype:  bytearray
    @return: Mutated seed bytes.
    """
    global _pending_sample_id, _current_masked, group, sample

    if _current_token_ids is None:
        raise RuntimeError("fuzz() called before fuzz_count() — no cached token ids")

    if sample % TRAIN_CFG.group_size == 0:
        _current_masked, _, _ = _random_mask(list(_current_token_ids))
        group += 1

    masked = _current_masked

    if masked is None or len(masked) <= 3:
        _pending_sample_id = None
        return buf

    result = TRAINER.infill(masked)

    out_buf = TRAINER.encode(result.infilled_ids)
    if len(out_buf) > max_size:
        out_buf = out_buf[:max_size]

    ref_lp = (
        TRAINER.ref_logprob(result.x_t, result.y_t)
        if TRAIN_CFG.kl_coef > 0.0
        else None
    )

    sample_id = STORE.new_sample_id()
    STORE.log(
        sample_id    = sample_id,
        group_id     = group,
        x_t          = result.x_t,
        y_t          = result.y_t,
        log_prob     = result.old_logprob,
        ref_log_prob = ref_lp,
    )

    _pending_sample_id = sample_id
    sample += 1
    return bytearray(out_buf)


def post_run() -> None:
    """
    Called by AFL++ after the target process exits and trace_bits is populated.

    Reads edge coverage from the SHM segment and the child's exit code (written
    by exit_hook.so to RLM_EXIT_FILE).  Reward is:

        -1.0              if exit code != 0 or file absent (signal crash)
        R_cov ∈ [0.5, 1]  if exit code == 0 (valid execution, TF-IDF coverage)

    IDF state is always updated from the bitmap regardless of exit code, so
    the frequency weights reflect all executions, not just valid ones.
    """
    global _pending_sample_id

    if _pending_sample_id is None:
        return

    bitmap     = _trace_bits_view.copy()
    cov_reward = IDF.reward(bitmap)   # always update IDF state

    exit_code = _read_exit_code()
    reward    = cov_reward if exit_code == 0 else -1.0

    STORE.patch_reward(_pending_sample_id, reward)
    _pending_sample_id = None


def _read_exit_code() -> int | None:
    """Read the exit code written by exit_hook.so to RLM_EXIT_FILE.

    Returns None if the file is absent (target terminated via signal, not exit()).
    """
    if _exit_code_path is None:
        return None
    try:
        with open(_exit_code_path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Masking strategy (AFL integration concern — stays in rlm.py)
# ---------------------------------------------------------------------------

def _random_mask(token_ids: list[int]) -> tuple[list[int], list[int], int]:
    """Apply one random mask mutation to a token sequence.

    Modes (equal probability):
      0 — RANDOM_INSERT:    insert 1..AFL_CFG.mask_count MASK tokens at random positions
      1 — RANDOM_OVERWRITE: replace 1..AFL_CFG.mask_count tokens in place

    @param token_ids: Original token sequence.
    @return: (masked_ids, mask_positions, mask_mode)
    """
    result = list(token_ids)
    mode   = random.randint(0, 1)
    mask   = TRAINER.mask_token

    if mode == 0:
        count     = random.randint(1, AFL_CFG.mask_count)
        positions = []
        for _ in range(count):
            pos = random.randint(0, len(result))
            result.insert(pos, mask)
            positions.append(pos)
    else:
        positions = []
        if result:
            count     = random.randint(1, min(AFL_CFG.mask_count, len(result)))
            positions = random.sample(range(len(result)), count)
            for pos in positions:
                result[pos] = mask

    return result, positions, mode


# ---------------------------------------------------------------------------
# Finetune scheduling (AFL integration concern — stays in rlm.py)
# ---------------------------------------------------------------------------

def _run_finetune() -> None:
    """Retrieve the completed rollout group and call TRAINER.finetune()."""
    records = STORE.close_group()
    if not records:
        log.info("[rlm] finetune triggered — store empty, skipping")
        return
    TRAINER.finetune(records)
