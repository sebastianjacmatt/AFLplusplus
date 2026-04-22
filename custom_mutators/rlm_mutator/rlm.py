#!/usr/bin/env python
# encoding: utf-8
"""AFL++ custom mutator adapter for rlm_mutator.

This module is intentionally thin. It owns only:

    - AFL++ hook callbacks: init, deinit, queue_get, fuzz_count, fuzz, post_run
    - SHM attachment and TF-IDF reward assembly
    - Exit-code recovery via exit_hook.so output file
    - Finetune scheduling based on queue_get cadence

Mutation logic lives in mutator.py.
Model ownership lives in base_trainer.py and its PPO/GRPO subclasses.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

from config import AFLConfig, ModelConfig, TrainingConfig, load_config
from mutator import Mutator
from rewarding import OnlineIDF, attach_trace_bits
from rollout import RolloutBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singletons — assigned in init()
# ---------------------------------------------------------------------------

TRAINER: Optional[object] = None
BUFFER: Optional[RolloutBuffer] = None
MUTATOR: Optional[Mutator] = None
IDF: Optional[OnlineIDF] = None
MODEL_CFG: Optional[ModelConfig] = None
AFL_CFG: Optional[AFLConfig] = None
TRAIN_CFG: Optional[TrainingConfig] = None

_trace_bits_view = None
_exit_code_path: Optional[str] = None
_queue_get_count: int = 0
_finetune_pending: bool = False


# ---------------------------------------------------------------------------
# Trainer resolution
# ---------------------------------------------------------------------------

def _resolve_trainer_cls(algorithm: str):
    """Resolve PPO/GRPO trainer class.

    Falls back to BaseTrainer if ppo.py or grpo.py has not yet been created.
    This keeps rlm.py importable while the loss subclasses are still stubs or
    absent. Finetune will then raise from BaseTrainer.compute_loss once called.
    """
    try:
        if algorithm == "grpo":
            from grpo import GRPOTrainer
            return GRPOTrainer
        from ppo import PPOTrainer
        return PPOTrainer
    except ImportError:
        from base_trainer import BaseTrainer
        log.warning(
            "[rlm] %s trainer module not found; falling back to BaseTrainer",
            algorithm.upper(),
        )
        return BaseTrainer


# ---------------------------------------------------------------------------
# AFL++ hooks
# ---------------------------------------------------------------------------

def init(seed: int) -> None:
    """Called once when AFL++ starts the Python custom mutator."""
    global TRAINER, BUFFER, MUTATOR, IDF, MODEL_CFG, AFL_CFG, TRAIN_CFG
    global _trace_bits_view, _exit_code_path, _queue_get_count, _finetune_pending

    MODEL_CFG, AFL_CFG, TRAIN_CFG = load_config()
    random.seed(seed)

    trainer_cls = _resolve_trainer_cls(TRAIN_CFG.algorithm)
    TRAINER = trainer_cls(MODEL_CFG, TRAIN_CFG)
    BUFFER = RolloutBuffer()
    MUTATOR = Mutator(TRAINER, BUFFER, AFL_CFG, TRAIN_CFG)
    IDF = OnlineIDF(bitmap_size=AFL_CFG.bitmap_size, alpha=AFL_CFG.idf_alpha)

    _trace_bits_view = attach_trace_bits(AFL_CFG.bitmap_size)

    # Set before the forkserver starts so the child inherits it.
    _exit_code_path = f"/tmp/rlm_exit_{os.getpid()}"
    os.environ["RLM_EXIT_FILE"] = _exit_code_path

    _queue_get_count = 0
    _finetune_pending = False

    log.info(
        "[rlm] init complete — model=%s device=%s bitmap=%d algorithm=%s",
        MODEL_CFG.model_name_or_path,
        MODEL_CFG.resolve_device(),
        AFL_CFG.bitmap_size,
        TRAIN_CFG.algorithm,
    )


def deinit() -> None:
    """Called once before AFL++ exits."""
    global BUFFER
    if BUFFER is not None and len(BUFFER) > 0:
        flushed = BUFFER.flush()
        if flushed:
            log.info("[rlm] deinit — dropped %d completed records", len(flushed))



def splice_optout() -> bool:
    """Disable AFL++ splice mode; add_buf remains unused in fuzz()."""
    return True



def queue_get(filename: str) -> bool:
    """Called by AFL++ when selecting the next seed from the queue."""
    global _queue_get_count, _finetune_pending
    _queue_get_count += 1

    if AFL_CFG is None:
        raise RuntimeError("queue_get() called before init()")

    if _queue_get_count > 0 and _queue_get_count % AFL_CFG.finetune_every == 0:
        _finetune_pending = True

    return True



def fuzz_count(buf: bytearray) -> int:
    """Prepare one seed for a fixed number of fuzz() calls."""
    global _finetune_pending

    if MUTATOR is None or AFL_CFG is None:
        raise RuntimeError("fuzz_count() called before init()")

    if _finetune_pending:
        _finetune_pending = False
        MUTATOR.maybe_finetune()

    MUTATOR.on_new_seed(buf)
    return AFL_CFG.fuzz_count



def fuzz(buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
    """Generate one mutation through Mutator and return bytes to AFL++."""
    del add_buf

    if MUTATOR is None:
        raise RuntimeError("fuzz() called before init()")

    out = MUTATOR.generate(max_size)
    return bytearray(out) if out is not None else buf



def post_run() -> None:
    """Assemble scalar reward from bitmap novelty and exit status."""
    if MUTATOR is None or IDF is None or _trace_bits_view is None:
        raise RuntimeError("post_run() called before init()")

    bitmap = _trace_bits_view.copy()
    cov_reward = IDF.reward(bitmap)  # always updates IDF state

    exit_code = _read_exit_code()
    reward = cov_reward if exit_code == 0 else -1.0
    MUTATOR.on_post_run(reward, coverage_reward=cov_reward, exit_code=exit_code)


# ---------------------------------------------------------------------------
# Exit-code helper
# ---------------------------------------------------------------------------

def _read_exit_code() -> int | None:
    """Read the exit code written by exit_hook.so to RLM_EXIT_FILE.

    Returns None when the file is absent or unreadable, which is treated as a
    crash / abnormal termination by post_run().
    """
    if _exit_code_path is None:
        return None
    try:
        with open(_exit_code_path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None
