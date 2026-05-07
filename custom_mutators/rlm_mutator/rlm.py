#!/usr/bin/env python
# encoding: utf-8
"""AFL++ custom mutator adapter for rlm_mutator.

This module is intentionally thin. It owns only:

    - AFL++ hook callbacks: init, deinit, queue_get, fuzz_count, fuzz, post_run
    - Rewarder construction and exit-code path setup
    - Finetune scheduling based on queue_get cadence

Mutation logic lives in mutator.py.
Reward observation/shaping lives in rewarding.py.
Model ownership lives in base_trainer.py and its PPO/GRPO subclasses.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

from config import AFLConfig, ModelConfig, TrainingConfig, load_config
from mutator import Mutator
from rewarding import CoverageRewarder, OnlineIDF
from rollout import RolloutBuffer, RolloutLogger

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
REWARDER: Optional[CoverageRewarder] = None
MODEL_CFG: Optional[ModelConfig] = None
AFL_CFG: Optional[AFLConfig] = None
TRAIN_CFG: Optional[TrainingConfig] = None

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
    global TRAINER, BUFFER, MUTATOR, REWARDER, MODEL_CFG, AFL_CFG, TRAIN_CFG
    global _queue_get_count, _finetune_pending

    MODEL_CFG, AFL_CFG, TRAIN_CFG = load_config()
    random.seed(seed)

    trainer_cls = _resolve_trainer_cls(TRAIN_CFG.algorithm)
    TRAINER = trainer_cls(MODEL_CFG, TRAIN_CFG)
    rollout_logger = RolloutLogger(
        output_dir = TRAINER.args.output_dir,
        enabled    = TRAIN_CFG.enable_logging,
        log_fn     = TRAINER.log,
    )
    BUFFER = RolloutBuffer(logger=rollout_logger)
    MUTATOR = Mutator(TRAINER, BUFFER, AFL_CFG, TRAIN_CFG)
    # Prefer RLM_EXIT_FILE from the shell wrapper — it's guaranteed to be in
    # AFL's env before the forkserver starts. Fall back to a /tmp path only
    # when unset (e.g. when running outside run_afl.sh).
    exit_code_path = os.environ.get("RLM_EXIT_FILE")
    if not exit_code_path:
        exit_code_path = f"/tmp/rlm_exit_{os.getpid()}"
        os.environ["RLM_EXIT_FILE"] = exit_code_path
        log.warning("[rlm] RLM_EXIT_FILE not set by shell; using fallback %s", exit_code_path)

    REWARDER = CoverageRewarder(
        idf = OnlineIDF(
            bitmap_size=AFL_CFG.bitmap_size,
            alpha=AFL_CFG.idf_alpha,
        ),
        bitmap_size = AFL_CFG.bitmap_size,
        exit_code_path = exit_code_path,
        invalid_coverage_scale = AFL_CFG.invalid_coverage_scale,
        invalid_exit_penalty   = AFL_CFG.invalid_exit_penalty,
        missing_exit_penalty   = AFL_CFG.missing_exit_penalty,
    )

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
    """Called once before AFL++ exits — flush rollout CSV and save model checkpoint."""
    if MUTATOR is None or BUFFER is None:
        return

    records = BUFFER.flush()
    if records:
        log.info("[rlm] deinit — logged %d remaining rollout records", len(records))

    log.info("[rlm] deinit — saving model checkpoint to %s", MUTATOR.trainer.args.output_dir)
    MUTATOR.trainer.save_model()



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

    if MUTATOR is None or REWARDER is None:
        raise RuntimeError("fuzz() called before init()")

    REWARDER.clear_exit_code()
    out = MUTATOR.fuzz_one(max_size)
    return out



def post_run() -> None:
    """Assemble scalar reward from bitmap novelty and exit status."""
    if MUTATOR is None or REWARDER is None:
        raise RuntimeError("post_run() called before init()")

    reward_result = REWARDER.compute()
    MUTATOR.on_post_run(reward_result)
