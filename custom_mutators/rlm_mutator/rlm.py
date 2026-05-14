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

import dataclasses
import json
import logging
import os
import random
import sys
from typing import Optional

from config import AFLConfig, ModelConfig, TrainingConfig, load_config
from mutator import Mutator
from rewarding import (
    ExitCodeRewarder,
    Rewarder,
    StderrValidityRewarder,
    TFIDFCoverageRewarder,
)
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
REWARDER: Optional[Rewarder] = None
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
    # Prefer RLM_EXIT_FILE / RLM_STDERR_FILE from the shell wrapper — they're
    # guaranteed to be in AFL's env before the forkserver starts. Fall back to
    # /tmp paths only when unset (e.g. when running outside run_afl.sh).
    exit_code_path = os.environ.get("RLM_EXIT_FILE")
    if not exit_code_path:
        exit_code_path = f"/tmp/rlm_exit_{os.getpid()}"
        os.environ["RLM_EXIT_FILE"] = exit_code_path
        log.warning("[rlm] RLM_EXIT_FILE not set by shell; using fallback %s", exit_code_path)

    stderr_path = os.environ.get("RLM_STDERR_FILE")
    if not stderr_path:
        stderr_path = f"/tmp/rlm_stderr_{os.getpid()}"
        os.environ["RLM_STDERR_FILE"] = stderr_path
        log.warning("[rlm] RLM_STDERR_FILE not set by shell; using fallback %s", stderr_path)

    REWARDER = Rewarder(
        tf_idf = TFIDFCoverageRewarder(
            bitmap_size = AFL_CFG.bitmap_size,
            alpha       = AFL_CFG.idf_alpha,
        ),
        exit_code = ExitCodeRewarder(
            exit_code_path = exit_code_path,
        ),
        validity = StderrValidityRewarder(
            stderr_path = stderr_path,
        ),
        validity_bonus = AFL_CFG.validity_bonus,
    )

    if TRAIN_CFG.sft_corpus_path:
        if TRAIN_CFG.sft_warmup_steps > 0:
            log.info(
                "[rlm] SFT warmup: %d steps from %s",
                TRAIN_CFG.sft_warmup_steps, TRAIN_CFG.sft_corpus_path,
            )
            # sft_warmup also stores records in TRAINER._mlm_corpus
            TRAINER.sft_warmup(MUTATOR._span_masker, TRAIN_CFG.sft_corpus_path, TRAIN_CFG.sft_warmup_steps)
            TRAINER.snapshot_ref()
            log.info("[rlm] SFT warmup done; pi_ref re-anchored to SFT checkpoint")
        elif TRAIN_CFG.mlm_coef > 0.0:
            # No SFT warmup, but MLM aux loss needs the corpus
            TRAINER.load_mlm_corpus(MUTATOR._span_masker, TRAIN_CFG.sft_corpus_path)

    _queue_get_count = 0
    _finetune_pending = False

    output_dir = TRAINER.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config_out = os.path.join(output_dir, "rlm_config.json")
    with open(config_out, "w") as fh:
        json.dump(
            {
                **dataclasses.asdict(MODEL_CFG),
                **dataclasses.asdict(AFL_CFG),
                **dataclasses.asdict(TRAIN_CFG),
            },
            fh,
            indent=2,
        )


def deinit() -> None:
    """Called once before AFL++ exits — flush rollout CSV and save model checkpoint."""
    if MUTATOR is None or BUFFER is None:
        return

    records = BUFFER.flush()
    if records:
        log.warning("[rlm] deinit — %d rollout records unflushed", len(records))

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

    groups_until_ft = AFL_CFG.finetune_every - (_queue_get_count % AFL_CFG.finetune_every or AFL_CFG.finetune_every)
    m = MUTATOR
    sys.stderr.write(
        "\r[rlm] ft_in=%-2d  valid=%-5s  clip=%-5s  kl=%-7s  usable=%-4s   " % (
            groups_until_ft,
            ("%.1f%%" % (m._last_validity_rate * 100)) if m and m._last_validity_rate is not None else "n/a",
            ("%.3f"  %  m._last_clip_frac)             if m and m._last_clip_frac     is not None else "n/a",
            ("%+.4f" %  m._last_kl_div)                if m and m._last_kl_div        is not None else "n/a",
            str(m._last_usable_groups)                 if m and m._last_usable_groups  is not None else "n/a",
        )
    )
    sys.stderr.flush()

    return True



def fuzz_count(buf: bytearray) -> int:
    """Prepare one seed for a fixed number of fuzz() calls."""
    global _finetune_pending

    if MUTATOR is None or AFL_CFG is None:
        raise RuntimeError("fuzz_count() called before init()")

    if _finetune_pending:
        _finetune_pending = False
        # maybe_finetune advances the IDF snapshot (CovRL Eq. 6) before training,
        # so the next collection phase scores against IDF_t.
        MUTATOR.maybe_finetune(REWARDER)

    MUTATOR.on_new_seed(buf)
    if not MUTATOR.should_fuzz():
        # Mutator marked the seed unfuzzable (e.g., 0 tokens after
        # tokenisation). Returning 0 tells AFL to skip fuzz() for this seed.
        return 0
    return AFL_CFG.fuzz_count



def fuzz(buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
    """Generate one mutation through Mutator and return bytes to AFL++.

    Returns a bytearray (not bytes) so AFL's py_bytes() takes the
    PyByteArray_AsString fast path. Returning bytes makes that call fail
    silently and leaves a stale Python TypeError in the interpreter state
    that surfaces as a delayed segfault during a later C-API operation.
    """
    del add_buf

    if MUTATOR is None or REWARDER is None:
        raise RuntimeError("fuzz() called before init()")

    REWARDER.clear_exit_code()
    out = MUTATOR.fuzz_one(max_size)
    return bytearray(out)



def post_run() -> None:
    """Score the just-executed mutation if Mutator has a pending sample.

    Mutator gates the actual reward + DF work on the pending-sample id so
    AFL's calibration / dry-run / trim stages — which fire post_run without
    a preceding fuzz() — don't pollute TF-IDF or read SHM/exit-file state
    that doesn't correspond to one of our samples.
    """
    if MUTATOR is None or REWARDER is None:
        raise RuntimeError("post_run() called before init()")
    MUTATOR.on_post_run(REWARDER)
