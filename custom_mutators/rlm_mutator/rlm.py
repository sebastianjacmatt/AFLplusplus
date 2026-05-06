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
IDF: Optional[OnlineIDF] = None
MODEL_CFG: Optional[ModelConfig] = None
AFL_CFG: Optional[AFLConfig] = None
TRAIN_CFG: Optional[TrainingConfig] = None

_trace_bits_view = None
_exit_code_path: Optional[str] = None
_queue_get_count: int = 0
_finetune_pending: bool = False

# post_run diagnostics: tally exit-code outcomes so we can verify the hook
# pipeline without a per-call log flood. Summaries emit every N post_runs.
_post_run_calls: int = 0
_exit_hits:      int = 0
_exit_miss:      int = 0
_exit_code_hist: dict[int, int] = {}
_POST_RUN_LOG_EVERY: int = 256


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
    rollout_logger = RolloutLogger(
        output_dir = TRAINER.args.output_dir,
        enabled    = TRAIN_CFG.enable_logging,
        log_fn     = TRAINER.log,
    )
    BUFFER = RolloutBuffer(logger=rollout_logger)
    MUTATOR = Mutator(TRAINER, BUFFER, AFL_CFG, TRAIN_CFG)
    IDF = OnlineIDF(bitmap_size=AFL_CFG.bitmap_size, alpha=AFL_CFG.idf_alpha)

    # Prefer RLM_EXIT_FILE from the shell wrapper — it's guaranteed to be in
    # AFL's env before the forkserver starts. Fall back to a /tmp path only
    # when unset (e.g. when running outside run_afl.sh).
    _exit_code_path = os.environ.get("RLM_EXIT_FILE")
    if not _exit_code_path:
        _exit_code_path = f"/tmp/rlm_exit_{os.getpid()}"
        os.environ["RLM_EXIT_FILE"] = _exit_code_path
        log.warning("[rlm] RLM_EXIT_FILE not set by shell; using fallback %s", _exit_code_path)

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

    if MUTATOR is None:
        raise RuntimeError("fuzz() called before init()")

    _clear_exit_code_file()
    out = MUTATOR.fuzz_one(max_size)
    return out



def post_run() -> None:
    """Assemble scalar reward from bitmap novelty and exit status."""
    global _trace_bits_view
    global _post_run_calls, _exit_hits, _exit_miss

    if MUTATOR is None or IDF is None:
        raise RuntimeError("post_run() called before init()")

    if _trace_bits_view is None:
        if AFL_CFG is None:
            raise RuntimeError("post_run() called before AFL config is available")
        _trace_bits_view = attach_trace_bits(AFL_CFG.bitmap_size)

    bitmap = _trace_bits_view.copy()
    cov_reward = IDF.reward(bitmap)  # always updates IDF state

    exit_code = _read_exit_code()
    reward = _shape_reward(cov_reward, exit_code)
    MUTATOR.on_post_run(reward, coverage_reward=cov_reward, exit_code=exit_code)

    _post_run_calls += 1
    if exit_code is None:
        _exit_miss += 1
    else:
        _exit_hits += 1
        _exit_code_hist[exit_code] = _exit_code_hist.get(exit_code, 0) + 1

    if _post_run_calls % _POST_RUN_LOG_EVERY == 0:
        hit_rate = _exit_hits / _post_run_calls if _post_run_calls else 0.0
        log.info(
            "[rlm] post_run tally: calls=%d hits=%d miss=%d hit_rate=%.2f hist=%s last(code=%s reward=%.4f cov=%.4f)",
            _post_run_calls, _exit_hits, _exit_miss, hit_rate,
            dict(sorted(_exit_code_hist.items())),
            exit_code, reward, cov_reward,
        )


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
    finally:
        _clear_exit_code_file()


def _clear_exit_code_file() -> None:
    """Remove any stale exit-code file before/after a child execution."""
    if _exit_code_path is None:
        return
    try:
        os.remove(_exit_code_path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("[rlm] failed to remove exit-code file %s: %s", _exit_code_path, exc)


def _shape_reward(cov_reward: float, exit_code: int | None) -> float:
    """Preserve some coverage signal even for invalid executions.

    A hard gate to -1.0 collapses most invalid samples onto one reward, which
    kills GRPO group variance. Instead:
      - valid executions keep the full coverage reward
      - invalid executions keep a scaled coverage component minus a penalty
      - missing exit codes receive a slightly larger penalty
    """
    if AFL_CFG is None:
        raise RuntimeError("_shape_reward() called before AFL config is available")

    if exit_code == 0:
        return cov_reward

    penalty = (
        AFL_CFG.missing_exit_penalty
        if exit_code is None
        else AFL_CFG.invalid_exit_penalty
    )
    return AFL_CFG.invalid_coverage_scale * cov_reward - penalty
