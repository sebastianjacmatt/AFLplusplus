"""AFL++ Python custom mutator: rllm.

Primary adapter at the AFL++ boundary. This is the *only* file in the
package that is aware of the AFL++ Python ABI; every other module is a
plain Python component composed here in `init()`. See docs/design.md
(Hexagonal Architecture, §2.1; AFL++ ABI conformance, §6).
"""

import os

from base_trainer import BaseTrainer
from config import Config, load_config
from mutator import Mutator
from policy_gradient_algorithms import build_algorithm
from rewarder import Rewarder, StderrValidityRewarder, TFIDFCoverageRewarder
from rollout import RolloutBuffer, RolloutDataset


CFG: Config
BUFFER: RolloutBuffer
MUTATOR: Mutator
REWARDER: Rewarder
TRAINER: BaseTrainer

_queue_get_count: int = 0


def init(seed: int) -> None:
    global CFG, BUFFER, MUTATOR, REWARDER, TRAINER

    CFG = load_config()

    stderr_path = os.environ.get("RLM_STDERR_FILE")
    if not stderr_path:
        raise RuntimeError("RLM_STDERR_FILE is not set")

    BUFFER = RolloutBuffer(CFG)
    MUTATOR = Mutator(CFG, BUFFER, seed=seed)
    REWARDER = Rewarder(
        CFG,
        tf_idf=TFIDFCoverageRewarder(
            bitmap_size=CFG.bitmap_size,
            alpha=CFG.idf_alpha,
        ),
        validity=StderrValidityRewarder(
            stderr_path=stderr_path,
        ),
    )
    TRAINER = BaseTrainer(CFG, MUTATOR, algorithm=build_algorithm(CFG))


def fuzz_count(buf: bytearray) -> int:
    MUTATOR.mask(buf)
    return CFG.fuzz_count


def fuzz(buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
    return bytearray(MUTATOR.mutate(max_size))


def post_run() -> None:
    MUTATOR.collect(REWARDER)


def queue_get(filename: str) -> bool:
    global _queue_get_count
    _queue_get_count += 1
    if _queue_get_count % CFG.finetune_every == 0:
        REWARDER.update_cycle()
        TRAINER.finetune(RolloutDataset(BUFFER))
    return True


def deinit() -> None:
    TRAINER.save_checkpoint()
    BUFFER.flush()
