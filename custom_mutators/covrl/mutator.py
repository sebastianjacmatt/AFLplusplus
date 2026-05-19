from collections import deque
from typing import Optional

from .config import Config


class Mutator:
    def __init__(self, cfg: Config, collection: str, trainer: str):
        assert collection in ("interesting", "all")
        assert trainer in ("covrl", "rllm")
        assert cfg.group_size >= 1
        assert cfg.fuzz_count % cfg.group_size == 0, (
            "fuzz_count must be a multiple of group_size"
        )
        self.cfg = cfg
        self.collection = collection
        self.trainer_kind = trainer

        self._pending = deque()
        self._last = None
        self._group_id = -1
        self._queue_get_count = 0

        self.masking = None
        self.model = None
        self.rollout = None
        self.trainer = None

    def init(self, seed: int) -> None:
        from .masking import LLMModel, Masking
        from .rollout import RolloutBuffer

        self.masking = Masking(self.cfg)
        self.rollout = RolloutBuffer(self.cfg)
        self.model = LLMModel(self.cfg)
        self.trainer = self._build_trainer()

    def queue_get(self, filename: str) -> bool:
        self._queue_get_count += 1
        if self._queue_get_count % self.cfg.finetune_every == 0:
            self.trainer.finetune(self.rollout)
        return True

    def fuzz_count(self, buf: bytearray) -> int:
        tokens = self.masking.tokenize(bytes(buf))
        num_groups = self.cfg.fuzz_count // self.cfg.group_size

        masks, group_ids = [], []
        for _ in range(num_groups):
            self._group_id += 1
            mask = self.masking.sample_mask(tokens)
            masks.extend([mask] * self.cfg.group_size)
            group_ids.extend([self._group_id] * self.cfg.group_size)

        xs = [self.masking.encode_input(m) for m in masks]
        ys = self.model.batch_generate(xs, max_length=self.cfg.max_output_length)

        self._pending.clear()
        for mask, x, y, gid in zip(masks, xs, ys, group_ids):
            self._pending.append((mask, x, y, gid))
        return self.cfg.fuzz_count

    def fuzz(self, buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
        if not self._pending:
            self.fuzz_count(buf)
        mask, x, y, gid = self._pending.popleft()
        out = self.masking.decode_output(mask, y, max_size=max_size)
        self._last = (x, y, out, gid)
        return bytearray(out)

    def post_run(self) -> None:
        from .utils import calc_reward, read_coverage, validity

        x, y, out, gid = self._last
        R = calc_reward(validity(out), read_coverage())
        self.rollout.add(x, y, R, group_id=gid)
        if self.collection == "all":
            self.rollout.commit_last()

    def queue_new_entry(self, new: str, orig: Optional[str]) -> None:
        if self.collection == "interesting":
            self.rollout.commit_last()

    def deinit(self) -> None:
        self.rollout.flush()
        if self.model is not None:
            self.model.save(self.cfg.checkpoint_dir)

    def _build_trainer(self):
        from .base_trainer import CovRLTrainer, RLLMTrainer
        from .policy_gradient_algorithms import grpo, ppo

        if self.trainer_kind == "covrl":
            return CovRLTrainer(self.cfg, self.model, policy_loss=ppo.loss)
        loss = grpo.loss if self.cfg.__dict__.get("policy_loss") == "grpo" else ppo.loss
        return RLLMTrainer(self.cfg, self.model, policy_loss=loss)