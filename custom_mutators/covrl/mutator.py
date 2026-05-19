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
        self._has_pending_sample = False
        self._group_id = -1
        self._queue_get_count = 0
        self._finetune_pending = False

        self.masking = None
        self.model = None
        self.rollout = None
        self.trainer = None

    def init(self, seed: int) -> None:
        from .masking import LLMModel, Masking
        from .rollout import RolloutBuffer

        self.masking = Masking(self.cfg)
        self.rollout = RolloutBuffer(self.cfg)
        # Share the tokenizer to keep sentinel / pad / eos ids consistent
        # between Masking and LLMModel.
        self.model = LLMModel(self.cfg, tokenizer=self.masking.tokenizer)
        self.trainer = self._build_trainer()

    def queue_get(self, filename: str) -> bool:
        self._queue_get_count += 1
        if self._queue_get_count % self.cfg.finetune_every == 0:
            # TODO: pre-train filtering of incomplete / zero-variance groups
            # and reference-policy snapshot for KL — deferred per baseline CovRL.
            self._finetune_pending = True
        return True

    def fuzz_count(self, buf: bytearray) -> int:
        if self._finetune_pending:
            self._finetune_pending = False
            self.trainer.finetune(self.rollout)

        tokens = self.masking.tokenize(bytes(buf))
        if not tokens:
            raise RuntimeError(
                f"seed tokenises to 0 tokens (buf_len={len(buf)}); "
                "no masking strategy defined for empty input"
            )

        num_groups = self.cfg.fuzz_count // self.cfg.group_size
        group_size = self.cfg.group_size

        # Sample one mask per group; gather all xs + group ids.
        mps = []
        gids: list[int] = []
        for _ in range(num_groups):
            self._group_id += 1
            mp = self.masking.mask(tokens)
            if not mp.spans:
                raise RuntimeError(
                    f"no spans sampled from seed (n_tokens={len(tokens)}); "
                    "seed too short for masking"
                )
            mps.append(mp)
            gids.append(self._group_id)

        # One mega-batch generate() across all groups. PPO (group_size=1)
        # batches across mutations; GRPO (group_size>1) batches across groups
        # AND across the num_return_sequences samples inside each group.
        # Use max(budgets) as max_new_tokens — span counts cluster tightly
        # under fixed mask_ratio, so padding waste is small relative to the
        # throughput win of a single batched call.
        max_new_tokens = max(
            self.masking.generation_budget(mp, self.cfg.max_new_tokens_per_span)
            for mp in mps
        )
        xs = [mp.input_ids for mp in mps]
        ys_flat = self.model.batch_generate(
            xs, n_samples=group_size, max_new_tokens=max_new_tokens
        )

        # HF num_return_sequences layout: [mps[0]·gs, mps[1]·gs, ...].
        # Replicate masks / gids to align.
        mps_flat = [mp for mp in mps for _ in range(group_size)]
        gids_flat = [gid for gid in gids for _ in range(group_size)]
        outs = self.masking.batch_decode(mps_flat, ys_flat)

        self._pending.clear()
        for mp, y, out, gid in zip(mps_flat, ys_flat, outs, gids_flat):
            self._pending.append((mp.input_ids, y, out, gid))
        return self.cfg.fuzz_count

    def fuzz(self, buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
        if not self._pending:
            self.fuzz_count(buf)
        x, y, out, gid = self._pending.popleft()
        if len(out) > max_size:
            raise RuntimeError(
                "generated program exceeds AFL max_size "
                f"(generated={len(out)}, max_size={max_size})"
            )
        self._last = (x, y, out, gid)
        self._has_pending_sample = True
        return bytearray(out)

    def post_run(self) -> None:
        # AFL fires post_run after every target execution (calibration, dry-run,
        # trim included). Without this gate, those non-fuzz runs would attribute
        # their coverage to the last fuzz() sample and pollute the rollout.
        if not self._has_pending_sample:
            return
        self._has_pending_sample = False

        from .utils import calc_reward, read_coverage, validity

        x, y, out, gid = self._last
        R = calc_reward(validity(out), read_coverage())
        # TODO: extend rollout schema with old_logprob (π_old(y|x)) captured at
        # generation time, and switch to a two-phase log/patch_reward flow.
        # Deferred per baseline CovRL — old_logprob is recomputed at train time.
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