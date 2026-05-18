"""GRPO loss (group-relative policy optimisation).

Advantages are computed by normalising rewards *within* groups of
``cfg.grpo.group_size`` consecutive samples — no value head required.
The batch is required to contain whole groups; ``config.py`` enforces
``train_batch_size % grpo.group_size == 0`` at load time.
"""

from typing import TYPE_CHECKING, Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler

from .base import clipped_surrogate, masked_mean, token_logprobs

if TYPE_CHECKING:
    from config import Config


class GRPOSampler(Sampler[int]):
    """Yield indices that keep every contiguous block of ``group_size`` together.

    Group order is shuffled each epoch; within-group order is fixed. The
    dataset is required to satisfy ``len(dataset) % group_size == 0`` —
    ``config.load_config`` guarantees this for both ``train_batch_size``
    and ``fuzz_count``. See docs/design.md §6.4.
    """

    def __init__(
        self,
        dataset: Dataset,
        group_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.n = len(dataset)  # type: ignore[arg-type]
        self.group_size = group_size
        self.generator = generator
        if self.n % group_size != 0:
            raise ValueError(
                f"dataset length {self.n} is not a multiple of group_size {group_size}"
            )

    def __iter__(self) -> Iterator[int]:
        num_groups = self.n // self.group_size
        order = torch.randperm(num_groups, generator=self.generator).tolist()
        for g in order:
            start = g * self.group_size
            for i in range(self.group_size):
                yield start + i

    def __len__(self) -> int:
        return self.n


class GRPOAlgorithm:
    def make_sampler(self, dataset: Dataset, cfg: "Config") -> Sampler:
        return GRPOSampler(dataset, cfg.grpo.group_size)

    def loss(
        self,
        inputs: dict[str, torch.Tensor],
        model_out,
        ref_out,
        cfg: "Config",
    ) -> torch.Tensor:
        labels = inputs["labels"]
        new_lp, mask = token_logprobs(model_out.logits, labels)
        ref_lp, _ = token_logprobs(ref_out.logits, labels)
        old_lp = inputs["old_logprobs"]

        ratio = torch.exp(new_lp - old_lp)

        rewards = inputs["rewards"]
        advantages = self._group_relative_advantages(rewards, cfg)
        advantages = advantages.unsqueeze(1).expand_as(ratio)

        surrogate = clipped_surrogate(ratio, advantages, cfg.clip_epsilon)
        policy_loss = -masked_mean(surrogate, mask)

        kl = masked_mean(new_lp - ref_lp, mask)

        return policy_loss + cfg.kl_coef * kl

    @staticmethod
    def _group_relative_advantages(
        rewards: torch.Tensor, cfg: "Config"
    ) -> torch.Tensor:
        G = cfg.grpo.group_size
        B = rewards.shape[0]
        if B % G != 0:
            raise ValueError(
                f"batch size {B} is not a multiple of group_size {G}"
            )

        grouped = rewards.view(B // G, G)
        means = grouped.mean(dim=1, keepdim=True)
        stds = grouped.std(dim=1, keepdim=True)
        norm = (grouped - means) / (stds + cfg.grpo.norm_epsilon)
        advantages = norm.reshape(B)

        if cfg.grpo.advantage_clip > 0.0:
            advantages = torch.clamp(
                advantages,
                -cfg.grpo.advantage_clip,
                cfg.grpo.advantage_clip,
            )
        return advantages
