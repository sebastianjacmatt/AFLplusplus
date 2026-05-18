"""PPO loss (clipped surrogate + entropy bonus + KL penalty).

Critic-free formulation: advantages are reward minus the batch-mean
baseline (REINFORCE with baseline) rather than GAE over a value head.
``cfg.ppo.gae_lambda`` and ``cfg.ppo.value_coef`` are placeholders for a
future value-head extension and are not consumed here.
"""

from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import Dataset, Sampler

from .base import clipped_surrogate, masked_mean, token_logprobs

if TYPE_CHECKING:
    from config import Config


class PPOAlgorithm:
    def make_sampler(self, dataset: Dataset, cfg: "Config") -> Optional[Sampler]:
        return None

    def pre_finetune(self, dataset, cfg) -> None:
        return None

    def save_auxiliary(self, output_dir: str) -> None:
        return None

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
        advantages = rewards - rewards.mean()
        advantages = advantages.unsqueeze(1).expand_as(ratio)

        surrogate = clipped_surrogate(ratio, advantages, cfg.clip_epsilon)
        policy_loss = -masked_mean(surrogate, mask)

        kl = masked_mean(new_lp - ref_lp, mask)
        entropy = -masked_mean(new_lp, mask)

        return (
            policy_loss
            + cfg.kl_coef * kl
            - cfg.ppo.entropy_coef * entropy
        )
