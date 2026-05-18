"""CriticDecorator: wraps any PolicyGradientAlgorithm with critic-guided reward.

Phase 1 (pre_finetune)  — trains the critic as a supervised CE classifier on
                          the collected rollout data.
Phase 2 (loss)          — queries the frozen critic; blends predicted and raw
                          rewards; delegates the actual loss to the wrapped
                          algorithm (PPO or GRPO).
make_sampler            — delegated unchanged, so GRPOSampler is inherited.
save_auxiliary          — checkpoints the critic.

See docs/design2.md §2 (Decorator pattern) and §3.2.
"""

import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler

from rollout import RolloutCollator

if TYPE_CHECKING:
    from config import Config
    from critic import Critic
    from rollout import RolloutDataset
    from .base import PolicyGradientAlgorithm

# T5 pad token is 0 for all codet5p-family models.
_T5_PAD_ID = 0


class CriticDecorator:
    """Decorator that adds critic-guided reward to any PolicyGradientAlgorithm.

    The wrapped algorithm's ``loss`` and ``make_sampler`` are delegated
    without modification — GRPO group ordering is therefore inherited
    automatically (docs/design2.md §2).
    """

    def __init__(
        self,
        wrapped: "PolicyGradientAlgorithm",
        critic: "Critic",
        cfg: "Config",
    ) -> None:
        self.wrapped = wrapped
        self.critic = critic
        self._critic_opt = optim.AdamW(
            critic.parameters(),
            lr=cfg.critic_cfg.critic_lr,
        )
        device = torch.device(cfg.resolve_device())
        self.critic.to(device)
        self._device = device

    # ------------------------------------------------------------------
    # Phase 1: critic training
    # ------------------------------------------------------------------

    def pre_finetune(
        self,
        dataset: "RolloutDataset",
        cfg: "Config",
    ) -> None:
        """Train the critic for one cycle on the collected rollout.

        Uses actual generated tokens (from the rollout) as input and the
        raw environment reward converted to a bucket label as the target.
        This trains on the same distribution the critic evaluates during
        actor training (design2.md §6, deviation from CovRL).
        """
        loader = DataLoader(
            dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            collate_fn=RolloutCollator(pad_token_id=_T5_PAD_ID),
        )
        self.critic.train()
        for _ in range(cfg.critic_cfg.critic_epochs):
            for batch in loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                critic_ids, critic_mask = self._build_critic_inputs(batch)
                targets = self._rewards_to_labels(batch["rewards"])
                loss, _ = self.critic(critic_ids, critic_mask, labels=targets)
                self._critic_opt.zero_grad()
                loss.backward()
                self._critic_opt.step()
        self.critic.eval()

        # Forward to wrapped algorithm (no-op for PPO / GRPO).
        self.wrapped.pre_finetune(dataset, cfg)

    # ------------------------------------------------------------------
    # Phase 2: actor loss with critic-blended rewards
    # ------------------------------------------------------------------

    def loss(
        self,
        inputs: dict[str, torch.Tensor],
        model_out,
        ref_out,
        cfg: "Config",
    ) -> torch.Tensor:
        # Critic predicts reward bucket for the actor's current greedy output.
        cur_tokens = model_out.logits.argmax(dim=-1)        # (B, T_labels)
        critic_ids, critic_mask = self._build_critic_inputs(inputs, cur_tokens)
        with torch.no_grad():
            critic_logits = self.critic(critic_ids, critic_mask)   # (B, num_labels)
        pred_buckets = critic_logits.argmax(dim=-1)
        critic_rewards = torch.tensor(
            [self.critic.label_to_reward(b.item()) for b in pred_buckets],
            dtype=torch.float32,
            device=inputs["rewards"].device,
        )

        # Blend: weight=1.0 → pure critic (CovRL); weight=0.0 → raw reward.
        w = cfg.critic_cfg.reward_weight
        blended = w * critic_rewards + (1.0 - w) * inputs["rewards"]

        # Delegate to wrapped algorithm with substituted rewards.
        modified = {**inputs, "rewards": blended}
        base_loss = self.wrapped.loss(modified, model_out, ref_out, cfg)

        # Optional MLM regularisation. model_out.loss is T5's seq-to-seq CE
        # loss (computed by T5ForConditionalGeneration when labels are passed).
        # cfg.mlm_coef defaults to 0.0, so this term vanishes unless set.
        return base_loss + cfg.mlm_coef * model_out.loss

    # ------------------------------------------------------------------
    # Sampler — pure delegation so GRPOSampler is inherited
    # ------------------------------------------------------------------

    def make_sampler(
        self,
        dataset,
        cfg: "Config",
    ) -> Optional[Sampler]:
        return self.wrapped.make_sampler(dataset, cfg)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_auxiliary(self, output_dir: str) -> None:
        """Save the critic backbone and classifier head to output_dir/critic."""
        save_path = os.path.join(output_dir, "critic")
        os.makedirs(save_path, exist_ok=True)
        self.critic.backbone.save_pretrained(save_path)
        torch.save(
            self.critic.classifier.state_dict(),
            os.path.join(save_path, "classifier.pt"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_critic_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        gen_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate masked context with generation for critic input.

        ``labels`` carries -100 for padding positions (HF convention).
        We strip those to 0 (T5 pad token) and mask them out in the
        attention mask so the critic does not attend to label-padding.
        """
        raw_labels = inputs["labels"]              # (B, T) with -100 padding
        gen_mask   = (raw_labels >= 0).long()      # 1 = real token, 0 = padding

        if gen_tokens is None:
            # Critic training: use actual rollout tokens.
            token_ids = raw_labels.clamp(min=0)    # -100 → 0 (T5 pad)
        else:
            # Actor training: use current model's greedy predictions,
            # zeroed at positions where labels are padded.
            token_ids = gen_tokens * gen_mask

        critic_ids  = torch.cat([inputs["input_ids"],      token_ids], dim=1)
        critic_mask = torch.cat([inputs["attention_mask"], gen_mask],  dim=1)
        return critic_ids, critic_mask

    def _rewards_to_labels(self, rewards: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [self.critic.reward_to_label(r.item()) for r in rewards],
            dtype=torch.long,
            device=rewards.device,
        )
