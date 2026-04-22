"""GRPO trainer for rlm_mutator.

Subclass of BaseTrainer that overrides HF Trainer.compute_loss with the
group-relative clipped objective (Shao et al., 2024).  Critic-free: the
advantage is computed from rewards within each group, using the group_id
field logged during mutation.

Behaviour log-probs (`old_log_prob`) are stored by Mutator as mean per-token
log-probabilities.  The current-policy log-prob uses the same convention so
that `ratio = exp(log_prob - old_log_prob)` is a scalar per sample.  We do
not divide the log probabilities beyond that masked mean.
"""

import torch
import torch.nn.functional as F

from base_trainer import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """GRPO finetuner for the mutator policy.

    One masked context x_g is reused across training_cfg.grpo.group_size
    samples (enforced by Mutator).  The group structure survives into
    compute_loss via inputs['group_id'], which RolloutCollator emits.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels         = inputs["labels"]
        old_log_prob   = inputs["old_log_prob"]
        reward         = inputs["reward"]
        group_id       = inputs["group_id"]
        ref_log_prob   = inputs["ref_log_prob"]

        # Forward through the current actor. HF seq2seq models produce logits
        # aligned 1:1 with `labels`; decoder-shift is applied internally.
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        logits = outputs.logits                         # [B, L_y, V]

        # Per-token log pi_theta(y_t | x_t), masked and averaged over non-pad tokens.
        token_logprob = F.log_softmax(logits, dim=-1)
        safe_labels   = labels.masked_fill(labels == -100, 0)
        gathered      = token_logprob.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        mask          = (labels != -100).float()
        seq_len       = mask.sum(dim=-1).clamp(min=1.0)
        log_prob      = (gathered * mask).sum(dim=-1) / seq_len  # [B]

        advantages = self._advantage(reward, group_id)            # [B]

        # Clipped group-relative objective.
        clip_eps      = self.training_cfg.clip_epsilon
        ratio         = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss    = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        kl_coef = self.training_cfg.kl_coef
        if kl_coef > 0.0:
            kl   = log_prob - ref_log_prob                        # token-level log-ratio
            loss = actor_loss + kl_coef * kl.mean()
        else:
            kl   = torch.zeros_like(log_prob)
            loss = actor_loss

        with torch.no_grad():
            self._log_train_scalars(actor_loss, advantages, ratio, kl, reward, group_id, clip_eps)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Group-relative advantage
    # ------------------------------------------------------------------

    def _advantage(self, reward: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        """Per-group z-score normalisation of rewards.

        For each group g present in the batch:
            A_{g,j} = (r_{g,j} - mean(r_g)) / (std(r_g) + norm_epsilon).

        Singleton groups collapse to zero std; the epsilon keeps division
        stable and the advantage is 0 for that sample.
        """
        eps = self.training_cfg.grpo.norm_epsilon
        advantages = torch.zeros_like(reward)
        for gid in group_id.unique():
            sel = (group_id == gid)
            rg  = reward[sel]
            mean_r = rg.mean()
            std_r  = rg.std(unbiased=False) if rg.numel() > 1 else torch.zeros_like(mean_r)
            advantages[sel] = (rg - mean_r) / (std_r + eps)
        return advantages

    # ------------------------------------------------------------------
    # Training-time scalar logging (prefix train/)
    # ------------------------------------------------------------------

    def _log_train_scalars(
        self,
        actor_loss: torch.Tensor,
        advantages: torch.Tensor,
        ratio:      torch.Tensor,
        kl:         torch.Tensor,
        reward:     torch.Tensor,
        group_id:   torch.Tensor,
        clip_eps:   float,
    ) -> None:
        clip_frac = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float().mean().item()
        logs = {
            "train/actor_loss":           actor_loss.item(),
            "train/advantage_mean":       advantages.mean().item(),
            "train/advantage_std":        advantages.std(unbiased=False).item() if advantages.numel() > 1 else 0.0,
            "train/advantage_min":        advantages.min().item(),
            "train/advantage_max":        advantages.max().item(),
            "train/ratio_mean":           ratio.mean().item(),
            "train/ratio_std":            ratio.std(unbiased=False).item() if ratio.numel() > 1 else 0.0,
            "train/ratio_min":            ratio.min().item(),
            "train/ratio_max":            ratio.max().item(),
            "train/kl_mean":              kl.mean().item(),
            "train/clip_fraction":        clip_frac,
            "train/reward_mean_in_batch": reward.mean().item(),
            "train/group_count_in_batch": float(group_id.unique().numel()),
        }
        self.log(logs)
