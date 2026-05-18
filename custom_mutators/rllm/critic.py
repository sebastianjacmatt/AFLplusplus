"""Discriminative reward model (critic) for CovRL-style Actor-Critic training.

A T5 encoder with a linear classification head that predicts which reward
bucket a (masked_context ∥ generated_tokens) pair will achieve.

The 8-bucket default reproduces CovRL-Fuzz's ``score_to_label`` /
``label_to_score`` from ``covrl/utils/base_utils.py`` exactly.

See docs/design2.md §3.1.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel

if TYPE_CHECKING:
    from config import CriticConfig


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

def score_to_label(reward: float, thresholds: list) -> int:
    """Map a float reward to the first bucket whose upper bound >= reward."""
    for i, t in enumerate(thresholds):
        if reward <= t:
            return i
    return len(thresholds)


def build_label_to_score(bucket_values: list) -> dict:
    """Build the inverse mapping: bucket index → representative scalar."""
    return {i: float(v) for i, v in enumerate(bucket_values)}


# ---------------------------------------------------------------------------
# Critic model
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """T5 encoder + classification head.

    Input:  cat(masked_context_ids, generated_token_ids)
    Output: logits over ``num_labels`` reward buckets

    The encoder's CLS-position representation (index 0 of
    ``last_hidden_state``) is pooled and fed to a Dropout + Linear head.
    CrossEntropyLoss is computed when ``labels`` are provided.
    """

    def __init__(self, cfg: "CriticConfig") -> None:
        super().__init__()
        if cfg.model_name:
            # Warm-start from a pretrained encoder (e.g. codet5p-220m encoder half).
            self.backbone = T5EncoderModel.from_pretrained(cfg.model_name)
        else:
            # Random T5-small backbone — cheaper, matches CovRL's CriticModel.
            self.backbone = T5EncoderModel(T5Config())

        d_model = self.backbone.config.d_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, cfg.num_labels),
        )
        self.loss_fct = nn.CrossEntropyLoss()

        # Bucket helpers — kept on the model so callers don't need the config.
        self._thresholds    = list(cfg.bucket_thresholds)
        self._label_to_score = build_label_to_score(cfg.bucket_values)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0]           # CLS pool → (B, d_model)
        logits = self.classifier(hidden)    # (B, num_labels)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, logits
        return logits

    # ------------------------------------------------------------------
    # Convenience wrappers so callers don't need to hold the config
    # ------------------------------------------------------------------

    def reward_to_label(self, reward: float) -> int:
        return score_to_label(reward, self._thresholds)

    def label_to_reward(self, label: int) -> float:
        return self._label_to_score[label]
