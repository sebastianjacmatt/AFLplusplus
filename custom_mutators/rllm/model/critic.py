"""Critic model for CovRL training.

T5 encoder + classification head that maps (masked_input, generated_output)
pairs to discretized reward classes. Trained before the actor each finetune
cycle; its logits drive the PPO advantage estimate in _train_actor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5EncoderModel

NUM_REWARD_CLASSES = 8

# Ordered reward values matching the 8 label indices.
# Labels 0-1 are validity penalties; 2-7 are coverage-weighted positive rewards.
LABEL_TO_REWARD: list[float] = [-1.0, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def score_to_label(score: float) -> int:
    """Map a continuous reward scalar to the nearest label index."""
    if score < -0.5:
        return 0
    if score < 0.0:
        return 1
    # Positive range: 0.5 increments starting at 0.5
    for i, threshold in enumerate([0.5, 0.6, 0.7, 0.8, 0.9], start=2):
        if score <= threshold:
            return i
    return 7


class CriticModel(nn.Module):
    """T5 encoder + 8-class head scoring (input, generation) pairs.

    Input to forward() is the concatenation of the sentinelized encoder
    input and the model's generated token ids — same layout as the CovRL
    reference (covrl/models/critic.py).

    # TODO: consider sharing encoder weights with Model._hf.encoder (model/llm.py)
    #       to halve VRAM; requires passing the live encoder rather than re-loading.
    """

    def __init__(self, model_name_or_path: str, device: str = "auto"):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name_or_path)
        d_model = self.encoder.config.d_model
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, NUM_REWARD_CLASSES)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        cls = hidden[:, 0, :]
        logits = self.head(self.dropout(cls))
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
