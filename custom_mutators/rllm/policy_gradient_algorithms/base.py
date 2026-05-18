"""Protocol and shared helpers for policy-gradient strategies.

See docs/design.md §2.4 (Strategy pattern) and §3.6.
"""

from typing import TYPE_CHECKING, Optional, Protocol

import torch
from torch.utils.data import Dataset, Sampler

if TYPE_CHECKING:
    from config import Config


LABEL_IGNORE = -100


class PolicyGradientAlgorithm(Protocol):
    """Structural interface implemented by PPO / GRPO / future algorithms.

    Two hooks (docs/design.md §2.5, §6.3):

    * ``loss`` — the policy-gradient loss for one batch.
    * ``make_sampler`` — optional custom training sampler. Default ``None``
      tells ``BaseTrainer`` to defer to HF Trainer's default.
    """

    def loss(
        self,
        inputs: dict[str, torch.Tensor],
        model_out,
        ref_out,
        cfg: "Config",
    ) -> torch.Tensor: ...

    def make_sampler(
        self,
        dataset: Dataset,
        cfg: "Config",
    ) -> Optional[Sampler]: ...


def token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = LABEL_IGNORE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-prob of labels under logits. Returns (logprobs, mask)."""
    mask = (labels != ignore_index).float()
    safe = labels.masked_fill(labels == ignore_index, 0)
    lp = (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, safe.unsqueeze(-1))
        .squeeze(-1)
    )
    return lp, mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def clipped_surrogate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Per-token PPO-style clipped surrogate (lower bound)."""
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return torch.min(surr1, surr2)
