"""Dr.GRPO — GRPO without the normalization biases (Liu et al. 2025).

Two fixes vs GRPO: drop the **/std** in the group advantage, and drop the
per-response **length normalization** in the loss. The length fix is already our
default (`GRPOHF._aggregate` = masked-sum, no length division), so Dr.GRPO is
exactly the advantage swap — the whole variant is this file:
"""

from __future__ import annotations

from training.advantages import drgrpo_advantage
from training.grpo import GRPOTrainer


class DrGRPOTrainer(GRPOTrainer):
    advantage_fn = staticmethod(drgrpo_advantage)
