"""Group-relative advantage strategies for GRPO variants.

Pure functions over a 1-D reward array (the ``G`` infill rewards of one
mask-group). The trainer's ``advantage_fn`` is one of these, selected by config
``method``. No model / torch state here.
"""

from __future__ import annotations

import numpy as np


def zscore_advantage(rewards, eps: float = 1e-6) -> np.ndarray:
    """GRPO default: ``(r - mean) / (std + eps)`` within the group."""
    x = np.asarray(rewards, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def drgrpo_advantage(rewards, eps: float = 1e-6) -> np.ndarray:
    """Dr.GRPO: mean-subtract only — **no std normalization** (drops the
    difficulty/std bias that GRPO's z-score introduces)."""
    x = np.asarray(rewards, dtype=np.float64)
    return x - x.mean()


ADVANTAGES = {"zscore": zscore_advantage, "drgrpo": drgrpo_advantage}

flat_advantage = zscore_advantage   # back-compat alias
