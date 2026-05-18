"""Policy-gradient Strategy implementations and the build_algorithm factory.

The factory absorbs the ``cfg.policy_gradient_algorithm`` branch so the
adapter in ``rllm.py`` stays algorithm-agnostic (docs/design.md §3.7).
Adding a new algorithm = add a class here + add a sub-config dataclass
in ``config.py`` + extend this factory. No other file changes.
"""

from typing import TYPE_CHECKING

from .base import PolicyGradientAlgorithm
from .grpo import GRPOAlgorithm
from .ppo import PPOAlgorithm

if TYPE_CHECKING:
    from config import Config


def build_algorithm(cfg: "Config") -> PolicyGradientAlgorithm:
    if cfg.policy_gradient_algorithm == "ppo":
        return PPOAlgorithm()
    if cfg.policy_gradient_algorithm == "grpo":
        return GRPOAlgorithm()
    raise ValueError(
        f"Unknown policy_gradient_algorithm: {cfg.policy_gradient_algorithm!r}"
    )


__all__ = ["PolicyGradientAlgorithm", "PPOAlgorithm", "GRPOAlgorithm", "build_algorithm"]
