"""Policy-gradient Strategy implementations and the build_algorithm factory.

The factory absorbs the ``cfg.policy_gradient_algorithm`` branch so the
adapter in ``rllm.py`` stays algorithm-agnostic (docs/design.md §3.7).

When ``cfg.use_critic`` is True the selected algorithm is wrapped in a
``CriticDecorator`` (docs/design2.md §2), giving PPO+Critic or GRPO+Critic
without modifying either base algorithm.
"""

from typing import TYPE_CHECKING

from .base import PolicyGradientAlgorithm
from .critic_decorator import CriticDecorator
from .grpo import GRPOAlgorithm
from .ppo import PPOAlgorithm

if TYPE_CHECKING:
    from config import Config


def build_algorithm(cfg: "Config") -> PolicyGradientAlgorithm:
    if cfg.policy_gradient_algorithm == "ppo":
        algo: PolicyGradientAlgorithm = PPOAlgorithm()
    elif cfg.policy_gradient_algorithm == "grpo":
        algo = GRPOAlgorithm()
    else:
        raise ValueError(
            f"Unknown policy_gradient_algorithm: {cfg.policy_gradient_algorithm!r}"
        )

    if cfg.use_critic:
        from critic import Critic
        algo = CriticDecorator(algo, Critic(cfg.critic_cfg), cfg)

    return algo


__all__ = [
    "PolicyGradientAlgorithm",
    "PPOAlgorithm",
    "GRPOAlgorithm",
    "CriticDecorator",
    "build_algorithm",
]
