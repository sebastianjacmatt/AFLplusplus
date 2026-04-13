"""GRPO trainer stub — not yet implemented."""
from abstract_trainer import Trainer


class GRPOTrainer(Trainer):
    """
    Group Relative Policy Optimisation trainer.

    Rewards within each group (same AFL++ src: seed) are normalised before
    the policy update, removing the need for a learned critic baseline.
    This mirrors DeepSeek-R1's GRPO formulation applied to coverage rewards.

    TODO: implement.
    """

    def __init__(self, actor, config):
        raise NotImplementedError("GRPOTrainer is not yet implemented")

    def _prepare_data():
        pass

    def _train_grpo():
        pass

    def _grpo_loss():
        pass

    def finetune(self):
        raise NotImplementedError

    def get_actor(self):
        raise NotImplementedError
