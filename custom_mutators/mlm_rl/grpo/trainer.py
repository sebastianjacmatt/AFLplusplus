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

    def __init__(
        self,
        actor,
        tokenizer,
        device,
        save_dir="./covrl_checkpoints",
        train_batch_size=4,
        learning_rate=2e-5,
        mask_probability=0.15,
        n_showmap_workers=8,
        group_size=8,
    ):
        raise NotImplementedError("GRPOTrainer is not yet implemented")

    def finetune(self):
        raise NotImplementedError

    def get_actor(self):
        raise NotImplementedError
