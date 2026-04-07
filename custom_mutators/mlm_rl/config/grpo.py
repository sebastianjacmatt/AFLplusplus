from dataclasses import dataclass


@dataclass
class GRPOConfig:
    # Number of mutations sampled per seed for group-relative reward normalisation.
    # Each group shares a group_id (AFL++ src: field); rewards are normalised
    # within the group before the policy update.
    group_size: int = 8

    train_batch_size: int = 4
    learning_rate: float = 2e-5
