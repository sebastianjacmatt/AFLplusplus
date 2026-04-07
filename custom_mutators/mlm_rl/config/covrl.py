from dataclasses import dataclass


@dataclass
class CovRLConfig:
    train_batch_size: int = 4
    learning_rate: float = 2e-5
