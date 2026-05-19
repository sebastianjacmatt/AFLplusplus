import json
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class Config:
    model_path: str = ""
    tokenizer_path: str = ""
    rewarder_path: str = ""

    fuzz_count: int = 32
    group_size: int = 1
    iter_cycle: int = 1000
    finetune_every: int = 1000

    mask_ratio: float = 0.15
    max_input_length: int = 512
    max_output_length: int = 256

    batch_size: int = 8
    learning_rate: float = 1e-5
    ppo_epsilon: float = 0.2
    grpo_epsilon_std: float = 1e-8

    corpus_dir: str = ""
    checkpoint_dir: str = "./checkpoints"
    rollout_path: str = "./rollouts.jsonl"

    device: str = "cuda"
    seed: int = 0

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        path = path or os.environ.get("COVRL_CONFIG")
        if not path:
            return cls()
        with open(path, "r") as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)