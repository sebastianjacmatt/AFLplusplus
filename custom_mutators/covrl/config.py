import json
import os
from dataclasses import dataclass, fields, asdict
from typing import Optional


@dataclass
class Config:
    model_path: str = ""
    tokenizer_path: str = ""
    rewarder_path: str = ""

    # AFL scheduling
    fuzz_count: int = 32
    group_size: int = 1
    iter_cycle: int = 1000
    finetune_every: int = 1000

    # T5 span corruption
    mask_ratio: float = 0.15
    mean_span_length: float = 3.0
    min_span_length: int = 1
    max_span_length: int = 5
    max_input_length: int = 512

    # Generation
    max_new_tokens_per_span: int = 16
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-5
    ppo_epsilon: float = 0.2
    grpo_epsilon_std: float = 1e-8

    # Paths / runtime
    corpus_dir: str = ""
    checkpoint_dir: str = "./checkpoints"
    rollout_path: str = "./rollouts.jsonl"
    device: str = "auto"
    seed: int = 0

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

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