from dataclasses import dataclass, field
from typing import Optional

import torch

from config.afl    import AFLConfig
from config.covrl  import CovRLConfig
from config.grpo   import GRPOConfig
from config.lora   import LoRAConfig


@dataclass
class Config:
    # Model
    model_name: str = "Salesforce/codet5p-220m"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Tokenisation / generation
    mask_token: int       = 4       # T5 sentinel; must match TOKENIZER.mask_token_id
    model_max_length: int = 768
    mask_probability: float = 0.15
    n_samples: int        = 32      # top_k for contrastive search
    penalty_alpha: float  = 0.6     # degeneration penalty for contrastive search
    sample_method: str    = "greedy"  # "greedy" | "contrastive"

    # Sub-configs — AFL is always required; exactly one of covrl/grpo must be set.
    afl:   AFLConfig            = field(default_factory=AFLConfig)
    covrl: Optional[CovRLConfig] = field(default_factory=CovRLConfig)
    grpo:  Optional[GRPOConfig]  = None
    lora:  Optional[LoRAConfig]  = None

    def __post_init__(self):
        if self.covrl is not None and self.grpo is not None:
            raise ValueError("Config: set exactly one of covrl or grpo, not both")
        if self.covrl is None and self.grpo is None:
            raise ValueError("Config: one of covrl or grpo must be set")
