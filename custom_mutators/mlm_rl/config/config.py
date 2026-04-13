from dataclasses import dataclass, field
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# AFL-side config
# ---------------------------------------------------------------------------

@dataclass
class AFLConfig:
    # Seeds processed between each finetune trigger (CovRL SYNC_INTERVAL=100).
    finetune_interval: int = 2

    # Mutation budget returned by fuzz_count() per seed.
    fuzz_count: int = 32

    # Max mask tokens inserted or overwritten per mutation step.
    mask_count: int = 3

    # Root directory for actor/critic checkpoints and showmap tmp files.
    save_dir: str = "./covrl_checkpoints"

    # Parallel afl-showmap threads (ThreadPoolExecutor). subprocess.run releases
    # the GIL during os.waitpid so threads achieve true parallelism.
    n_showmap_workers: int = 8

    # AFL++ coverage map size (2^17 = 131072).
    bitmap_size: int = 131072


# ---------------------------------------------------------------------------
# Training algorithm sub-configs
# ---------------------------------------------------------------------------

@dataclass
class CovRLConfig:
    train_batch_size: int  = 4
    learning_rate:  float  = 2e-5
    idf_alpha:      float  = 0.6     # EMA smoothing factor for IDF update (CovRL Eq. 5)


@dataclass
class GRPOConfig:
    # Mutations sampled per seed for group-relative reward normalisation.
    # Rewards are normalised within the group (same AFL++ src: field) before
    # the policy update — no learned critic baseline needed.
    group_size:      int   = 8
    train_batch_size: int  = 4
    learning_rate:  float  = 2e-5


@dataclass
class LoRAConfig:
    r:             int   = 8
    lora_alpha:    int   = 16
    lora_dropout:  float = 0.1
    # T5/CodeT5+ attention projection names targeted by LoRA adapters.
    target_modules: List[str] = field(default_factory=lambda: ["q", "v"])


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Model
    model_name: str = "Salesforce/codet5p-220m"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Tokenisation / generation
    mask_token:       int   = 4       # T5 sentinel; must match TOKENIZER.mask_token_id
    model_max_length: int   = 768
    mask_probability: float = 0.15
    n_samples:        int   = 32      # top_k for contrastive search
    penalty_alpha:    float = 0.6     # degeneration penalty
    sample_method:    str   = "greedy"  # "greedy" | "contrastive"

    # AFL-side settings (always required)
    afl: AFLConfig = field(default_factory=AFLConfig)

    # Training algorithm — set exactly one
    covrl: Optional[CovRLConfig] = field(default_factory=CovRLConfig)
    grpo:  Optional[GRPOConfig]  = None
    lora:  Optional[LoRAConfig]  = None

    def __post_init__(self):
        if self.covrl is not None and self.grpo is not None:
            raise ValueError("Config: set exactly one of covrl or grpo, not both")
        if self.covrl is None and self.grpo is None:
            raise ValueError("Config: one of covrl or grpo must be set")


# ---------------------------------------------------------------------------
# Active config — edit here to configure a run
# ---------------------------------------------------------------------------

CONFIG = Config(
    afl=AFLConfig(
        finetune_interval=2,
        fuzz_count=32,
        mask_count=3,
        save_dir="./covrl_checkpoints",
        n_showmap_workers=8,
        bitmap_size=131072,
    ),
    covrl=CovRLConfig(
        train_batch_size=4,
        learning_rate=2e-5,
        idf_alpha=0.6,
    ),
    # To switch to GRPO: set covrl=None and uncomment below
    # covrl=None,
    # grpo=GRPOConfig(group_size=8, train_batch_size=4, learning_rate=2e-5),
)
