"""Configuration for rllm.

Single unified ``Config`` dataclass plus algorithm-specific sub-configs
(``PPOConfig`` / ``GRPOConfig``). Exactly one sub-config is populated at
load time based on ``policy_gradient_algorithm``.

Usage::

    export RLLM_CONFIG=/path/to/rllm_config.json
    afl-fuzz ...

    cfg = load_config()
"""

import json
import os
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import torch
from transformers import HfArgumentParser


# ---------------------------------------------------------------------------
# Algorithm-specific sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95


@dataclass
class GRPOConfig:
    group_size: int = 8
    norm_epsilon: float = 1e-8
    advantage_clip: float = 0.0


@dataclass
class CriticConfig:
    # Backbone: encoder half of a T5-family model; "" → random T5Config (cheap).
    model_name: str = "Salesforce/codet5p-220m"
    critic_lr: float = 1e-4
    critic_epochs: int = 1
    num_labels: int = 8
    # How much of the reward comes from the critic vs. the raw environment signal.
    # 1.0 = pure critic (CovRL-style); 0.0 = no critic influence.
    reward_weight: float = 1.0
    # len(bucket_thresholds) must equal num_labels - 1.
    bucket_thresholds: list = field(default_factory=lambda: [-0.5, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9])
    # len(bucket_values) must equal num_labels.
    bucket_values: list = field(default_factory=lambda: [-1.0, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


# ---------------------------------------------------------------------------
# Unified config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Model / generation
    model_name_or_path: str = "Salesforce/codet5p-220m"
    device: str = "auto"
    max_length: int = 512
    mask_probability: float = 0.15
    mean_span_length: float = 3.0
    min_span_length: int = 1
    max_span_length: int = 5
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    max_new_tokens_per_mask: int = 16
    whole_word_masking: bool = True

    # AFL++ integration
    fuzz_count: int = 512
    mask_count: int = 3
    mask_strategy: Literal["span", "scatter"] = "span"
    bitmap_size: int = 65536
    idf_alpha: float = 0.6
    finetune_every: int = 64
    validity_bonus: float = 0.0

    # Training (shared)
    policy_gradient_algorithm: Literal["ppo", "grpo"] = "grpo"
    learning_rate: float = 1e-4
    train_batch_size: int = 8
    num_train_epochs: int = 1
    kl_coef: float = 0.0
    clip_epsilon: float = 0.2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: str = "q,v"
    enable_logging: bool = True
    logging_steps: int = 1
    warmup_ratio: float = 0.0
    bf16: bool = False
    ref_update_every: int = 1
    mlm_coef: float = 0.0
    sft_corpus_path: str = ""
    sft_warmup_steps: int = 0
    sft_adapter_path: str = ""

    # Populated by load_config based on policy_gradient_algorithm.
    ppo:  Optional[PPOConfig]  = None
    grpo: Optional[GRPOConfig] = None

    # Critic (optional). Populated by load_config when use_critic=True.
    use_critic:  bool                    = False
    critic_cfg:  Optional[CriticConfig]  = None

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def group_size(self) -> int:
        return self.grpo.group_size if self.grpo is not None else 1

    @property
    def lora_target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.lora_target_modules.split(",") if m.strip()]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_PARSER = HfArgumentParser(Config)


def _sub_kwargs(raw: dict, cls) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def load_config(config_path: str | None = None) -> Config:
    """Load Config from JSON or return dataclass defaults.

    Resolution: explicit arg > RLLM_CONFIG env var > defaults.
    """
    path = config_path or os.environ.get("RLLM_CONFIG", "")
    if path:
        with open(path) as fh:
            raw = json.load(fh)
        (cfg,) = _PARSER.parse_dict(raw, allow_extra_keys=True)
    else:
        raw = {}
        cfg = Config()

    if cfg.policy_gradient_algorithm == "grpo":
        cfg.grpo = GRPOConfig(**_sub_kwargs(raw, GRPOConfig))
        cfg.ppo = None
        if cfg.train_batch_size % cfg.grpo.group_size != 0:
            raise ValueError(
                f"train_batch_size ({cfg.train_batch_size}) must be a multiple of "
                f"grpo.group_size ({cfg.grpo.group_size})."
            )
        if cfg.fuzz_count % cfg.grpo.group_size != 0:
            raise ValueError(
                f"fuzz_count ({cfg.fuzz_count}) must be a multiple of "
                f"grpo.group_size ({cfg.grpo.group_size})."
            )
    elif cfg.policy_gradient_algorithm == "ppo":
        cfg.ppo = PPOConfig(**_sub_kwargs(raw, PPOConfig))
        cfg.grpo = None
    else:
        raise ValueError(
            f"policy_gradient_algorithm must be 'ppo' or 'grpo', "
            f"got {cfg.policy_gradient_algorithm!r}."
        )

    if cfg.use_critic:
        cfg.critic_cfg = CriticConfig(**_sub_kwargs(raw, CriticConfig))
        if len(cfg.critic_cfg.bucket_thresholds) != cfg.critic_cfg.num_labels - 1:
            raise ValueError(
                f"len(bucket_thresholds) ({len(cfg.critic_cfg.bucket_thresholds)}) "
                f"must equal num_labels - 1 ({cfg.critic_cfg.num_labels - 1})."
            )
        if len(cfg.critic_cfg.bucket_values) != cfg.critic_cfg.num_labels:
            raise ValueError(
                f"len(bucket_values) ({len(cfg.critic_cfg.bucket_values)}) "
                f"must equal num_labels ({cfg.critic_cfg.num_labels})."
            )

    return cfg
