"""Configuration for rlm_mutator.

Three dataclass groups, each owned by a distinct layer:

    ModelConfig    — model identity, device, generation knobs
    AFLConfig      — AFL++ integration: mutation budget, bitmap, IDF schedule
    TrainingConfig — RL shared hyper-parameters, plus optional PPOConfig / GRPOConfig sub-config

Usage (from JSON file pointed at by RLM_CONFIG env var)::

    export RLM_CONFIG=/path/to/rlm_config.json
    afl-fuzz ...

Usage (programmatic defaults, no file needed)::

    model_cfg, afl_cfg, train_cfg = load_config()

JSON example (GRPO)::

    {
        "model_name_or_path": "Salesforce/codet5p-220m",
        "device": "cuda",
        "algorithm": "grpo",
        "group_size": 8,
        "bitmap_size": 131072,
        "fuzz_count": 512
    }

HfArgumentParser treats each dataclass as a flat namespace so all fields from
all three groups can coexist in a single JSON object.
"""

import os
from dataclasses import dataclass, field, fields
from typing import Literal, Optional

import torch
from transformers import HfArgumentParser


# ---------------------------------------------------------------------------
# Model / generation
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        default="Salesforce/codet5p-220m",
        metadata={"help": "HuggingFace model id or local path passed to from_pretrained."},
    )
    device: str = field(
        default="auto",
        metadata={"help": "'auto' resolves to cuda if available, else cpu."},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum token length for encoder input and decoder output."},
    )
    mask_probability: float = field(
        default=0.15,
        metadata={"help": "Fraction of input tokens corrupted by CodeT5 masked span prediction."},
    )
    mean_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean corrupted span length for CodeT5 masked span prediction."},
    )
    min_span_length: int = field(
        default=1,
        metadata={"help": "Minimum corrupted span length for CodeT5 masked span prediction."},
    )
    max_span_length: int = field(
        default=5,
        metadata={"help": "Maximum corrupted span length for CodeT5 masked span prediction."},
    )
    top_k: int = field(
        default=50,
        metadata={"help": "top-k filter for multinomial sampling (0 disables)."},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Nucleus-sampling threshold for multinomial sampling."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for multinomial sampling."},
    )
    max_new_tokens_per_mask: int = field(
        default=16,
        metadata={"help": "Maximum generated tokens allowed per masked span prediction."},
    )
    whole_word_masking: bool = field(
        default=True,
        metadata={
            "help": (
                "Sample masked spans in word units before subword expansion "
                "(CodeT5 §3.2 / CodeT5+ §3.1: 'sample spans before subword "
                "tokenization to avoid masking partial words'). False reverts "
                "to legacy token-level span sampling."
            )
        },
    )

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ---------------------------------------------------------------------------
# AFL++ integration
# ---------------------------------------------------------------------------

@dataclass
class AFLConfig:
    fuzz_count: int = field(
        default=512,
        metadata={
            "help": (
                "Total fuzz() calls AFL++ will schedule per seed. "
                "Must be divisible by group_size when using GRPO. "
                "Example: 32 groups × 16 samples = 512."
            )
        },
    )
    mask_count: int = field(
        default=3,
        metadata={"help": "Maximum original tokens replaced inside one masked span."},
    )
    mask_strategy: Literal["span", "scatter"] = field(
        default="span",
        metadata={"help": "Masking strategy. 'span' is safer and closer to CodeT5 pretraining."},
    )
    bitmap_size: int = field(
        default=65536,
        metadata={"help": "AFL++ coverage bitmap size in bytes (2^16 or 2^17)."},
    )
    idf_alpha: float = field(
        default=0.6,
        metadata={"help": "IDF momentum rate alpha (CovRL Eq. 7). Higher = more inertia."},
    )
    finetune_every: int = field(
        default=64,
        metadata={"help": "queue_get() calls between finetune cycles."},
    )
    validity_bonus: float = field(
        default=0.0,
        metadata={
            "help": (
                "Flat validity credit added to the valid-program reward branch. "
                "Final reward = validity_bonus + (1 - validity_bonus) * R_cov. "
                "0.0 = pure coverage (CovRL default). 0.3 creates a 3-level reward "
                "floor: syntax=-1.0 / semantic=-0.5 / valid-zero-cov=0.3 / valid-high=~0.85, "
                "widening within-group advantage variance for all-valid groups."
            )
        },
    )


# ---------------------------------------------------------------------------
# Algorithm-specific sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO-specific hyper-parameters (actor-critic).

    Set TrainingConfig.algorithm='ppo' and populate this sub-config.
    group_size is implicitly 1 for PPO — each fuzz() call gets an independent mask.
    """
    value_coef: float = field(
        default=0.5,
        metadata={"help": "Critic value loss coefficient in the PPO objective."},
    )
    entropy_coef: float = field(
        default=0.01,
        metadata={"help": "Entropy bonus coefficient to encourage exploration."},
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "GAE-Lambda smoothing for advantage estimation."},
    )


@dataclass
class GRPOConfig:
    """GRPO-specific hyper-parameters (group-relative policy optimisation).

    Set TrainingConfig.algorithm='grpo' and populate this sub-config.
    """
    group_size: int = field(
        default=8,
        metadata={
            "help": (
                "Number of fuzz() calls sharing one masked input context x_t. "
                "Rewards are normalised within the group before the policy update. "
                "AFL_CFG.fuzz_count must be divisible by group_size."
            )
        },
    )
    norm_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Std stabiliser in GRPO group advantage normalisation."},
    )


# ---------------------------------------------------------------------------
# Shared training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Shared RL training hyper-parameters.

    algorithm selects PPO or GRPO; ppo/grpo fields carry the algorithm-specific
    sub-config.  Exactly one of ppo or grpo must be set (enforced at load time).
    """
    algorithm: Literal["ppo", "grpo"] = field(
        default="grpo",
        metadata={"help": "RL algorithm: 'ppo' (actor-critic) or 'grpo' (group-relative)."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "AdamW learning rate."},
    )
    train_batch_size: int = field(
        default=8,
        metadata={
            "help": (
                "Samples per gradient update. For GRPO this must be a multiple "
                "of group_size so each batch contains complete reward groups."
            )
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "number of training epochs"},
    )
    kl_coef: float = field(
        default=0.0,
        metadata={"help": "KL penalty coefficient beta (requires reference model if > 0)."},
    )
    clip_epsilon: float = field(
        default=0.2,
        metadata={"help": "PPO/GRPO importance-ratio clip bound epsilon."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank. 0 disables LoRA (full fine-tune)."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA scaling alpha."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability."},
    )
    lora_target_modules: str = field(
        default="q,v",
        metadata={"help": "Comma-separated attention projection names for LoRA adapters."},
    )
    enable_logging: bool = field(
        default=True,
        metadata={
            "help": (
                "Master logging switch. True: report_to='tensorboard', rollout CSV "
                "written, per-step train/rollout scalars emitted via self.log(). "
                "False: report_to='none', no CSV, no scalar emissions."
            )
        },
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Number of optimizer steps between Trainer log entries."},
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of training steps used for linear LR warmup per finetune cycle. "
                "0.1 = first 10% of steps ramp from 0 → learning_rate. Stabilises early "
                "updates when advantage estimates are noisiest (few valid samples in rollout)."
            )
        },
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Train in bfloat16. Halves the logits tensor (B × L × vocab) that drives "
                "OOM on long training runs. Requires Ampere+ GPU (RTX 30xx / A100+)."
            )
        },
    )
    ref_update_every: int = field(
        default=1,
        metadata={
            "help": (
                "Re-anchor pi_ref every N finetune cycles. Default 1 updates after every "
                "cycle (original behaviour). Setting >1 holds the reference fixed for N "
                "cycles so the KL term pulls toward a less-drifted policy, providing a "
                "weaker validity anchor when valid_rate collapses."
            )
        },
    )

    # Algorithm-specific sub-configs — set by load_config(), not via JSON directly.
    ppo:  Optional[PPOConfig]  = field(default=None, metadata={"help": "PPO sub-config."})
    grpo: Optional[GRPOConfig] = field(default=None, metadata={"help": "GRPO sub-config."})

    @property
    def lora_target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.lora_target_modules.split(",") if m.strip()]

    @property
    def group_size(self) -> int:
        """Convenience accessor: 1 for PPO, GRPOConfig.group_size for GRPO."""
        if self.grpo is not None:
            return self.grpo.group_size
        return 1


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# HfArgumentParser flattens all three dataclasses into one namespace.
# PPOConfig and GRPOConfig fields are handled manually in load_config().
_PARSER = HfArgumentParser((ModelConfig, AFLConfig, TrainingConfig))


def load_config(
    config_path: str | None = None,
) -> tuple[ModelConfig, AFLConfig, TrainingConfig]:
    """Load config from a JSON file or return dataclass defaults.

    Resolution order:
      1. Explicit ``config_path`` argument.
      2. ``RLM_CONFIG`` environment variable.
      3. Dataclass defaults (no file required).

    For GRPO, ``AFLConfig.fuzz_count`` must be divisible by ``GRPOConfig.group_size``.

    @param config_path: Optional path to a JSON file understood by HfArgumentParser.
    @return: (ModelConfig, AFLConfig, TrainingConfig)
    """
    path = config_path or os.environ.get("RLM_CONFIG", "")
    if path:
        import json
        with open(path) as fh:
            raw = json.load(fh)
        model_cfg, afl_cfg, train_cfg = _PARSER.parse_dict(raw, allow_extra_keys=True)
    else:
        raw = {}
        model_cfg, afl_cfg, train_cfg = ModelConfig(), AFLConfig(), TrainingConfig()

    if train_cfg.logging_steps < 1:
        raise ValueError(
            f"TrainingConfig.logging_steps ({train_cfg.logging_steps}) must be >= 1."
        )

    if model_cfg.max_new_tokens_per_mask < 1:
        raise ValueError(
            f"ModelConfig.max_new_tokens_per_mask ({model_cfg.max_new_tokens_per_mask}) must be >= 1."
        )

    if not 0.0 <= model_cfg.mask_probability <= 1.0:
        raise ValueError(
            f"ModelConfig.mask_probability ({model_cfg.mask_probability}) must be in [0, 1]."
        )

    if model_cfg.min_span_length < 1:
        raise ValueError(
            f"ModelConfig.min_span_length ({model_cfg.min_span_length}) must be >= 1."
        )

    if model_cfg.max_span_length < model_cfg.min_span_length:
        raise ValueError(
            f"ModelConfig.max_span_length ({model_cfg.max_span_length}) must be >= "
            f"ModelConfig.min_span_length ({model_cfg.min_span_length})."
        )

    if not model_cfg.min_span_length <= model_cfg.mean_span_length <= model_cfg.max_span_length:
        raise ValueError(
            f"ModelConfig.mean_span_length ({model_cfg.mean_span_length}) must be between "
            f"ModelConfig.min_span_length ({model_cfg.min_span_length}) and "
            f"ModelConfig.max_span_length ({model_cfg.max_span_length})."
        )

    if afl_cfg.mask_count < 1:
        raise ValueError(
            f"AFLConfig.mask_count ({afl_cfg.mask_count}) must be >= 1."
        )

    # Attach algorithm sub-config — flat JSON keys like "group_size" are
    # picked up here, since HfArgumentParser only parses the three top-level
    # dataclasses.
    if train_cfg.algorithm == "grpo":
        train_cfg.grpo = GRPOConfig(**_sub_kwargs(raw, GRPOConfig))
        train_cfg.ppo  = None
        if train_cfg.train_batch_size % train_cfg.grpo.group_size != 0:
            raise ValueError(
                f"TrainingConfig.train_batch_size ({train_cfg.train_batch_size}) must be a multiple of "
                f"GRPOConfig.group_size ({train_cfg.grpo.group_size})."
            )
        if afl_cfg.fuzz_count % train_cfg.grpo.group_size != 0:
            raise ValueError(
                f"AFLConfig.fuzz_count ({afl_cfg.fuzz_count}) must be divisible by "
                f"GRPOConfig.group_size ({train_cfg.grpo.group_size})."
            )
    else:
        train_cfg.ppo  = PPOConfig(**_sub_kwargs(raw, PPOConfig))
        train_cfg.grpo = None

    return model_cfg, afl_cfg, train_cfg


def _sub_kwargs(raw: dict, cls) -> dict:
    """Select the top-level raw-dict keys that match ``cls`` dataclass fields."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}
