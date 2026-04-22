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
        metadata={"help": "Fraction of max_length used as decoder budget per mutation."},
    )
    sample_method: Literal["greedy", "contrastive"] = field(
        default="greedy",
        metadata={"help": "Generation strategy: 'greedy' or 'contrastive' search."},
    )
    top_k: int = field(
        default=8,
        metadata={"help": "top-k for contrastive search (ignored for greedy)."},
    )
    penalty_alpha: float = field(
        default=0.6,
        metadata={"help": "Degeneration penalty alpha for contrastive search."},
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
        metadata={"help": "Maximum mask tokens inserted or overwritten per mutation."},
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
                "Samples per gradient update. For GRPO this must equal "
                "group_size so each batch contains exactly one reward group."
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

    if model_cfg.sample_method == "contrastive":
        if model_cfg.penalty_alpha <= 0.0:
            raise ValueError(
                "ModelConfig.penalty_alpha must be > 0 when sample_method='contrastive'."
            )
        if model_cfg.top_k <= 1:
            raise ValueError(
                "ModelConfig.top_k must be > 1 when sample_method='contrastive'."
            )

    # Attach algorithm sub-config — flat JSON keys like "group_size" are
    # picked up here, since HfArgumentParser only parses the three top-level
    # dataclasses.
    if train_cfg.algorithm == "grpo":
        train_cfg.grpo = GRPOConfig(**_sub_kwargs(raw, GRPOConfig))
        train_cfg.ppo  = None
        if train_cfg.train_batch_size != train_cfg.grpo.group_size:
            raise ValueError(
                f"TrainingConfig.train_batch_size ({train_cfg.train_batch_size}) must equal "
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
