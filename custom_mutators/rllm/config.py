"""Configuration for the rllm custom mutator.

Presets live in ``configs/*.json``. The ``RLLM_CONFIG`` env var selects
which file to load (default: ``configs/default.json``); relative paths
resolve against this module's directory so they work regardless of AFL's
working directory. The loaded config is mirrored to
``<out>/rllm_config.json`` per run so each fuzzing run records the exact
settings it used.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MutatorConfig:
    """Combined config for mutator cadence, model loading, and masking."""

    # --- AFL mutator cadence ---
    fuzz_count: int = 16
    finetune_every: int = 100

    # --- Model ---
    model_name_or_path: str = "Salesforce/codet5p-220m"
    sampling_method: str = "contrastive"
    max_new_tokens: int = 64
    device: str = "auto"

    # Nucleus-sampling knobs (used when sampling_method == "nucleus")
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    no_repeat_ngram_size: int = 3

    # Contrastive-search knobs (used when sampling_method == "contrastive")
    penalty_alpha: float = 0.6
    contrastive_top_k: int = 32

    # --- Masking (T5 / CodeT5 defaults) ---
    corruption_rate: float = 0.15
    mean_span_length: float = 3.0
    min_span_length: int = 1
    max_span_length: int = 5
    whole_word_masking: bool = True


def load_config() -> MutatorConfig:
    """Load config from the JSON path in ``RLLM_CONFIG`` (default ``configs/default.json``).

    Also mirrors the resolved config to ``<out>/rllm_config.json`` for
    audit. Called once from ``rllm.init`` after AFL has set
    ``AFL_CUSTOM_INFO_OUT``.
    """
    here = Path(__file__).resolve().parent
    rel = os.environ.get("RLLM_CONFIG", "configs/default.json")
    path = Path(rel) if os.path.isabs(rel) else here / rel
    with open(path) as f:
        data = json.load(f)
    cfg = MutatorConfig(**{k: v for k, v in data.items() if not k.startswith("_")})
    _mirror_to_out_dir(cfg, path)
    return cfg


def _mirror_to_out_dir(cfg: MutatorConfig, source_path: Path) -> None:
    """Write the loaded config to ``<out>/rllm_config.json``."""
    out_dir = os.environ.get("AFL_CUSTOM_INFO_OUT")
    if not out_dir:
        return
    try:
        with open(Path(out_dir) / "rllm_config.json", "w") as f:
            json.dump(
                {"_source": str(source_path), **asdict(cfg)},
                f,
                indent=2,
            )
    except OSError:
        pass
