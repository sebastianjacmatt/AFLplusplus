"""Configuration for the rllm custom mutator.

One flat ``MutatorConfig`` dataclass holding every knob the mutator, model,
and masking modules need. Values are read once in ``rllm.init`` and treated
as immutable thereafter — edit defaults here (or replace :func:`load_config`
when wiring env/CLI loading later) before AFL launches the forkserver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MutatorConfig:
    """Combined config for mutator cadence, model loading, and masking."""

    # --- AFL mutator cadence ---
    fuzz_count: int = 16
    """Number of mutations produced per AFL seed (one batched generate)."""

    finetune_every: int = 100
    """Seeds-between-finetune cadence. Unused in mutation-only mode; the
    Mutator still tracks it so the CovRL extension drops in without edits."""

    # --- Model ---
    model_name_or_path: str = "Salesforce/codet5p-220m"
    sampling_method: str = "contrastive"
    """``"contrastive"`` (CovRL §4 default — higher validity) or ``"nucleus"``."""
    max_new_tokens: int = 64
    device: str = "auto"

    # Nucleus-sampling knobs (used when sampling_method == "nucleus")
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    no_repeat_ngram_size: int = 3

    # Contrastive-search knobs (used when sampling_method == "contrastive")
    # CovRL §4 reports penalty_alpha=0.6 and top_k=32 as their setup.
    penalty_alpha: float = 0.6
    contrastive_top_k: int = 32

    # --- Masking (T5 / CodeT5 defaults) ---
    corruption_rate: float = 0.15
    mean_span_length: float = 3.0
    min_span_length: int = 1
    max_span_length: int = 5
    whole_word_masking: bool = True


def load_config() -> MutatorConfig:
    """Return the default config. Replace this function (or edit the dataclass
    defaults) when wiring env-var / CLI loading; ``rllm.init`` calls it once."""
    return MutatorConfig()
