"""Configuration for the rllm custom mutator.

Presets live in ``configs/*.json``. The ``RLLM_CONFIG`` env var selects
which file to load (default: ``configs/grpo.json``); relative paths
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
    inference_batch_size: int = 1
    finetune_every: int = 100

    # --- Model ---
    model_name_or_path: str = "Salesforce/codet5p-220m"
    sampling_method: str = "contrastive"
    max_new_tokens: int = 64
    max_seq_len: int = 768               # truncate seed tokens before mask/generate (CovRL model_max_length; bounds O(seq²) attention/VRAM)
    device: str = "auto"

    # Nucleus-sampling knobs (used when sampling_method == "nucleus")
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    no_repeat_ngram_size: int = 3

    # Contrastive-search knobs (used when sampling_method == "contrastive")
    penalty_alpha: float = 0.6
    contrastive_top_k: int = 32

    # --- Masking (random span corruption; CovRL/TLAFL overwrite + T5 spans) ---
    corruption_rate: float = 0.15
    mean_span_length: float = 3.0
    min_span_length: int = 1
    max_span_length: int = 5
    whole_word_masking: bool = True
    # CovRL-Fuzz alignment: cap masks per mutation to UR(3)+1 = {1,2,3}.
    # Set to 0 to disable the cap (pure corruption_rate behaviour).
    max_masks: int = 3

    # --- GRPO trainer (active when finetune_every > 0; else pure LLM mutation) ---
    # Per random mask the mutator samples `group_size` infills (one GRPO group);
    # the trainer full-fine-tunes the model on the z-scored group advantage.
    group_size: int = 16                  # G infills per mask (M = fuzz_count // G masks/seed)
    lr: float = 1e-4                      # AdamW learning rate
    kappa: float = 0.05                   # GRPO k3 KL coefficient (vs rollout policy)
    eps_low: float = 0.2                  # GRPO clip lower bound
    eps_high: float = 0.2                 # GRPO clip upper bound (raise for Clip-Higher/DAPO)
    # KL-to-frozen-reference anchor (validity stability; docs/coverage_signal.md §3 share
    # lever). The `kappa` KL is vs the rollout policy → inert on-policy; this SEPARATE term
    # pulls the policy toward a frozen snapshot (the base model's valid prior), countering
    # the unanchored random-walk that drives validity collapse. 0 = off (no ref model).
    # `ref_update_every` advances the snapshot to the current policy every N cycles
    # (0 = anchor to the base forever; keep low only if the policy stays healthy).
    kl_ref_coef: float = 0.0
    ref_update_every: int = 0
    max_train_infills: int = 1024         # cap infills trained per cycle (0 = all)
    method: str = "grpo"                  # grpo | drgrpo | … (selects trainer subclass + advantage)
    train_batch_size: int = 16            # HF Trainer forward batch (memory/speed; NOT #updates)
    fp16: bool = False                    # HF Trainer mixed precision
    # NOTE: one optimizer step per cycle is a structural constant (single on-policy
    # REINFORCE update, ratio ≡ 1) — not a config knob. Multi-update is a future opt-in.

    # --- Coverage reward (data/rewarding.py:TFIDFCoverageRewarder, CovRL Eq. 2/5) ---
    # Active when coverage_reward is set and a trainer is attached. Reads the
    # AFL++ bitmap via __AFL_SHM_ID; degrades to validity-only (+1) if the SHM
    # attach fails (e.g. no instrumentation).
    coverage_reward: bool = True
    bitmap_size: int = 65536              # AFL++ default MAP_SIZE (2**16); match the target's map
    idf_alpha: float = 0.6               # IDF momentum (CovRL α)
    validity_bonus: float = 0.5          # b in valid reward = b + (1-b)·R_cov ∈ [b, 1]
    # Reward for a timeout (hang). A hang wastes the full exec budget and games the
    # validity floor, so it is the hardest penalty. Detected via exit_hook's
    # SIGUSR1 marker (requires AFL_KILL_SIGNAL=SIGUSR1 — set in run_rllm.sh).
    timeout_reward: float = -1.0

    # B1 delta-vs-parent coverage (docs/coverage_signal.md §3): credit only edges new vs
    # the parent seed, not absolute coverage. Strips the parent's inherited edges (the
    # shared interpreter floor + the parent's own rare edges) that pin absolute Σtf·idf
    # ~constant within a mask group (between-seed variance → cancelled by GRPO's
    # within-group baseline). Measured ~800× the within-group coverage signal at no
    # validity cost. Each queue entry's edge set is cached from the AFL bitmap when AFL
    # creates the entry (mutator.queue_new_entry) and looked up when it's fuzzed
    # (fuzz_count) — pure SHM read, no subprocess. False ⇒ absolute CovRL reward.
    delta_coverage: bool = False

    # Observation aid: write a per-mutation .cur_input.diff (ANSI-highlighted
    # parent→mutant). Costs a difflib pass + file write *per mutation* (O(n²) on
    # long seeds). Pure debugging output — disable on speed runs; no effect on
    # mutation behaviour or training.
    write_diff_sidecar: bool = True

    # Per-infill sample log (<out>/rllm_samples.txt): one row per executed
    # mutation — validity class, coverage R_cov, total reward, and (seed,
    # positions) group id. Raw data for offline GRPO group-variance analysis.
    # ~one buffered write per mutation (values already computed) — negligible
    # vs generation. Disable for pure-speed runs.
    log_samples: bool = True

    # Per-cycle GRPO policy-optimization diagnostics in rllm_train.tsv (ratio
    # mean/max/std, clip-fraction, KL, advantage mean/std, entropy) — the signals
    # that distinguish GRPO variants over time. Entropy reuses the existing
    # log_softmax (≈free, once/cycle). NOTE: grad_norm is deferred (no clean
    # transformers 4.29.2 hook); `log_grad_norm` to be added when one exists.
    log_train_entropy: bool = True


def load_config() -> MutatorConfig:
    """Load config from the JSON path in ``RLLM_CONFIG`` (default ``configs/grpo.json``).

    Also mirrors the resolved config to ``<out>/rllm_config.json`` for
    audit. Called once from ``rllm.init`` after AFL has set
    ``AFL_CUSTOM_INFO_OUT``.
    """
    here = Path(__file__).resolve().parent
    rel = os.environ.get("RLLM_CONFIG", "configs/grpo.json")
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
