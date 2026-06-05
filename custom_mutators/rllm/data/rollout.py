"""GRPO rollout collection + dataset during fuzzing.

Per fuzzed seed the mutator emits ``M`` random masks × ``G`` infills (block
layout); each executed infill gets a scalar reward. ``RolloutBuffer`` groups
infills by mask (one GRPO group per mask). At the finetune-cycle boundary
``build_dataset`` z-scores rewards within each group and flattens to a
``RolloutDataset`` of per-infill samples that the HF-``Trainer`` consumes.

Only the data plumbing lives here — the reward itself (validity + coverage) is
computed by the mutator and passed to :meth:`commit`.

A **sample is a dict**: required ``{enc, target, advantage}`` (+ ``logp_old``
filled by the trainer's no-grad pass) and an open set of optional fields.
``GroupCollator`` pads ``enc``/``target`` (with masks) and then pads/stacks every
other present field generically by shape — so a later method can add a per-sample
field with no collator change.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np
import torch


class RolloutBuffer:
    """Groups infills by mask across the seeds of one finetune cycle."""

    def __init__(self) -> None:
        # seed_id -> {"token_ids": [...], "masks": {positions_tuple: [infill, ...]}}
        self._buf: dict[int, dict] = {}
        self._seed_id = -1
        self._pending: tuple | None = None   # (enc, target, positions) in flight

    def begin_seed(self, token_ids: Sequence[int]) -> None:
        """Open a new seed's rollout (call once per ``fuzz_count``)."""
        self._seed_id += 1
        self._buf[self._seed_id] = {"token_ids": list(token_ids), "masks": {}}

    def set_pending(self, enc, target, positions) -> None:
        """Record the infill about to execute (call in ``fuzz``)."""
        self._pending = (enc, target, list(positions))

    def has_pending(self) -> bool:
        """True if an infill with a non-empty target awaits a reward."""
        return self._pending is not None and bool(self._pending[1])

    def commit(self, reward: float) -> None:
        """Attach ``reward`` to the in-flight infill's mask-group (call in
        ``post_run``). No-op if nothing is pending or the target was empty."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        enc, target, positions = pending
        sr = self._buf.get(self._seed_id)
        if sr is not None and target:
            sr["masks"].setdefault(tuple(positions), []).append(
                {"enc": enc, "target": target, "reward": reward}
            )

    def build_dataset(self, advantage_fn: Callable, max_samples: int = 0) -> "RolloutDataset":
        """Flatten to a ``RolloutDataset``: per mask-group (≥2 infills) compute
        per-infill advantages via ``advantage_fn`` (z-scored G rewards); one
        sample ``{enc, target, advantage}`` per infill. ``enc`` is shared by
        reference within a group, so memory is ~#groups, not #infills.
        ``max_samples`` subsamples (0 = keep all)."""
        samples: list[dict] = []
        stds: list[float] = []                       # per-group reward spread (the GRPO signal)
        abs_adv: list[float] = []
        n_dead = n_all_invalid = n_all_valid = n_all_valid_live = 0
        for sr in self._buf.values():
            for infills in sr["masks"].values():
                if len(infills) < 2:                 # need ≥2 for a group baseline
                    continue
                rewards = [inf["reward"] for inf in infills]
                adv = advantage_fn(rewards)
                r = np.asarray(rewards, dtype=float)
                sd = float(r.std())
                stds.append(sd)
                abs_adv.extend(abs(float(a)) for a in adv)
                if sd < 1e-6:           n_dead += 1          # zero variance ⇒ advantage≡0 ⇒ no gradient
                if r.max() <= 0.0:      n_all_invalid += 1   # no valid infill (all syntax/semantic)
                elif r.min() > 0.0:                          # all valid: reward spread == R_cov spread
                    n_all_valid += 1
                    if sd >= 1e-6:      n_all_valid_live += 1 # carries a real coverage gradient
                for i, inf in enumerate(infills):
                    samples.append({"enc": inf["enc"], "target": inf["target"],
                                    "advantage": float(adv[i])})
        # GRPO group health: only groups with real reward spread teach anything.
        g = len(stds)
        self.group_health = {
            "n_groups": g,
            "grp_std_med": round(float(np.median(stds)), 4) if g else 0.0,
            "pct_dead": round(100.0 * n_dead / g, 1) if g else 0.0,
            "pct_all_inval": round(100.0 * n_all_invalid / g, 1) if g else 0.0,
            "pct_all_valid": round(100.0 * n_all_valid / g, 1) if g else 0.0,
            # Of the all-valid groups, the fraction carrying a coverage gradient (reward
            # spread = R_cov spread when validity is constant). This is the delta-coverage
            # payoff that grp_std_med (median over ALL groups, dominated by dead all-invalid
            # ones) structurally hides.
            "pct_allvalid_live": round(100.0 * n_all_valid_live / n_all_valid, 1) if n_all_valid else 0.0,
            "mean_abs_adv": round(float(np.mean(abs_adv)), 4) if abs_adv else 0.0,
        }
        if max_samples and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        return RolloutDataset(samples)

    def clear(self) -> None:
        self._buf.clear()


class RolloutDataset(torch.utils.data.Dataset):
    """In-memory list of sample dicts; the trainer fills ``logp_old`` in place."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return self.samples[i]

    def set_field(self, name: str, values: Sequence) -> None:
        """Write a per-sample field (e.g. ``logp_old``) back onto every sample."""
        for s, v in zip(self.samples, values):
            s[name] = v


def _pad_seqs(seqs, pad_value, dtype):
    """Pad a list of variable-length sequences to (B, max_len); return (tensor, mask)."""
    n = len(seqs)
    L = max((len(s) for s in seqs), default=0)
    padded = torch.full((n, L), pad_value, dtype=dtype)
    mask = torch.zeros((n, L), dtype=torch.float)
    for i, s in enumerate(seqs):
        if len(s):
            padded[i, : len(s)] = torch.as_tensor(list(s), dtype=dtype)
            mask[i, : len(s)] = 1.0
    return padded, mask


class GroupCollator:
    """Pad ``enc``/``target`` (with masks); pad/stack every other field by shape.

    Field-agnostic for the non-required fields: a per-token list (e.g. ``logp_old``)
    is padded to batch-max with 0.0; a scalar (e.g. ``advantage``) is stacked. New
    optional fields go through the same generic path — no collator change needed.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        out: dict[str, torch.Tensor] = {}
        out["input_ids"], out["attention_mask"] = _pad_seqs(
            [b["enc"] for b in batch], self.pad_id, torch.long)
        out["labels"], out["labels_mask"] = _pad_seqs(
            [b["target"] for b in batch], self.pad_id, torch.long)
        for k in batch[0]:
            if k in ("enc", "target"):
                continue
            vals = [b[k] for b in batch]
            if isinstance(vals[0], (list, tuple)):       # per-token field (e.g. logp_old)
                out[k], _ = _pad_seqs(vals, 0.0, torch.float)
            else:                                        # scalar (e.g. advantage)
                out[k] = torch.as_tensor(vals, dtype=torch.float)
        return out
