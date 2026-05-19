"""Rollout buffer + Dataset/Sampler/Collator for CovRL training.

The two-stage buffer encodes the CollectInteresting vs CollectAll split:
    add()         — stages one rollout (overwrites any prior staged entry)
    commit_last() — promotes the staged entry into D_T

CollectAll calls commit_last() immediately after add() (in mutator.post_run).
CollectInteresting calls commit_last() in mutator.queue_new_entry — only when
AFL flagged the run as new coverage; uncommitted rollouts are overwritten by
the next add().

Records are plain dicts so extra fields can be added later without API churn —
pass them as keyword args to add() and they land in the record.

Field names match rlm_mutator's so RolloutCollator can stay unchanged when the
deferred fields (log_prob, ref_log_prob) come online.
"""

from typing import Any, Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler

from .config import Config


# ---------------------------------------------------------------------------
# Buffer — written by mutator.post_run / mutator.queue_new_entry
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Two-stage in-memory buffer.

    Single-threaded (AFL drives the mutator from one thread), no locking.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._staged: Optional[dict] = None
        self._committed: list[dict] = []

    # TODO (#3): two-phase log/patch_reward — fuzz() logs a partial record
    # keyed by sample_id; post_run() patches the reward in. Deferred per
    # baseline CovRL.
    def add(self, x: Any, y: Any, R: float, group_id: int, **extra: Any) -> None:
        """Stage a completed rollout. Overwrites any prior staged entry.

        TODO (#2, #8): `extra` is reserved for log_prob (π_old at generation
        time) and ref_log_prob (KL anchor). Deferred per baseline CovRL —
        adding them is a one-line change at the call site.
        """
        self._staged = {
            "x_t":      _coerce(x),
            "y_t":      _coerce(y),
            "reward":   float(R),
            "group_id": int(group_id),
            **extra,
        }

    def commit_last(self) -> None:
        """Promote the staged entry into D_T. No-op if nothing is staged."""
        if self._staged is None:
            return
        self._committed.append(self._staged)
        self._staged = None

    def flush(self) -> list[dict]:
        """Drain committed records and clear. Staged entry is dropped."""
        records = self._committed
        self._committed = []
        self._staged = None
        return records

    def __len__(self) -> int:
        return len(self._committed)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._committed)


# ---------------------------------------------------------------------------
# Dataset / Sampler / Collator — consumed by base_trainer
# ---------------------------------------------------------------------------

class RolloutDataset(Dataset):
    """torch Dataset over a drained list of rollout records."""

    def __init__(self, records: list[dict]):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        return self._records[idx]


class GroupedBatchSampler(Sampler):
    """Yield minibatches made of complete GRPO groups.

    `batch_size` must be a multiple of `group_size`; every group in the dataset
    must have exactly `group_size` members. This matches CollectAll, where
    every fuzz commits and groups stay intact. Under CollectInteresting groups
    may legitimately be partial — the trainer should pick a different sampler
    in that case.

    Per principle #7, no silent filtering of incomplete or zero-variance
    groups — partial groups raise so we learn when the assumption breaks.
    """

    def __init__(self, dataset: RolloutDataset, batch_size: int, group_size: int):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of group_size ({group_size})"
            )

        groups: dict[int, list[int]] = {}
        group_order: list[int] = []
        for idx in range(len(dataset)):
            gid = int(dataset[idx]["group_id"])
            if gid not in groups:
                groups[gid] = []
                group_order.append(gid)
            groups[gid].append(idx)

        for gid in group_order:
            if len(groups[gid]) != group_size:
                raise ValueError(
                    f"group {gid} has {len(groups[gid])} samples, expected {group_size}"
                )

        if len(dataset) % batch_size != 0:
            raise ValueError(
                f"dataset size ({len(dataset)}) must be divisible by batch_size ({batch_size})"
            )

        self._batches: list[list[int]] = []
        batch: list[int] = []
        for gid in group_order:
            batch.extend(groups[gid])
            if len(batch) == batch_size:
                self._batches.append(batch)
                batch = []

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._batches:
            yield list(batch)

    def __len__(self) -> int:
        return len(self._batches)


class RolloutCollator:
    """Pad and stack rollout records into batched tensors for HF Trainer.

    Output keys:
        input_ids       [B, L_x]  pad = tokenizer.pad_token_id
        attention_mask  [B, L_x]  1/0
        labels          [B, L_y]  pad = -100  (HF: loss ignores -100)
        reward          [B]       float32
        group_id        [B]       int64

    TODO (#2, #8): old_log_prob / ref_log_prob will be added here when the
    rollout schema gains those fields.
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        if self.eos_id is None:
            raise ValueError("tokenizer must define eos_token_id")

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        max_x = max(len(r["x_t"]) for r in batch)
        max_y = max(len(r["y_t"]) for r in batch) + 1  # +1 for appended EOS

        input_ids      = torch.full((b, max_x), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_x),             dtype=torch.long)
        labels         = torch.full((b, max_y), -100,        dtype=torch.long)

        for i, r in enumerate(batch):
            x = r["x_t"]
            y = r["y_t"] + [self.eos_id]
            input_ids[i, : len(x)]      = torch.tensor(x, dtype=torch.long)
            attention_mask[i, : len(x)] = 1
            labels[i, : len(y)]         = torch.tensor(y, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "reward":         torch.tensor([r["reward"]   for r in batch], dtype=torch.float32),
            "group_id":       torch.tensor([r["group_id"] for r in batch], dtype=torch.long),
        }


def _coerce(v: Any) -> Any:
    """Normalise tensor / numpy values to plain Python lists for storage."""
    if hasattr(v, "tolist"):
        return v.tolist()
    return v
