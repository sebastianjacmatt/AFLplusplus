"""Rollout data pipeline for rlm_mutator.

Three classes feed the HF Trainer from the AFL++ mutation loop:

    RolloutBuffer    — two-phase in-memory store (log → patch_reward → flush)
    RolloutDataset   — torch Dataset over a flushed record list
    RolloutCollator  — pads and stacks records into batched tensors

Record schema (one dict per sample):

    {
        "sample_id":       str,    # "s00000000"
        "group_id":        int,    # rollout-unique group identifier for GRPO
        "x_t":             list,   # masked encoder input token IDs
        "y_t":             list,   # generated decoder token IDs
        "log_prob":        float,  # mean per-token log-prob under the behaviour actor
        "reward":          float,  # scalar reward from post_run (exit-code gated)
        "coverage_reward": float | None,  # raw TF-IDF component before exit-code gate
        "exit_code":       int | None,    # target exit code, None on crash / missing file
        "ref_log_prob":    float | None,  # reference model log-prob (KL term, optional)
        "value_pred":      float | None,  # critic value estimate (PPO, optional)
    }
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# RolloutBuffer — written by mutator.generate() and mutator.on_post_run()
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Single-threaded in-memory two-phase store.

    AFL++ drives the Python mutator from one thread, so no locking is needed.
    """

    def __init__(self):
        self._records: dict[str, dict] = {}
        self._counter: int = 0

    def new_sample_id(self) -> str:
        sid = f"s{self._counter:08d}"
        self._counter += 1
        return sid

    def log(
        self,
        sample_id:    str,
        group_id:     int,
        x_t:          list,
        y_t:          list,
        log_prob:     float,
        ref_log_prob: Optional[float] = None,
        value_pred:   Optional[float] = None,
    ) -> None:
        """Record a new mutant immediately after generation. reward fills in post_run()."""
        self._records[sample_id] = {
            "sample_id":       sample_id,
            "group_id":        group_id,
            "x_t":             x_t,
            "y_t":             y_t,
            "log_prob":        log_prob,
            "reward":          None,
            "coverage_reward": None,
            "exit_code":       None,
            "ref_log_prob":    ref_log_prob,
            "value_pred":      value_pred,
        }

    def patch_reward(
        self,
        sample_id:       str,
        reward:          float,
        coverage_reward: Optional[float] = None,
        exit_code:       Optional[int]   = None,
    ) -> None:
        rec = self._records.get(sample_id)
        if rec is not None:
            rec["reward"]          = reward
            rec["coverage_reward"] = coverage_reward
            rec["exit_code"]       = exit_code

    def flush(self) -> list[dict]:
        """Return all records with a populated reward and clear the store."""
        complete = [r for r in self._records.values() if r["reward"] is not None]
        self._records.clear()
        return complete

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# RolloutDataset — injected into HF Trainer via trainer.set_rollout_dataset()
# ---------------------------------------------------------------------------

class RolloutDataset(Dataset):
    """torch Dataset wrapping a flushed rollout record list.

    Records stay as Python dicts; tensorisation happens in RolloutCollator.
    """

    def __init__(self, records: list[dict]):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        return self._records[idx]


# ---------------------------------------------------------------------------
# GroupedBatchSampler — keeps each GRPO reward group in one batch
# ---------------------------------------------------------------------------

class GroupedBatchSampler(Sampler[list[int]]):
    """Yield one full GRPO group per batch, preserving first-seen group order."""

    def __init__(self, dataset: RolloutDataset, group_size: int):
        self._batches: list[list[int]] = []

        groups: dict[int, list[int]] = {}
        group_order: list[int] = []
        for idx in range(len(dataset)):
            try:
                gid = int(dataset[idx]["group_id"])
            except KeyError as exc:
                raise ValueError("GroupedBatchSampler requires dataset records with 'group_id'.") from exc

            if gid not in groups:
                groups[gid] = []
                group_order.append(gid)
            groups[gid].append(idx)

        for gid in group_order:
            batch = groups[gid]
            if len(batch) != group_size:
                raise ValueError(
                    f"Group {gid} has {len(batch)} samples, expected {group_size}."
                )
            self._batches.append(batch)

    def __iter__(self):
        for batch in self._batches:
            yield list(batch)

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# RolloutCollator — callable passed as HF Trainer's data_collator
# ---------------------------------------------------------------------------

class RolloutCollator:
    """Pads and stacks rollout records into batched tensors.

    Output keys (all torch.Tensor):
      input_ids         [B, L_x]    pad = tokenizer.pad_token_id
      attention_mask    [B, L_x]    1/0
      labels            [B, L_y]    pad = -100 (HF convention: loss ignores pads)
      old_log_prob      [B]         float32
      reward            [B]         float32
      group_id          [B]         int64      rollout-unique GRPO group id
      ref_log_prob      [B]         float32    (0.0 if record has None)
      value_pred        [B]         float32    (0.0 if record has None)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id    = tokenizer.pad_token_id
        self.eos_id    = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        b = len(batch)
        max_x = max(len(r["x_t"]) for r in batch)
        max_y = max(len(r["y_t"]) for r in batch) + 1   # + eos

        input_ids      = torch.full((b, max_x), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_x), dtype=torch.long)
        labels         = torch.full((b, max_y), -100, dtype=torch.long)

        for i, r in enumerate(batch):
            x = r["x_t"]
            y = r["y_t"] + [self.eos_id]
            input_ids[i, :len(x)]      = torch.tensor(x, dtype=torch.long)
            attention_mask[i, :len(x)] = 1
            labels[i, :len(y)]         = torch.tensor(y, dtype=torch.long)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "old_log_prob":   torch.tensor([r["log_prob"] for r in batch], dtype=torch.float32),
            "reward":         torch.tensor([r["reward"]   for r in batch], dtype=torch.float32),
            "group_id":       torch.tensor([r["group_id"] for r in batch], dtype=torch.long),
            "ref_log_prob":   torch.tensor([r["ref_log_prob"] or 0.0 for r in batch], dtype=torch.float32),
            "value_pred":     torch.tensor([r["value_pred"]   or 0.0 for r in batch], dtype=torch.float32),
        }
