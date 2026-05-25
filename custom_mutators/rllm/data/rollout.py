"""Rollout dataset construction for CovRL training.

Reads the AFL++ queue (already-generated u16 files), scores each entry via
afl-showmap, and returns a RolloutDataset that both critic and actor consume
via the HF Trainer. No model inference happens here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.rewarding import Rewarding
from model.tokenizer import Tokenizer


class RolloutDataset(Dataset):
    """Flat dataset of (original_ids, reward) pairs from the AFL++ queue.

    Both critic and actor consume this same object; each trainer method
    applies its own masking via its DataCollator.
    """

    def __init__(self, records: list[dict]):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        return self._records[idx]


def generate_rollout(
    tokenizer: Tokenizer,
    rewarding: Rewarding,
    queue_dir: Path,
) -> RolloutDataset:
    """Score AFL++ queue files with afl-showmap and return a shared dataset.

    For each queue file:
      1. Parse u16 token buffer → JS bytes (detokenize only, no masking)
      2. Run afl-showmap → (validity, bitmap)
      3. Compute TF-IDF reward
      4. Store (original_ids, reward) as one dataset record

    IDF is updated in-place on rewarding after each call.

    # TODO: mix in a broader pre-training dataset at 4:1 ratio relative to
    #       queue size (CovRL finetuner.py:277 — train_dataset.sample(len(mutations) * 4))
    """
    records: list[dict] = []
    valid_bitmaps: list[np.ndarray] = []

    for path in sorted(queue_dir.glob("id:*")):
        token_ids = tokenizer.parse_u16(path.read_bytes())
        if not token_ids:
            continue

        js_bytes = tokenizer.detokenize(token_ids)
        validity, bitmap = rewarding.run(js_bytes, path.name)
        reward = rewarding.score(validity, bitmap)

        records.append({"original_ids": token_ids, "reward": reward})
        if bitmap is not None:
            valid_bitmaps.append(bitmap)

    if valid_bitmaps:
        rewarding.update_idf(np.vstack(valid_bitmaps))

    return RolloutDataset(records)
