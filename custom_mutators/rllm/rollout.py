"""Trajectory storage for the rllm rollout.

Two collaborating types plus a collator (docs/design.md §3.4):

* ``RolloutBuffer`` — append-only, mutable, owned by the ``Mutator``.
  One entry per ``fuzz()`` call.
* ``RolloutDataset`` — read-only ``torch.utils.data.Dataset`` snapshot
  consumed by ``BaseTrainer``.
* ``RolloutCollator`` — pads variable-length rollout entries into a
  batched dict (HF Trainer feeds batches via a DataLoader).

Split rationale: the buffer's append API is wrong for training and the
dataset's read API is wrong for the mutator. SRP, one type per role.
"""

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from config import Config


class RolloutBuffer:
    """Append-only trajectory store owned by the Mutator."""

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self._entries: list[dict] = []

    def append(self, entry: dict) -> None:
        self._entries.append(entry)

    def set_last_reward(self, reward: float) -> None:
        if self._entries:
            self._entries[-1]["reward"] = reward

    def snapshot(self) -> list[dict]:
        """Shallow copy of entries — consumed by RolloutDataset."""
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def flush(self) -> None:
        """Final drain on shutdown."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class RolloutDataset(Dataset):
    """Read-only snapshot of completed trajectories for one training cycle.

    Entries with ``reward is None`` (post_run hasn't fired yet, or attribution
    was missed) are dropped — the trainer only sees complete trajectories.
    """

    def __init__(self, buffer: RolloutBuffer) -> None:
        self._entries = [
            e for e in buffer.snapshot() if e.get("reward") is not None
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        e = self._entries[idx]
        return {
            "input_ids":    e["masked_input"],
            "labels":       e["gen_tokens"],
            "old_logprobs": e["logprobs"],
            "reward":       torch.tensor(e["reward"], dtype=torch.float32),
        }


class RolloutCollator:
    """Pad variable-length rollout entries into batched tensors."""

    LABEL_PAD = -100  # HF convention: ignored in CE loss

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=self.LABEL_PAD,
        )
        old_logprobs = pad_sequence(
            [b["old_logprobs"] for b in batch],
            batch_first=True,
            padding_value=0.0,
        )
        rewards = torch.stack([b["reward"] for b in batch])

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "old_logprobs":   old_logprobs,
            "rewards":        rewards,
        }
