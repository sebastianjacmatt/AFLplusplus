"""Dataset and collator classes for CovRL critic and actor training.

Both consume the same RolloutDataset from data/rollout.py. Masking is applied
fresh on each __getitem__ call (stochastic augmentation), matching CovRL's
ActorDataset / CriticDataset pattern (actor_dataset.py, critic_dataset.py).
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data.masking import Masking
from data.rollout import RolloutDataset
from model.critic import score_to_label


class CriticDataset(Dataset):
    """(T5_masked_input ++ T5_target_labels, reward_label) pairs.

    Input layout mirrors CovRL CriticDataset.__getitem__ (critic_dataset.py:55):
      input_ids      = cat(masked_encoder_input, T5_target_ids)
      attention_mask = cat(encoder_attn,          ones_like(target))
      labels         = score_to_label(reward)   — scalar long

    Masking is re-sampled on every __getitem__ call for augmentation.
    """

    def __init__(self, rollout: RolloutDataset, masking: Masking):
        self._rollout = rollout
        self._masking = masking

    def __len__(self) -> int:
        return len(self._rollout)

    def __getitem__(self, idx: int) -> dict:
        item = self._rollout[idx]
        mp = self._masking.mask(item["original_ids"])

        if not mp.masked:
            # Rare: program too short to mask. Return as-is; gradient is a no-op.
            ids = item["original_ids"]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "labels": torch.tensor(score_to_label(item["reward"]), dtype=torch.long),
            }

        target = self._masking.target_ids(mp)
        combined = mp.input_ids + target
        return {
            "input_ids": torch.tensor(combined, dtype=torch.long),
            "attention_mask": torch.ones(len(combined), dtype=torch.long),
            "labels": torch.tensor(score_to_label(item["reward"]), dtype=torch.long),
        }


class CriticCollator:
    """Pads a batch of CriticDataset items.

    Mirrors CovRL CriticDataCollator (critic_dataset.py:62-76).
    Labels are scalars — stacked, not padded.
    """

    def __call__(self, features: list[dict]) -> dict:
        input_ids = pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=0
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        labels = torch.stack([f["labels"] for f in features])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ActorDataset(Dataset):
    """Masked seq2seq inputs for actor (PPO + CE) training.

    Item layout mirrors CovRL ActorDataset.__getitem__ (actor_dataset.py:114-128):
      input_ids         = T5_masked_encoder_input
      attention_mask    = ones_like(input_ids)
      decoder_input_ids = T5_target_ids
      labels            = T5_target_ids

    reward is not in the item — the critic scores the actor's live predictions
    during _compute_actor_loss, not from stored values.
    """

    def __init__(self, rollout: RolloutDataset, masking: Masking, pad_token_id: int):
        self._rollout = rollout
        self._masking = masking
        self._pad = pad_token_id

    def __len__(self) -> int:
        return len(self._rollout)

    def __getitem__(self, idx: int) -> dict:
        item = self._rollout[idx]
        mp = self._masking.mask(item["original_ids"])

        if not mp.masked:
            ids = item["original_ids"]
            t = torch.tensor(ids, dtype=torch.long)
            return {
                "input_ids": t,
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "decoder_input_ids": t,
                "labels": t,
            }

        target = self._masking.target_ids(mp)
        return {
            "input_ids": torch.tensor(mp.input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(mp.input_ids), dtype=torch.long),
            "decoder_input_ids": torch.tensor(target, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }


class ActorCollator:
    """Pads a batch of ActorDataset items.

    Mirrors CovRL ActorDataCollator (actor_dataset.py:130-143).
    labels are padded with -100 so HF cross-entropy ignores padding positions.
    """

    def __init__(self, pad_token_id: int = 0):
        self._pad = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        input_ids = pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=self._pad
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        decoder_input_ids = pad_sequence(
            [f["decoder_input_ids"] for f in features],
            batch_first=True,
            padding_value=self._pad,
        )
        labels = pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
