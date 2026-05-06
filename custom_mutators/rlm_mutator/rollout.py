"""Rollout data pipeline for rlm_mutator.

Three classes feed the HF Trainer from the AFL++ mutation loop:

    RolloutBuffer    — two-phase in-memory store (log → patch_reward → flush/log)
    RolloutDataset   — torch Dataset over a flushed record list
    RolloutCollator  — pads and stacks records into batched tensors

Record schema (one dict per sample):

    {
        "finetune_id"      int     # last finetune id
        "sample_id":       str,    # "s00000000"
        "group_id":        int,    # rollout-unique group identifier for GRPO
        "reward":          float,  # scalar reward from post_run (exit-code gated)
        "coverage_reward": float | None,  # raw TF-IDF component before exit-code gate
        "exit_code":       int | None,    # target exit code, None on crash / missing file
        "log_prob":        float,  # mean per-token log-prob under the behaviour actor
        "ref_log_prob":    float | None,  # reference model log-prob (KL term, optional)
        executed_program: Optional[bytes] = None,
    }
"""

import csv
import os
import statistics
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# RolloutBuffer — written by mutator.fuzz_one() and mutator.on_post_run()
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Single-threaded in-memory two-phase store.

    AFL++ drives the Python mutator from one thread, so no locking is needed.
    """

    def __init__(self, logger: "RolloutLogger | None" = None):
        self._records: dict[str, dict] = {}
        self._counter: int = 0
        self._logger = logger

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
        executed_program: Optional[bytes] = None,
        ref_log_prob: Optional[float] = None,
        value_pred:   Optional[float] = None,
    ) -> None:
        """Record a new mutant immediately after generation. reward fills in post_run()."""
        self._records[sample_id] = {
            "sample_id":       sample_id,
            "group_id":        group_id,
            "x_t":             x_t,
            "y_t":             y_t,
            "executed_program": executed_program,
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
        """Return all rewarded records, log them once, and clear the store."""
        complete = [r for r in self._records.values() if r["reward"] is not None]
        if self._logger is not None:
            self._logger.log_rollout(complete)
        self._records.clear()
        return complete

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Rollout logging
# ---------------------------------------------------------------------------

class RolloutLogger:
    """Writes rollout CSV/artifacts and rollout/ scalar summaries."""

    _CSV_HEADER = (
        "finetune_id", "sample_id", "group_id", "reward",
        "coverage_reward", "exit_code", "log_prob", "ref_log_prob",
        "executed_program_path",
    )

    def __init__(
        self,
        output_dir: str,
        enabled: bool,
        log_fn: Callable[[dict], None] | None = None,
    ):
        self.enabled = enabled
        self.log_fn = log_fn
        self._finetune_id = 0
        self._rollout_csv_path = os.path.join(output_dir, "rollout_samples.csv")
        self._artifact_root = os.path.join(output_dir, "rollout_artifacts")

    def log_rollout(self, records: list[dict]) -> None:
        """Dump per-sample rows and rollout/ scalars for one dataset."""
        if not records or not self.enabled:
            return

        finetune_id = self._finetune_id
        self._finetune_id += 1

        output_dir = os.path.dirname(self._rollout_csv_path)
        os.makedirs(output_dir, exist_ok=True)
        write_header = not os.path.exists(self._rollout_csv_path)
        with open(self._rollout_csv_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(self._CSV_HEADER)
            for r in records:
                artifact_paths = self._write_rollout_artifacts(finetune_id, r)
                writer.writerow((
                    finetune_id,
                    r["sample_id"],
                    r["group_id"],
                    r["reward"],
                    _csv_opt(r.get("coverage_reward")),
                    _csv_opt(r.get("exit_code")),
                    r["log_prob"],
                    _csv_opt(r.get("ref_log_prob")),
                    artifact_paths["executed_program_path"],
                ))

        if self.log_fn is not None:
            self.log_fn(_rollout_scalars(finetune_id, records))

    def _write_rollout_artifacts(self, finetune_id: int, record: dict) -> dict[str, str]:
        """Persist replay/debug artifacts for one rollout sample."""
        artifact_dir = os.path.join(self._artifact_root, f"finetune_{finetune_id:06d}")
        os.makedirs(artifact_dir, exist_ok=True)

        sample_id = record["sample_id"]
        executed_path = _write_bytes(
            os.path.join(artifact_dir, f"{sample_id}.executed.bin"),
            record.get("executed_program"),
        )
        return {
            "executed_program_path": executed_path,
        }


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


def _rollout_scalars(finetune_id: int, records: list[dict]) -> dict[str, float]:
    rewards   = [r["reward"] for r in records]
    cov_vals  = [r["coverage_reward"] for r in records if r.get("coverage_reward") is not None]
    n_samples = len(records)
    n_groups  = len({r["group_id"] for r in records})
    valid_rate = sum(1 for r in records if r.get("exit_code") == 0) / n_samples
    error_rate = sum(1 for r in records if (r.get("exit_code") or 0) != 0) / n_samples

    groups: dict[int, list[float]] = {}
    group_exit_codes: dict[int, set[int | None]] = {}
    for r in records:
        groups.setdefault(r["group_id"], []).append(r["reward"])
        group_exit_codes.setdefault(r["group_id"], set()).add(r.get("exit_code"))
    group_means = [statistics.fmean(g) for g in groups.values()]
    group_stds  = [statistics.pstdev(g) if len(g) > 1 else 0.0 for g in groups.values()]
    mixed_exit_groups = sum(1 for exits in group_exit_codes.values() if len(exits) > 1)
    all_valid_groups  = sum(1 for exits in group_exit_codes.values() if exits == {0})
    all_invalid_groups = sum(1 for exits in group_exit_codes.values() if 0 not in exits)

    return {
        "rollout/finetune_id":          float(finetune_id),
        "rollout/reward_mean":          statistics.fmean(rewards),
        "rollout/reward_std":           statistics.pstdev(rewards) if n_samples > 1 else 0.0,
        "rollout/reward_min":           min(rewards),
        "rollout/reward_max":           max(rewards),
        "rollout/valid_rate":           valid_rate,
        "rollout/error_rate":           error_rate,
        "rollout/coverage_reward_mean": statistics.fmean(cov_vals) if cov_vals else 0.0,
        "rollout/group_reward_mean":    statistics.fmean(group_means),
        "rollout/group_reward_std_mean": statistics.fmean(group_stds) if group_stds else 0.0,
        "rollout/mixed_exit_group_rate": mixed_exit_groups / n_groups if n_groups else 0.0,
        "rollout/all_valid_group_rate": all_valid_groups / n_groups if n_groups else 0.0,
        "rollout/all_invalid_group_rate": all_invalid_groups / n_groups if n_groups else 0.0,
        "rollout/n_samples":            float(n_samples),
        "rollout/n_groups":             float(n_groups),
    }


def _csv_opt(v) -> str:
    """Render None as empty string for CSV; otherwise str()."""
    return "" if v is None else str(v)


def _write_bytes(path: str, data: bytes | None) -> str:
    if data is None:
        return ""
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# ---------------------------------------------------------------------------
# GroupedBatchSampler — keeps each GRPO reward group in one batch
# ---------------------------------------------------------------------------

class GroupedBatchSampler(Sampler[list[int]]):
    """Yield minibatches made from complete GRPO groups.

    The minibatch size is controlled by ``batch_size``. If ``batch_size`` equals
    ``group_size``, each batch contains one group; if it is larger, each batch
    contains multiple complete groups.
    """

    def __init__(self, dataset: RolloutDataset, batch_size: int, group_size: int):
        self._batches: list[list[int]] = []

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, got {batch_size}."
            )
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size <= 0
        ):
            raise ValueError(
                f"group_size should be a positive integer value, got {group_size}."
            )
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of group_size ({group_size})."
            )

        groups: dict[int, list[int]] = {}
        group_order: list[int] = []
        for idx in range(len(dataset)):
            try:
                gid = int(dataset[idx]["group_id"])
            except KeyError as exc:
                raise ValueError(
                    f"GroupedBatchSampler requires record {idx} to have 'group_id'."
                ) from exc

            if gid not in groups:
                groups[gid] = []
                group_order.append(gid)
            groups[gid].append(idx)

        for gid in group_order:
            group = groups[gid]
            if len(group) != group_size:
                raise ValueError(
                    f"Group {gid} has {len(group)} samples, expected {group_size}."
                )

        if len(dataset) % batch_size != 0:
            raise ValueError(
                f"dataset size ({len(dataset)}) must be divisible by batch_size ({batch_size}) "
                "so every optimizer step uses a full GRPO batch."
            )

        batch: list[int] = []
        for gid in group_order:
            group = groups[gid]
            batch.extend(group)
            if len(batch) == batch_size:
                self._batches.append(batch)
                batch = []

        if batch:
            raise ValueError(
                f"Internal error: built a partial batch of {len(batch)} samples; "
                f"expected full batches of {batch_size}."
            )

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
