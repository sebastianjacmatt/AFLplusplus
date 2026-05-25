"""CovRL training orchestrator.

Common interface between the mutator and the training loop. Called by
Mutator._maybe_finetune() every cfg.finetune_every queue cycles.

Training order mirrors CovRL-Fuzz/covrl/models/inferencer.py:finetune():
  1. generate_rollout  — score queue files with afl-showmap, build shared dataset
  2. train_critic      — HF Trainer + CriticDataset + CriticCollator, every iteration
  3. finetune_actor    — HF ActorTrainer + ActorDataset + ActorCollator, skip first iter
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from config import MutatorConfig
from data.masking import Masking
from data.rewarding import Rewarding
from data.rollout import RolloutDataset, generate_rollout
from model.critic import LABEL_TO_REWARD, CriticModel
from model.llm import Model
from training.datasets import ActorCollator, ActorDataset, CriticCollator, CriticDataset


class _ActorTrainer(Trainer):
    """HF Trainer subclass that substitutes a custom PPO+CE loss.

    Mirrors CovRL's ActorTrainer (finetuner.py:18-29). loss_fn is passed
    directly rather than via compute_loss_func (added in transformers>=4.41)
    so this works on older versions too.
    """

    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = self._loss_fn(outputs, inputs)
        return (loss, outputs) if return_outputs else loss


class CovRLTrainer:
    """Orchestrates critic + actor updates on each finetune() call."""

    def __init__(
        self,
        model: Model,
        masking: Masking,
        rewarding: Rewarding,
        cfg: MutatorConfig,
        queue_dir: Path,
    ):
        self.model = model
        self.masking = masking
        self.rewarding = rewarding
        self.cfg = cfg
        self.queue_dir = queue_dir
        self.critic: CriticModel | None = None
        self._iteration = 0

        out_dir = os.environ.get("AFL_CUSTOM_INFO_OUT", "/tmp")
        self._model_output_dir = os.path.join(out_dir, "model_output")

    def _make_training_args(self, epochs: int) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self._model_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.cfg.train_batch_size,
            learning_rate=self.cfg.learning_rate,
            evaluation_strategy="no",
            save_strategy="no",
            fp16=False,
            no_cuda=False,
        )

    def finetune(self) -> None:
        """Entry point called by Mutator._maybe_finetune()."""
        if self.critic is None:
            self.critic = CriticModel(self.cfg.model_name_or_path, self.cfg.device)

        dataset = generate_rollout(self.model.tokenizer, self.rewarding, self.queue_dir)
        if len(dataset) == 0:
            return

        self._train_critic(dataset)
        if self._iteration > 0:
            # Actor skipped on first iteration — mirrors inferencer.py:71
            self._train_actor(dataset)

        self._iteration += 1

    def _train_critic(self, dataset: RolloutDataset) -> None:
        """Train critic with standard HF Trainer.

        Mirrors CovRL FineTuner.train_critic (finetuner.py:112-132).
        """
        trainer = Trainer(
            model=self.critic,
            args=self._make_training_args(self.cfg.critic_epochs),
            train_dataset=CriticDataset(dataset, self.masking),
            data_collator=CriticCollator(),
        )
        trainer.train()

    def _compute_actor_loss(self, cur_outputs, inputs) -> torch.Tensor:
        """PPO + CE loss. Mirrors finetuner.py:compute_actor_loss (lines 134-173).

        cur_outputs: forward pass result from the current actor (run by _ActorTrainer).
        inputs: collated batch dict {input_ids, attention_mask, labels, ...}.
        _prev_actor and critic are captured from self via closure.
        """
        device = cur_outputs.logits.device

        cur_predictions = cur_outputs.logits.argmax(dim=-1)
        critic_inputs = {
            "input_ids": torch.cat([inputs["input_ids"], cur_predictions], dim=1),
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.ones_like(cur_predictions)], dim=1
            ),
        }

        with torch.no_grad():
            prev_outputs = self._prev_actor(**inputs)
            critic_scores = self.critic(**critic_inputs)

        # Reward from critic argmax — finetuner.py:157-159
        pred_labels = critic_scores.argmax(dim=-1)
        reward = torch.tensor(
            [LABEL_TO_REWARD[l.item()] for l in pred_labels],
            dtype=torch.float,
            device=device,
        ).view(-1, 1, 1)

        # PPO ratio + clip [0.8, 1.2] — finetuner.py:161-169
        log_softmax = nn.LogSoftmax(dim=-1)
        cur_log = log_softmax(cur_outputs.logits)
        prev_log = log_softmax(prev_outputs.logits)
        ratio = torch.exp(cur_log - prev_log)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        ppo_loss = -torch.min(ratio * reward, clipped * reward).mean()

        # Final loss: PPO + CE — finetuner.py:172
        return ppo_loss + cur_outputs.loss.mean()

    def _train_actor(self, dataset: RolloutDataset) -> None:
        """Fine-tune actor with PPO+CE via _ActorTrainer.

        Mirrors CovRL FineTuner.finetune_actor (finetuner.py:175-205).
        prev_actor is a frozen snapshot of the current weights used to compute
        the PPO probability ratio.
        """
        self._prev_actor = copy.deepcopy(self.model._hf)
        self._prev_actor.eval()
        self.critic.eval()

        try:
            trainer = _ActorTrainer(
                loss_fn=self._compute_actor_loss,
                model=self.model._hf,
                args=self._make_training_args(self.cfg.actor_epochs),
                train_dataset=ActorDataset(
                    dataset, self.masking, self.model.tokenizer.pad_token_id
                ),
                data_collator=ActorCollator(self.model.tokenizer.pad_token_id),
            )
            trainer.train()
        finally:
            del self._prev_actor
            self.model._hf.eval()
