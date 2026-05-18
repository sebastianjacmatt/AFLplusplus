"""BaseTrainer: HF Trainer extended with a Strategy-injected PG loss.

Template Method (HF ``Trainer``) provides the training loop; Strategy
(``PolicyGradientAlgorithm``) provides the loss. See docs/design.md §2.5
(Template Method + Strategy) and §3.5 (BaseTrainer responsibility).

This file owns three loop-level concerns:

  * The reference-policy snapshot ``pi_ref`` (for KL terms).
  * The forward pass of both current and reference policies, whose outputs
    are handed to the strategy.
  * On-policy bookkeeping: clear the rollout buffer at the end of each
    finetune cycle so the next rollout is sampled from the updated policy.

Algorithm-specific math lives in ``policy_gradient_algorithms/``.
"""

import copy
from typing import TYPE_CHECKING

import torch
from transformers import Trainer, TrainingArguments

from rollout import RolloutCollator, RolloutDataset

if TYPE_CHECKING:
    from config import Config
    from mutator import Mutator
    from policy_gradient_algorithms.base import PolicyGradientAlgorithm


class BaseTrainer(Trainer):
    """HF Trainer with policy-gradient ``compute_loss`` and a reference policy."""

    def __init__(
        self,
        cfg: "Config",
        mutator: "Mutator",
        algorithm: "PolicyGradientAlgorithm",
        output_dir: str = "rllm_output",
    ) -> None:
        self.cfg = cfg
        self.mutator = mutator
        self.algorithm = algorithm
        self._finetune_count = 0
        self.ref_model = self._snapshot_reference(mutator.model)

        args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.train_batch_size,
            num_train_epochs=cfg.num_train_epochs,
            warmup_ratio=cfg.warmup_ratio,
            bf16=cfg.bf16,
            logging_steps=cfg.logging_steps,
            report_to="tensorboard" if cfg.enable_logging else "none",
            save_strategy="no",
            remove_unused_columns=False,
        )

        super().__init__(
            model=mutator.model,
            args=args,
            data_collator=RolloutCollator(mutator.tokenizer.pad_token_id),
            tokenizer=mutator.tokenizer,
        )

    def _get_train_sampler(self):
        """Delegate to the Strategy; fall back to HF's default if it returns None."""
        custom = self.algorithm.make_sampler(self.train_dataset, self.cfg)
        return custom if custom is not None else super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Forward current + reference policies; delegate to the Strategy."""
        model_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        with torch.no_grad():
            ref_out = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )

        loss = self.algorithm.loss(inputs, model_out, ref_out, self.cfg)
        return (loss, model_out) if return_outputs else loss

    def finetune(self, dataset: RolloutDataset) -> None:
        """One on-policy training cycle."""
        if len(dataset) == 0:
            return

        self.train_dataset = dataset
        self.model.train()
        self.train()
        self.model.eval()

        self._finetune_count += 1
        if self._finetune_count % self.cfg.ref_update_every == 0:
            self.ref_model = self._snapshot_reference(self.model)

        self.mutator.buffer.clear()

    def save_checkpoint(self) -> None:
        self.save_model()

    @staticmethod
    def _snapshot_reference(model: torch.nn.Module) -> torch.nn.Module:
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad = False
        return ref
