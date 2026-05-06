"""BaseTrainer for rlm_mutator.

Subclasses HuggingFace `Trainer`.  Owns the model, the LoRA wrapper (applied
conditionally from cfg), and the reference model used for KL penalties override.
Policy Gradient Algorithm overides `compute_loss`

(see docs/design.md):
BaseTrainer handles training and models 
Mutator handles tokenization, rollouts and dataset creation.
"""

import copy
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config  import ModelConfig, TrainingConfig
from rollout import GroupedBatchSampler, RolloutCollator, RolloutDataset


class BaseTrainer(Trainer):
    """
    Construction end-to-end from config:
      1. Load tokenizer + base model.
      2. Wrap with LoRA iff training_cfg.lora_r > 0.
      3. Build TrainingArguments + RolloutCollator from training_cfg.
      4. Hand everything to HF Trainer via super().__init__.
      5. Initialise _ref_model to a full deepcopy (so ref_logprob works before
         the first finetune cycle has run).
    """

    def __init__(self, model_cfg: ModelConfig, training_cfg: TrainingConfig):
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
        model     = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.model_name_or_path)
        output_dir = _resolve_output_dir() # gets afl output dir

        if training_cfg.grpo is not None and training_cfg.train_batch_size % training_cfg.grpo.group_size != 0:
            raise ValueError(
                f"TrainingConfig.train_batch_size ({training_cfg.train_batch_size}) must be a multiple of "
                f"GRPOConfig.group_size ({training_cfg.grpo.group_size})."
            )

        if training_cfg.lora_r > 0:
            from peft import LoraConfig, TaskType, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r              = training_cfg.lora_r,
                lora_alpha     = training_cfg.lora_alpha,
                lora_dropout   = training_cfg.lora_dropout,
                target_modules = training_cfg.lora_target_modules_list,
                task_type      = TaskType.SEQ_2_SEQ_LM,
            ))

        args = TrainingArguments(
            output_dir                  = output_dir,
            per_device_train_batch_size = training_cfg.train_batch_size,
            learning_rate               = training_cfg.learning_rate,
            num_train_epochs            = training_cfg.num_train_epochs,
            save_strategy               = "no",
            report_to                   = "tensorboard" if training_cfg.enable_logging else "none",
            logging_strategy            = "steps" if training_cfg.enable_logging else "no",
            logging_steps               = training_cfg.logging_steps if training_cfg.enable_logging else 500,
            remove_unused_columns       = False,   # keep custom fields in the batch dict
        )

        super().__init__(
            model         = model,
            args          = args,
            data_collator = RolloutCollator(tokenizer),
            tokenizer     = tokenizer,
        )

        self.model_cfg    = model_cfg
        self.training_cfg = training_cfg
        self._ref_model   = None
        self.snapshot_ref()

    def snapshot_ref(self) -> None:
        """Re-anchor pi_ref to the current actor weights.

        First call: full deepcopy (needed for both full-FT and LoRA to allocate
        the reference model).  Subsequent calls with LoRA active: copy only the
        adapter weights into the existing ref, leaving the frozen base intact.
        """
        if self._ref_model is None:
            self._ref_model = copy.deepcopy(self.model)
        elif self.training_cfg.lora_r > 0:
            _copy_lora_weights(self.model, self._ref_model)
        else:
            self._ref_model = copy.deepcopy(self.model)

    def ref_logprob(self, x_t: list[int], y_t: list[int]) -> float:
        """mean per-token log-probability pi_ref(y_t | x_t)."""
        return self.sequence_logprob(self._ref_model, x_t, y_t)

    def sequence_logprob(self, model, x_t: list[int], y_t: list[int]) -> float:
        """Mean per-token log pi(y_t | x_t) under the given model.

        Uses the seq2seq negative log-likelihood exposed by HF models when
        `labels` is passed.  Returns -loss (so higher = more likely).
        """
        input_ids   = torch.tensor([x_t], dtype=torch.long, device=model.device)
        decoder_ids = torch.tensor(
            [y_t + [self.tokenizer.eos_token_id]], dtype=torch.long, device=model.device,
        )
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=decoder_ids)
        return -out.loss.item()

    def kl_divergence(self, logprob, ref_logprob):
        """Token-level log-ratio used as a KL approximation.

        PPO/GRPO sum/average this across a batch.  Full-distribution KL
        would need the softmax tensor, which is not stored in rollouts.
        """
        return logprob - ref_logprob

    # ------------------------------------------------------------------
    # Dataset injection — called by Mutator.maybe_finetune()
    # ------------------------------------------------------------------

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # custom grouped sampler for grpo
        if self.training_cfg.grpo is not None:
            group_size = self.training_cfg.grpo.group_size
            if self.args.per_device_train_batch_size % group_size != 0:
                raise ValueError(
                    f"per_device_train_batch_size ({self.args.per_device_train_batch_size}) must be a multiple of GRPO group_size ({group_size})."
                )
            return DataLoader(
                self.train_dataset,
                batch_sampler = GroupedBatchSampler(
                    dataset=    self.train_dataset, 
                    batch_size= self.args.per_device_train_batch_size,
                    group_size= group_size
                    ),
                collate_fn    = self.data_collator,
            )

        return DataLoader(
            self.train_dataset,
            batch_size = self.args.per_device_train_batch_size,
            sampler    = SequentialSampler(self.train_dataset),
            collate_fn = self.data_collator,
            drop_last  = False,
        )

    def set_rollout_dataset(self, dataset: RolloutDataset) -> None:
        self.train_dataset = dataset


    # ------------------------------------------------------------------
    # compute_loss — overridden by PPOTrainer / GRPOTrainer
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """ compute_loss: overridden by specific Policy Gradient Algorithm """
        raise NotImplementedError("Policy Gradient Algorithm must override compute_loss")

def _copy_lora_weights(src, dst) -> None:
    """Copy only lora_ keys from src state_dict into dst — O(|phi|), not O(|theta|)."""
    dst.load_state_dict(
        {k: v for k, v in src.state_dict().items() if "lora_" in k},
        strict=False,
    )

def _resolve_output_dir() -> str:
    """Choose a durable run-local trainer output directory.

    Preference order:
      1. Explicit RLM_OUTPUT_DIR override.
      2. AFL's custom-mutator output dir for this run.
      3. AFL's general output dir.
      4. /tmp fallback for non-AFL testing.
    """
    base_dir = (
        os.environ.get("RLM_OUTPUT_DIR")
        or os.environ.get("AFL_CUSTOM_INFO_OUT")
        or os.environ.get("__AFL_OUT_DIR")
    )
    if base_dir:
        return os.path.join(base_dir, "rlm_trainer")
    return "/tmp/rlm_trainer"
