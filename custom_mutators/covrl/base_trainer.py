"""Cycle-level finetuning for the CovRL / RLLM mutator.

Public API:
    BaseTrainer  — finetune(rollout) called by Mutator at each cycle boundary.
    CovRLTrainer — trains an 8-class critic on raw D_T, then runs a PPO mutator
                   update using R_learned (critic predictions).
    RLLMTrainer  — runs a PPO or GRPO mutator update using R_direct (stored R);
                   no critic.

The policy loss (ppo.loss / grpo.loss) is injected at construction. The HF
Trainer wiring is shared between CovRL and RLLM — they differ only in (a)
whether a critic is trained, and (b) which reward function is plugged in.
"""

import copy
from typing import Callable

from .config import Config
from .rollout import RolloutBuffer

_MUTATOR_TRAINER_CLS = None


class BaseTrainer:
    def __init__(self, cfg: Config, model, policy_loss: Callable):
        self.cfg = cfg
        self.model = model
        self.policy_loss = policy_loss
        self._cycle = 0

    def finetune(self, rollout: RolloutBuffer) -> None:
        raise NotImplementedError


class CovRLTrainer(BaseTrainer):
    """CovRL: 8-class critic trained on raw D_T; PPO mutator update on R_learned."""

    def __init__(self, cfg, model, policy_loss):
        super().__init__(cfg, model, policy_loss)
        self.rewarder = None

    def finetune(self, rollout: RolloutBuffer) -> None:
        from .utils import mix_corpus

        if self.rewarder is None:
            self.rewarder = self._build_rewarder()

        d_mix = mix_corpus(rollout, self.cfg)

        # Critic is trained on raw D_T (no corpus mix), per design.
        self._train_rewarder(rollout)

        # Mutator update is skipped on the first cycle per CovRL spec.
        if self._cycle > 0:
            _run_mutator_finetune(
                self.cfg,
                self.model,
                d_mix,
                policy_loss=self.policy_loss,
                reward_fn=_LearnedReward(self.rewarder, self.cfg),
            )
        self._cycle += 1

    def _build_rewarder(self):
        from transformers import AutoModelForSequenceClassification

        path = self.cfg.rewarder_path or self.cfg.model_path
        rewarder = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=8
        )
        rewarder.to(self.cfg.device)
        return rewarder

    def _train_rewarder(self, rollout: RolloutBuffer) -> None:
        from transformers import Trainer, TrainingArguments

        from .utils import rewarder_collator, rewarder_dataset

        args = TrainingArguments(
            output_dir=f"{self.cfg.checkpoint_dir}/rewarder",
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=1,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )
        Trainer(
            model=self.rewarder,
            args=args,
            train_dataset=rewarder_dataset(rollout, self.cfg),
            data_collator=rewarder_collator(self.cfg),
        ).train()


class RLLMTrainer(BaseTrainer):
    """RLLM: PPO or GRPO mutator update with R_direct; no critic."""

    def finetune(self, rollout: RolloutBuffer) -> None:
        from .utils import mix_corpus

        d_mix = mix_corpus(rollout, self.cfg)
        _run_mutator_finetune(
            self.cfg,
            self.model,
            d_mix,
            policy_loss=self.policy_loss,
            reward_fn=_direct_reward,
        )
        self._cycle += 1


def _direct_reward(inputs):
    """R_direct: return the stored R from the batch dict."""
    return inputs["R"]


class _LearnedReward:
    """R_learned: query critic for class predictions, map to scalar."""

    def __init__(self, rewarder, cfg: Config):
        self.rewarder = rewarder
        self.cfg = cfg

    def __call__(self, inputs):
        raise NotImplementedError(
            "critic inference: forward (x, y) through rewarder, argmax over 8 "
            "classes, map class → scalar reward"
        )


def _run_mutator_finetune(cfg, model, d_mix, policy_loss, reward_fn):
    """One mutator training pass with L = E[L_policy + L_CE]."""
    from transformers import TrainingArguments

    from .utils import mutator_collator

    prev = _snapshot(model.model)
    args = TrainingArguments(
        output_dir=cfg.checkpoint_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=1,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    TrainerCls = _mutator_trainer_class()
    TrainerCls(
        model=model.model,
        args=args,
        train_dataset=d_mix,
        data_collator=mutator_collator(model.tokenizer),
        prev_model=prev,
        policy_loss=policy_loss,
        reward_fn=reward_fn,
        cfg=cfg,
    ).train()


def _snapshot(model):
    snap = copy.deepcopy(model)
    for p in snap.parameters():
        p.requires_grad_(False)
    snap.eval()
    return snap


def _mutator_trainer_class():
    """Build the HF Trainer subclass lazily (cached) so transformers/torch stay
    lazy imports."""
    global _MUTATOR_TRAINER_CLS
    if _MUTATOR_TRAINER_CLS is not None:
        return _MUTATOR_TRAINER_CLS

    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    def _seq_log_prob(logits, labels):
        """Per-sample log p(y|x) summed over non-ignored label positions."""
        log_probs = F.log_softmax(logits, dim=-1)
        mask = labels != -100
        safe = labels.masked_fill(~mask, 0)
        gathered = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        return (gathered * mask.float()).sum(dim=-1)

    class _MutatorTrainer(Trainer):
        def __init__(self, *args, prev_model, policy_loss, reward_fn, cfg, **kw):
            super().__init__(*args, **kw)
            self.prev_model = prev_model
            self.policy_loss = policy_loss
            self.reward_fn = reward_fn
            self.cfg = cfg

        def compute_loss(self, model, inputs, return_outputs=False, **_):
            fwd = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": inputs["labels"],
            }

            out = model(**fwd)
            log_p_theta = _seq_log_prob(out.logits, inputs["labels"])

            with torch.no_grad():
                prev_logits = self.prev_model(**fwd).logits
            log_p_prev = _seq_log_prob(prev_logits, inputs["labels"])

            ratios = torch.exp(log_p_theta - log_p_prev)
            R_hat = self.reward_fn(inputs)

            policy = self.policy_loss(
                ratios=ratios,
                R_hat=R_hat,
                group_ids=inputs.get("group_id"),
                cfg=self.cfg,
            )
            ce = -log_p_theta.mean()
            loss = policy + ce
            return (loss, out) if return_outputs else loss

    _MUTATOR_TRAINER_CLS = _MutatorTrainer
    return _MUTATOR_TRAINER_CLS
