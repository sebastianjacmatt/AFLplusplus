"""Actor and Trainer for rlm_mutator.

Owns everything model-related:

    - Model and tokenizer loading (from_pretrained)
    - Sentinel conversion for T5-style infilling
    - Generation and old log-prob computation
    - Optional reference model scoring for KL penalty
    - LoRA adapter wrapping (via peft)
    - finetune() stub — override in PPO/GRPO subclasses

rlm.py owns AFL integration: masking, store logging, reward patching,
and finetune scheduling.  It calls Trainer.infill(masked_ids) and
Trainer.finetune(records) and nothing else.
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import ModelConfig, TrainingConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Infill result
# ---------------------------------------------------------------------------

@dataclass
class InfillResult:
    """Token-level output from one infill call.

    @param infilled_ids: Reconstructed full token sequence (encoder tokens with
                         mask positions replaced by generated predictions).
    @param old_logprob:  Mean per-token log-probability of y_t under the actor
                         at generation time.  Used as the behaviour policy log-prob
                         in PPO/GRPO importance-sampling ratio.
    @param x_t:          Padded encoder input IDs with T5 extra-token sentinels
                         substituted for MASK positions.  Stored as the context
                         for the training dataset.
    @param y_t:          Decoder output token IDs (excluding EOS and decoder-start).
                         Stored as the action for the training dataset.
    """
    infilled_ids: list[int]
    old_logprob:  float
    x_t:          list[int]
    y_t:          list[int]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Loads the actor (and optionally a frozen reference model) and exposes
    infill() for mutation and finetune() for RL updates.

    Subclass and override finetune() to implement PPO or GRPO.  The base
    implementation logs the batch stats and returns without updating weights.

    @param model_cfg:    ModelConfig — identity, device, generation settings.
    @param training_cfg: TrainingConfig — algorithm, LoRA, optimiser settings.
    """

    def __init__(self, model_cfg: ModelConfig, training_cfg: TrainingConfig):
        self._model_cfg    = model_cfg
        self._training_cfg = training_cfg
        self._device       = model_cfg.resolve_device()

        log.info("[trainer] loading %s on %s", model_cfg.model_name_or_path, self._device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
        self.actor     = AutoModelForSeq2SeqLM.from_pretrained(
            model_cfg.model_name_or_path
        ).to(self._device)

        if training_cfg.lora_r > 0:
            self.actor = self._wrap_lora(self.actor, training_cfg)

        self.actor.eval()

        self._vocab_size  = self.tokenizer.vocab_size
        self._mask_token  = self.tokenizer.mask_token_id
        self._unk_token   = self.tokenizer.unk_token_id
        self._pad_token   = self.tokenizer.pad_token_id
        self._eos_token   = self.tokenizer.eos_token_id
        self._max_pred    = round(model_cfg.max_length * model_cfg.mask_probability)

        self._ref_model: AutoModelForSeq2SeqLM | None = None

    # ------------------------------------------------------------------
    # Public properties (read by rlm.py for masking)
    # ------------------------------------------------------------------

    @property
    def mask_token(self) -> int:
        """Token ID used by the masking strategy in rlm.py."""
        return self._mask_token

    # ------------------------------------------------------------------
    # Tokenisation helpers (called by rlm.py)
    # ------------------------------------------------------------------

    def tokenize(self, buf: bytearray) -> list[int]:
        """Decode AFL++ bytes to UTF-8 and tokenize without special tokens.

        @param buf: Raw bytes from AFL++.
        @return:    List of token IDs.
        """
        text = buf.decode("utf-8", errors="replace")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode(self, token_ids: list[int]) -> bytes:
        """Detokenize a token ID sequence back to UTF-8 bytes for AFL++.

        @param token_ids: List of token IDs.
        @return:          UTF-8 encoded bytes.
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.encode("utf-8")

    # ------------------------------------------------------------------
    # Infill (called by rlm.py per fuzz() call)
    # ------------------------------------------------------------------

    def infill(self, masked_token_ids: list[int]) -> InfillResult:
        """Generate predictions for all MASK positions in one forward+decode pass.

        Steps:
          1. Convert MASK sentinels to T5-style extra tokens.
          2. Pad to a batch of 1 and run actor.generate() with output_scores=True.
          3. Compute old_logprob from per-step scores (no extra forward pass).
          4. Reconstruct the full token sequence.

        @param masked_token_ids: Token sequence with self.mask_token sentinels.
        @return: InfillResult with infilled_ids, old_logprob, x_t, y_t.
        """
        converted, mask_dict = self._sentinelize(masked_token_ids)

        if not mask_dict:
            # No masks applied — return identity with zero logprob metadata.
            return InfillResult(
                infilled_ids = list(masked_token_ids),
                old_logprob  = 0.0,
                x_t          = list(masked_token_ids),
                y_t          = [],
            )

        if len(converted) > self._model_cfg.max_length - 3:
            log.warning(
                "[trainer] sequence length %d exceeds max_length %d; truncating",
                len(converted), self._model_cfg.max_length,
            )
            converted = converted[:self._model_cfg.max_length - 3]

        padded    = converted + [self._pad_token]
        attn_mask = [1] * len(converted) + [0]

        input_ids_t = torch.tensor([padded],    dtype=torch.long, device=self._device)
        attn_mask_t = torch.tensor([attn_mask], dtype=torch.long, device=self._device)

        with torch.no_grad():
            outputs = self._generate(input_ids_t, attn_mask_t)

        y_t, old_logprob = self._extract_logprob(outputs)
        infilled_ids     = self._reconstruct(padded, mask_dict, outputs.sequences.tolist()[0])

        return InfillResult(
            infilled_ids = infilled_ids,
            old_logprob  = old_logprob,
            x_t          = padded,
            y_t          = y_t,
        )

    def ref_logprob(self, x_t: list[int], y_t: list[int]) -> float:
        """Compute log-prob of y_t under the frozen reference model.

        Returns 0.0 if no reference model is loaded (kl_coef == 0).

        @param x_t: Encoder input token IDs (from InfillResult.x_t).
        @param y_t: Decoder token IDs (from InfillResult.y_t).
        @return:    Mean per-token log-prob under the reference model.
        """
        if self._ref_model is None:
            return 0.0

        input_ids_t  = torch.tensor([x_t], dtype=torch.long, device=self._device)
        decoder_ids  = torch.tensor([y_t + [self._eos_token]], dtype=torch.long, device=self._device)

        with torch.no_grad():
            out = self._ref_model(input_ids=input_ids_t, labels=decoder_ids)

        # out.loss is mean NLL; negate to get mean log-prob
        return -out.loss.item()

    # ------------------------------------------------------------------
    # Finetune (override in PPO / GRPO subclasses)
    # ------------------------------------------------------------------

    def finetune(self, records: list[dict]) -> None:
        """Run one RL update over a completed rollout group.

        Base implementation logs summary statistics and returns.
        Subclasses should override this to implement PPO or GRPO weight updates.

        @param records: List of store records from RLStore.close_group().
                        Each dict contains sample_id, group_id, x_t, y_t,
                        log_prob, reward, ref_log_prob, value_pred, interesting.
        """
        if not records:
            return
        rewards = [r["reward"] for r in records]
        log.info(
            "[trainer] finetune stub — %d records | reward mean=%.4f min=%.4f max=%.4f",
            len(records),
            sum(rewards) / len(rewards),
            min(rewards),
            max(rewards),
        )

    # ------------------------------------------------------------------
    # Reference model management
    # ------------------------------------------------------------------

    def load_ref_model(self) -> None:
        """Load a frozen copy of the actor as the KL reference model.

        Call after init if training_cfg.kl_coef > 0.  The reference model
        is never updated and stays on the same device as the actor.
        """
        self._ref_model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_cfg.model_name_or_path
        ).to(self._device)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        log.info("[trainer] reference model loaded (frozen)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sentinelize(self, token_ids: list[int]) -> tuple[list[int], dict]:
        """Replace MASK sentinels with T5-style extra tokens.

        @return: (converted_ids, mask_dict) where mask_dict maps
                 extra_token_id -> original_position_in_converted.
        """
        converted    = list(token_ids)
        mask_dict    = {}
        sentinel_n   = 0

        for i, tok in enumerate(converted):
            if tok == self._mask_token or tok == self._unk_token:
                sentinel_n   += 1
                extra_id      = self._vocab_size - sentinel_n
                mask_dict[extra_id] = i
                converted[i]        = extra_id

        return converted, mask_dict

    def _generate(self, input_ids_t, attn_mask_t):
        cfg = self._model_cfg
        if cfg.sample_method == "contrastive":
            return self.actor.generate(
                input_ids      = input_ids_t,
                attention_mask = attn_mask_t,
                do_sample      = True,
                penalty_alpha  = cfg.penalty_alpha,
                top_k          = cfg.top_k,
                eos_token_id   = self._eos_token,
                no_repeat_ngram_size = 3,
                min_length     = 1,
                max_length     = self._max_pred,
                output_scores  = True,
                return_dict_in_generate = True,
            )
        return self.actor.generate(
            input_ids      = input_ids_t,
            attention_mask = attn_mask_t,
            eos_token_id   = self._eos_token,
            no_repeat_ngram_size = 3,
            max_length     = self._max_pred,
            output_scores  = True,
            return_dict_in_generate = True,
        )

    def _extract_logprob(self, outputs) -> tuple[list[int], float]:
        """Collect generated token IDs and compute mean per-token log-prob."""
        sequences    = outputs.sequences
        scores       = outputs.scores
        y_t          = []
        log_prob_sum = 0.0

        for t, step_score in enumerate(scores):
            tok = sequences[0, t + 1].item()   # +1 skips decoder_start_token
            if tok == self._eos_token:
                break
            lp = F.log_softmax(step_score[0], dim=-1)[tok].item()
            log_prob_sum += lp
            y_t.append(tok)

        old_logprob = log_prob_sum / max(len(y_t), 1)
        return y_t, old_logprob

    def _reconstruct(self, padded: list[int], mask_dict: dict, predictions: list[int]) -> list[int]:
        """Splice generated tokens back at their masked positions."""
        result_dict  = {extra_id: (pos, []) for extra_id, pos in mask_dict.items()}
        prev_mask_id = None

        for pred in predictions:
            if pred in mask_dict:
                prev_mask_id = pred
            elif pred > (self._vocab_size - 100) or pred == self._eos_token:
                prev_mask_id = None
            elif prev_mask_id is not None:
                result_dict[prev_mask_id][1].append(pred)

        new_inputs = []
        prev_pos   = 0
        for pos, preds in sorted(result_dict.values()):
            new_inputs.extend(padded[prev_pos:pos] + preds)
            prev_pos = pos + 1

        remaining = padded[prev_pos:]
        trim = next(
            (i for i, t in enumerate(remaining) if t == self._pad_token),
            len(remaining),
        )
        new_inputs.extend(remaining[:trim])

        return new_inputs if new_inputs else list(padded)

    @staticmethod
    def _wrap_lora(model, training_cfg: TrainingConfig):
        """Apply LoRA adapters via peft if lora_r > 0."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft is required for LoRA; pip install peft")

        lora_cfg = LoraConfig(
            task_type      = TaskType.SEQ_2_SEQ_LM,
            r              = training_cfg.lora_r,
            lora_alpha     = training_cfg.lora_alpha,
            lora_dropout   = training_cfg.lora_dropout,
            target_modules = training_cfg.lora_target_modules_list,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        return model
