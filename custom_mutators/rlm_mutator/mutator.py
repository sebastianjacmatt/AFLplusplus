"""Mutator for rlm_mutator.

Owns everything that lives between AFL++ bytes and the trainer:

    - Tokenisation (byte <-> token-id)
    - Random masking (insert / overwrite)
    - Infill via the actor's generate() (T5-style sentinels)
    - Rollout buffer logging (reward filled later by post_run)
    - Finetune trigger (flushes buffer, runs trainer.train, re-anchors ref)

State that used to live as rlm.py module globals (cached seed tokens, current
masked context, group/sample counters, pending sample id) lives here as
instance attributes.

Trainer owns the model; Mutator accesses it through self.trainer.model.
"""

import logging
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config   import AFLConfig, TrainingConfig
from rollout  import RolloutBuffer
from base_trainer import BaseTrainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Infill result (token-level output from one infill call)
# ---------------------------------------------------------------------------

@dataclass
class InfillResult:
    """Token-level output from one infill call.

    @param infilled_ids: Reconstructed full token sequence (encoder tokens with
                         mask positions replaced by generated predictions).
    @param old_logprob:  Mean per-token log-probability of y_t under the actor
                         at generation time.  Used as the behaviour policy
                         log-prob in PPO/GRPO importance-sampling ratios.
    @param x_t:          Padded encoder input IDs with T5 extra-token sentinels
                         substituted for MASK positions.  Stored as context.
    @param y_t:          Decoder output token IDs (excluding EOS and decoder start).
    """
    infilled_ids: list[int]
    old_logprob:  float
    x_t:          list[int]
    y_t:          list[int]


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------

class Mutator:
    """Drives mutation for one AFL++ custom-mutator process.

    @param trainer:       BaseTrainer instance; owns model + tokenizer.
    @param buffer:        RolloutBuffer; receives per-sample records.
    @param afl_cfg:       AFLConfig; masking and fuzz budget knobs.
    @param training_cfg:  TrainingConfig; group_size for GRPO, kl_coef for ref scoring.
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        buffer: RolloutBuffer,
        afl_cfg: AFLConfig,
        training_cfg: TrainingConfig,
    ):
        self.trainer      = trainer
        self.buffer       = buffer
        self.afl_cfg      = afl_cfg
        self.training_cfg = training_cfg

        tok = trainer.tokenizer
        self._mask_token = tok.mask_token_id
        self._unk_token  = tok.unk_token_id
        self._pad_token  = tok.pad_token_id
        self._eos_token  = tok.eos_token_id
        self._vocab_size = tok.vocab_size
        self._max_pred   = round(trainer.model_cfg.max_length * trainer.model_cfg.mask_probability)
        self._max_new_tokens_per_mask = trainer.model_cfg.max_new_tokens_per_mask

        # Per-seed state (reset in on_new_seed)
        self._tokens:   list[int] | None = None
        self._masked:   list[int] | None = None
        self._group:    int               = -1
        self._sample:   int               = 0
        self._pending_sample_id: str | None = None
        self._next_group_id: int = 0

    # ------------------------------------------------------------------
    # Convenience accessor — mirrors design's trainer.model indirection
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self.trainer.model

    # ------------------------------------------------------------------
    # Called from rlm.py hooks
    # ------------------------------------------------------------------

    def on_new_seed(self, buf: bytearray) -> None:
        """fuzz_count(): tokenize once, reset per-seed state."""
        self._tokens = self.tokenize(buf)
        self._masked = None
        self._group  = -1
        self._sample = 0

    def generate(self, max_size: int) -> bytes | None:
        """fuzz(): one mutation.  Returns mutated bytes, or None to keep original.

        Re-masks at each `sample % group_size == 0` boundary so that every
        group_size samples in a group share one x_t (GRPO precondition). The
        recorded group_id is monotonic across seeds so grouped training never
        merges unrelated samples from different seeds.
        """
        if self._tokens is None:
            raise RuntimeError("Mutator.generate() called before on_new_seed()")

        if self._sample % self.training_cfg.group_size == 0:
            self._masked = self._random_mask(list(self._tokens))
            self._group = self._next_group_id
            self._next_group_id += 1

        masked = self._masked
        if masked is None or len(masked) <= 3:
            self._pending_sample_id = None
            return None

        result  = self.infill(masked)
        out_buf = self.encode(result.infilled_ids)
        if len(out_buf) > max_size:
            out_buf = out_buf[:max_size]

        ref_lp = (
            self.trainer.ref_logprob(result.x_t, result.y_t)
            if self.training_cfg.kl_coef > 0.0
            else None
        )

        sample_id = self.buffer.new_sample_id()
        self.buffer.log(
            sample_id        = sample_id,
            group_id         = self._group,
            x_t              = result.x_t,
            y_t              = result.y_t,
            log_prob         = result.old_logprob,
            masked_program   = self.render(masked, skip_special_tokens=False),
            generated_infill = self.render(result.y_t, skip_special_tokens=False),
            executed_program = bytes(out_buf),
            ref_log_prob     = ref_lp,
        )

        self._pending_sample_id = sample_id
        self._sample += 1
        return out_buf

    def on_post_run(
        self,
        reward:          float,
        coverage_reward: float | None = None,
        exit_code:       int   | None = None,
    ) -> None:
        """post_run(): patch reward + diagnostic fields on the pending sample."""
        if self._pending_sample_id is not None:
            self.buffer.patch_reward(
                self._pending_sample_id,
                reward,
                coverage_reward = coverage_reward,
                exit_code       = exit_code,
            )
        self._pending_sample_id = None

    def maybe_finetune(self) -> None:
        """Drain the rollout buffer and run one trainer.train() cycle."""
        records = self.buffer.flush()
        if not records:
            log.info("[mutator] maybe_finetune — buffer empty, skipping")
            return
        self.trainer.log_rollout_dataset(records)   # CSV + aggregated rollout/ scalars
        self.trainer.set_rollout_dataset(records)
        self.trainer.train()                        # HF Trainer rebuilds optimizer each call
        self.trainer.snapshot_ref()                 # re-anchor pi_ref after weights updated

    # ------------------------------------------------------------------
    # Byte <-> token-id conversion
    # ------------------------------------------------------------------

    def tokenize(self, buf: bytearray) -> list[int]:
        text = buf.decode("utf-8", errors="replace")
        return self.trainer.tokenizer.encode(text, add_special_tokens=False)

    def encode(self, token_ids: list[int]) -> bytes:
        text = self.trainer.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.encode("utf-8")

    def render(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        """Render token IDs to text for logging/debugging artifacts."""
        return self.trainer.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Random masking (insert / overwrite)
    # ------------------------------------------------------------------

    def _random_mask(self, token_ids: list[int]) -> list[int]:
        """Apply one mask mutation to a token sequence.

        The default 'span' strategy is intentionally conservative and closer
        to CodeT5/T5 span corruption than scattered token masking:
          - replace one contiguous span of 1..mask_count tokens with one MASK
          - optionally create a single insertion hole with insert_mask_prob

        The legacy 'scatter' strategy keeps the old random insert/overwrite
        behaviour for experiments.
        """
        result = list(token_ids)
        mask   = self._mask_token
        cap    = self.afl_cfg.mask_count

        if self.afl_cfg.mask_strategy == "scatter":
            mode = random.randint(0, 1)
            if mode == 0:
                count = random.randint(1, cap)
                for _ in range(count):
                    pos = random.randint(0, len(result))
                    result.insert(pos, mask)
            else:
                if result:
                    count     = random.randint(1, min(cap, len(result)))
                    positions = random.sample(range(len(result)), count)
                    for pos in positions:
                        result[pos] = mask
            return result

        if not result:
            return result

        if random.random() < self.afl_cfg.insert_mask_prob:
            pos = random.randint(0, len(result))
            result.insert(pos, mask)
            return result

        span_len = random.randint(1, min(cap, len(result)))
        start = random.randint(0, len(result) - span_len)
        result[start:start + span_len] = [mask]
        return result

    # ------------------------------------------------------------------
    # Infill — generate y_t ~ pi_theta(. | x_t)
    # ------------------------------------------------------------------

    def infill(self, masked_token_ids: list[int]) -> InfillResult:
        """Generate predictions for all MASK positions in one forward+decode pass."""
        converted, mask_dict = self._sentinelize(masked_token_ids)

        if not mask_dict:
            return InfillResult(
                infilled_ids = list(masked_token_ids),
                old_logprob  = 0.0,
                x_t          = list(masked_token_ids),
                y_t          = [],
            )

        max_len = self.trainer.model_cfg.max_length
        if len(converted) > max_len - 3:
            log.warning(
                "[mutator] sequence length %d exceeds max_length %d; truncating",
                len(converted), max_len,
            )
            converted = converted[:max_len - 3]

        max_pred = min(
            self._max_pred,
            max(2, len(mask_dict) * self._max_new_tokens_per_mask + 2),
        )

        padded    = converted + [self._pad_token]
        attn_mask = [1] * len(converted) + [0]

        device = self.model.device
        input_ids_t = torch.tensor([padded],    dtype=torch.long, device=device)
        attn_mask_t = torch.tensor([attn_mask], dtype=torch.long, device=device)

        self.model.eval()
        with torch.no_grad():
            outputs = self._generate(input_ids_t, attn_mask_t, max_pred)

        y_t, old_logprob = self._extract_logprob(outputs)
        infilled_ids     = self._reconstruct(padded, mask_dict, outputs.sequences.tolist()[0])

        return InfillResult(
            infilled_ids = infilled_ids,
            old_logprob  = old_logprob,
            x_t          = padded,
            y_t          = y_t,
        )

    # ------------------------------------------------------------------
    # Internal infill helpers (ported from legacy trainer.py)
    # ------------------------------------------------------------------

    def _sentinelize(self, token_ids: list[int]) -> tuple[list[int], dict]:
        """Replace MASK sentinels with T5-style extra tokens."""
        converted  = list(token_ids)
        mask_dict  = {}
        sentinel_n = 0
        for i, tok in enumerate(converted):
            if tok == self._mask_token or tok == self._unk_token:
                sentinel_n += 1
                extra_id    = self._vocab_size - sentinel_n
                mask_dict[extra_id] = i
                converted[i]        = extra_id
        return converted, mask_dict

    def _generate(self, input_ids_t, attn_mask_t, max_pred: int):
        cfg = self.trainer.model_cfg
        if cfg.sample_method == "contrastive":
            return self.model.generate(
                input_ids               = input_ids_t,
                attention_mask          = attn_mask_t,
                do_sample               = False,
                penalty_alpha           = cfg.penalty_alpha,
                top_k                   = cfg.top_k,
                eos_token_id            = self._eos_token,
                no_repeat_ngram_size    = 3,
                min_length              = 1,
                max_length              = max_pred,
                output_scores           = True,
                return_dict_in_generate = True,
            )
        if cfg.sample_method == "sampling":
            return self.model.generate(
                input_ids               = input_ids_t,
                attention_mask          = attn_mask_t,
                do_sample               = True,
                temperature             = cfg.temperature,
                top_p                   = cfg.top_p,
                top_k                   = cfg.top_k,
                eos_token_id            = self._eos_token,
                no_repeat_ngram_size    = 3,
                max_length              = max_pred,
                output_scores           = True,
                return_dict_in_generate = True,
            )
        return self.model.generate(
            input_ids               = input_ids_t,
            attention_mask          = attn_mask_t,
            do_sample               = False,
            eos_token_id            = self._eos_token,
            no_repeat_ngram_size    = 3,
            max_length              = max_pred,
            output_scores           = True,
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
