"""Mutator for rlm_mutator.

Owns everything that lives between AFL++ bytes and the trainer:

    - Tokenisation (byte <-> token-id)
    - CodeT5-style masked span prediction via masking.CodeT5SpanMasker
    - Decoder sampling via the actor's generate()
    - Rollout buffer writes (reward filled later by post_run)
    - Finetune trigger (flushes buffer, runs trainer.train, re-anchors ref)

State that used to live as rlm.py module globals (cached seed tokens, current
masked context, group/sample counters, pending sample id) lives here as
instance attributes.

Trainer owns the model; Mutator accesses it through self.trainer.model.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config   import AFLConfig, TrainingConfig
from rollout  import RolloutBuffer, RolloutDataset
from masking  import CodeT5MaskedProgram, CodeT5SpanMasker
from rewarding import RewardResult, Rewarder
from base_trainer import BaseTrainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Masked span prediction result
# ---------------------------------------------------------------------------

@dataclass
class MaskedSpanPrediction:
    """Token-level output from one masked span prediction call.

    @param predicted_ids: Reconstructed full token sequence after replacing
                          sentinels with sampled decoder spans.
    @param old_logprob:  Mean per-token log-probability of y_t under the actor
                         at generation time.  Used as the behaviour policy
                         log-prob in Policy Gradient importance-sampling ratios.
    @param x_t:          Encoder input IDs with T5 extra-token sentinels marking
                         corrupted spans.  Stored as context.
    @param y_t:          Decoder output token IDs (excluding EOS and decoder start).
    """
    predicted_ids: list[int]
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
        self._eos_token  = tok.eos_token_id
        self._max_new_tokens_per_span = trainer.model_cfg.max_new_tokens_per_mask
        self._span_masker = CodeT5SpanMasker(
            tok,
            corruption_rate   = trainer.model_cfg.mask_probability,
            mean_span_length  = trainer.model_cfg.mean_span_length,
            min_span_length   = trainer.model_cfg.min_span_length,
            max_span_length   = trainer.model_cfg.max_span_length,
        )
        # Per-seed state (reset in on_new_seed)
        self._tokens:   list[int]                  | None = None
        self._masked:   CodeT5MaskedProgram        | None = None
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

    def fuzz_one(self, max_size: int) -> bytes:
        """fuzz(): one mutation. Returns mutated bytes.

        Re-masks at each `sample % group_size == 0` so that every
        group_size samples in a group share one x_t (GRPO precondition). The
        recorded group_id is monotonic across seeds so grouped training never
        merges unrelated samples from different seeds.
        """
        if self._tokens is None:
            raise RuntimeError("Mutator.fuzz_one() called before on_new_seed()")

        # mask seed
        if self._sample % self.training_cfg.group_size == 0:
            self._masked = self._span_masker.mask(list(self._tokens))
            self._group = self._next_group_id
            self._next_group_id += 1

        masked = self._masked
        if masked is None or not masked.masked:
            raise RuntimeError("Mutator.fuzz_one() has no masked program")
        
        # predict masked spans and return reconstructed program
        result  = self.masked_span_prediction(masked)
        # encode back into bytes
        out_buf = self.encode(result.predicted_ids)
        
        if len(out_buf) > max_size:
            raise RuntimeError(
                "Mutator.fuzz_one() generated a program larger than AFL max_size "
                f"(generated={len(out_buf)}, max_size={max_size}). "
                "Adjust generation budget, masking parameters, or AFL max_size."
            )

        ref_lp = None
        if self.training_cfg.kl_coef > 0.0:
            try:
                ref_lp = self.trainer.ref_logprob(result.x_t, result.y_t)
            except Exception as exc:
                raise RuntimeError("Could not calculate reference log-prob for KL") from exc


        sample_id = self.buffer.new_sample_id()
        self.buffer.log(
            sample_id        = sample_id,
            group_id         = self._group,
            x_t              = result.x_t,
            y_t              = result.y_t,
            log_prob         = result.old_logprob,
            executed_program = bytes(out_buf),
            ref_log_prob     = ref_lp,
        )

        self._pending_sample_id = sample_id
        self._sample += 1
        return out_buf

    def on_post_run(
        self,
        reward_result: RewardResult,
    ) -> None:
        """post_run(): patch reward + diagnostic fields on the pending sample."""
        if self._pending_sample_id is not None:
            self.buffer.patch_reward(
                self._pending_sample_id,
                reward_result,
            )
        self._pending_sample_id = None

    def maybe_finetune(self, rewarder: Rewarder) -> None:
        """Drain the rollout buffer and run one trainer.train() cycle.

        Advances the TF-IDF cycle (CovRL Eq. 6: blends accumulated DF into
        IDF_t) before training, so the next collection phase scores against
        the refreshed snapshot. Rewards already in the buffer were produced
        under IDF_{t-1} and are not recomputed.
        """
        records = self.buffer.flush()
        if not records:
            log.info("[mutator] maybe_finetune — buffer empty, skipping")
            return
        rewarder.tf_idf.update_cycle()
        dataset = RolloutDataset(records)
        self.trainer.set_rollout_dataset(dataset)
        self.trainer.train()                        # HF Trainer rebuilds optimizer each call
        self.trainer.snapshot_ref()                 # re-anchor pi_ref after weights updated

    # ------------------------------------------------------------------
    # Byte <-> token-id conversion
    # ------------------------------------------------------------------

    def tokenize(self, buf: bytearray) -> list[int]:
        "decodes bytes and tokenizes"
        text = buf.decode("utf-8", errors="replace")
        return self.trainer.tokenizer.encode(text, add_special_tokens=False)

    def encode(self, token_ids: list[int]) -> bytes:
        "de-tokenizes and encodes into bytes"
        text = self.trainer.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.encode("utf-8")

    def render(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        # --todo;?--
        """Render token IDs to text for logging/debugging artifacts."""
        return self.trainer.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Masked span prediction — sample y_t ~ pi_theta(. | x_t)
    # ------------------------------------------------------------------

    def masked_span_prediction(self, masked_program: CodeT5MaskedProgram) -> MaskedSpanPrediction:
        """Generate predictions for all MASK spans in one forward+decode pass."""
        if not masked_program.spans:
            raise RuntimeError("no spans found in masked program, likley because the seed was too short")

        masked_input_ids = list(masked_program.input_ids)
        max_len = self.trainer.model_cfg.max_length
        if len(masked_input_ids) > max_len - 3:
            raise RuntimeError(
                "Masked span prediction input exceeds model max_length budget "
                f"(input_tokens={len(masked_input_ids)}, max_length={max_len}, "
                f"reserved_tokens=3). Adjust max_length or masking parameters."
            )

        if not masked_program.spans:
            raise RuntimeError(
                "Masked span prediction has no spans after validation. "
                "Adjust seed filtering or masking parameters."
            )

        max_new_tokens = self._span_masker.generation_budget(
            masked_program,
            self._max_new_tokens_per_span,
        )

        device = self.model.device
        input_ids_t = torch.tensor([masked_input_ids], dtype=torch.long, device=device)
        attn_mask_t = torch.ones_like(input_ids_t)

        self.model.eval()
        with torch.no_grad():
            cfg = self.trainer.model_cfg
            # TODO: Revisit greedy decoding for deterministic MSP baselines.
            # TODO: Revisit contrastive search if we need low-entropy non-RL ablations.
            outputs = self.model.generate(
                input_ids               = input_ids_t,
                attention_mask          = attn_mask_t,
                do_sample               = True,
                temperature             = cfg.temperature,
                top_p                   = cfg.top_p,
                top_k                   = cfg.top_k,
                eos_token_id            = self._eos_token,
                no_repeat_ngram_size    = 3,
                max_new_tokens          = max_new_tokens,
                output_scores           = True,
                return_dict_in_generate = True,
            )

        y_t, old_logprob = self._extract_logprob(outputs)
        generated_ids    = outputs.sequences.tolist()[0]
        predicted_ids    = self._span_masker.reconstruct(masked_program, generated_ids)

        return MaskedSpanPrediction(
            predicted_ids = predicted_ids,
            old_logprob  = old_logprob,
            x_t          = masked_input_ids,
            y_t          = y_t,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
