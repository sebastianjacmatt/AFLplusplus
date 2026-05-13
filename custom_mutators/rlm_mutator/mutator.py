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


@dataclass
class _CachedSample:
    """One pre-generated sample waiting to be returned by fuzz_one().

    Filled by _fill_group_cache() at the start of each GRPO group: one
    batched generate produces group_size of these, then fuzz_one() pops them
    one by one. Holds everything fuzz_one needs to log + return.
    """
    x_t:         list[int]
    y_t:         list[int]
    old_logprob: float
    out_buf:     bytes
    ref_lp:      float | None


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
            corruption_rate    = trainer.model_cfg.mask_probability,
            mean_span_length   = trainer.model_cfg.mean_span_length,
            min_span_length    = trainer.model_cfg.min_span_length,
            max_span_length    = trainer.model_cfg.max_span_length,
            whole_word_masking = trainer.model_cfg.whole_word_masking,
        )
        # Per-seed state (reset in on_new_seed)
        self._tokens:   list[int]                  | None = None
        self._masked:   CodeT5MaskedProgram        | None = None
        self._group:    int               = -1
        self._sample:   int               = 0
        self._pending_sample_id: str | None = None
        self._next_group_id: int = 0
        # Group-batched generation cache: one entry per pending sample in the
        # current GRPO group. Refilled at every group boundary in fuzz_one.
        self._group_cache: list[_CachedSample] = []
        # Set in on_new_seed when the seed tokenizes to zero tokens (empty
        # buf or all-non-decodable bytes). rlm.fuzz_count reads should_fuzz()
        # and returns 0 to AFL so fuzz() is never called for an unmaskable seed.
        self._skip_seed: bool = False

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
        self._group_cache.clear()
        if not self._tokens:
            log.warning(
                "[mutator] seed tokenises to 0 tokens (buf_len=%d); marking unfuzzable",
                len(buf),
            )
            self._skip_seed = True
        else:
            self._skip_seed = False

    def should_fuzz(self) -> bool:
        """rlm.fuzz_count gates on this: returns False iff on_new_seed marked
        the seed unfuzzable (currently: empty after tokenisation)."""
        return not self._skip_seed

    def fuzz_one(self, max_size: int) -> bytes:
        """fuzz(): one mutation. Returns mutated bytes.

        Generation is batched per GRPO group. At every `sample % group_size == 0`
        boundary we re-mask once, run one batched generate (and one batched
        ref_logprob if needed), and stash group_size pre-generated samples in
        self._group_cache. Subsequent fuzz_one calls in the same group pop from
        the cache without touching the model. The recorded group_id is monotonic
        across seeds so grouped training never merges unrelated samples from
        different seeds.
        """
        if self._tokens is None:
            raise RuntimeError("Mutator.fuzz_one() called before on_new_seed()")

        if self._sample % self.training_cfg.group_size == 0:
            self._masked = self._span_masker.mask(list(self._tokens))
            self._group = self._next_group_id
            self._next_group_id += 1
            self._fill_group_cache(self._masked)

        if not self._group_cache:
            raise RuntimeError("Mutator.fuzz_one() has empty group cache")

        cached = self._group_cache.pop(0)

        if len(cached.out_buf) > max_size:
            raise RuntimeError(
                "Mutator.fuzz_one() generated a program larger than AFL max_size "
                f"(generated={len(cached.out_buf)}, max_size={max_size}). "
                "Adjust generation budget, masking parameters, or AFL max_size."
            )

        sample_id = self.buffer.new_sample_id()
        # Diagnostic: log first 8 sampled tokens per fuzz_one. Within a GRPO group
        # all samples share x_t, so identical y_t prefixes across the group point
        # at sampling collapse (model too peaky / RNG not advancing); diverging
        # y_t with identical executed_program bytes points at tokenizer-decode
        # collapse. Cheap to leave in; trim or move to debug level once stable.
        log.info(
            "[mut] %s group=%d logp=%.3f y_t[:8]=%s",
            sample_id,
            self._group,
            cached.old_logprob,
            cached.y_t[:8],
        )
        self.buffer.log(
            sample_id        = sample_id,
            group_id         = self._group,
            x_t              = cached.x_t,
            y_t              = cached.y_t,
            log_prob         = cached.old_logprob,
            executed_program = cached.out_buf,
            ref_log_prob     = cached.ref_lp,
        )

        self._pending_sample_id = sample_id
        self._sample += 1
        return cached.out_buf

    def _fill_group_cache(self, masked: CodeT5MaskedProgram) -> None:
        """Run one batched generate for the whole group, then one batched
        ref_logprob, and stash group_size cached samples for subsequent
        fuzz_one calls."""
        if masked is None or not masked.masked:
            raise RuntimeError("Mutator._fill_group_cache() has no masked program")

        group_size = self.training_cfg.group_size
        predictions = self.masked_span_prediction_batch(masked, group_size)

        # Recompute old_log_prob via the actor's raw forward pass so it uses the
        # same unfiltered logit distribution as compute_loss. outputs.scores from
        # generate() are top_k/top_p filtered, which concentrates probability mass
        # and inflates log-probs relative to the raw distribution. That mismatch
        # makes ratio = exp(new - old) << 1.0 at step 0, clipping every gradient.
        old_lps = self.trainer.sequence_logprob_batch(
            self.trainer.model,
            [p.x_t for p in predictions],
            [p.y_t for p in predictions],
        )

        if self.training_cfg.kl_coef > 0.0:
            try:
                ref_lps = self.trainer.ref_logprob_batch(
                    [p.x_t for p in predictions],
                    [p.y_t for p in predictions],
                )
            except Exception as exc:
                raise RuntimeError("Could not calculate reference log-prob for KL") from exc
        else:
            ref_lps = [None] * len(predictions)

        self._group_cache = [
            _CachedSample(
                x_t         = p.x_t,
                y_t         = p.y_t,
                old_logprob = old_lp,
                out_buf     = self.encode(p.predicted_ids),
                ref_lp      = ref_lp,
            )
            for p, old_lp, ref_lp in zip(predictions, old_lps, ref_lps)
        ]

    def on_post_run(self, rewarder: Rewarder) -> None:
        """post_run(): score the just-executed sample if there is one.

        AFL fires post_run after every target execution, including its own
        calibration / dry-run / trim stages where we did not call fuzz()
        and have no sample to score. Gating compute() and observe_last_seed()
        on a pending sample id keeps those stages from snapshotting SHM,
        consuming exit-hook output, or polluting TF-IDF DF.
        """
        if self._pending_sample_id is None:
            return
        reward_result = rewarder.compute()
        rewarder.tf_idf.observe_last_seed()
        self.buffer.patch_reward(self._pending_sample_id, reward_result)
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
        """Single-sample wrapper around masked_span_prediction_batch."""
        return self.masked_span_prediction_batch(masked_program, 1)[0]

    def masked_span_prediction_batch(
        self,
        masked_program: CodeT5MaskedProgram,
        n_samples: int,
    ) -> list[MaskedSpanPrediction]:
        """Generate `n_samples` predictions sharing one masked context in one
        forward+decode pass. The encoder runs once over masked_program; the
        decoder samples n_samples independent trajectories via HF's
        `num_return_sequences`. This is the GRPO-group-aligned batch path."""
        if not masked_program.spans:
            raise RuntimeError("no spans found in masked program, likley because the seed was too short")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        masked_input_ids = list(masked_program.input_ids)
        max_len = self.trainer.model_cfg.max_length
        if len(masked_input_ids) > max_len - 3:
            raise RuntimeError(
                "Masked span prediction input exceeds model max_length budget "
                f"(input_tokens={len(masked_input_ids)}, max_length={max_len}, "
                f"reserved_tokens=3). Adjust max_length or masking parameters."
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
                num_return_sequences    = n_samples,
            )

        results: list[MaskedSpanPrediction] = []
        for i in range(n_samples):
            y_t_i, lp_i = self._extract_logprob_at(outputs, i)
            seq_i = outputs.sequences[i].tolist()
            predicted_ids_i = self._span_masker.reconstruct(masked_program, seq_i)
            results.append(MaskedSpanPrediction(
                predicted_ids = predicted_ids_i,
                old_logprob   = lp_i,
                x_t           = masked_input_ids,
                y_t           = y_t_i,
            ))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_logprob_at(self, outputs, beam_idx: int) -> tuple[list[int], float]:
        """Collect generated token IDs and mean per-token log-prob for one
        trajectory in a (possibly batched) generate output. With
        num_return_sequences=N, sequences has shape (N, T) and each
        scores[t] has shape (N, vocab) — index by beam_idx."""
        sequences    = outputs.sequences
        scores       = outputs.scores
        y_t          = []
        log_prob_sum = 0.0
        for t, step_score in enumerate(scores):
            tok = sequences[beam_idx, t + 1].item()   # +1 skips decoder_start_token
            if tok == self._eos_token:
                break
            lp = F.log_softmax(step_score[beam_idx], dim=-1)[tok].item()
            log_prob_sum += lp
            y_t.append(tok)
        old_logprob = log_prob_sum / max(len(y_t), 1)
        return y_t, old_logprob
