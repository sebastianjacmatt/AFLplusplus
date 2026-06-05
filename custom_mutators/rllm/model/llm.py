"""Seq2seq LLM wrapper for rllm — CodeT5/T5 masked span infill.

Owns the LLM end of the mutator pipeline:

    - Loads any HF AutoModelForSeq2SeqLM checkpoint (CodeT5, CodeT5+, T5).
    - Holds the matching ``Tokenizer`` (exposed as ``self.tokenizer``) so the
      mutator can do ``model.tokenizer.tokenize(buf)`` and
      ``model.tokenizer.reconstruct(mp, y)`` without separate wiring.
    - Batched masked-span infill via :meth:`batch_generate`: ``@torch.no_grad``
      eval-mode generation that pads a list of masked inputs into one
      ``model.generate`` call, eating HF's per-call overhead once per seed.
    - Checkpoint save for the eventual CovRL trainer to call from ``deinit``.

The class is stateless w.r.t. decoding strategy: caller (``rllm._build_mutator``)
picks contrastive vs nucleus and hands the corresponding ``gen_kwargs`` dict in.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, LogitsProcessor, LogitsProcessorList

from model.tokenizer import Tokenizer


class _ForceNonEmptySpan(LogitsProcessor):
    """Forbid a sentinel / EOS *immediately after* a span-opening sentinel, so
    each masked span gets ≥1 content token.

    Without it the infiller can emit an empty span (sentinel straight to the next
    sentinel/eos) → the masked token is **deleted**, which is usually invalid and
    gives a whole within-mask group the same broken outcome (zero reward variance
    → no GRPO signal). Opening sentinels are the ones present in the encoder
    input; the trailing terminator sentinel is *not* in that set, so EOS stays
    allowed after the last span.
    """

    def __init__(self, opening_sentinels, forbid_ids):
        self._opening = torch.tensor(sorted(opening_sentinels), dtype=torch.long)
        self._forbid = torch.tensor(sorted(forbid_ids), dtype=torch.long)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        opened = torch.isin(input_ids[:, -1], self._opening.to(input_ids.device))   # (B,)
        if opened.any():
            block = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
            block[self._forbid.to(scores.device)] = True
            scores = scores.masked_fill(opened.unsqueeze(1) & block.unsqueeze(0), float("-inf"))
        return scores


class Model:
    """Seq2seq LLM wrapped for batched masked-span infill.

    Construction:
      - ``model_name_or_path``: HF id or local path (e.g. ``"Salesforce/codet5p-220m"``).
      - ``gen_kwargs``: HF-generate args specific to the chosen decoding
        strategy. Picked by ``rllm._build_mutator`` based on
        ``cfg.sampling_method`` — see that branch for the contrastive vs
        nucleus dicts. Splatted into every ``model.generate`` call alongside
        the always-needed args (input/attention/max_new_tokens/eos/pad/N).
      - ``max_new_tokens``: decoder budget (overridable per ``batch_generate`` call).
      - ``device``: ``"auto"`` (cuda when available, else cpu) or explicit.
    """

    def __init__(
        self,
        model_name_or_path: str,
        gen_kwargs: dict[str, Any],
        max_new_tokens: int = 64,
        device: str = "auto",
    ):
        self.tokenizer = Tokenizer(model_name_or_path)
        self._hf = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self._hf.eval()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._hf.to(device)

        self.gen_kwargs = dict(gen_kwargs)
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def batch_generate(
        self,
        input_ids_list: Sequence[Sequence[int]],
        n_samples: int = 1,
        max_new_tokens: int | None = None,
        no_empty_spans: bool = True,
    ) -> list[list[int]]:
        """One batched generate across many masked inputs.

        ``input_ids_list[i]`` is the sentinelized encoder input for mask ``i``;
        all entries are right-padded to the longest length with the tokenizer's
        pad token, and an attention mask is built so padded positions don't
        influence generation.

        Returns ``len(input_ids_list) * n_samples`` token sequences (each a
        ``list[int]`` including HF's leading decoder-start). With
        ``n_samples > 1`` (GRPO later), samples for input ``i`` occupy
        ``results[i*n_samples : (i+1)*n_samples]`` per HF's
        ``num_return_sequences`` ordering.
        """
        if not input_ids_list:
            return []
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        max_len = max(len(ids) for ids in input_ids_list)
        padded: list[list[int]] = []
        attn: list[list[int]] = []
        for ids in input_ids_list:
            pad_n = max_len - len(ids)
            padded.append(list(ids) + [pad_id] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)

        input_ids_t = torch.tensor(padded, dtype=torch.long, device=self.device)
        attn_mask_t = torch.tensor(attn, dtype=torch.long, device=self.device)

        proc = None
        if no_empty_spans:
            opening = {t for ids in input_ids_list for t in ids if self.tokenizer.is_sentinel_id(t)}
            if opening:
                forbid = set(self.tokenizer.sentinel_ids) | {eos_id}
                proc = LogitsProcessorList([_ForceNonEmptySpan(opening, forbid)])

        outputs = self._hf.generate(
            input_ids            = input_ids_t,
            attention_mask       = attn_mask_t,
            max_new_tokens       = max_new_tokens or self.max_new_tokens,
            eos_token_id         = eos_id,
            pad_token_id         = pad_id,
            num_return_sequences = n_samples,
            logits_processor     = proc,
            **self.gen_kwargs,
        )
        return [seq.tolist() for seq in outputs]

    # ------------------------------------------------------------------
    # Teacher-forced log-prob (for the GRPO trainer's logπ_fill)
    # ------------------------------------------------------------------

    def target_logprob(self, enc_input_ids: Sequence[int], target_ids: Sequence[int]) -> torch.Tensor:
        """``Σ_t log p(target_t | target_<t, enc)`` — **differentiable** teacher-
        forced log-prob of a span-infill target given its masked encoder input.

        ``target_ids`` is the decoder target (sentinel-delimited content, ending
        in EOS), **without** HF's leading decoder-start token — pass
        ``generated[1:]`` trimmed at EOS. Used for ``logπ_fill`` (current at loss
        time, "old" captured under ``no_grad`` at rollout time).
        """
        enc = torch.tensor([list(enc_input_ids)], dtype=torch.long, device=self.device)
        tgt = torch.tensor([list(target_ids)], dtype=torch.long, device=self.device)
        dec_in = self._hf._shift_right(tgt)
        logits = self._hf(input_ids=enc, decoder_input_ids=dec_in).logits      # (1, T, V)
        logp = torch.log_softmax(logits, dim=-1)
        return logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1).sum()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str | None = None) -> None:
        """Save the underlying HF model to ``output_dir``.

        For the mutation-only milestone weights are unchanged from the
        pretrained checkpoint so this is effectively a no-op when
        ``output_dir`` is ``None``. Kept on the interface so the CovRL trainer
        can call it from ``Mutator.deinit`` without further wiring.
        """
        if output_dir is None:
            return
        self._hf.save_pretrained(output_dir)
