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

The class is intentionally stateless w.r.t. masking — it does not know about
sentinels, spans, or word boundaries; those concerns live in ``tokenizer.py``
and ``masking.py``. Adding a ``batch_generate_with_logprobs`` for PPO old-log-
prob capture is the only extension expected for the CovRL milestone.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForSeq2SeqLM

from model.tokenizer import Tokenizer


class Model:
    """Seq2seq LLM wrapped for batched masked-span infill.

    Construction:
      - ``model_name_or_path``: HF id or local path (e.g. ``"Salesforce/codet5p-220m"``).
      - Generation knobs: ``max_new_tokens``, ``temperature``, ``top_p``,
        ``top_k``, ``no_repeat_ngram_size``. Applied to every ``batch_generate``
        call unless overridden per-call.
      - ``device``: ``"auto"`` (cuda when available, else cpu) or an explicit
        torch device string.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        no_repeat_ngram_size: int = 3,
        device: str = "auto",
    ):
        self.tokenizer = Tokenizer(model_name_or_path)
        self._hf = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self._hf.eval()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._hf.to(device)

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.no_repeat_ngram_size = no_repeat_ngram_size

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def batch_generate(
        self,
        input_ids_list: Sequence[Sequence[int]],
        n_samples: int = 1,
        max_new_tokens: int | None = None,
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

        outputs = self._hf.generate(
            input_ids            = input_ids_t,
            attention_mask       = attn_mask_t,
            do_sample            = True,
            temperature          = self.temperature,
            top_p                = self.top_p,
            top_k                = self.top_k,
            max_new_tokens       = max_new_tokens or self.max_new_tokens,
            eos_token_id         = eos_id,
            pad_token_id         = pad_id,
            no_repeat_ngram_size = self.no_repeat_ngram_size,
            num_return_sequences = n_samples,
        )
        return [seq.tolist() for seq in outputs]

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
