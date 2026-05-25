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
from transformers import AutoModelForSeq2SeqLM

from model.tokenizer import Tokenizer


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
    def generate(
        self,
        input_ids: Sequence[int],
        n_samples: int = 1,
        max_new_tokens: int | None = None,
    ) -> list[list[int]]:
        """Generate ``n_samples`` outputs for one masked input.

        ``input_ids`` is the sentinelized encoder input for a single mask.
        Returns ``n_samples`` token sequences (each a ``list[int]`` including
        HF's leading decoder-start) per HF's ``num_return_sequences`` ordering.

        Sequential single-input call matches CovRL's predict-per-havoc-iter
        protocol exactly and dodges the in-step ``repeat_interleave(top_k)``
        memory blowup of batched contrastive search.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

        input_ids_t = torch.tensor(
            [list(input_ids)], dtype=torch.long, device=self.device,
        )

        outputs = self._hf.generate(
            input_ids            = input_ids_t,
            max_new_tokens       = max_new_tokens or self.max_new_tokens,
            eos_token_id         = self.tokenizer.eos_token_id,
            pad_token_id         = self.tokenizer.pad_token_id,
            num_return_sequences = n_samples,
            **self.gen_kwargs,
        )
        return [seq.tolist() for seq in outputs]

    # Batched path — revive when we move off contrastive search to nucleus /
    # GRPO. With contrastive, HF's `repeat_interleave(top_k, dim=0)` per decode
    # step makes the in-step batch `chunk_size * top_k`; at fuzz_count=512,
    # top_k=32 that's a 16,384-row intermediate that OOMs even on a 24 GB
    # 3090. Nucleus has no such expansion so batching across inputs becomes
    # a clean throughput win. To revive, restore `chunk_size: int = 16` on
    # `__init__` and `self._pending_outputs` in `mutator.Mutator.__init__`.
    #
    # @torch.no_grad()
    # def batch_generate(
    #     self,
    #     input_ids_list: Sequence[Sequence[int]],
    #     n_samples: int = 1,
    #     max_new_tokens: int | None = None,
    # ) -> list[list[int]]:
    #     if not input_ids_list:
    #         return []
    #     if n_samples < 1:
    #         raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    #
    #     pad_id = self.tokenizer.pad_token_id
    #     eos_id = self.tokenizer.eos_token_id
    #     chunk = self.chunk_size if self.chunk_size > 0 else len(input_ids_list)
    #
    #     results: list[list[int]] = []
    #     for start in range(0, len(input_ids_list), chunk):
    #         batch = input_ids_list[start : start + chunk]
    #         max_len = max(len(ids) for ids in batch)
    #
    #         padded: list[list[int]] = []
    #         attn: list[list[int]] = []
    #         for ids in batch:
    #             pad_n = max_len - len(ids)
    #             padded.append(list(ids) + [pad_id] * pad_n)
    #             attn.append([1] * len(ids) + [0] * pad_n)
    #
    #         input_ids_t = torch.tensor(padded, dtype=torch.long, device=self.device)
    #         attn_mask_t = torch.tensor(attn, dtype=torch.long, device=self.device)
    #
    #         outputs = self._hf.generate(
    #             input_ids            = input_ids_t,
    #             attention_mask       = attn_mask_t,
    #             max_new_tokens       = max_new_tokens or self.max_new_tokens,
    #             eos_token_id         = eos_id,
    #             pad_token_id         = pad_id,
    #             num_return_sequences = n_samples,
    #             **self.gen_kwargs,
    #         )
    #         results.extend(seq.tolist() for seq in outputs)
    #
    #     return results

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
