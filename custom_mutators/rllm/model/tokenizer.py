"""CodeT5/T5 tokenizer wrapper for rllm.

Owns every tokenizer-aware concern in the mutator pipeline:

    - bytes <-> token-id conversion (UTF-8, lossy on invalid)
    - sentinel-id resolution for ``<extra_id_N>``
    - BPE word-boundary detection (``Ġ`` / ``▁``) used by masking for whole-word spans
    - sentinel-aware reconstruct: splice model-generated spans back into a
      masked program and decode the result to bytes

``masking.py`` stays decoupled: it operates on ``list[int]`` only and consumes
the data this module exposes (``sentinel_ids``, ``word_starts``, special-token
IDs) as plain values rather than holding an HF tokenizer reference.
"""

from __future__ import annotations

from typing import Sequence

from transformers import AutoTokenizer


# T5/CodeT5 conventionally expose 100 sentinel tokens (<extra_id_0>..<extra_id_99>).
# Treated as a tokenizer-capacity limit on simultaneous spans, not an MSP knob.
_MAX_SENTINEL_SPANS = 100


class Tokenizer:
    """CodeT5/T5 tokenizer wrapper.

    The HF tokenizer is held internally and not exposed; callers must go
    through this class so masking and the model never pick up tokenizer-
    specific behaviour directly.
    """

    def __init__(self, model_name_or_path: str):
        self._hf = AutoTokenizer.from_pretrained(model_name_or_path)
        self._sentinel_ids = [
            self._resolve_sentinel_id(i)
            for i in range(_MAX_SENTINEL_SPANS + 1)
        ]
        self._sentinel_id_set = set(self._sentinel_ids)

    # ------------------------------------------------------------------
    # bytes <-> token IDs
    # ------------------------------------------------------------------

    def tokenize(self, buf) -> list[int]:
        """Decode bytes as UTF-8 (lossy on invalid) and encode to token IDs."""
        text = bytes(buf).decode("utf-8", errors="replace")
        return self._hf.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: Sequence[int]) -> bytes:
        """Decode token IDs (special tokens stripped) and encode as UTF-8."""
        text = self._hf.decode(list(token_ids), skip_special_tokens=True)
        return text.encode("utf-8")

    # ------------------------------------------------------------------
    # sentinel-aware reconstruct
    # ------------------------------------------------------------------

    def reconstruct(self, masked_program, generated_ids: Sequence[int]) -> bytes:
        """Splice ``generated_ids`` back into ``masked_program`` and detokenize.

        ``masked_program`` must expose ``input_ids: list[int]`` (the
        sentinelized encoder input) and ``spans`` (each with ``sentinel_id``).
        ``generated_ids`` is a T5-style sentinel-delimited target as produced
        by HF encoder-decoder generate; missing sentinels are treated as empty
        replacements and any trailing terminal sentinel is dropped.
        """
        if not masked_program.spans:
            return self.detokenize(masked_program.input_ids)

        valid = {span.sentinel_id for span in masked_program.spans}
        spans_by_id = self._parse_generated_spans(generated_ids, valid)

        spliced: list[int] = []
        for token_id in masked_program.input_ids:
            if token_id in spans_by_id:
                spliced.extend(spans_by_id[token_id])
            elif not self.is_sentinel_id(token_id):
                spliced.append(token_id)
        return self.detokenize(spliced)

    # ------------------------------------------------------------------
    # Data consumed by masking (no HF tokenizer leaks across the boundary)
    # ------------------------------------------------------------------

    @property
    def sentinel_ids(self) -> list[int]:
        """Ordered sentinel IDs (<extra_id_0>, <extra_id_1>, ...)."""
        return list(self._sentinel_ids)

    def is_sentinel_id(self, token_id: int) -> bool:
        return token_id in self._sentinel_id_set

    def word_starts(self, token_ids: Sequence[int]) -> list[int]:
        """Token positions where a new word starts.

        Word i covers ``[starts[i], starts[i+1])``; the trailing entry is
        ``len(token_ids)``. A token starts a word iff it is at position 0 or
        its surface form begins with a BPE space marker — ``Ġ`` for byte-level
        BPE (CodeT5 / CodeT5+ / GPT-2 / RoBERTa) or ``▁`` for SentencePiece
        (T5). Subword continuations and adjacent punctuation are treated as
        part of the preceding word, matching CodeT5 §3.2 ("avoid masking
        partial sub-tokens") without needing a fast-tokenizer round-trip.
        """
        n = len(token_ids)
        if n == 0:
            return [0]
        tokens = self._hf.convert_ids_to_tokens(list(token_ids))
        starts = [0]
        for i in range(1, n):
            t = tokens[i]
            if t and (t[0] == "Ġ" or t[0] == "▁"):
                starts.append(i)
        starts.append(n)
        return starts

    # ------------------------------------------------------------------
    # Special-token IDs
    # ------------------------------------------------------------------

    @property
    def pad_token_id(self) -> int:
        return self._hf.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._hf.eos_token_id

    @property
    def unk_token_id(self) -> int | None:
        return getattr(self._hf, "unk_token_id", None)

    @property
    def decoder_start_token_id(self) -> int | None:
        return getattr(self._hf, "decoder_start_token_id", None)

    @property
    def vocab_size(self) -> int:
        return int(self._hf.vocab_size)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_sentinel_id(self, idx: int) -> int:
        token = f"<extra_id_{idx}>"
        token_id = self._hf.convert_tokens_to_ids(token)
        unk_id = getattr(self._hf, "unk_token_id", None)
        if token_id is not None and token_id != unk_id:
            return int(token_id)
        # T5/CodeT5 tokenizers conventionally place extra_id_0 at vocab_size - 1.
        return int(self._hf.vocab_size) - idx - 1

    def _parse_generated_spans(
        self,
        generated_ids: Sequence[int],
        valid_sentinel_ids: set[int],
    ) -> dict[int, list[int]]:
        spans: dict[int, list[int]] = {sid: [] for sid in valid_sentinel_ids}
        current: int | None = None
        eos = self.eos_token_id
        pad = self.pad_token_id
        for pos, token_id in enumerate(generated_ids):
            # HF encoder-decoder generators emit the decoder-start token at
            # pos 0. For T5-family models that token is pad_token_id, so the
            # eos/pad break below would terminate the parse on the very first
            # iteration if we didn't skip pos 0 unconditionally. Safe because
            # sequences[0] is always decoder-start by HF convention.
            if pos == 0:
                continue
            if token_id == eos or token_id == pad:
                break
            if self.is_sentinel_id(token_id):
                current = token_id if token_id in valid_sentinel_ids else None
                continue
            if current is not None:
                spans[current].append(token_id)
        return spans
