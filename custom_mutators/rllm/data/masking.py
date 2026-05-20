"""T5-style span-corruption masking for rllm.

Operates purely on token-id sequences. The module has no tokenizer
dependency — sentinel IDs and the optional word-start function (for whole-
word masking) are injected at construction time. CodeT5/T5-specific concerns
(``<extra_id_N>`` resolution, BPE marker detection, bytes ↔ tokens, sentinel-
aware reconstruct) live in ``model/tokenizer.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class MaskedSpan:
    """One original token span replaced by a sentinel in the encoder input."""

    start: int
    end: int
    sentinel_id: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MaskedProgram:
    """Result of T5-style span corruption.

    ``input_ids`` is the encoder-side masked program. ``spans`` keeps enough
    metadata for sentinel-aware reconstruction without recovering positions
    from the sentinelized input.
    """

    original_ids: list[int]
    input_ids: list[int]
    spans: list[MaskedSpan]

    @property
    def sentinel_ids(self) -> list[int]:
        return [span.sentinel_id for span in self.spans]

    @property
    def masked(self) -> bool:
        return bool(self.spans)


class Masking:
    """Sample contiguous spans and replace each with a sentinel.

    Pure span-selection over ``list[int]``: knows nothing about CodeT5,
    bytes, or HF tokenizers. Tokenizer-derived values it needs are injected
    as plain data at construction:

      * ``sentinel_ids`` — ordered list of sentinel token IDs (from
        ``Tokenizer.sentinel_ids``). Length must be ``max_spans + 1``; the
        trailing ID is reserved as the T5 target-sequence terminator used by
        :meth:`target_ids`.
      * ``word_starts_fn`` — optional callable ``list[int] -> list[int]``
        returning word-start positions (with a trailing ``len(tokens)``
        entry); when provided, spans are sampled at the word level.

    Default T5 knobs: ``corruption_rate=0.15``, ``mean/min/max span = 3/1/5``.
    """

    def __init__(
        self,
        sentinel_ids: Sequence[int],
        word_starts_fn: Callable[[Sequence[int]], list[int]] | None = None,
        corruption_rate: float = 0.15,
        mean_span_length: float = 3.0,
        min_span_length: int = 1,
        max_span_length: int = 5,
        rng: random.Random | None = None,
    ):
        if not 0.0 <= corruption_rate <= 1.0:
            raise ValueError(
                f"corruption_rate must be in [0, 1], got {corruption_rate}."
            )
        if mean_span_length <= 0:
            raise ValueError(
                f"mean_span_length must be > 0, got {mean_span_length}."
            )
        if min_span_length < 1:
            raise ValueError(
                f"min_span_length must be >= 1, got {min_span_length}."
            )
        if max_span_length < min_span_length:
            raise ValueError(
                f"max_span_length ({max_span_length}) must be >= "
                f"min_span_length ({min_span_length})."
            )
        if not min_span_length <= mean_span_length <= max_span_length:
            raise ValueError(
                f"mean_span_length ({mean_span_length}) must be between "
                f"min_span_length ({min_span_length}) and "
                f"max_span_length ({max_span_length})."
            )
        if len(sentinel_ids) < 2:
            raise ValueError(
                "sentinel_ids must contain at least 2 IDs "
                f"(1 span + 1 terminator), got {len(sentinel_ids)}."
            )

        self._sentinel_ids = list(sentinel_ids)
        # Last sentinel reserved as the T5 target-sequence terminator.
        self._max_spans = len(self._sentinel_ids) - 1
        self._word_starts_fn = word_starts_fn
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.rng = rng or random

    # ------------------------------------------------------------------
    # Forward — token sequence -> masked program
    # ------------------------------------------------------------------

    def mask(self, token_ids: Sequence[int]) -> MaskedProgram:
        """Return a sentinel-corrupted program for MSP-style infilling."""
        original = list(token_ids)
        n = len(original)

        # Tiny-input guard: nothing to mask.
        if n == 0 or n < self.min_span_length or self.corruption_rate == 0.0:
            return MaskedProgram(
                original_ids=original,
                input_ids=list(original),
                spans=[],
            )

        if self._word_starts_fn is not None:
            noise_mask = self._random_word_spans_noise_mask(original)
        else:
            noise_mask = self._random_spans_noise_mask(n)

        input_ids: list[int] = []
        spans: list[MaskedSpan] = []
        idx = 0
        sentinel_idx = 0

        while idx < n:
            if not noise_mask[idx]:
                input_ids.append(original[idx])
                idx += 1
                continue

            start = idx
            while idx < n and noise_mask[idx]:
                idx += 1
            end = idx

            sentinel_id = self._sentinel_ids[sentinel_idx]
            sentinel_idx += 1
            input_ids.append(sentinel_id)
            spans.append(MaskedSpan(start=start, end=end, sentinel_id=sentinel_id))

        return MaskedProgram(
            original_ids=original,
            input_ids=input_ids,
            spans=spans,
        )

    # ------------------------------------------------------------------
    # MLM target + decoder budget
    # ------------------------------------------------------------------

    def target_ids(self, masked_program: MaskedProgram) -> list[int]:
        """Build the supervised MSP decoder target from the original spans."""
        target: list[int] = []
        for span in masked_program.spans:
            target.append(span.sentinel_id)
            target.extend(masked_program.original_ids[span.start:span.end])
        if masked_program.spans:
            target.append(self._sentinel_ids[len(masked_program.spans)])
        return target

    def generation_budget(
        self,
        masked_program: MaskedProgram,
        max_new_tokens_per_span: int,
    ) -> int:
        """Return a decoder budget for sentinel-delimited span prediction.

        Each predicted span needs room for its leading sentinel and generated
        content. T5-style targets also end with one extra sentinel, and
        generation may append EOS.
        """
        if max_new_tokens_per_span < 1:
            raise ValueError(
                f"max_new_tokens_per_span must be >= 1, got {max_new_tokens_per_span}."
            )
        n_spans = len(masked_program.spans)
        if n_spans == 0:
            return 1
        return n_spans * (max_new_tokens_per_span + 1) + 2

    # ------------------------------------------------------------------
    # Internals — span sampling
    # ------------------------------------------------------------------

    def _random_spans_noise_mask(self, length: int) -> list[bool]:
        """T5-style random contiguous noise spans at the token level.

        Choose a total noise-token budget from ``corruption_rate``, draw span
        lengths around ``mean_span_length`` within ``[min, max]``, and
        collapse each span into one sentinel downstream.
        """
        num_noise_tokens = int(round(length * self.corruption_rate))
        num_noise_tokens = min(max(num_noise_tokens, 1), length)
        mask: list[bool] = []
        masked = 0
        spans = 0
        idx = 0

        while idx < length:
            remaining_tokens = length - idx
            remaining_noise = num_noise_tokens - masked
            remaining_sentinels = self._max_spans - spans

            if remaining_noise <= 0 or remaining_sentinels <= 0:
                mask.extend([False] * remaining_tokens)
                break

            density = remaining_noise / remaining_tokens
            if self.rng.random() > density:
                mask.append(False)
                idx += 1
                continue

            span_len = self._sample_span_length()
            span_len = min(span_len, remaining_noise, remaining_tokens)
            mask.extend([True] * span_len)
            masked += span_len
            spans += 1
            idx += span_len

            if idx < length:
                mask.append(False)
                idx += 1

        return mask[:length]

    def _random_word_spans_noise_mask(self, token_ids: list[int]) -> list[bool]:
        """T5-style random span sampling at the WORD level.

        CodeT5 §3.2 / CodeT5+ §3.1 specify sampling spans before subword
        tokenization to avoid masking partial words. We approximate that on
        already-tokenized input via ``word_starts_fn`` (provided by the
        tokenizer), sampling spans in word units and expanding each masked
        word to all its underlying subword tokens. The corruption budget is
        still expressed in tokens (paper specifies token-level rate).
        """
        n_tokens = len(token_ids)
        if n_tokens == 0:
            return []

        assert self._word_starts_fn is not None
        word_starts = self._word_starts_fn(token_ids)
        n_words = len(word_starts) - 1
        if n_words == 0:
            return [False] * n_tokens

        target_noise_tokens = int(round(n_tokens * self.corruption_rate))
        target_noise_tokens = min(max(target_noise_tokens, 1), n_tokens)

        word_mask = [False] * n_words
        masked_tokens = 0
        spans_used = 0
        word_idx = 0

        while word_idx < n_words:
            remaining_words = n_words - word_idx
            remaining_noise = target_noise_tokens - masked_tokens
            remaining_sentinels = self._max_spans - spans_used
            if remaining_noise <= 0 or remaining_sentinels <= 0:
                break

            remaining_tokens = n_tokens - word_starts[word_idx]
            density = remaining_noise / max(remaining_tokens, 1)
            if self.rng.random() > density:
                word_idx += 1
                continue

            span_word_len = self._sample_span_length()
            span_word_len = min(span_word_len, remaining_words)
            span_tok_len = word_starts[word_idx + span_word_len] - word_starts[word_idx]

            for w in range(word_idx, word_idx + span_word_len):
                word_mask[w] = True
            masked_tokens += span_tok_len
            spans_used += 1
            word_idx += span_word_len

            # T5-style separator: skip one word so adjacent spans don't merge.
            if word_idx < n_words:
                word_idx += 1

        # The token-level path guarantees ≥1 masked span (density at the last
        # token hits 1.0); the word-level variant computes density over a
        # multi-token word slice, so the walk can finish all-False on small
        # inputs. Force one masked word in that case.
        if not any(word_mask) and n_words > 0:
            forced_w = self.rng.randint(0, n_words - 1)
            word_mask[forced_w] = True

        token_mask = [False] * n_tokens
        for w in range(n_words):
            if word_mask[w]:
                for t in range(word_starts[w], word_starts[w + 1]):
                    token_mask[t] = True
        return token_mask

    def _sample_span_length(self) -> int:
        """Sample a bounded span length with mass centered near the mean."""
        lengths = list(range(self.min_span_length, self.max_span_length + 1))
        weights = [
            1.0 / (abs(length - self.mean_span_length) + 1.0)
            for length in lengths
        ]
        return self.rng.choices(lengths, weights=weights, k=1)[0]
