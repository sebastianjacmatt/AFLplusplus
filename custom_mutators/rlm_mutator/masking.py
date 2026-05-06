"""CodeT5/T5-style masked span prediction helpers.

CodeT5's masked span prediction objective follows the T5 denoising shape:
contiguous token spans are replaced in the encoder input by ordered sentinel
tokens, and the decoder predicts the missing spans separated by the same
sentinels.  This module owns that masking/reconstruction logic so the AFL
mutator can treat span corruption as a mutation interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CodeT5MaskedSpan:
    """One original token span replaced by a sentinel in the encoder input."""

    start: int
    end: int
    sentinel_id: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CodeT5MaskedProgram:
    """Result of CodeT5/T5 span corruption.

    ``input_ids`` is the encoder-side masked program.  ``spans`` keeps enough
    metadata for reconstruction/debugging without needing to recover positions
    from the sentinelized input.
    """

    original_ids: list[int]
    input_ids: list[int]
    spans: list[CodeT5MaskedSpan]

    @property
    def sentinel_ids(self) -> list[int]:
        return [span.sentinel_id for span in self.spans]

    @property
    def masked(self) -> bool:
        return bool(self.spans)


class CodeT5SpanMasker:
    """Apply CodeT5/T5-style masked span prediction corruption.

    The default corruption parameters mirror the usual T5 setup: corrupt about
    15% of input tokens with an average corrupted span length of 3.  Identifier
    aware CodeT5 objectives are intentionally out of scope here.
    """

    def __init__(
        self,
        tokenizer,
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

        self.tokenizer = tokenizer
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.rng = rng or random

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.decoder_start_token_id = getattr(tokenizer, "decoder_start_token_id", None)

        #chat-note; T5/CodeT5 normally expose 100 sentinels, but max spans is a
        # tokenizer capacity limit rather than an MSP objective parameter.
        self._max_sentinel_spans = 100
        self._sentinel_ids = [
            self._resolve_sentinel_id(i)
            for i in range(self._max_sentinel_spans + 1)
        ]
        self._sentinel_id_set = set(self._sentinel_ids)

    def mask(self, token_ids: list[int]) -> CodeT5MaskedProgram:
        """Return a sentinel-corrupted program for MSP-style infilling."""
        original = list(token_ids)
        n_tokens = len(original)

        #chat-note; This is a tiny-input guard, not a parameter from the paper.
        min_tokens = self.min_span_length
        if n_tokens < min_tokens or n_tokens == 0 or self.corruption_rate == 0.0:
            return CodeT5MaskedProgram(
                original_ids=original,
                input_ids=list(original),
                spans=[],
            )

        noise_mask = self._random_spans_noise_mask(n_tokens)
        input_ids: list[int] = []
        spans: list[CodeT5MaskedSpan] = []
        idx = 0
        sentinel_idx = 0

        while idx < n_tokens:
            if not noise_mask[idx]:
                input_ids.append(original[idx])
                idx += 1
                continue

            start = idx
            while idx < n_tokens and noise_mask[idx]:
                idx += 1
            end = idx

            sentinel_id = self._sentinel_ids[sentinel_idx]
            sentinel_idx += 1
            input_ids.append(sentinel_id)
            spans.append(CodeT5MaskedSpan(start=start, end=end, sentinel_id=sentinel_id))

        return CodeT5MaskedProgram(
            original_ids=original,
            input_ids=input_ids,
            spans=spans,
        )

    def _mask(self, token_ids: list[int]) -> list[int]:
        """Compatibility wrapper returning only encoder-side masked IDs."""
        return self.mask(token_ids).input_ids

    def target_ids(self, masked_program: CodeT5MaskedProgram) -> list[int]:
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
        masked_program: CodeT5MaskedProgram,
        max_new_tokens_per_span: int,
    ) -> int:
        """Return a decoder budget for sentinel-delimited span prediction.

        Each predicted span needs room for its leading sentinel and generated
        content.  T5-style targets also end with one extra sentinel, and
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

    def reconstruct(
        self,
        masked_program: CodeT5MaskedProgram,
        generated_ids: Sequence[int],
    ) -> list[int]:
        """Replace each encoder sentinel with the generated span for it.

        Missing sentinels are interpreted as empty replacements.  Extra
        generated sentinels, including T5's terminal sentinel, delimit spans but
        are not copied into the reconstructed program.
        """
        if not masked_program.spans:
            return list(masked_program.input_ids)

        generated_spans = self._parse_generated_spans(
            generated_ids,
            valid_sentinel_ids=set(masked_program.sentinel_ids),
        )

        reconstructed: list[int] = []
        for token_id in masked_program.input_ids:
            if token_id in generated_spans:
                reconstructed.extend(generated_spans[token_id])
            elif not self.is_sentinel_id(token_id):
                reconstructed.append(token_id)
        return reconstructed

    def _parse_generated_spans(
        self,
        generated_ids: Sequence[int],
        valid_sentinel_ids: set[int],
    ) -> dict[int, list[int]]:
        spans = {sentinel_id: [] for sentinel_id in valid_sentinel_ids}
        current_sentinel: int | None = None

        for pos, token_id in enumerate(generated_ids):
            if pos == 0 and token_id == self.decoder_start_token_id:
                continue
            if token_id == self.eos_token_id or token_id == self.pad_token_id:
                break
            if self.is_sentinel_id(token_id):
                current_sentinel = token_id if token_id in valid_sentinel_ids else None
                continue
            if current_sentinel is not None:
                spans[current_sentinel].append(token_id)

        return spans

    def is_sentinel_id(self, token_id: int) -> bool:
        return token_id in self._sentinel_id_set

    def _random_spans_noise_mask(self, length: int) -> list[bool]:
        """Sample T5-style random contiguous noise spans.

        This mirrors the T5 span-corruption recipe: choose a total noise-token
        budget, choose span lengths around ``mean_span_length`` with the
        configured min/max bounds, then collapse each span into one sentinel.
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
            remaining_sentinels = self._max_sentinel_spans - spans

            if remaining_noise <= 0:
                mask.extend([False] * remaining_tokens)
                break
            if remaining_sentinels <= 0:
                mask.extend([False] * remaining_tokens)
                break

            expected_noise_density = remaining_noise / remaining_tokens
            if self.rng.random() > expected_noise_density:
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

    def _sample_span_length(self) -> int:
        """Sample a bounded span length with mass centered near the mean."""
        weights = []
        lengths = range(self.min_span_length, self.max_span_length + 1)
        for length in lengths:
            weights.append(1.0 / (abs(length - self.mean_span_length) + 1.0))
        return self.rng.choices(list(lengths), weights=weights, k=1)[0]

    def _resolve_sentinel_id(self, idx: int) -> int:
        token = f"<extra_id_{idx}>"
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if token_id is not None and token_id != unk_id:
            return int(token_id)

        # T5/CodeT5 tokenizers conventionally place extra_id_0 at vocab_size - 1.
        vocab_size = int(self.tokenizer.vocab_size)
        return vocab_size - idx - 1
