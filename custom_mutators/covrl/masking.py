"""T5-style masked span prediction over a HuggingFace tokenizer.

Owns tokenisation, span sampling, encoder-input construction, T5 supervised
target construction (`target_ids`), and reconstruction of the original program
from the decoder output.

Naming + structure mirrors rlm_mutator/masking.py so lessons learned across the
two implementations stay portable. Whole-word masking (CodeT5 §3.2) is
deferred — quality improvement, not required for baseline CovRL.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import Config


# ---------------------------------------------------------------------------
# Mask structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedSpan:
    """One original-token span replaced by a sentinel in the encoder input."""

    start: int
    end: int
    sentinel_id: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MaskedProgram:
    """Output of T5 span corruption.

    original_ids — pre-mask tokens (used by `target_ids` and `reconstruct`)
    input_ids    — encoder-side ids with sentinels replacing original spans (x_t)
    spans        — span metadata; sentinel order matches input_ids
    """

    original_ids: list[int]
    input_ids: list[int]
    spans: list[MaskedSpan]

    @property
    def masked(self) -> bool:
        return bool(self.spans)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class Masking:
    """T5 span corruption + reconstruction over a HuggingFace tokenizer."""

    _MAX_SENTINEL_SPANS = 100  # T5 / CodeT5 convention

    def __init__(self, cfg: Config, rng: Optional[random.Random] = None):
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.rng = rng or random
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_path or cfg.model_path
        )

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None or self.eos_token_id is None:
            raise RuntimeError("tokenizer must define pad_token_id and eos_token_id")

        self._sentinel_ids = [
            self._resolve_sentinel_id(i)
            for i in range(self._MAX_SENTINEL_SPANS + 1)
        ]
        self._sentinel_id_set = set(self._sentinel_ids)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, buf: bytes) -> list[int]:
        text = buf.decode("utf-8", errors="replace")
        return self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.cfg.max_input_length,
            truncation=True,
        ).input_ids

    # ------------------------------------------------------------------
    # Mask construction (was: sample_mask + encode_input — now one call)
    # ------------------------------------------------------------------

    def mask(self, tokens: list[int]) -> MaskedProgram:
        """Sample a T5 span-corruption mask and build the encoder input."""
        original = list(tokens)
        n = len(original)
        if n < self.cfg.min_span_length or self.cfg.mask_ratio == 0.0:
            return MaskedProgram(
                original_ids=original, input_ids=list(original), spans=[]
            )

        noise = self._random_spans_noise_mask(n)

        input_ids: list[int] = []
        spans: list[MaskedSpan] = []
        i = 0
        sentinel_idx = 0
        while i < n:
            if not noise[i]:
                input_ids.append(original[i])
                i += 1
                continue
            start = i
            while i < n and noise[i]:
                i += 1
            end = i
            sentinel_id = self._sentinel_ids[sentinel_idx]
            sentinel_idx += 1
            input_ids.append(sentinel_id)
            spans.append(MaskedSpan(start=start, end=end, sentinel_id=sentinel_id))

        return MaskedProgram(
            original_ids=original, input_ids=input_ids, spans=spans
        )

    # ------------------------------------------------------------------
    # T5 supervised target — used by SFT corpus mixing
    # ------------------------------------------------------------------

    def target_ids(self, mp: MaskedProgram) -> list[int]:
        """T5 supervised target: <s_0> span_0 <s_1> span_1 … <s_N>."""
        target: list[int] = []
        for span in mp.spans:
            target.append(span.sentinel_id)
            target.extend(mp.original_ids[span.start:span.end])
        if mp.spans:
            target.append(self._sentinel_ids[len(mp.spans)])
        return target

    # ------------------------------------------------------------------
    # Generation budget — per-mask, exact
    # ------------------------------------------------------------------

    def generation_budget(self, mp: MaskedProgram, max_new_tokens_per_span: int) -> int:
        """Decoder budget for sentinel-delimited span prediction.

        Each predicted span needs its leading sentinel and content; the target
        ends with one terminal sentinel; generation may append EOS.
        """
        n = len(mp.spans)
        if n == 0:
            return 1
        return n * (max_new_tokens_per_span + 1) + 2

    # ------------------------------------------------------------------
    # Reconstruction (decoder output → original tokens with infills)
    # ------------------------------------------------------------------

    def reconstruct(self, mp: MaskedProgram, generated_ids: Sequence[int]) -> list[int]:
        """Replace each encoder sentinel with the corresponding generated span.

        Missing sentinels → empty replacement. Extra / terminal sentinels are
        treated as span boundaries, not content.
        """
        if not mp.spans:
            return list(mp.input_ids)
        infills = self._parse_generated_spans(
            generated_ids,
            valid_sentinel_ids={s.sentinel_id for s in mp.spans},
        )
        out: list[int] = []
        for tok in mp.input_ids:
            if tok in infills:
                out.extend(infills[tok])
            elif tok not in self._sentinel_id_set:
                out.append(tok)
        return out

    def decode(self, mp: MaskedProgram, generated_ids: Sequence[int]) -> bytes:
        full = self.reconstruct(mp, generated_ids)
        text = self.tokenizer.decode(full, skip_special_tokens=True)
        return text.encode("utf-8", errors="replace")

    def batch_decode(
        self, mps: list[MaskedProgram], generated_ids_list: list[list[int]]
    ) -> list[bytes]:
        fulls = [self.reconstruct(mp, y) for mp, y in zip(mps, generated_ids_list)]
        texts = self.tokenizer.batch_decode(fulls, skip_special_tokens=True)
        return [t.encode("utf-8", errors="replace") for t in texts]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _random_spans_noise_mask(self, length: int) -> list[bool]:
        """T5 span-corruption noise mask: density walk, mean span length."""
        num_noise = int(round(length * self.cfg.mask_ratio))
        num_noise = min(max(num_noise, 1), length)

        mask: list[bool] = []
        masked = 0
        spans = 0
        i = 0
        while i < length:
            remaining_tokens = length - i
            remaining_noise = num_noise - masked
            remaining_sentinels = self._MAX_SENTINEL_SPANS - spans
            if remaining_noise <= 0 or remaining_sentinels <= 0:
                mask.extend([False] * remaining_tokens)
                break
            density = remaining_noise / remaining_tokens
            if self.rng.random() > density:
                mask.append(False)
                i += 1
                continue
            span_len = self._sample_span_length()
            span_len = min(span_len, remaining_noise, remaining_tokens)
            mask.extend([True] * span_len)
            masked += span_len
            spans += 1
            i += span_len
            # Separator so adjacent spans don't collapse into one sentinel.
            if i < length:
                mask.append(False)
                i += 1
        return mask[:length]

    def _sample_span_length(self) -> int:
        lengths = list(range(self.cfg.min_span_length, self.cfg.max_span_length + 1))
        weights = [1.0 / (abs(L - self.cfg.mean_span_length) + 1.0) for L in lengths]
        return self.rng.choices(lengths, weights=weights, k=1)[0]

    def _parse_generated_spans(
        self,
        generated_ids: Sequence[int],
        valid_sentinel_ids: set[int],
    ) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {sid: [] for sid in valid_sentinel_ids}
        current: Optional[int] = None
        for tok in generated_ids:
            if tok == self.eos_token_id or tok == self.pad_token_id:
                break
            if tok in self._sentinel_id_set:
                current = tok if tok in valid_sentinel_ids else None
                continue
            if current is not None:
                out[current].append(tok)
        return out

    def _resolve_sentinel_id(self, idx: int) -> int:
        token = f"<extra_id_{idx}>"
        tid = self.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if tid is not None and tid != unk_id:
            return int(tid)
        # T5 / CodeT5 fallback: extra_id_0 conventionally at vocab_size - 1.
        return int(self.tokenizer.vocab_size) - idx - 1


# ---------------------------------------------------------------------------
# LLM wrapper — Option A: one generate() per group, exact budget
# ---------------------------------------------------------------------------


class LLMModel:
    """HF seq2seq wrapper.

    `batch_generate(xs, n_samples, max_new_tokens)` runs ONE generate() call
    across all xs, using HF's `num_return_sequences` for intra-input sampling.
    Returns a flat list of `len(xs) * n_samples` cleaned sequences (decoder
    start token stripped, truncated at first EOS) in HF's layout:
    `[xs[0]·n_samples, xs[1]·n_samples, ...]`.

    This is the one call per `fuzz_count` cycle. It collapses to the right
    shape for both PPO (group_size=1: batch across mutations) and GRPO
    (group_size>1: batch across groups + intra-group num_return_sequences).

    Sharing the tokenizer with `Masking` (via constructor arg) is required —
    otherwise sentinel / pad / eos ids can drift between the two.
    """

    def __init__(self, cfg: Config, tokenizer=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self._torch = torch
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            cfg.tokenizer_path or cfg.model_path
        )
        self.device = cfg.resolve_device()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        if self.pad_id is None or self.eos_id is None:
            raise RuntimeError("tokenizer must define pad_token_id and eos_token_id")

    def batch_generate(
        self,
        xs: list[list[int]],
        n_samples: int,
        max_new_tokens: int,
    ) -> list[list[int]]:
        """One mega-batch generate(). Returns `len(xs) * n_samples` cleaned
        sequences in `[xs[0]·n_samples, xs[1]·n_samples, ...]` order."""
        torch = self._torch

        max_len = max(len(x) for x in xs)
        input_ids = torch.full(
            (len(xs), max_len), self.pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(xs), max_len), dtype=torch.long, device=self.device
        )
        for i, x in enumerate(xs):
            input_ids[i, : len(x)] = torch.tensor(
                x, dtype=torch.long, device=self.device
            )
            attention_mask[i, : len(x)] = 1

        with torch.no_grad():
            out = self.model.generate(
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                do_sample            = True,
                temperature          = self.cfg.temperature,
                top_p                = self.cfg.top_p,
                top_k                = self.cfg.top_k,
                eos_token_id         = self.eos_id,
                max_new_tokens       = max_new_tokens,
                num_return_sequences = n_samples,
            )

        return [self._clean(seq) for seq in out.tolist()]

    def _clean(self, seq: list[int]) -> list[int]:
        """Strip decoder_start (index 0, = pad_token_id for T5); truncate at
        first EOS. Without this, raw `generate()` output stored as `y_t` would
        train the model to emit decoder-start / pads as labels."""
        out: list[int] = []
        for tok in seq[1:]:
            if tok == self.eos_id:
                break
            out.append(tok)
        return out

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)