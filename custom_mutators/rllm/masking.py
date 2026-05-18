"""Span masking and sentinel-based unmasking for CodeT5-style models.

Split out from ``mutator.py`` because masking is a distinct axis of
change from policy/model interaction (docs/design.md §2.2 SRP). The
``Masker`` owns the sentinel vocabulary and the encode/decode pair so
the mutator can stay focused on the model and the rollout.
"""

import random
from typing import Optional

import torch


class Masker:
    def __init__(self, tokenizer, cfg) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sentinel_ids: list[int] = [
            tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            for i in range(cfg.mask_count)
        ]
        self.sentinel_set: set[int] = set(self.sentinel_ids)

    def apply(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace up to mask_count non-overlapping spans with sentinel tokens."""
        tokens = input_ids.tolist()
        n = len(tokens)
        spans: list[tuple[int, int]] = []
        for _ in range(self.cfg.mask_count):
            length = random.randint(self.cfg.min_span_length, self.cfg.max_span_length)
            length = max(1, min(length, n))
            start = random.randint(0, max(n - length, 0))
            spans.append((start, start + length))
        spans.sort()

        out: list[int] = []
        cursor = 0
        k = 0
        for s, e in spans:
            if s < cursor:
                continue
            out.extend(tokens[cursor:s])
            out.append(self.sentinel_ids[k])
            cursor = e
            k += 1
        out.extend(tokens[cursor:])
        return torch.tensor(out, dtype=input_ids.dtype)

    def unmask(self, masked: torch.Tensor, generated: torch.Tensor) -> bytes:
        """Splice generated spans back into the masked context, decode to bytes."""
        spans: dict[int, list[int]] = {}
        active: Optional[int] = None
        for tok in generated.tolist():
            if tok in self.sentinel_set:
                active = tok
                spans.setdefault(active, [])
            elif active is not None:
                if tok == self.tokenizer.eos_token_id:
                    break
                spans[active].append(tok)

        filled: list[int] = []
        for tok in masked.tolist():
            if tok in self.sentinel_set:
                filled.extend(spans.get(tok, []))
            else:
                filled.append(tok)

        return self.tokenizer.decode(
            filled, skip_special_tokens=True
        ).encode("utf-8", errors="replace")
