"""Masked-span generation policy.

Owns the policy model and tokenizer at runtime. The hot-path optimisation
lives in ``mask()``: it runs the *full* fuzz_count batch through
``model.generate`` in a single call, so each subsequent ``mutate()`` is
just a list pop. See docs/design.md §3.2 (Mutator responsibility) and §2
(amortise everything possible outside the inner loop).

Span masking and sentinel-unmasking are delegated to ``masking.Masker``
(SRP — they are a separate axis of change).
"""

import random
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import Config
from masking import Masker
from rewarder import Rewarder
from rollout import RolloutBuffer


class Mutator:
    def __init__(
        self,
        cfg: Config,
        buffer: RolloutBuffer,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.buffer = buffer
        self.device = cfg.resolve_device()

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        self.masker = Masker(self.tokenizer, cfg)

        self._pending: list[dict] = []
        self._idx: int = 0

    @torch.no_grad()
    def mask(self, buf: bytes) -> None:
        """Mask the input and pre-generate fuzz_count completions as one batch."""
        text = bytes(buf).decode("utf-8", errors="replace")
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
        ).input_ids[0]

        masked_ids = self.masker.apply(tokens)
        encoder_input = masked_ids.unsqueeze(0).to(self.device)

        out = self.model.generate(
            encoder_input,
            num_return_sequences=self.cfg.fuzz_count,
            max_new_tokens=self.cfg.max_new_tokens_per_mask * self.cfg.mask_count,
            do_sample=True,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            temperature=self.cfg.temperature,
            return_dict_in_generate=True,
            output_scores=True,
        )

        scores = torch.stack(out.scores, dim=1)
        gen = out.sequences[:, -scores.shape[1]:]
        logprobs = (
            torch.log_softmax(scores, dim=-1)
            .gather(-1, gen.unsqueeze(-1))
            .squeeze(-1)
        )

        masked_cpu = masked_ids.cpu()
        self._pending = []
        for i in range(self.cfg.fuzz_count):
            self._pending.append({
                "bytes": self.masker.unmask(masked_cpu, gen[i].cpu()),
                "masked_input": masked_cpu,
                "gen_tokens": gen[i].cpu(),
                "logprobs": logprobs[i].cpu(),
            })
        self._idx = 0

    def mutate(self, max_size: int) -> bytes:
        """Pop the next pre-generated completion, record trajectory, return bytes."""
        sample = self._pending[self._idx]
        self._idx += 1
        out_bytes = sample["bytes"][:max_size]
        self.buffer.append({
            "output_bytes": out_bytes,
            "masked_input": sample["masked_input"],
            "gen_tokens": sample["gen_tokens"],
            "logprobs": sample["logprobs"],
            "reward": None,
        })
        return out_bytes

    def collect(self, rewarder: Rewarder) -> None:
        """Attach the reward of the just-executed sample to the last rollout entry."""
        self.buffer.set_last_reward(rewarder.score())
