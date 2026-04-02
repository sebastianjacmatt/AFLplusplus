"""T5-style random span masking helpers, shared across CovRL datasets.

Both CriticDataset and ActorDataset apply identical Poisson(λ=3.0) span
masking — they differ only in how they package the output:
  CriticDataset — concatenated encoder-only (masked_input ++ label_span)
  ActorDataset  — seq2seq split (encoder input / decoder labels)

Concrete dataset classes must set on self before calling any method:
  self.tokenizer        — HF tokenizer
  self.max_length       — int (tokenizer.model_max_length)
  self.mask_probability — float (e.g. 0.15)
  self.poisson_lambda   — float (e.g. 3.0)
"""
import numpy as np
import torch


class SpanMaskingMixin:
    """
    Ported from CovRL-Fuzz/covrl/models/actor_dataset.py.

    Span lengths are drawn from a multinomial approximation of Poisson(lambda).
    Noise spans are replaced with sentinel tokens (vocab_size - N) in the
    masked input; the complement spans become the decoder label sequence.
    """

    def _random_spans_noise_mask(self, length):
        """
        Return a bool array of length `length` where True marks noise tokens.
        """
        num_noise_tokens = max(
            1, min(length - 1, int(round(length * self.mask_probability)))
        )
        num_noise_spans = max(1, int(round(num_noise_tokens / self.poisson_lambda)))

        def _segment(total, n):
            splits = np.cumsum(np.random.multinomial(total - n, [1.0 / n] * n)) + 1
            return np.diff(np.insert(splits, 0, 0))

        noise_lens    = _segment(num_noise_tokens, num_noise_spans)
        nonnoise_lens = _segment(length - num_noise_tokens, num_noise_spans)

        span_lengths       = np.empty(num_noise_spans * 2, dtype=int)
        span_lengths[0::2] = nonnoise_lens
        span_lengths[1::2] = noise_lens
        mask               = np.zeros(length, dtype=bool)
        mask[np.cumsum(span_lengths)[:-1]] = 1
        return np.cumsum(mask) % 2 == 1

    def _create_sentinel_ids(self, mask_indices):
        mask_indices[:, -1] = 1
        start_indices       = (
            mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        )
        start_indices[:, 0] = mask_indices[:, 0]
        sentinel_ids        = np.where(
            start_indices != 0,
            len(self.tokenizer) - np.cumsum(start_indices, axis=-1),
            0,
        )
        return sentinel_ids - mask_indices + start_indices

    def _filter_input_ids(self, input_ids, sentinel_ids):
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        filtered       = input_ids_full[input_ids_full >= 0]
        labels         = np.concatenate(
            [filtered.copy(), [self.tokenizer.eos_token_id] * 2]
        )
        attn_mask = np.ones(filtered.shape[0], dtype=int)
        return filtered, attn_mask, labels

    def _pad(self, token_ids, pad_id=None):
        pad_id = pad_id if pad_id is not None else self.tokenizer.pad_token_id
        padded = np.pad(token_ids, (0, max(0, 1)), constant_values=pad_id)
        return padded[: self.max_length]

    def _mask_tokens(self, inputs):
        """
        Apply span masking to a single tokenized sequence.

        Returns (masked_input_ids, label_ids, attention_mask) as 1-D long
        tensors padded to self.max_length.

        label_ids are padded with tokenizer.pad_token_id, not -100.
        Callers that need the HF Seq2Seq convention (-100 padding) must
        replace pad positions after calling this method.
        """
        inputs   = inputs.squeeze()[: self.max_length]
        is_token = ~(inputs == self.tokenizer.eos_token_id)

        mask_indices = np.asarray(
            [self._random_spans_noise_mask(is_token.sum().item() + 1)]
        )

        inp_sentinel                 = self._create_sentinel_ids(mask_indices.astype(np.int8))
        masked_ids, attn_mask, _     = self._filter_input_ids(inputs.numpy(), inp_sentinel)

        lbl_sentinel                 = self._create_sentinel_ids((~mask_indices).astype(np.int8))
        _, _, label_ids              = self._filter_input_ids(inputs.numpy(), lbl_sentinel)

        masked_ids_t = torch.tensor(self._pad(masked_ids),          dtype=torch.long)
        attn_mask_t  = torch.tensor(self._pad(attn_mask, pad_id=0), dtype=torch.long)
        label_ids_t  = torch.tensor(self._pad(label_ids, pad_id=0), dtype=torch.long)
        return masked_ids_t, label_ids_t, attn_mask_t
