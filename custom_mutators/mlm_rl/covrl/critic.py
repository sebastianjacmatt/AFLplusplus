"""CovRL critic: T5-encoder reward classifier, dataset, and collator.

The critic is trained as a supervised classifier that predicts which reward
bucket a (span-masked input, label-span) pair falls into.  After training,
its class predictions are consumed by ActorTrainer as the PPO reward signal
(r(W*) in paper Eq. 8).
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import T5EncoderModel, T5PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.masking import SpanMaskingMixin
from utils.rewarding import score_to_label, NUM_LABELS


class CriticModel(T5PreTrainedModel):
    """
    T5-encoder backbone → mean pool → Dropout → Linear(NUM_LABELS).

    config should be the actor's T5Config so that d_model, num_heads, and
    vocabulary size are consistent (e.g. d_model=1024 for codet5p-220m).

    Mean pooling over non-padding positions is used rather than a first-token
    CLS heuristic — T5 has no dedicated CLS token.
    """

    def __init__(self, config, dropout_rate=0.1):
        super().__init__(config)
        self.backbone   = T5EncoderModel(config)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(config.d_model, NUM_LABELS),
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        encoder_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden      = encoder_out.last_hidden_state            # (B, seq, d_model)

        # Mean pool over non-padding positions
        mask   = attention_mask.unsqueeze(-1).float()          # (B, seq, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, d_model)

        logits = self.classifier(pooled)                       # (B, NUM_LABELS)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class CriticDataset(Dataset, SpanMaskingMixin):
    """
    Each item (all tensors on CPU — device placement is the HFTrainer's job):
      input_ids      — masked_input ++ label_span (concatenated on seq dim)
      attention_mask — corresponding mask
      labels         — score_to_label(reward) as a scalar long tensor

    The concatenated format lets the encoder-only critic attend jointly over
    both what was masked and what the actor predicted for those spans.
    Span masking via SpanMaskingMixin mirrors CovRL-Fuzz ActorDataset.
    """

    def __init__(self, dataset, tokenizer,
                 mask_probability=0.15, poisson_lambda=3.0):
        self.dataset          = dataset.reset_index(drop=True)
        self.tokenizer        = tokenizer
        self.mask_probability = mask_probability
        self.poisson_lambda   = poisson_lambda
        self.max_length       = tokenizer.model_max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text   = self.dataset["data"][idx]
        reward = (
            self.dataset["reward"][idx]
            if "reward" in self.dataset.columns
            else 0.0
        )

        encoded               = self.tokenizer(text, return_tensors="pt", truncation=True)
        masked_ids, label_ids, attn_mask = self._mask_tokens(encoded["input_ids"])

        return {
            "input_ids":      torch.cat((masked_ids, label_ids),                 dim=-1),
            "attention_mask": torch.cat((attn_mask, torch.ones_like(label_ids)), dim=-1),
            "labels":         torch.tensor(score_to_label(reward), dtype=torch.long),
        }


class CriticDataCollator:
    def __call__(self, features):
        input_ids      = pad_sequence(
            [f["input_ids"]      for f in features], batch_first=True, padding_value=0
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        labels = torch.stack([f["labels"] for f in features])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
