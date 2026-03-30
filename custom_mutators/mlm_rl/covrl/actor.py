"""CovRL PPO actor: dataset, collator, and custom HFTrainer subclass.

ActorTrainer.compute_loss implements the CovRL PPO-like objective
(paper Eq. 7 + Eq. 8):

  R(x,y)_t  = r(W*) + log(π_t / π_{t-1})             [KL-regularised reward]
  L_CovRL   = -E[min(ρ · R, clip(ρ, 1-ε, 1+ε) · R)]  [clipped IS ratio]
  L_total   = L_CovRL + L_CE                           [CE prevents forgetting]

where ρ = exp(log π_t - log π_{t-1}) and r(W*) is the scalar critic estimate.
Both previous_actor and critic are frozen before ActorTrainer is constructed.
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import Trainer as HFTrainer

from utils.masking import SpanMaskingMixin
from utils.rewarding import label_to_score


class ActorDataset(Dataset, SpanMaskingMixin):
    """
    Seq2seq dataset for the PPO actor update.

    Each item:
      input_ids      — span-masked encoder input
      attention_mask — encoder attention mask
      labels         — decoder targets (complement spans), padded with -100
      rewards        — scalar float reward (carried through for ActorTrainer)

    Identical span masking to CriticDataset via SpanMaskingMixin.
    Output format differs: seq2seq encoder/decoder split rather than the
    concatenated single-sequence format used by the encoder-only critic.

    The rewards field is popped by ActorTrainer before the model forward pass
    so that the seq2seq model never sees it.
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

        # Replace pad_token_id padding with -100 (HF Seq2Seq CE-loss convention:
        # positions with -100 are excluded from the loss computation).
        label_ids = label_ids.masked_fill(
            label_ids == self.tokenizer.pad_token_id, -100
        )

        return {
            "input_ids":      masked_ids,
            "attention_mask": attn_mask,
            "labels":         label_ids,
            "rewards":        torch.tensor(float(reward), dtype=torch.float),
        }


class ActorDataCollator:
    """Pads a batch of ActorDataset items for the HF Seq2SeqLM forward pass."""

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids      = pad_sequence(
            [f["input_ids"]      for f in features],
            batch_first=True, padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True, padding_value=0,
        )
        # Labels padded with -100 so trailing positions are ignored in CE loss
        labels  = pad_sequence(
            [f["labels"]  for f in features],
            batch_first=True, padding_value=-100,
        )
        rewards = torch.stack([f["rewards"] for f in features])
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "rewards":        rewards,
        }


class ActorTrainer(HFTrainer):
    """
    HFTrainer subclass that replaces standard CE loss with the CovRL PPO loss.

    previous_actor and critic must already be frozen (eval + requires_grad=False)
    before this trainer is constructed — see PPOTrainer._snapshot_actor and
    PPOTrainer._finetune_actor_with_ppo_like_loss.
    """

    _PPO_CLIP = 0.2   # ε in clip(ρ, 1-ε, 1+ε); matches CovRL paper's [0.8, 1.2]

    def __init__(self, previous_actor, critic, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_actor = previous_actor
        self._critic         = critic

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop rewards — not part of the seq2seq model's forward signature.
        # The actor PPO loss derives r(W*) from the frozen critic below.
        inputs.pop("rewards", None)

        labels = inputs["labels"]   # (B, dec_len), -100 for padding

        # --- forward current actor ---
        cur_outputs = model(**inputs)
        ce_loss     = cur_outputs.loss

        # --- forward frozen previous actor (old policy, no grad) ---
        with torch.no_grad():
            prev_outputs = self._previous_actor(**inputs)

        # --- per-sequence mean log-probability under each policy ---
        cur_lp  = self._seq_log_prob(cur_outputs.logits,  labels)   # (B,)
        prev_lp = self._seq_log_prob(prev_outputs.logits, labels)   # (B,)

        # --- IS ratio ---
        log_ratio = cur_lp - prev_lp   # (B,)
        ratio     = log_ratio.exp()    # ρ = π_t / π_{t-1}

        # --- critic reward r(W*) ---
        # Build concatenated (masked_input ++ label_span) input that the critic
        # was trained on, then argmax its class logits → scalar reward per sequence.
        with torch.no_grad():
            safe_labels  = labels.clamp(min=0)   # replace -100 with 0 for cat
            critic_ids   = torch.cat([inputs["input_ids"], safe_labels],       dim=-1)
            critic_attn  = torch.cat([
                inputs["attention_mask"],
                (labels != -100).long(),
            ], dim=-1)
            critic_out    = self._critic(input_ids=critic_ids, attention_mask=critic_attn)
            critic_labels = critic_out.logits.argmax(-1)                       # (B,)
            r_critic      = torch.tensor(
                [label_to_score(int(l)) for l in critic_labels],
                dtype=torch.float, device=ratio.device,
            )

        # --- KL-regularised advantage (Eq. 8) ---
        advantage = r_critic + log_ratio

        # --- clipped PPO objective (Eq. 7) ---
        clipped  = ratio.clamp(1.0 - self._PPO_CLIP, 1.0 + self._PPO_CLIP)
        ppo_loss = -torch.min(ratio * advantage, clipped * advantage).mean()

        total_loss = ppo_loss + ce_loss
        return (total_loss, cur_outputs) if return_outputs else total_loss

    @staticmethod
    def _seq_log_prob(logits, labels):
        """
        Mean per-token log-probability of the label tokens under logits.

        Mean (not sum) is used for numerical stability across varying sequence
        lengths — with sum, the IS ratio is dominated by sequence length.

        @param logits: (B, dec_len, vocab)
        @param labels: (B, dec_len), -100 = ignore
        @return:       (B,)
        """
        log_probs  = F.log_softmax(logits, dim=-1)
        mask       = (labels != -100)
        gather_ids = labels.clamp(min=0).unsqueeze(-1)
        token_lp   = log_probs.gather(-1, gather_ids).squeeze(-1)
        token_lp   = token_lp * mask.float()
        return token_lp.sum(-1) / mask.sum(-1).float().clamp(min=1)
