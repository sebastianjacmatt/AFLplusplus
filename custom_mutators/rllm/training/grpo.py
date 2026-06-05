"""GRPO trainer on ``transformers.Trainer`` — composable base.

Two pieces:
  * ``GRPOHF(transformers.Trainer)`` — the loss. ``compute_loss`` keeps **per-token**
    logp ``(B,T)`` + mask and routes through overridable hooks ``_ratio``,
    ``_surrogate``, ``_kl`` and a final ``_aggregate``. The **defaults reproduce the
    previous pure-GRPO numerics exactly** (masked-sum ⇒ sequence-level clipped
    surrogate + k3 KL). A variant (Dr.GRPO, Clip-Higher, GSPO, …) overrides one hook.
  * ``GRPOTrainer`` — manager held by the mutator. Per finetune cycle it builds the
    ``RolloutDataset`` (``data/rollout.py``), fills ``logp_old`` with a no-grad pass,
    and runs a **fresh** ``GRPOHF`` (HF owns batching/optimizer/grad-accum/AMP — the
    batched forward replaces the old sequential loop). A method = a ``GRPOTrainer``
    subclass setting ``hf_cls`` + ``advantage_fn`` (see ``training/drgrpo.py``).
"""

from __future__ import annotations

import copy
import math

import torch
from transformers import Trainer, TrainingArguments

from data.rollout import GroupCollator
from training.advantages import zscore_advantage


def _trim_target(generated, pad_id, eos_id):
    """``generated[1:]`` (drop HF's decoder-start), truncated at the first EOS/pad."""
    out = []
    for tok in list(generated)[1:]:
        if tok == eos_id:
            out.append(tok)
            break
        if tok == pad_id:
            break
        out.append(tok)
    return out


def _per_token_logp(hf, input_ids, attn, labels, with_entropy=False):
    """``(B,T)`` teacher-forced per-token log p(label_t | label_<t, enc). With
    ``with_entropy`` also return per-token policy entropy ``(B,T)`` computed from the
    same ``log_softmax`` — no second forward, no extra materialized tensor."""
    dec_in = hf._shift_right(labels)
    logits = hf(input_ids=input_ids, attention_mask=attn, decoder_input_ids=dec_in).logits
    logp = torch.log_softmax(logits, -1)
    tok_lp = logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    if with_entropy:
        return tok_lp, -(logp.exp() * logp).sum(-1)
    return tok_lp


class GRPOHF(Trainer):
    """GRPO loss as a ``transformers.Trainer``. Override one hook to get a variant."""

    def __init__(self, hf_model, train_dataset, collator, *, lr, kappa,
                 eps_low, eps_high, batch_size, grad_accum, epochs, fp16,
                 log_entropy=True, kl_ref_coef=0.0):
        self.kappa, self.eps_low, self.eps_high = kappa, eps_low, eps_high
        self.kl_ref_coef = kl_ref_coef
        self.log_entropy = log_entropy
        self._diag = None     # lazily-init on-device accumulators; drained once/cycle
        args = TrainingArguments(
            output_dir="/tmp/rllm_trainer",
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            weight_decay=0.0,
            fp16=fp16,
            lr_scheduler_type="constant",
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,     # keep advantage / logp_old / masks
            dataloader_pin_memory=False,
        )
        super().__init__(model=hf_model, args=args,
                         train_dataset=train_dataset, data_collator=collator)

    # ---- overridable hooks (defaults == pure-GRPO) ----
    def _aggregate(self, per_token, mask):
        """Per-token → per-sample. Default: masked sum (no length-norm)."""
        return (per_token * mask).sum(-1)

    def _ratio(self, ptl, ptl_old, mask):
        """Default: sequence-level ratio ``exp(Σlogp − Σlogp_old)``."""
        return torch.exp(self._aggregate(ptl, mask) - self._aggregate(ptl_old, mask))

    def _surrogate(self, ratio, adv):
        """Default: clipped surrogate, decoupled ε_low/ε_high."""
        return -torch.min(ratio * adv,
                          torch.clamp(ratio, 1 - self.eps_low, 1 + self.eps_high) * adv)

    def _kl(self, ptl, ptl_old, mask):
        """Default: sequence-level k3 estimator (≥0)."""
        d = self._aggregate(ptl_old, mask) - self._aggregate(ptl, mask)
        return torch.exp(d) - d - 1.0

    def _kl_ref(self, ptl, ptl_ref, mask):
        """KL(π‖ref) vs a *frozen reference* policy — **per-token mean** k3 (≥0). Unlike
        ``_kl`` (sequence-summed, vs the rollout policy ⇒ inert on-policy), this is vs a
        separate frozen snapshot, so it has a real gradient pulling the policy back toward
        the reference's (valid) distribution — the stability lever. Token-*averaged* (not
        summed) so the scale is O(1) and length-invariant: a sequence-sum KL(π‖base) runs
        ~100+ (the policy's sampled outputs sit far in the diffuse base's tail), which made
        ``kl_ref_coef`` untunable and length-confounded. ``d`` is clamped before ``exp``
        so the k3 estimator can't blow up — a diffuse policy assigning lower prob than ref
        to its own samples drives ``d>0`` and ``exp(d)`` exploded to ~500 in a long
        base-anchored run; a slow-snapshot ref (``ref_update_every>0``) keeps ``d`` small
        so the clamp is only a safety net."""
        d = (ptl_ref - ptl).clamp(max=10.0)                  # per-token (B,T); cap exp() blow-up
        kl_tok = torch.exp(d) - d - 1.0                      # per-token k3 (≥0)
        return (kl_tok * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)   # per-sample token-mean

    def compute_loss(self, model, inputs, return_outputs=False):
        # Deterministic policy forward ⇒ ρ≡1 at the on-policy step. T5 uses
        # *functional* dropout (F.dropout(p=self.dropout, training=self.training) in
        # attention/FFN) that nn.Dropout.p=0 does NOT disable; eval() turns
        # training=False so F.dropout is a no-op. Grads still flow (eval ≠ no_grad).
        model.eval()
        mask = inputs["labels_mask"]
        ent = None
        if self.log_entropy:
            ptl, ent = _per_token_logp(model, inputs["input_ids"],
                inputs["attention_mask"], inputs["labels"], with_entropy=True)
        else:
            ptl = _per_token_logp(model, inputs["input_ids"],
                inputs["attention_mask"], inputs["labels"])
        ptl_old = inputs["logp_old"]
        ratio = self._ratio(ptl, ptl_old, mask)
        surr = self._surrogate(ratio, inputs["advantage"])
        kl = self._kl(ptl, ptl_old, mask)
        loss = surr + self.kappa * kl
        kl_ref = None
        if self.kl_ref_coef > 0.0 and "logp_ref" in inputs:
            kl_ref = self._kl_ref(ptl, inputs["logp_ref"], mask)   # anchor to the frozen ref
            loss = loss + self.kl_ref_coef * kl_ref
        loss = loss.mean()
        self._accumulate(ratio, kl, inputs["advantage"], ent, mask, kl_ref)
        return (loss, None) if return_outputs else loss

    # ---- per-cycle diagnostics: accumulate across micro-batches, drain once ----
    def _accumulate(self, ratio, kl, adv, ent, mask, kl_ref=None):
        """Fold this micro-batch's (sequence-level) ratio/kl/advantage — and optional
        per-token entropy and the ref-anchor KL — into on-device running sums. Cheap
        reductions only; the single host sync happens in ``diagnostics`` at cycle end."""
        with torch.no_grad():
            d = self._diag
            if d is None:
                z = lambda: torch.zeros((), device=ratio.device)
                d = self._diag = {k: z() for k in (
                    "n", "ratio_sum", "ratio_sq", "ratio_max", "clip",
                    "kl_sum", "kl_ref_sum", "adv_sum", "adv_sq", "ent_sum", "ent_tok")}
            d["n"] += ratio.numel()
            d["ratio_sum"] += ratio.sum()
            d["ratio_sq"] += (ratio * ratio).sum()
            d["ratio_max"] = torch.maximum(d["ratio_max"], ratio.max())
            d["clip"] += ((ratio < 1 - self.eps_low) | (ratio > 1 + self.eps_high)).sum()
            d["kl_sum"] += kl.sum()
            if kl_ref is not None:
                d["kl_ref_sum"] += kl_ref.sum()
            d["adv_sum"] += adv.sum()
            d["adv_sq"] += (adv * adv).sum()
            if ent is not None:
                d["ent_sum"] += (ent * mask).sum()
                d["ent_tok"] += mask.sum()

    def diagnostics(self):
        """Summarize the accumulated policy-optimization signals for this cycle. One
        host sync. ``{}`` if no micro-batch ran. These are the columns that let you
        compare GRPO variants over time in ``rllm_train.tsv``."""
        d = self._diag
        if not d or float(d["n"]) == 0:
            return {}
        n = d["n"]
        r_mean, a_mean = d["ratio_sum"] / n, d["adv_sum"] / n
        out = {
            "ratio_mean": float(r_mean),
            "ratio_max": float(d["ratio_max"]),
            "ratio_std": float((d["ratio_sq"] / n - r_mean * r_mean).clamp_min(0).sqrt()),
            "clip_frac": float(d["clip"] / n),
            "kl_mean": float(d["kl_sum"] / n),
            "adv_mean": float(a_mean),
            "adv_std": float((d["adv_sq"] / n - a_mean * a_mean).clamp_min(0).sqrt()),
        }
        if self.kl_ref_coef > 0.0:
            out["kl_ref_mean"] = float(d["kl_ref_sum"] / n)   # anchor strength (vs frozen ref)
        if float(d["ent_tok"]) > 0:
            out["entropy_mean"] = float(d["ent_sum"] / d["ent_tok"])
        return out


class GRPOTrainer:
    """Manager: builds the dataset + logp_old, runs a fresh ``GRPOHF`` per cycle."""

    hf_cls = GRPOHF
    advantage_fn = staticmethod(zscore_advantage)

    def __init__(self, model, *, lr=1e-4, kappa=0.05, eps_low=0.2, eps_high=0.2,
                 max_train_infills=1024, batch_size=16, fp16=False, log_entropy=True,
                 kl_ref_coef=0.0, ref_update_every=0):
        self.model = model
        # Deterministic policy forward: the GRPO ratio needs logp_old == logp at the
        # on-policy step, but HF Trainer runs the forward in train mode — dropout would
        # make them differ and blow the ratio up. The shared model is inference+RL only,
        # so disable dropout globally (also keeps logp_old == training-time logp).
        for mod in model._hf.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        self.max_train_infills = max_train_infills
        self._collator = GroupCollator(model.tokenizer.pad_token_id)
        # KL-to-frozen-reference anchor (validity stability). The kappa KL is vs the
        # rollout policy → inert on-policy (ρ≡1); this is a SEPARATE k3 KL vs a frozen
        # snapshot that pulls the policy back toward a valid reference, countering the
        # unanchored random-walk behind validity collapse. Snapshot here = the base model
        # (its valid prior); `ref_update_every` cycles advance it to the current policy
        # (0 = anchor to base forever). Off (no ref model) when kl_ref_coef == 0.
        self.kl_ref_coef = kl_ref_coef
        self.ref_update_every = ref_update_every
        self._ref_hf = None
        self._cycle = 0
        if kl_ref_coef > 0.0:
            self._ref_hf = copy.deepcopy(model._hf).eval()
            for p in self._ref_hf.parameters():
                p.requires_grad_(False)
        # epochs/grad_accum are structural (one optimizer step per cycle), not knobs;
        # finetune sets grad_accum to span the whole dataset → a single update.
        self._cfg = dict(lr=lr, kappa=kappa, eps_low=eps_low, eps_high=eps_high,
                         batch_size=batch_size, grad_accum=1, epochs=1, fp16=fp16,
                         log_entropy=log_entropy, kl_ref_coef=kl_ref_coef)

    @torch.no_grad()
    def _old_logprobs(self, ds):
        """One no-grad eval pass → per-sample per-token ``logp_old`` (the cycle-start
        policy = ratio/clip reference). Written back onto the dataset samples."""
        hf = self.model._hf
        was_training = hf.training
        hf.eval()
        dev = next(hf.parameters()).device
        bs = self._cfg["batch_size"]
        vals = []
        for i in range(0, len(ds), bs):
            batch = self._collator([ds[j] for j in range(i, min(i + bs, len(ds)))])
            ptl = _per_token_logp(hf, batch["input_ids"].to(dev),
                                  batch["attention_mask"].to(dev), batch["labels"].to(dev))
            m = batch["labels_mask"]
            for r in range(ptl.size(0)):
                T = int(m[r].sum().item())
                vals.append(ptl[r, :T].tolist())
        if was_training:
            hf.train()
        ds.set_field("logp_old", vals)

    @torch.no_grad()
    def _ref_logprobs(self, ds):
        """One no-grad pass through the frozen reference → per-sample ``logp_ref`` (the
        KL-anchor target). Same shape/contract as ``logp_old`` but from the snapshot."""
        ref = self._ref_hf
        dev = next(ref.parameters()).device
        bs = self._cfg["batch_size"]
        vals = []
        for i in range(0, len(ds), bs):
            batch = self._collator([ds[j] for j in range(i, min(i + bs, len(ds)))])
            ptl = _per_token_logp(ref, batch["input_ids"].to(dev),
                                  batch["attention_mask"].to(dev), batch["labels"].to(dev))
            m = batch["labels_mask"]
            for r in range(ptl.size(0)):
                T = int(m[r].sum().item())
                vals.append(ptl[r, :T].tolist())
        ds.set_field("logp_ref", vals)

    def finetune(self, buffer):
        ds = buffer.build_dataset(self.advantage_fn, self.max_train_infills)
        if len(ds) == 0:
            return {}
        self._old_logprobs(ds)
        if self._ref_hf is not None:
            self._ref_logprobs(ds)              # logp_ref for the KL anchor
        cfg = dict(self._cfg)
        # Default = ONE optimizer step per cycle: accumulate every minibatch's
        # gradient, then step once. This is a single on-policy REINFORCE step
        # (ratio ≡ 1, no multi-step drift) — the pre-refactor numerics — while the
        # forward stays batched (the speedup). Multi-epoch / multi-step PPO (where
        # the clip/KL begin to bite) is a later opt-in.
        cfg["grad_accum"] = math.ceil(len(ds) / cfg["batch_size"])
        hf = self.hf_cls(self.model._hf, ds, self._collator, **cfg)
        out = hf.train()
        diag = hf.diagnostics()                 # drain accumulators before freeing hf
        del hf                                  # drop the per-cycle optimizer state
        self.model._hf.eval()                   # generation: dropout-free + deterministic
        if torch.cuda.is_available():
            torch.cuda.empty_cache()            # release reserved memory for generation
        self._cycle += 1
        if (self._ref_hf is not None and self.ref_update_every > 0
                and self._cycle % self.ref_update_every == 0):
            self._ref_hf.load_state_dict(self.model._hf.state_dict())   # advance the anchor
        info = {"loss": float(out.training_loss), "n_infills": len(ds),
                "seeds": len(buffer._buf)}
        info.update(getattr(buffer, "group_health", {}))   # GRPO group-variance health
        info.update(diag)                                  # policy-optimization diagnostics
        return info
