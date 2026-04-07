# LoRA Adapter Integration Plan

## Overview

Apply LoRA adapters to both the actor and critic to reduce the number of trainable
parameters per finetune cycle while preserving full-model expressiveness via frozen
base weights.

## Architecture read

| Component | What it is | Current training |
|---|---|---|
| `actor` | Full T5 seq2seq model (codet5p-220m) | Updated by `ActorTrainer` via PPO-like loss |
| `critic` | `CriticModel` (T5EncoderModel backbone + classifier head) | Updated by plain `HFTrainer` |
| `_previous_actor` | Frozen deep copy of actor | π_{t-1} reference policy, not trained |

## Design principle — optional configs at the seam

The single control point is two optional parameters on `PPOTrainer.__init__`:

```python
actor_lora_config: Optional[LoraConfig] = None
critic_lora_config: Optional[LoraConfig] = None
```

`None` keeps the current full fine-tuning behaviour unchanged.
Passing a `LoraConfig` enables LoRA for that component.
No changes are needed to datasets, collators, or `ActorTrainer.compute_loss` —
PEFT models are drop-in `nn.Module` replacements.

## Actor

In `PPOTrainer.__init__`, after the actor is stored:

```python
if actor_lora_config is not None:
    from peft import get_peft_model
    self.actor = get_peft_model(self.actor, actor_lora_config)
```

`get_peft_model` freezes all base weights and injects trainable LoRA adapters.
Both forward passes in `ActorTrainer.compute_loss` (`model(**inputs)` and
`self._previous_actor(**inputs)`) work transparently.
`actor.save_pretrained()` saves adapter weights only (a few MB vs. hundreds).

### Snapshot (`_snapshot_actor`)

`copy.deepcopy(self.actor)` works on PEFT models — they are plain `nn.Module`s.
The deep copy captures the current LoRA weights as the "old policy" state (π_{t-1}),
which is what the PPO IS-ratio needs.

## Critic

The critic has a complication: the `classifier` head (`nn.Sequential(Dropout, Linear)`)
lives outside the T5 backbone and must be trained and saved alongside the adapters.
PEFT's `modules_to_save` handles this — listed modules are cloned, kept full-precision
and trainable, and included in `save_pretrained` output alongside the adapter weights.

Critic LoRA config must always include:

```python
LoraConfig(
    ...,
    modules_to_save=["classifier"],
)
```

In `setup_critic()`, after `CriticModel` is constructed:

```python
if self._critic_lora_config is not None:
    from peft import get_peft_model
    self.critic = get_peft_model(self.critic, self._critic_lora_config)
```

`critic.save_pretrained()` then saves adapter weights + the `classifier` head.

## Suggested starting configs (CodeT5+ / T5)

T5 attention projection names: `"q"`, `"k"`, `"v"`, `"o"`.

```python
from peft import LoraConfig

ACTOR_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
)

CRITIC_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
```

`target_modules=["q", "v"]` (Q and V projections only) is the standard conservative
starting point. Expand to `["q", "k", "v", "o"]` if more capacity is needed.

## Files that need changes

| File | Change |
|---|---|
| `covrl/trainer.py` | Add `actor_lora_config`, `critic_lora_config` params to `__init__`; call `get_peft_model` in `__init__` (actor) and `setup_critic()` (critic); store `_critic_lora_config` on self |
| `mlm_rl.py` | Pass `LoraConfig` instances when constructing `PPOTrainer` |
| `covrl/actor.py` | No changes |
| `covrl/critic.py` | No changes |
| `environment.yml` | Add `peft` dependency |

## Files that do NOT need changes

- `covrl/actor.py` — `ActorTrainer.compute_loss` all three forward passes work
  transparently with PEFT models.
- `covrl/critic.py` — `CriticModel` forward is unchanged; PEFT wraps it externally.
- All dataset and collator classes — data pipeline is unaffected.
- `finetune()` orchestration in `trainer.py` — no changes to cycle logic.
