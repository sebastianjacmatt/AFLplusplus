from transformers import Trainer


class RolloutBuffer:
    def __init__(self):
        self.records = []
        self.pending_idx = None

    def add(self, record: dict) -> int:
        self.records.append(record)
        idx = len(self.records) - 1
        self.pending_idx = idx
        return idx

    def patch_reward(self, reward: float, exit_code=None):
        if self.pending_idx is None:
            return
        self.records[self.pending_idx]["reward"] = reward
        self.records[self.pending_idx]["exit_code"] = exit_code
        self.pending_idx = None

    def flush(self) -> list[dict]:
        ready = [r for r in self.records if r.get("reward") is not None]
        self.records = [r for r in self.records if r.get("reward") is None]
        self.pending_idx = None
        return ready


class Mutator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = load_tokenizer(cfg)
        self.actor = load_actor(cfg)
        self.old_actor = clone_frozen(self.actor)
        self.ref_actor = clone_frozen(self.actor) if cfg.kl_coef > 0 else None
        self.buffer = RolloutBuffer()

    def infill(self, masked_token_ids: list[int]) -> dict:
        x_t = list(masked_token_ids)
        y_t, old_logprob = generate_with_logprob(
            model=self.old_actor,
            tokenizer=self.tokenizer,
            x_t=x_t,
            cfg=self.cfg,
        )
        infilled_ids = merge_infill(x_t, y_t)
        return {
            "x_t": x_t,
            "y_t": y_t,
            "old_logprob": old_logprob,
            "infilled_ids": infilled_ids,
        }

    def record_sample(self, group_id: int, sample_id: int, x_t, y_t, old_logprob):
        self.buffer.add(
            {
                "group_id": group_id,
                "sample_id": sample_id,
                "x_t": x_t,
                "y_t": y_t,
                "old_logprob": old_logprob,
                "reward": None,
                "exit_code": None,
            }
        )

    def patch_reward(self, reward: float, exit_code=None):
        self.buffer.patch_reward(reward, exit_code)

    def flush_records(self) -> list[dict]:
        return self.buffer.flush()

    def sync_old_actor(self):
        copy_weights(self.actor, self.old_actor)
        freeze(self.old_actor)

    def sync_ref_actor(self):
        if self.ref_actor is None:
            return
        copy_weights(self.actor, self.ref_actor)
        freeze(self.ref_actor)


class RolloutDataset:
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class RolloutCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        return pad_and_stack(features, self.pad_token_id)


class BaseRLTrainer(Trainer):
    def __init__(self, cfg, mutator: Mutator, *args, **kwargs):
        self.cfg = cfg
        self.mutator = mutator
        super().__init__(*args, **kwargs)

    def build_dataset(self, records: list[dict]):
        return RolloutDataset(records)

    def get_collator(self):
        return RolloutCollator(self.mutator.tokenizer.pad_token_id)

    def current_logprob(self, model, batch):
        return score_logprob(model, batch["x_t"], batch["y_t"])

    def ref_logprob(self, batch):
        if self.mutator.ref_actor is None:
            return None
        return score_logprob(self.mutator.ref_actor, batch["x_t"], batch["y_t"])

    def policy_ratio(self, logprob, old_logprob):
        return exp(logprob - old_logprob)

    def kl_term(self, logprob, ref_logprob):
        return logprob - ref_logprob

    def finetune(self, records: list[dict]):
        self.train_dataset = self.build_dataset(records)
        self.data_collator = self.get_collator()
        self.train()


class PPOTrainer(BaseRLTrainer):
    def compute_loss(self, model, batch, return_outputs=False):
        logprob = self.current_logprob(model, batch)
        ratio = self.policy_ratio(logprob, batch["old_logprob"])
        clipped = clip(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)

        value_pred = critic_value(batch["x_t"]) if self.cfg.use_critic else 0.0
        advantage = batch["reward"] - value_pred

        policy_loss = -mean(minimum(ratio * advantage, clipped * advantage))

        value_loss = 0.0
        if self.cfg.use_critic:
            value_loss = self.cfg.value_coef * mse(value_pred, batch["reward"])

        kl_loss = 0.0
        if self.cfg.kl_coef > 0 and self.mutator.ref_actor is not None:
            ref_logprob = self.ref_logprob(batch)
            kl_loss = self.cfg.kl_coef * mean(self.kl_term(logprob, ref_logprob))

        loss = policy_loss + value_loss + kl_loss
        return loss


class GRPOTrainer(BaseRLTrainer):
    def compute_loss(self, model, batch, return_outputs=False):
        logprob = self.current_logprob(model, batch)
        ratio = self.policy_ratio(logprob, batch["old_logprob"])
        clipped = clip(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon)

        advantage = self._group_advantage(batch["reward"], batch["group_id"])

        policy_loss = -mean(minimum(ratio * advantage, clipped * advantage))

        kl_loss = 0.0
        if self.cfg.kl_coef > 0 and self.mutator.ref_actor is not None:
            ref_logprob = self.ref_logprob(batch)
            kl_loss = self.cfg.kl_coef * mean(self.kl_term(logprob, ref_logprob))

        loss = policy_loss + kl_loss
        return loss

    def _group_advantage(self, rewards, group_ids):
        return normalize_within_group(rewards, group_ids, self.cfg.norm_epsilon)


def build_trainer(cfg, mutator: Mutator):
    common = {
        "cfg": cfg,
        "mutator": mutator,
        "model": mutator.actor,
        "args": cfg.training_args,
        "tokenizer": mutator.tokenizer,
    }
    if cfg.algorithm == "ppo":
        return PPOTrainer(**common)
    if cfg.algorithm == "grpo":
        return GRPOTrainer(**common)
    raise ValueError("unknown algorithm")


# rlm.py sketch

cfg = load_config()
mutator = Mutator(cfg)
trainer = build_trainer(cfg, mutator)

def fuzz(masked_token_ids, group_id, sample_id):
    out = mutator.infill(masked_token_ids)
    mutator.record_sample(group_id, sample_id, out["x_t"], out["y_t"], out["old_logprob"])
    return out["infilled_ids"]

def post_run(reward, exit_code):
    mutator.patch_reward(reward, exit_code)

def maybe_finetune():
    records = mutator.flush_records()
    if records:
        trainer.finetune(records)
        mutator.sync_old_actor()