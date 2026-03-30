import copy
import os

import torch
from abstract_trainer import Trainer


class PPOTrainer(Trainer):
    """
    Actor-critic PPO trainer for CovRL.

    Owns the finetune cycle counter so that covrl.py only needs to call
    finetune() and get_actor() — it does not need to track cycle state.

    Cycle semantics (matches original CovRL finetuner.py):
      cycle 0  — critic warmup only; actor is not updated because the critic
                 has not yet produced meaningful value estimates.
      cycle N>0 — critic update then actor PPO update using critic as baseline.
    """

    def __init__(self, actor, tokenizer, device, save_dir="./covrl_checkpoints"):
        """
        @type  actor:     transformers.AutoModelForSeq2SeqLM
        @param actor:     Pre-loaded actor model from covrl.py init().

        @type  tokenizer: transformers.AutoTokenizer
        @param tokenizer: Tokenizer shared with covrl.py.

        @type  device:    str
        @param device:    "cuda" or "cpu", passed from DEVICE in covrl.py.

        @type  save_dir:  str
        @param save_dir:  Root directory for actor/critic checkpoints.
        """
        self.actor = actor
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self._finetune_cycle_index = 0
        self.critic = None
        self._previous_actor = None

    def finetune(self, corpus_dir):
        """
        Run one staged CovRL finetuning cycle over the AFL++ queue at corpus_dir.

        cycle 0  — critic warmup only.  The critic must see at least one
                   training pass before its scores are used as PPO rewards.
        cycle N>0 — critic update then actor PPO update.

        @type  corpus_dir: str or None
        @param corpus_dir: Path to the AFL++ output queue directory.
                           None until Stage 2 corpus loading is implemented.
        """
        # TODO (Stage 2): load corpus, compute rewards, build datasets:
        #   mutation_dataset   = load_and_reward(corpus_dir, update_idf=True)
        #   sampled_train_data = sample_train_data(len(mutation_dataset) * 4)

        # --- critic update (every cycle) ---
        critic_dataset = self._make_critic_dataset(corpus_dir)
        self._train_critic(critic_dataset)

        # --- actor PPO update (cycle 1 onward) ---
        if self._finetune_cycle_index > 0:
            actor_dataset = self._make_actor_dataset(corpus_dir)
            self._snapshot_actor()
            self._finetune_actor_with_ppo_like_loss(
                actor_dataset=actor_dataset,
                critic=self.critic,
                previous_actor=self._previous_actor,
            )

        self._finetune_cycle_index += 1

    def get_actor(self):
        """
        Return the current actor model.

        Called by covrl.py _finetune() to hot-swap ACTOR after a cycle.
        The returned model is already on the correct device.
        """
        return self.actor

    def save_checkpoint(self, save_dir):
        # TODO (Stage 2): save actor + critic weights
        pass

    def load_checkpoint(self, save_dir):
        # TODO (Stage 2): load actor + critic weights
        pass

    def setup_critic(self):
        # TODO (Stage 2): T5EncoderModel + dropout + linear head (8-class cross-entropy)
        pass

    def get_critic(self):
        # TODO (Stage 2)
        pass

    def _snapshot_actor(self):
        # TODO (Stage 2): deep-copy actor weights into self._previous_actor before PPO update
        pass

    def _make_critic_dataset(self, corpus_dir):
        # TODO (Stage 2): load corpus, run afl-showmap, assign labels via score_to_label(),
        # mix with 4:1 sampled clean training data; return CriticDataset
        pass

    def _make_actor_dataset(self, corpus_dir):
        # TODO (Stage 2): load corpus, run afl-showmap, assign rewards,
        # mix with 4:1 sampled clean training data; return ActorDataset (T5 span-masking, Poisson lambda=3.0)
        pass

    def _train_critic(self, critic_dataset):
        # TODO (Stage 2): cross-entropy training loop on 8-class labels
        pass

    def _finetune_actor_with_ppo_like_loss(self, actor_dataset, critic, previous_actor):
        # TODO (Stage 2): clipped ratio [0.8, 1.2] + CE loss
        pass

    def _get_latest_actor_checkpoint(self):
        # TODO (Stage 2): read path from save_dir
        pass

    def _get_latest_critic_checkpoint(self):
        # TODO (Stage 2): read path from save_dir
        pass
