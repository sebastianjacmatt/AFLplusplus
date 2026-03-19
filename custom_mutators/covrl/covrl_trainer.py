from abstract_trainer import Trainer

class PPOTrainer(Trainer):

    def record_sample(self, sample):
        """
        Accumulate (masked_input, infilled_output, coverage_reward) for the PPO batch.
        """
        pass

    def compute_advantages(self, rewards):
        """
        Compute advantages using the critic's value estimates as baseline.
        """
        pass

    def finetune(self, mutation_dataset, sampled_train_data):
        """
        Full PPO cycle:
          cycle 0 — critic warmup only (_finetune_cycle_index gate lives here)
          cycle N>0 — make datasets → train critic → snapshot actor → PPO update
        """
        pass

    def get_actor(self):
        # TODO (Stage 2)
        pass

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

    def _make_critic_dataset(self, mutation_dataset, sampled_train_data):
        # TODO (Stage 2): CriticDataset — tokenize JS, assign label via score_to_label()
        pass

    def _make_actor_dataset(self, mutation_dataset, sampled_train_data):
        # TODO (Stage 2): ActorDataset — T5 span-masking with Poisson noise (lambda=3.0)
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
