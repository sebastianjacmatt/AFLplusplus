"""High-level CovRL PPO orchestration.

PPOTrainer owns the finetune cycle counter, the critic, and the actor snapshot.
mlm_rl.py interacts with this class only via the two abstract methods defined
in abstract_trainer.Trainer:

  finetune() — run one staged training cycle
  get_actor() — return the (updated) actor for hot-swap

Cycle semantics:
  cycle 0  — critic warmup only; actor is not updated because the critic
             has not yet produced meaningful value estimates.
  cycle N>0 — critic update then actor PPO update using critic as baseline.

Data flow per cycle:
  Rewarder.compute()
      └─ returns mixed DataFrame (mutations + sampled orig at 4:1)
            ├─ _make_critic_dataset(mixed_df) → CriticDataset
            └─ _make_actor_dataset(mixed_df)  → ActorDataset

All queue loading, reward computation, and corpus mixing live in Rewarder.
The dataset builders are pure structural wrappers over already-prepared DataFrames.
"""
import copy
import logging
import os

log = logging.getLogger(__name__)
from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer as HFTrainer

from abstract_trainer import Trainer
from covrl.critic import CriticModel, CriticDataset, CriticDataCollator
from covrl.actor  import ActorDataset, ActorDataCollator, ActorTrainer
from utils.rewarding  import Rewarder


class PPOTrainer(Trainer):

    def __init__(self, actor, config):
        """
        @type  actor:   transformers.AutoModelForSeq2SeqLM
        @param actor:   Pre-loaded actor model from mlm_rl.py init().

        @type  config:  config.config.Config
        @param config:  Top-level run config; all hyperparameters and paths are
                        read from here (config.covrl, config.afl, config.device).
        """
        self.actor     = actor
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.device    = config.device
        self.save_dir  = config.afl.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self._finetune_cycle_index = 0
        self.critic          = None
        self._previous_actor = None
        self._mask_probability = config.mask_probability

        # HuggingFace TrainingArguments shared by critic and actor training.
        # Checkpoint saving is handled explicitly; HF auto-save is disabled.
        self._training_args = TrainingArguments(
            output_dir=os.path.join(config.afl.save_dir, "hf_output"),
            overwrite_output_dir=True,
            per_device_train_batch_size=config.covrl.train_batch_size,
            fp16=False,
            learning_rate=config.covrl.learning_rate,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            num_train_epochs=1,
            remove_unused_columns=False,
        )

        self._rewarder = Rewarder(config=config)

    # -------------------------------------------------------------------------
    # Trainer interface
    # -------------------------------------------------------------------------

    def finetune(self):
        """
        Run one staged CovRL finetuning cycle over the current AFL++ queue.

        Rewarder.compute() handles queue loading, reward computation, and
        orig-corpus mixing; the returned DataFrame is passed directly to the
        critic and actor dataset builders.
        """
        log.info("[finetune] cycle %d — preparing data", self._finetune_cycle_index)
        mixed_df = self._rewarder.compute()

        if mixed_df.empty:
            raise Exception("afl queue contains no mutation entries before finetuning")

        log.info("[finetune] training critic on %d rows", len(mixed_df))
        critic_dataset = self._make_critic_dataset(mixed_df)
        self._train_critic(critic_dataset)
        log.info("[finetune] critic done")

        if self._finetune_cycle_index > 0:
            self._snapshot_actor()
            log.info("[finetune] finetuning actor on %d rows", len(mixed_df))
            actor_dataset = self._make_actor_dataset(mixed_df)
            self._finetune_actor_with_ppo_like_loss(
                actor_dataset=actor_dataset,
                critic=self.critic,
                previous_actor=self._previous_actor,
            )
            log.info("[finetune] actor done")

        self._finetune_cycle_index += 1
        log.info("[finetune] cycle %d complete", self._finetune_cycle_index - 1)

    def get_actor(self):
        """Return the current actor model for hot-swap in mlm_rl.py."""
        return self.actor

    # -------------------------------------------------------------------------
    # Critic
    # -------------------------------------------------------------------------

    def setup_critic(self):
        """
        Instantiate CriticModel using the actor's T5Config and move to device.
        Called lazily on the first _train_critic() call.
        """
        self.critic = CriticModel(config=self.actor.config)
        self.critic.to(self.device)

    def get_critic(self):
        return self.critic

    def _make_critic_dataset(self, prepared_critic_df):
        """
        Wrap the prepared critic DataFrame in a CriticDataset.

        The DataFrame is already mixed (mutations + sampled orig at 4:1) and
        has a "reward" column on every row — all data work was done in
        _prepare_data.  This method is a structural wrapper only.

        @type  prepared_critic_df: pd.DataFrame
        @param prepared_critic_df: First element of the _prepare_data() tuple.

        @rtype:  CriticDataset
        """
        return CriticDataset(
            dataset=prepared_critic_df,
            tokenizer=self.tokenizer,
            mask_probability=self._mask_probability,
        )

    def _train_critic(self, critic_dataset):
        """Train critic for one epoch; save weights to save_dir/critic_final."""
        if self.critic is None:
            self.setup_critic()

        # Restore requires_grad — critic parameters are frozen in-place by
        # _finetune_actor_with_ppo_like_loss each cycle and never unfrozen.
        # .train() only restores dropout/batchnorm mode, not requires_grad.
        for p in self.critic.parameters():
            p.requires_grad_(True)
        self.critic.train()
        trainer = HFTrainer(
            model=self.critic,
            args=self._training_args,
            train_dataset=critic_dataset,
            data_collator=CriticDataCollator(),
        )
        trainer.train()

        self.critic.save_pretrained(os.path.join(self.save_dir, "critic_final"))

    # -------------------------------------------------------------------------
    # Actor
    # -------------------------------------------------------------------------

    def _snapshot_actor(self):
        """
        Deep-copy the current actor into self._previous_actor and freeze it.

        Must be called immediately before the PPO update so that π_{t-1} is
        held fixed while π_t is trained.
        """
        self._previous_actor = copy.deepcopy(self.actor)
        self._previous_actor.eval()
        for p in self._previous_actor.parameters():
            p.requires_grad_(False)

    def _make_actor_dataset(self, prepared_actor_df):
        """
        Wrap the prepared actor DataFrame in an ActorDataset.

        The DataFrame is the same mixed dataset as the critic (mutations +
        sampled orig at 4:1), matching CovRL's single self.dataset that both
        training paths consume.  Orig entries contribute CE anti-forgetting
        loss; r(W*) in ActorTrainer.compute_loss is derived dynamically from
        the frozen critic, making the reward=0.0 on orig rows immaterial to
        the PPO objective.  All data work was done in _prepare_data; this
        method is a structural wrapper only.

        @type  prepared_actor_df: pd.DataFrame
        @param prepared_actor_df: Second element of the _prepare_data() tuple.

        @rtype:  ActorDataset
        """
        return ActorDataset(
            dataset=prepared_actor_df,
            tokenizer=self.tokenizer,
            mask_probability=self._mask_probability,
        )

    def _finetune_actor_with_ppo_like_loss(self, actor_dataset, critic, previous_actor):
        """
        Update the actor for one epoch using the CovRL PPO-like objective.

        The critic and previous_actor are frozen before this call.
        ActorTrainer.compute_loss implements Eq. 7 + Eq. 8.

        Saves updated actor weights to save_dir/actor_final.
        """
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)

        self.actor.train()
        actor_trainer = ActorTrainer(
            previous_actor=previous_actor,
            critic=critic,
            model=self.actor,
            args=self._training_args,
            train_dataset=actor_dataset,
            data_collator=ActorDataCollator(
                pad_token_id=self.tokenizer.pad_token_id
            ),
        )
        actor_trainer.train()

        actor_path = os.path.join(self.save_dir, "actor_final")
        self.actor.save_pretrained(actor_path)
        self.tokenizer.save_pretrained(actor_path)