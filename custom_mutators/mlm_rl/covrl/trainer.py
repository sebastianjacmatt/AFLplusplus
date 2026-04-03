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
  _prepare_data()
      └─ returns (prepared_critic_df, prepared_actor_df)
            ├─ _make_critic_dataset(prepared_critic_df) → CriticDataset
            └─ _make_actor_dataset(prepared_actor_df)   → ActorDataset

All queue loading, reward computation, and corpus mixing live exclusively in
_prepare_data.  The dataset builders are pure structural wrappers over
already-prepared DataFrames.
"""
import copy
import logging
import os

import pandas as pd

log = logging.getLogger(__name__)
from transformers import TrainingArguments
from transformers import Trainer as HFTrainer

from abstract_trainer import Trainer
from covrl.critic import CriticModel, CriticDataset, CriticDataCollator
from covrl.actor  import ActorDataset, ActorDataCollator, ActorTrainer
from utils.data_utils import load_orig_corpus, load_mutation_corpus
from utils.rewarding  import Rewarder


class PPOTrainer(Trainer):

    def __init__(
        self,
        actor,
        tokenizer,
        device,
        save_dir="./covrl_checkpoints",
        train_batch_size=4,
        learning_rate=2e-5,
        mask_probability=0.15,
        n_showmap_workers=8,
    ):
        """
        @type  actor:              transformers.AutoModelForSeq2SeqLM
        @param actor:              Pre-loaded actor model from mlm_rl.py init().

        @type  tokenizer:          transformers.AutoTokenizer
        @param tokenizer:          Tokenizer shared with mlm_rl.py.

        @type  device:             str
        @param device:             "cuda" or "cpu", passed from DEVICE in mlm_rl.py.

        @type  save_dir:           str
        @param save_dir:           Root directory for actor/critic checkpoints.

        @type  train_batch_size:   int
        @param train_batch_size:   Per-device batch size for critic and actor training.

        @type  learning_rate:      float
        @param learning_rate:      Learning rate for critic and actor training.

        @type  mask_probability:   float
        @param mask_probability:   Fraction of tokens masked per span.
                                   Should match mlm_rl.py MASK_PROBABILITY.

        @type  n_showmap_workers:  int
        @param n_showmap_workers:  Number of parallel afl-showmap workers.  AFL++
                                   occupies one core; the remaining cores are free
                                   for showmap.  Uses multiprocessing spawn context
                                   (no AFL++ fork-server state inherited).
                                   Set to 1 to run sequentially (safest, slowest).

        """
        self.actor     = actor
        self.tokenizer = tokenizer
        self.device    = device
        self.save_dir  = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self._finetune_cycle_index = 0
        self.critic          = None
        self._previous_actor = None
        self._mask_probability = mask_probability

        # HuggingFace TrainingArguments shared by critic and actor training.
        # Checkpoint saving is handled explicitly; HF auto-save is disabled.
        self._training_args = TrainingArguments(
            output_dir=os.path.join(save_dir, "hf_output"),
            overwrite_output_dir=True,
            per_device_train_batch_size=train_batch_size,
            fp16=False,
            learning_rate=learning_rate,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            num_train_epochs=1,
            remove_unused_columns=False,
        )

        self._rewarder = Rewarder(
            tmp_dir=os.path.join(save_dir, "tmp"),
            n_workers=n_showmap_workers,
        )

    # -------------------------------------------------------------------------
    # Trainer interface
    # -------------------------------------------------------------------------

    def finetune(self):
        """
        Run one staged CovRL finetuning cycle over the current AFL++ queue.

        Data preparation runs once via _prepare_data; the resulting DataFrames
        are handed directly to _make_critic_dataset and _make_actor_dataset so
        that queue loading, reward computation, and corpus mixing are never
        duplicated between the two training paths.
        """
        log.info("[finetune] cycle %d — preparing data", self._finetune_cycle_index)
        prepared_critic_df, prepared_actor_df = self._prepare_data()

        if prepared_critic_df.empty:
            raise Exception("afl queue contains no mutation entries before finetuning")

        log.info("[finetune] training critic on %d rows", len(prepared_critic_df))
        critic_dataset = self._make_critic_dataset(prepared_critic_df)
        self._train_critic(critic_dataset)
        log.info("[finetune] critic done")

        if self._finetune_cycle_index > 0:
            self._snapshot_actor()
            log.info("[finetune] finetuning actor on %d rows", len(prepared_actor_df))
            actor_dataset = self._make_actor_dataset(prepared_actor_df)
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
    # Shared data preparation
    # -------------------------------------------------------------------------

    def _prepare_data(self):
        """
        Single shared preprocessing step for one finetune cycle.

        Reads the current AFL++ queue via data_utils and performs reward
        computation and corpus mixing.  Both _make_critic_dataset and
        _make_actor_dataset consume the returned DataFrames without any
        further data loading.

        Steps:
          1. Load all non-orig queue entries (coverage-increasing mutations).
          2. Compute rewards via self._rewarder (afl-showmap + IDF).
             TODO: remove this or at least raise rewards, as this should not really be possible: Falls back to reward=0.0 when self._rewarder is None.
          3. Load orig: queue entries as the clean reference corpus.
          4. Build one shared mixed dataset for both training paths:
               mutations + orig sampled at 4:1 relative to len(mutations).
          5. Return the mixed dataset as both prepared_critic_df and
             prepared_actor_df.

        Both training paths receive the same mixed dataset, mirroring CovRL's
        FineTuner.preprocess() which builds a single self.dataset consumed by
        both train_critic() and finetune_actor().

        TODO: I don't belive covrl does this.
        Orig entries in the mixed dataset receive reward=0.0.  For the actor
        this has no effect: ActorTrainer.compute_loss derives r(W*) from the
        frozen critic dynamically.  For the critic, orig entries train toward
        label 4 (score_to_label(0.0)) and contribute CE anti-forgetting signal.

        @rtype:  tuple[pd.DataFrame, pd.DataFrame]
        @return: (prepared_critic_df, prepared_actor_df) — same object both slots
        """
        # Step 1 — load all coverage-increasing (non-orig) queue entries
        mutations = load_mutation_corpus()
        log.info("[prepare_data] loaded %d mutations", len(mutations))

        if not mutations.empty:
            # Step 2 — compute rewards
            if self._rewarder is not None:
                log.info("[prepare_data] computing rewards via afl-showmap...")
                mutations = self._rewarder.compute(mutations)
                log.info("[prepare_data] rewards done")
            else:
                raise Exception("No rewards in current mutations")
                #mutations["reward"] = 0.0

        # Step 3 — load orig entries as clean reference corpus
        orig_corpus = load_orig_corpus()

        # Steps 4-5 — build one shared mixed dataset for both training paths
        n_mutations = len(mutations)
        if not orig_corpus.empty and n_mutations > 0:
            n_orig       = min(n_mutations * 4, len(orig_corpus))
            sampled_orig = orig_corpus.sample(n=n_orig, ignore_index=True)
            sampled_orig["reward"] = 0.0 # TODO: check that this is correct with covrl
            mixed_df = pd.concat([mutations, sampled_orig], ignore_index=True)
        else:
            mixed_df = mutations

        return mixed_df, mixed_df

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

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    def save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, save_dir):
        pass

    def _get_latest_actor_checkpoint(self):
        pass

    def _get_latest_critic_checkpoint(self):
        pass
