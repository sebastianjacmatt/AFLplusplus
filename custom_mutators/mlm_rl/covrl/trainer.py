"""High-level CovRL PPO orchestration.

PPOTrainer owns the finetune cycle counter, the critic, the actor snapshot,
and the accumulated mutation dataset.  mlm_rl.py interacts with this class
only via the two abstract methods defined in abstract_trainer.Trainer:

  finetune(corpus_dir) — run one staged training cycle
  get_actor()          — return the (updated) actor for hot-swap

Cycle semantics:
  cycle 0  — critic warmup only; actor is not updated because the critic
             has not yet produced meaningful value estimates.
  cycle N>0 — critic update then actor PPO update using critic as baseline.

Data flow per cycle:
  _prepare_data(corpus_dir)
      └─ returns (prepared_critic_df, prepared_actor_df)
            ├─ _make_critic_dataset(prepared_critic_df) → CriticDataset
            └─ _make_actor_dataset(prepared_actor_df)   → ActorDataset

All queue loading, reward computation, mutation accumulation, and corpus
mixing live exclusively in _prepare_data.  The dataset builders are pure
structural wrappers over already-prepared DataFrames.
"""
import copy
import os

import pandas as pd
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
        afl_showmap_path=None,
        interpreter_path=None,
        interpreter_target="v8",
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

        @type  afl_showmap_path:   str or None
        @param afl_showmap_path:   Absolute path to the afl-showmap binary.
                                   When None (or interpreter_path is None), a
                                   Rewarder is not created and all new mutation
                                   entries receive a 0.0 placeholder reward.

        @type  interpreter_path:   str or None
        @param interpreter_path:   Absolute path to the target interpreter binary
                                   (e.g. the jerry or d8 executable).

        @type  interpreter_target: str
        @param interpreter_target: Error dialect for stderr classification.
                                   One of "v8", "jsc", "chakra", "jerry".
                                   Passed to Rewarder at construction.
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
        )

        # Non-orig AFL++ queue entries accumulated across finetune cycles.
        # Schema: ["is_orig", "file_id", "data", "reward"].
        # "reward" is absent on the empty initialisation; it is added by
        # Rewarder.compute or by the 0.0 placeholder on the first cycle.
        # is_orig is always False here; the column is retained for schema
        # consistency with load_mutation_corpus and the shared DataFrame contract.
        self._mutation_dataset = pd.DataFrame(
            [], columns=["is_orig", "file_id", "data"]
        )

        # Rewarder encapsulates the afl-showmap pipeline and all IDF state.
        # None when showmap or interpreter paths are not yet configured.
        if afl_showmap_path is not None and interpreter_path is not None:
            self._rewarder = Rewarder(
                afl_showmap_path=afl_showmap_path,
                interpreter_path=interpreter_path,
                interpreter_target=interpreter_target,
                tmp_dir=os.path.join(save_dir, "tmp"),
            )
        else:
            self._rewarder = None

    # -------------------------------------------------------------------------
    # Trainer interface
    # -------------------------------------------------------------------------

    def finetune(self, corpus_dir):
        """
        Run one staged CovRL finetuning cycle over the AFL++ queue at corpus_dir.

        Data preparation runs once via _prepare_data; the resulting DataFrames
        are handed directly to _make_critic_dataset and _make_actor_dataset so
        that queue loading, reward computation, and corpus mixing are never
        duplicated between the two training paths.

        @type  corpus_dir: str or None
        @param corpus_dir: Path to the AFL++ output queue directory.
        """
        prepared_critic_df, prepared_actor_df = self._prepare_data(corpus_dir)

        critic_dataset = self._make_critic_dataset(prepared_critic_df)
        self._train_critic(critic_dataset)

        if self._finetune_cycle_index > 0:
            self._snapshot_actor()
            actor_dataset = self._make_actor_dataset(prepared_actor_df)
            self._finetune_actor_with_ppo_like_loss(
                actor_dataset=actor_dataset,
                critic=self.critic,
                previous_actor=self._previous_actor,
            )

        self._finetune_cycle_index += 1

    def get_actor(self):
        """Return the current actor model for hot-swap in mlm_rl.py."""
        return self.actor

    # -------------------------------------------------------------------------
    # Shared data preparation
    # -------------------------------------------------------------------------

    def _prepare_data(self, corpus_dir):
        """
        Single shared preprocessing step for one finetune cycle.

        Performs all queue I/O, reward computation, and corpus mixing exactly
        once per cycle.  Both _make_critic_dataset and _make_actor_dataset
        consume the returned DataFrames without any further data loading.

        Steps:
          1. Load new non-orig queue entries (incremental dedup via known file_ids).
          2. Compute rewards via self._rewarder, which runs afl-showmap and
             updates its IDF from the full accumulated bitmap history (CovRL
             update_idf alignment — full-history document-frequency weighting).
             Falls back to reward=0.0 when self._rewarder is None.
          3. Accumulate rewarded entries into self._mutation_dataset.
          4. Load orig: queue entries as the clean reference corpus.
          5. Build one shared mixed dataset for both training paths:
               self._mutation_dataset + orig sampled at 4:1 relative to
               len(self._mutation_dataset).
          6. Return the same mixed dataset as both prepared_critic_df and
             prepared_actor_df.

        Both training paths receive the same mixed dataset, mirroring CovRL's
        FineTuner.preprocess() which builds a single self.dataset consumed by
        both train_critic() and finetune_actor().

        Orig entries in the mixed dataset receive reward=0.0, which is a
        simplification relative to CovRL where the train/orig sample is also
        run through afl-showmap to obtain coverage-derived rewards.  For the
        actor this has no effect: ActorTrainer.compute_loss pops the rewards
        field and derives r(W*) from the frozen critic dynamically.  For the
        critic, orig entries always train toward label 4 (score_to_label(0.0))
        rather than a coverage-calibrated label.  Both training paths still
        receive correct CE anti-forgetting signal from the orig entries.

        Shared trainer state updated:
          self._mutation_dataset — accumulated non-orig entries with rewards
          self._rewarder         — IDF vector updated internally by Rewarder

        @type  corpus_dir: str or None
        @param corpus_dir: AFL++ output queue directory.

        @rtype:  tuple[pd.DataFrame, pd.DataFrame]
        @return: (prepared_critic_df, prepared_actor_df) — same object both slots
        """
        # Step 1 — load unseen mutation entries only
        new_mutations = load_mutation_corpus(
            corpus_dir,
            known_ids=set(self._mutation_dataset["file_id"].tolist()),
        )

        if not new_mutations.empty:
            # Step 2 — compute rewards via Rewarder (afl-showmap + full-history IDF)
            if self._rewarder is not None:
                new_mutations = self._rewarder.compute(new_mutations)
            else:
                new_mutations["reward"] = 0.0

            # Step 3 — accumulate
            if self._mutation_dataset.empty:
                self._mutation_dataset = new_mutations
            else:
                self._mutation_dataset = pd.concat(
                    [self._mutation_dataset, new_mutations], ignore_index=True
                )

        # Step 4 — load orig: entries as clean reference corpus
        orig_corpus = load_orig_corpus(corpus_dir)

        # Steps 5-6 — build one shared mixed dataset for both training paths
        n_mutations = len(self._mutation_dataset)
        if not orig_corpus.empty and n_mutations > 0:
            n_orig       = min(n_mutations * 4, len(orig_corpus))
            sampled_orig = orig_corpus.sample(n=n_orig, ignore_index=True)
            sampled_orig["reward"] = 0.0
            mixed_df = pd.concat(
                [self._mutation_dataset, sampled_orig], ignore_index=True
            )
        else:
            mixed_df = self._mutation_dataset

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