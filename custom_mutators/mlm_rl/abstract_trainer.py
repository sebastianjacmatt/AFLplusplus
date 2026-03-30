from abc import ABC, abstractmethod


class Trainer(ABC):
    """
    Abstract base for CovRL trainers.

    Defines the minimal interface required by the AFL++ custom mutator in
    mlm_rl.py.  Concrete trainers (PPO, GRPO, …) are responsible for all
    data loading, reward computation, and training logic internally.
    """

    @abstractmethod
    def finetune(self, corpus_dir):
        """
        Run one full training cycle over the AFL++ queue at corpus_dir.

        mlm_rl.py passes only the queue directory — data loading, reward
        computation, dataset construction, and the training loop are the
        trainer's responsibility.

        @type  corpus_dir: str or None
        @param corpus_dir: Path to the AFL++ output queue directory.
        """
        pass

    @abstractmethod
    def get_actor(self):
        """
        Return the current actor model after a finetune cycle.

        Called by mlm_rl.py _reload_actor() to hot-swap the global ACTOR.
        The returned model must already be on the correct device.
        """
        pass
