from abc import ABC, abstractmethod

class Trainer(ABC):
    """
    Abstract trainer for PPO and GRPO with optional LoRA support.
    """
    @abstractmethod
    def record_sample(self, sample):
        """
        Accumulate a training sample (mutation + coverage reward) for the next finetune cycle.
        """
        pass

    @abstractmethod
    def compute_advantages(self, rewards):
        """
        Derive per-sample advantages from a list of scalar rewards.
        PPO uses critic-estimated baselines; GRPO uses within-group normalisation.
        """
        pass

    @abstractmethod
    def finetune(self, mutation_dataset, sampled_train_data):
        """
        Run one full training cycle over the prepared datasets.
        """
        pass

    @abstractmethod
    def get_actor(self):
        """
        Return the current actor model (used by _reload_actor() after a finetune cycle).
        """
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir):
        """
        Persist model weights to save_dir.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, save_dir):
        """
        Restore model weights from save_dir.
        """
        pass