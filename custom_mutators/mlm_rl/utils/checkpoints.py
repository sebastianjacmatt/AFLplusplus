"""Save and load checkpoint helpers for actor and critic models.

Centralised here so that PPOTrainer and future GRPO trainers share the same
checkpoint format.  Both functions are stubs pending Stage 2 implementation.
"""
import os


def save_checkpoint(save_dir, actor, tokenizer, critic=None):
    """
    Persist actor weights (and optionally critic weights) to save_dir.

    @type  save_dir:  str
    @param save_dir:  Root checkpoint directory.

    @type  actor:     transformers.AutoModelForSeq2SeqLM
    @param actor:     Actor model to save.

    @type  tokenizer: transformers.AutoTokenizer
    @param tokenizer: Tokenizer to save alongside the actor.

    @type  critic:    CriticModel or None
    @param critic:    Critic to save; skipped when None.
    """
    # TODO (Stage 2): implement
    pass


def load_checkpoint(save_dir):
    """
    Load actor and critic from save_dir.

    @type  save_dir: str
    @param save_dir: Root checkpoint directory written by save_checkpoint.

    @rtype:  dict
    @return: {"actor": model, "tokenizer": tokenizer, "critic": model or None}
    """
    # TODO (Stage 2): implement
    pass
