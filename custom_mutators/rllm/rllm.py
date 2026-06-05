"""AFL++ custom mutator entry point for rllm.

Thin shim — module-level AFL hooks delegate to a single Mutator instance.
"""

import random

from config import load_config
from data.masking import Masking
from model.llm import Model
from mutator import Mutator

MUTATOR: Mutator | None = None


def init(seed):
    global MUTATOR
    random.seed(seed)
    MUTATOR = _build_mutator(seed)


def queue_get(filename):
    return MUTATOR.queue_get(filename)


def fuzz_count(buf):
    return MUTATOR.fuzz_count(buf)


def fuzz(buf, add_buf, max_size):
    return MUTATOR.fuzz(buf, add_buf, max_size)


def post_process(buf):
    return MUTATOR.post_process(buf)


def describe(max_description_len):
    return b"rllm"


def post_run():
    MUTATOR.post_run()


def splice_optout():
    pass


def queue_new_entry(filename_new_queue, filename_orig_queue):
    return MUTATOR.queue_new_entry(filename_new_queue, filename_orig_queue)


def deinit():
    MUTATOR.deinit()


def _build_mutator(seed) -> Mutator:
    cfg = load_config()

    if cfg.sampling_method == "contrastive":
        # CovRL §4: penalty_alpha=0.6, top_k=32. do_sample=False is HF's
        # contrastive-search trigger when penalty_alpha is set.
        model = Model(
            model_name_or_path = cfg.model_name_or_path,
            max_new_tokens     = cfg.max_new_tokens,
            device             = cfg.device,
            gen_kwargs = {
                "do_sample":            False,
                "penalty_alpha":        cfg.penalty_alpha,
                "top_k":                cfg.contrastive_top_k,
                "no_repeat_ngram_size": 3,
            },
        )
    elif cfg.sampling_method == "nucleus":
        model = Model(
            model_name_or_path = cfg.model_name_or_path,
            max_new_tokens     = cfg.max_new_tokens,
            device             = cfg.device,
            gen_kwargs = {
                "do_sample":            True,
                "temperature":          cfg.temperature,
                "top_p":                cfg.top_p,
                "top_k":                cfg.top_k,
                "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
            },
        )
    else:
        raise ValueError(
            f"cfg.sampling_method must be 'contrastive' or 'nucleus', "
            f"got {cfg.sampling_method!r}."
        )
    masking = Masking(
        sentinel_ids     = model.tokenizer.sentinel_ids,
        word_starts_fn   = model.tokenizer.word_starts if cfg.whole_word_masking else None,
        corruption_rate  = cfg.corruption_rate,
        mean_span_length = cfg.mean_span_length,
        min_span_length  = cfg.min_span_length,
        max_span_length  = cfg.max_span_length,
        max_masks        = cfg.max_masks,
        g                = cfg.group_size,
    )

    trainer = None
    if cfg.finetune_every > 0:
        # GRPO trainer (HF Trainer + RolloutDataset, full fine-tune). `method`
        # selects the variant subclass (one file each). Set finetune_every=0 for a
        # straight LLM mutator (no training). Reward from the rollout (data/rewarding.py).
        from training.grpo import GRPOTrainer
        trainers = {"grpo": GRPOTrainer}
        try:
            from training.drgrpo import DrGRPOTrainer
            trainers["drgrpo"] = DrGRPOTrainer
        except ImportError:
            pass
        if cfg.method not in trainers:
            raise ValueError(f"unknown method {cfg.method!r}; known: {sorted(trainers)}")
        trainer = trainers[cfg.method](
            model,
            lr                = cfg.lr,
            kappa             = cfg.kappa,
            eps_low           = cfg.eps_low,
            eps_high          = cfg.eps_high,
            max_train_infills = cfg.max_train_infills,
            batch_size        = cfg.train_batch_size,
            fp16              = cfg.fp16,
            log_entropy       = cfg.log_train_entropy,
            kl_ref_coef       = cfg.kl_ref_coef,
            ref_update_every  = cfg.ref_update_every,
        )
    return Mutator(cfg, masking, model, trainer=trainer)
