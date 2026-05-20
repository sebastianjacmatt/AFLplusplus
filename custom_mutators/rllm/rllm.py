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
    MUTATOR.queue_new_entry(filename_new_queue, filename_orig_queue)
    return False


def deinit():
    MUTATOR.deinit()


def _build_mutator(seed) -> Mutator:
    cfg = load_config()
    model = Model(
        model_name_or_path = cfg.model_name_or_path,
        max_new_tokens     = cfg.max_new_tokens,
        temperature        = cfg.temperature,
        top_p              = cfg.top_p,
        top_k              = cfg.top_k,
        device             = cfg.device,
    )
    masking = Masking(
        sentinel_ids     = model.tokenizer.sentinel_ids,
        word_starts_fn   = model.tokenizer.word_starts if cfg.whole_word_masking else None,
        corruption_rate  = cfg.corruption_rate,
        mean_span_length = cfg.mean_span_length,
        min_span_length  = cfg.min_span_length,
        max_span_length  = cfg.max_span_length,
    )
    return Mutator(cfg, masking, model, trainer=None)
