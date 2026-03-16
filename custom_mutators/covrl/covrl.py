#!/usr/bin/env python
# encoding: utf-8
"""
This AFL++ re implementation preserves the staged CovRL baseline while replacing
the original inter process TCP based architecture with a single custom mutator.

Because the original CovRL AFL 2.52b integration operates on token level
representations rather than raw byte buffers, this port caches the tokenized
representation of the currently selected AFL seed in fuzz_count(). This avoids
re tokenizing the same seed on every fuzz() call. Each fuzz() invocation then
applies a fresh mask based mutation to the cached token sequence and performs
token level infilling before re encoding the result to bytes for execution.

TODO: token MASK & token infilling
TODO: copy hex_to_dec(bytes(buf))/dec_to_hex(ids) from covrl.utils instead of decode_or_tokenize_once()
TODO: define config
TODO: rewarding in finetuning and dataset finetuning dataset creation
TODO: describe ambiguety in post_process()
TODO: splicing
TODO: description of bytes->tokens-encode->masking->tokens-decode->bytes

@author:     Sebastian Jacobsen Matthews
@contact:    sebastianjacmatt@gmail.com

@license:
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import random


def init(seed):
    """
    Called once when AFLFuzz starts up.

    @type seed: int
    @param seed: A 32 bit random value
    """
    random.seed(seed)

    global CONFIG
    global ACTOR
    global TOKENIZER
    global FINETUNER
    global SAVE_DIR

    global FUZZ_COUNTER_LIMIT
    global fuzz_counter
    global pending_new_queue_files
    global finetune_cycle_index

    global current_seed_token_ids
    global current_seed_metadata

    global last_actor_path
    global last_critic_path

    # TODO: define how CONFIG is loaded in the AFL++ custom mutator environment
    CONFIG = load_config()

    SAVE_DIR = CONFIG.save_dir

    ACTOR = Inferencer(
        conf=CONFIG,
        model_path=CONFIG.actor_path,
        critic_path=CONFIG.critic_path,
        sample_method="contrastive",
        save_dir=SAVE_DIR,
    )

    TOKENIZER = ACTOR.tokenizer
    FINETUNER = ACTOR.finetuner

    FUZZ_COUNTER_LIMIT = CONFIG.finetune_interval
    fuzz_counter = 0
    pending_new_queue_files = []
    finetune_index = 0

    current_seed_token_ids = None
    current_seed_metadata = None

    last_actor_path = CONFIG.actor_path
    last_critic_path = CONFIG.critic_path


def deinit():
    """
    Called once before AFLFuzz exits.
    """
    # TODO: persist any state that should survive shutdown
    pass


def queue_get(filename):
    '''
    Called at the beginning of each fuzz iteration to determine whether the
    test case should be fuzzed, we use this to determine if we should finetune or not based on fuzz_count

    @type filename: str
    @param filename: File name of the test case in the current queue entry

    @rtype: bool
    @return: Return True if the custom mutator decides to fuzz the test case,
        and False otherwise
    '''
    _maybe_finetune()
    return True


def fuzz_count(buf):
    """
    Called when AFL selects a seed and wants to know how many times fuzz()
    should be invoked for that seed.

    This caches the tokenized representation of the currently selected seed
    once per seed selection, so fuzz() can avoid repeated tokenization work.
    """
    global fuzz_counter
    global current_seed_token_ids
    global current_seed_metadata

    # TODO: define exact decoding semantics from AFL++ bytes to JS source
    current_seed_token_ids = decode_or_tokenize_once(buf)

    current_seed_metadata = {
        # TODO: add any metadata needed by the mutator
        "seed_length_bytes": len(buf),
        "seed_length_tokens": len(current_seed_token_ids),
    }
    # TODO: choose how many mutation attempts to perform per selected seed
    return N


def fuzz(buf, add_buf, max_size):
    """
    Called per fuzzing iteration.

    @type buf: bytearray
    @param buf: The buffer that should be mutated.

    @type add_buf: bytearray
    @param add_buf: A second buffer that can be used as mutation source.

    @type max_size: int
    @param max_size: Maximum size of the mutated output. The mutation must not
        produce data larger than max_size.

    @rtype: bytearray
    @return: A new bytearray containing the mutated data
    """
    global current_seed_token_ids

    if current_seed_token_ids is None:
        raise Exception("fuzz() is reached without a cached seed")

    base_token_ids = current_seed_token_ids.copy()

    splice_token_ids = None
    if add_buf:
        # TODO: only tokenize add_buf when splice mode is actually selected
        splice_token_ids = decode_or_tokenize_once(add_buf)

    masked_token_ids = _random_mask(
        token_ids=base_token_ids,
        splice_source=splice_token_ids,
        # TODO: define exact mutation modes and probabilities
    )

    mutated_token_ids = _actor_infill(masked_token_ids)

    out_buf = encode(mutated_token_ids)

    if len(out_buf) > max_size:
        # TODO: decide whether to truncate, resample, or reject oversized outputs
        out_buf = out_buf[:max_size]

    return out_buf

# def post_process(buf):
#     '''
#     Called just before the execution to write the test case in the format
#     expected by the target
#
#     @type buf: bytearray
#     @param buf: The buffer containing the test case to be executed
#
#     @rtype: bytearray
#     @return: The buffer containing the test case after
#     '''
#     return buf

def queue_new_entry(filename_new_queue, filename_orig_queue):
    """
    Called after adding a new test case to the queue.

    @type filename_new_queue: str
    @param filename_new_queue: File name of the new queue entry

    @type filename_orig_queue: str
    @param filename_orig_queue: File name of the original queue entry
    """
    global pending_new_queue_files

    # AFL has already decided the testcase is interesting and added it to queue.
    # This hook is only a notification point for later finetuning or bookkeeping.
    pending_new_queue_files.append(
        {
            "new": filename_new_queue,
            "orig": filename_orig_queue,
        }
    )


def _maybe_finetune():
    """
    Trigger fuzz_counter based finetuning when enough FUZZ_COUNTER_LIMIT attempts have elapsed.

    This preserves the staged CovRL design:
    mutation and execution happen online during fuzzing,
    reward computation and finetuning happen later in batch over the corpus.
    """
    global fuzz_counter
    global finetune_index

    fuzz_counter += 1

    if fuzz_counter < FUZZ_COUNTER_LIMIT:
        return

    corpus_dir = SAVE_DIR # define a proper way to store SAVE_DIR
    _finetune(corpus_dir)

    fuzz_counter = 0
    finetune_index += 1


def _finetune(corpus_dir):
    """
    Run one staged finetuning cycle over saved corpus files.
    """
    global ACTOR
    global FINETUNER
    global last_actor_path
    global last_critic_path
    global finetune_cycle_index

    dataset = load_saved_queue_files(corpus_dir)

    # each file becomes something like:
    # {
    #     "is_orig": bool,
    #     "file_id": str,
    #     "data": decoded_js_source,
    # }
    
    # TODO: fix proper rewarding
    mutation_dataset = Rewarding.update(dataset, is_update_idf=True)

    # Rewarding.update conceptually does:
    #
    # fit(dataset)
    #   for each testcase:
    #       run afl-showmap on interpreter and testcase
    #       read coverage bitmap
    #       classify validity
    #       syntax error   -> reward = -1.0
    #       semantic error -> reward = -0.5
    #       valid          -> reward deferred until coverage scoring
    #
    # update_idf(dataset, alpha)
    #   compute document frequency over unique coverage
    #   update IDF embedding with momentum
    #
    # get_reward(dataset)
    #   for valid cases only:
    #       tfidf_score = dot(bitmap, idf)
    #       reward = sigmoid(log(tfidf_score))
    #
    # return dataset with rewards

    # TODO: define how additional sampled training data is mixed in, 4:1 maybe from CovRL-Fuzz baseline
    sampled_train_data = sample_train_data()

    critic_dataset = make_critic_dataset(
        mutation_dataset=mutation_dataset,
        sampled_train_data=sampled_train_data,
    )

    # TODO: define training interface exactly
    train_critic(critic_dataset)

    if finetune_index > 0:
        actor_dataset = make_actor_dataset(
            mutation_dataset=mutation_dataset,
            sampled_train_data=sampled_train_data,
        )

        # TODO: define how critic and previous actor are passed into PPO style update
        finetune_actor_with_ppo_like_loss(
            actor_dataset=actor_dataset,
            critic=get_current_critic(),
            previous_actor=get_previous_actor(),
        )

    reload_actor()


def reload_actor():
    """
    Reload the newest actor and any dependent state after a finetuning cycle.
    """
    global ACTOR
    global TOKENIZER
    global FINETUNER
    global last_actor_path
    global last_critic_path
    global CONFIG
    global SAVE_DIR

    # TODO: define where newly trained actor and critic checkpoints are written
    last_actor_path = get_latest_actor_checkpoint()
    last_critic_path = get_latest_critic_checkpoint()

    ACTOR = Inferencer(
        conf=CONFIG,
        model_path=last_actor_path,
        critic_path=last_critic_path,
        sample_method="contrastive",
        save_dir=SAVE_DIR,
    )

    TOKENIZER = ACTOR.tokenizer
    FINETUNER = ACTOR.finetuner


def _random_mask(token_ids, splice_source=None):
    """
    Create a fresh masked mutation candidate from the cached current seed.

    This should mirror the runtime masking behavior of the original modified
    AFL 2.52b baseline, where fresh masking happens per mutation attempt.
    """
    # TODO: implement insert mask
    # TODO: implement overwrite mask
    # TODO: implement splice mode if needed
    # TODO: preserve sentinel or mask token conventions expected by Inferencer
    return masked_token_ids


def _actor_infill(masked_token_ids):
    """
    Run actor inference on a token sequence that already contains runtime masks.
    """
    # TODO: define exact ACTOR inference call signature
    return ACTOR.inference(masked_token_ids)


def decode_or_tokenize_once(buf):
    """
    Convert AFL++ byte buffer into the token representation expected by runtime
    masking and infilling.
    """
    # TODO: decode bytes to JS source if needed
    # TODO: tokenize using TOKENIZER
    return token_ids


def encode(token_ids):
    """
    Convert mutated token ids back into an AFL++ byte buffer.
    """
    # TODO: detokenize token ids to JS source if needed
    # TODO: encode JS source to bytes
    return out_buf

def load_saved_queue_files(corpus_dir):
    # TODO
    return dataset

def load_config():
    # TODO
    return config

def sample_train_data():
    # TODO
    return sampled_train_data

def make_critic_dataset(mutation_dataset, sampled_train_data):
    # TODO
    return critic_dataset

def make_actor_dataset(mutation_dataset, sampled_train_data):
    # TODO
    return actor_dataset

def train_critic(critic_dataset):
    # TODO
    pass

def finetune_actor_with_ppo_like_loss(actor_dataset, critic, previous_actor):
    # TODO
    pass

def get_current_critic():
    # TODO
    return critic

def get_previous_actor():
    # TODO
    return previous_actor

def get_latest_actor_checkpoint():
    # TODO
    return actor_path

def get_latest_critic_checkpoint():
    # TODO
    return critic_path