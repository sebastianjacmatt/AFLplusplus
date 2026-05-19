def init(seed: int) -> None:
    """ initializes the trainer, the rollout buffer, the mutator, rewarder using the CFG
    """
    global TRAINER, ROLLOUT, MUTATOR, REWARDER, CFG
    pass
def deinit() -> None:
    """
    Store Rollout and model checkpoint on exit
    """
    pass
def queue_get(filename: str) -> bool:
    """
    increments and initialize finetuning
    """
    _queue_get_count += 1
    if _queue_get_count > 0 and _queue_get_count % CFG.finetune_every:
        dataset = RolloutDataset(mutator.rollout)
        trainer.finetune(dataset)

def fuzz_count(buf: bytearray) -> int:
    """
    pass seed to mutator for CFG.fuzz_count amount of fuzz_one's
    """
    MUTATOR.mask(buf)
    return CFG.fuzz_count

def fuzz(buf: bytearray, add_buf: bytearray, max_size: int) -> bytearray:
    """
    fuzz one based on masked program from within Mutator
    """
    mut = MUTATOR.mutate(max_size)
    return bytearray(out)

def post_run() -> None:
    """
    Adds rewards to rollout
    """
    MUTATOR.collect(REWARDER)