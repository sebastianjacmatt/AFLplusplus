from dataclasses import dataclass


@dataclass
class AFLConfig:
    # Seeds processed between each finetune trigger (CovRL SYNC_INTERVAL=100).
    finetune_interval: int = 2

    # Mutation budget returned by fuzz_count() per seed.
    fuzz_count: int = 32

    # Max mask tokens inserted or overwritten per mutation step.
    mask_count: int = 3

    # Root directory for actor/critic checkpoints and showmap tmp files.
    save_dir: str = "./covrl_checkpoints"

    # Parallel afl-showmap threads (ThreadPoolExecutor). subprocess.run releases
    # the GIL during os.waitpid so threads achieve true parallelism.
    n_showmap_workers: int = 8
