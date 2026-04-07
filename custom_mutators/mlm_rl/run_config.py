from config.config import Config
from config.afl    import AFLConfig
from config.covrl  import CovRLConfig
from config.grpo   import GRPOConfig
from config.lora   import LoRAConfig

CONFIG = Config(
    afl=AFLConfig(
        finetune_interval=2,
        fuzz_count=32,
        mask_count=3,
        save_dir="./covrl_checkpoints",
        n_showmap_workers=8,
    ),
    covrl=CovRLConfig(
        train_batch_size=4,
        learning_rate=2e-5,
    ),
    # grpo=GRPOConfig(group_size=8, train_batch_size=4, learning_rate=2e-5),
    # lora=LoRAConfig(r=8, lora_alpha=16),
)
