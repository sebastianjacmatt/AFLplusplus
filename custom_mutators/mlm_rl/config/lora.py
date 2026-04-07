from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    # T5/CodeT5+ attention projection names targeted by LoRA adapters.
    target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
