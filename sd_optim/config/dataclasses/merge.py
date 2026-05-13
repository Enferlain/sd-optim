from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MergeConfig:
    model_paths: list[str] = field(default_factory=list)
    base_model_index: int = 0
    fallback_model_index: int | None = -1
    merge_method: str = "weighted_sum"
    device: str = "cuda"
    threads: int = 4
    merge_dtype: str = "fp32"
    save_dtype: str = "bf16"
    add_extra_keys: bool = False
