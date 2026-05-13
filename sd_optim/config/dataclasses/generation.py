from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    batch_size: int = 1
    save_imgs: bool = True
    img_average_type: str = "arithmetic"
    generator_concurrency_limit: int = 10
    generator_keepalive_interval: int = 60
    generator_total_timeout: int | float | None = 0
