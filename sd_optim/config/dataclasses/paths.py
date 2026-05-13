from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PathConfig:
    models_dir: str = ""
    configs_dir: str = ""
    conversion_dir: str = ""
    wildcards_dir: str = "wildcards"
    scorer_model_dir: str = ""
