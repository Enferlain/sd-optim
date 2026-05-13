from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecipeOptimizationConfig:
    recipe_path: str = ""
    target_nodes: Any = ""
    target_params: list[str] = field(default_factory=list)
