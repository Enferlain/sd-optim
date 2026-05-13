from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sd_optim.config.dataclasses.generation import GenerationConfig
from sd_optim.config.dataclasses.merge import MergeConfig
from sd_optim.config.dataclasses.optimizer import OptimizerConfig
from sd_optim.config.dataclasses.paths import PathConfig
from sd_optim.config.dataclasses.recipe import RecipeOptimizationConfig
from sd_optim.config.dataclasses.scoring import ScoringConfig
from sd_optim.config.dataclasses.visualization import VisualizationConfig


@dataclass
class SdOptimConfig:
    run_name: str = "${merge.merge_method}_${scoring.scorer_method}"
    hydra: dict[str, Any] = field(default_factory=dict)
    payloads: dict[str, Any] = field(default_factory=dict)
    optimization_guide: dict[str, Any] = field(default_factory=dict)

    webui_urls: dict[str, str] = field(
        default_factory=lambda: {
            "a1111": "http://localhost:7860",
            "forge": "http://localhost:7860",
            "reforge": "http://localhost:7860",
            "comfy": "http://localhost:8188",
            "swarm": "http://localhost:7801",
        }
    )
    webui: str = "forge"
    url: str = "${webui_urls[${webui}]}"

    save_merge_artifacts: bool = True
    save_best: bool = True
    reuse_cached_results: bool = False
    reuse_scan_legacy_pngs: bool = False
    fail_on_error: bool = True

    optimization_mode: str = "merge"
    paths: PathConfig = field(default_factory=PathConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    recipe_optimization: RecipeOptimizationConfig = field(default_factory=RecipeOptimizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    visualizations: VisualizationConfig = field(default_factory=VisualizationConfig)
