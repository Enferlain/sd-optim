from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from sd_optim.config.dataclasses.optimizer import OptimizerConfig
from sd_optim.config.dataclasses.run import SdOptimConfig
from sd_optim.config.schemas import register_all, register_sd_optim
from sd_optim.config.validation import validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_hydra() -> None:
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_structured_runtime_schema_contains_stable_config_sections() -> None:
    cfg = OmegaConf.structured(SdOptimConfig)

    assert isinstance(cfg.optimizer, DictConfig)
    assert isinstance(cfg.optimizer.optuna_config, DictConfig)
    assert isinstance(cfg.recipe_optimization, DictConfig)
    assert isinstance(cfg.paths, DictConfig)
    assert isinstance(cfg.merge, DictConfig)
    assert isinstance(cfg.generation, DictConfig)
    assert isinstance(cfg.scoring, DictConfig)
    assert cfg.optimizer.optuna_config.sampler.type == "tpe"
    assert cfg.paths.wildcards_dir == "wildcards"
    assert cfg.merge.merge_method == "weighted_sum"
    assert cfg.scoring.scorer_method == ["manual"]
    assert cfg.generation.generator_concurrency_limit == 10

    with pytest.raises(ConfigAttributeError):
        cfg.optimizer.not_a_real_optimizer_field = True

    with pytest.raises(ConfigAttributeError):
        cfg.scorer_method = ["manual"]

    with pytest.raises(ConfigAttributeError):
        cfg.models_dir = "models"

    with pytest.raises(ConfigAttributeError):
        cfg.merge_method = "ties_sum"


def test_hydra_composition_applies_schema_to_stable_runtime_sections() -> None:
    register_all()

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        cfg = compose(config_name="config")

    assert OmegaConf.is_struct(cfg.optimizer)
    assert OmegaConf.is_struct(cfg.optimizer.optuna_config)
    assert OmegaConf.is_struct(cfg.recipe_optimization)
    assert OmegaConf.is_struct(cfg.paths)
    assert OmegaConf.is_struct(cfg.merge)
    assert OmegaConf.is_struct(cfg.generation)
    assert OmegaConf.is_struct(cfg.scoring)
    assert cfg.optimizer.init_points == 55
    assert cfg.recipe_optimization.target_nodes == "&12"
    assert cfg.paths.models_dir
    assert cfg.merge.merge_method == "weighted_sum"
    assert cfg.scoring.scorer_method == ["manual"]

    with pytest.raises(ConfigAttributeError):
        cfg.recipe_optimization.not_a_real_recipe_field = "bad"


def test_config_template_loads_runtime_schema_and_names_model_directory() -> None:
    template = yaml.safe_load((PROJECT_ROOT / "conf" / "config.tmpl.yaml").read_text(encoding="utf-8"))

    assert "sd_optim_schema" in template["defaults"]
    assert "paths" in template
    assert "models_dir" in template["paths"]
    assert "configs_dir" in template["paths"]
    assert "merge" in template
    assert "merge_method" in template["merge"]
    assert "model_paths" in template["merge"]
    assert "recipe_optimization" in template
    assert "optimizer" in template
    assert "generation" in template
    assert "scoring" in template
    assert "models_dir" not in template
    assert "configs_dir" not in template
    assert "merge_method" not in template
    assert "model_paths" not in template
    assert "scorer_method" not in template
    assert "batch_size" not in template


def test_legacy_config_schema_module_reexports_current_schema_api() -> None:
    from sd_optim.config_schema import SdOptimConfig as LegacySdOptimConfig
    from sd_optim.config_schema import register_config_schemas

    assert LegacySdOptimConfig is SdOptimConfig
    assert register_config_schemas is register_sd_optim


def test_optimizer_config_semantic_validation_requires_one_optimizer() -> None:
    cfg = OmegaConf.structured(SdOptimConfig)
    cfg.optimizer = OmegaConf.structured(OptimizerConfig(bayes=True, optuna=True))

    with pytest.raises(ValueError, match="Exactly one optimizer"):
        validate_config(cfg)
