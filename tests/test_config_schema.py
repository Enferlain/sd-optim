from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from sd_optim.config_schema import SdOptimConfig, register_config_schemas


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
    assert cfg.optimizer.optuna_config.sampler.type == "tpe"
    assert cfg.scorer_method == ["manual"]
    assert cfg.generator_concurrency_limit == 10

    with pytest.raises(ConfigAttributeError):
        cfg.optimizer.not_a_real_optimizer_field = True


def test_hydra_composition_applies_schema_to_stable_runtime_sections() -> None:
    register_config_schemas()

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "conf")):
        cfg = compose(config_name="config")

    assert OmegaConf.is_struct(cfg.optimizer)
    assert OmegaConf.is_struct(cfg.optimizer.optuna_config)
    assert OmegaConf.is_struct(cfg.recipe_optimization)
    assert cfg.optimizer.init_points == 55
    assert cfg.recipe_optimization.target_nodes == "&12"
    assert cfg.scorer_method == ["manual"]

    with pytest.raises(ConfigAttributeError):
        cfg.recipe_optimization.not_a_real_recipe_field = "bad"


def test_config_template_loads_runtime_schema_and_names_model_directory() -> None:
    template = yaml.safe_load((PROJECT_ROOT / "conf" / "config.tmpl.yaml").read_text(encoding="utf-8"))

    assert "sd_optim_schema" in template["defaults"]
    assert "models_dir" in template
    assert "configs_dir" in template
    assert "recipe_optimization" in template
    assert "optimizer" in template
