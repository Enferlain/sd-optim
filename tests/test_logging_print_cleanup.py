from __future__ import annotations

import ast
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CORE_MODULES_WITHOUT_PRINTS = [
    "sd_optim/core/optimizer_base.py",
    "sd_optim/core/optimizer_runtime.py",
    "sd_optim/scorer.py",
    "sd_optim/optimizers/optuna/optimizer.py",
    "sd_optim/optimizers/bayes/optimizer.py",
    "sd_optim/optimizers/optuna/dashboard.py",
    "sd_optim/optimizers/optuna/objective.py",
    "sd_optim/optimizers/optuna/reporting.py",
    "sd_optim/optimizers/optuna/sampler_factory.py",
    "sd_optim/optimizers/optuna/study_manager.py",
    "sd_optim/optimizers/optuna/trial_logger.py",
    "sd_optim/utils/__init__.py",
    "sd_optim/utils/methods.py",
    "sd_optim/extensions/bundled/scorers/models/LumiAnatomy.py",
    "sd_optim/extensions/bundled/scorers/models/LumiAnatomyv2.py",
    "sd_optim/extensions/bundled/scorers/models/lumi_model.py",
    "sd_optim/extensions/bundled/scorers/models/predictlumi_model.py",
    "sd_optim/extensions/bundled/scorers/models/AestheticV25.py",
    "sd_optim/extensions/bundled/scorers/models/BLIP/blip.py",
    "sd_optim/extensions/bundled/scorers/models/BLIP/vit.py",
]


def _has_print_call(source: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        for node in ast.walk(tree)
    )


def test_hydra_job_logging_profile_includes_source_location() -> None:
    profile_path = PROJECT_ROOT / "conf" / "hydra" / "job_logging" / "sd_optim.yaml"
    assert profile_path.exists(), "Expected custom Hydra job logging profile to exist"

    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    fmt = data["formatters"]["detailed"]["format"]

    assert "%(filename)s" in fmt
    assert "%(lineno)d" in fmt
    assert "%(name)s" in fmt


def test_config_uses_custom_hydra_job_logging_profile() -> None:
    config_path = PROJECT_ROOT / "conf" / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8")

    assert "override hydra/job_logging: sd_optim" in config_text


def test_core_runtime_modules_do_not_call_print() -> None:
    offenders: list[str] = []

    for rel_path in CORE_MODULES_WITHOUT_PRINTS:
        full_path = PROJECT_ROOT / rel_path
        source = full_path.read_text(encoding="utf-8")
        if _has_print_call(source):
            offenders.append(rel_path)

    assert not offenders, f"print() calls found in core runtime modules: {offenders}"
