import logging
import subprocess
from pathlib import Path
from typing import Any

from sd_optim.core.optimizer_base import Optimizer
from sd_optim.optimizers.optuna.dashboard import start_dashboard_for_optimizer
from sd_optim.optimizers.optuna.reporting import (
    collect_optimization_history,
    create_visualization_report as generate_visualization_report,
    postprocess_study,
)
from sd_optim.optimizers.optuna.sampler_factory import configure_sampler, validate_optimizer_config
from sd_optim.optimizers.optuna.study_manager import initialize_optuna_state, optimize_study

logger = logging.getLogger(__name__)


class OptunaOptimizer(Optimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        initialize_optuna_state(self)

    def validate_optimizer_config(self) -> bool:
        return validate_optimizer_config(self.cfg)

    def _configure_sampler(self) -> Any:
        return configure_sampler(self.cfg)

    async def optimize(self) -> None:
        await optimize_study(self)

    async def postprocess(self) -> None:
        await postprocess_study(self)

    def get_best_parameters(self) -> dict[str, Any]:
        return self.study.best_params if self.study and hasattr(self.study, "best_params") else {}

    def get_optimization_history(self) -> list[dict[str, Any]]:
        return collect_optimization_history(self.study)

    def create_visualization_report(self, output_dir: str | Path | None = None) -> None:
        generate_visualization_report(self, output_dir=output_dir)

    def start_dashboard_background(self, port: int = 8080) -> subprocess.Popen | None:
        return start_dashboard_for_optimizer(self, port=port)
