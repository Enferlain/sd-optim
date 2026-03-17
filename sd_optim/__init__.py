from typing import Any

__all__ = ["OptunaOptimizer", "BayesOptimizer"]


def __getattr__(name: str) -> Any:
    if name == "OptunaOptimizer":
        from sd_optim.optimizers.optuna.optimizer import OptunaOptimizer

        return OptunaOptimizer
    if name == "BayesOptimizer":
        from sd_optim.bayes_optimizer import BayesOptimizer

        return BayesOptimizer
    raise AttributeError(f"module 'sd_optim' has no attribute '{name}'")
