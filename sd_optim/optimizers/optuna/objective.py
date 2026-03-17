from __future__ import annotations

import asyncio
import logging
import operator
from typing import TYPE_CHECKING, Any

import optuna
from optuna import Trial

from sd_optim.core.optimizer_cache import fail_on_error_enabled

if TYPE_CHECKING:
    from sd_optim.optuna_optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)


_CONDITION_OPERATORS = {
    "!=": operator.ne,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


def evaluate_condition(value: Any, condition: str) -> bool:
    try:
        parts = condition.split()
        if len(parts) != 2:
            return True

        operator_func = _CONDITION_OPERATORS.get(parts[0])
        threshold = float(parts[1])
        if operator_func is None:
            return True
        return bool(operator_func(value, threshold))
    except Exception as error:
        logger.warning("Error evaluating condition '%s' for value %s: %s", condition, value, error)
        return True


def suggest_parameters(optimizer: OptunaOptimizer, trial: Trial) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not optimizer.optimizer_pbounds:
        logger.error("Optimizer bounds not initialized in objective. Cannot suggest parameters.")
        raise optuna.exceptions.TrialPruned("Bounds not available")

    all_param_names = list(optimizer.optimizer_pbounds.keys())
    suggested_params: set[str] = set()

    def suggest_param(name: str) -> Any:
        if name in suggested_params:
            return params[name]

        if name in optimizer.child_to_parent:
            dep_info = optimizer.child_to_parent[name]
            parent_name = dep_info["parent"]
            parent_val = suggest_param(parent_name)
            if not evaluate_condition(parent_val, dep_info["condition"]):
                params[name] = dep_info["default"]
                logger.debug(
                    "Skipping suggest for '%s': parent '%s'=%s did not meet '%s'. Using default %s.",
                    name,
                    parent_name,
                    parent_val,
                    dep_info["condition"],
                    dep_info["default"],
                )
                suggested_params.add(name)
                return params[name]

        bound_config = optimizer.optimizer_pbounds[name]
        try:
            if isinstance(bound_config, dict) and "range" in bound_config:
                low, high = bound_config["range"]
                log = bound_config.get("log", False)
                step = bound_config.get("step")
                is_integer_range = isinstance(low, int) and isinstance(high, int) and (step is None or isinstance(step, int))
                if is_integer_range:
                    params[name] = trial.suggest_int(name, low, high, step=step or 1, log=log)
                    logger.debug("Suggesting for '%s': Int range [%s-%s], Step=%s, Log=%s", name, low, high, step or 1, log)
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high), step=step, log=log)
                    logger.debug("Suggesting for '%s': Float range [%s-%s], Step=%s, Log=%s", name, low, high, step, log)
            elif isinstance(bound_config, list):
                params[name] = trial.suggest_categorical(name, bound_config)
                logger.debug("Suggesting for '%s': Categorical %s", name, bound_config)
            elif isinstance(bound_config, tuple):
                if len(bound_config) != 2:
                    raise ValueError("Range tuple must have 2 values.")
                low, high = bound_config
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                    logger.debug("Suggesting for '%s': Simple Int range [%s-%s]", name, low, high)
                else:
                    params[name] = trial.suggest_float(name, float(low), float(high))
                    logger.debug("Suggesting for '%s': Simple Float range [%s-%s]", name, low, high)
            elif isinstance(bound_config, (int, float)):
                params[name] = bound_config
                logger.debug("Using fixed value for '%s': %s", name, bound_config)
            else:
                raise ValueError(f"Unsupported bounds format for '{name}': {bound_config}")
        except Exception as error:
            logger.error(
                "Error during parameter suggestion for '%s' with config %s: %s",
                name,
                bound_config,
                error,
                exc_info=True,
            )
            raise

        suggested_params.add(name)
        return params[name]

    for param_name in all_param_names:
        suggest_param(param_name)

    return params


def run_objective(optimizer: OptunaOptimizer, trial: Trial) -> float:
    params = suggest_parameters(optimizer, trial)
    try:
        result = asyncio.run(optimizer.sd_target_function(params))
        trial_scorer_summary = getattr(optimizer, "last_trial_scorer_summary", {})
        if isinstance(trial_scorer_summary, dict) and isinstance(trial_scorer_summary.get("aggregate"), dict):
            trial.set_user_attr("scorer_results", trial_scorer_summary.get("aggregate", {}))
            trial.set_user_attr("scorer_results_payloads", trial_scorer_summary.get("payloads", []))
            logger.debug("Stored aggregated scorer summary for trial %s", trial.number)
        elif hasattr(optimizer.scorer, "last_scorer_results") and optimizer.scorer.last_scorer_results:
            trial.set_user_attr("scorer_results", optimizer.scorer.last_scorer_results)
            logger.debug("Stored scorer_results for trial %s", trial.number)

        optimizer.trial_scores.append(result)
        if optimizer.early_stopping:
            if optimizer.trial_scores and result > (max(optimizer.trial_scores[:-1] or [0]) + optimizer.min_improvement):
                optimizer.no_improvement_count = 0
            else:
                optimizer.no_improvement_count += 1

            if optimizer.no_improvement_count >= optimizer.patience:
                logger.info("Early stopping triggered after %s trials", len(optimizer.trial_scores))
                raise optuna.exceptions.TrialPruned()

        return result
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as error:
        if fail_on_error_enabled(optimizer.cfg):
            logger.error("Error in objective function: %s", error, exc_info=True)
            raise
        logger.error("Error in objective function: %s", error, exc_info=True)
        return float("-inf")
