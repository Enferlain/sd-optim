import logging

from typing import Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig, ListConfig
from sd_interim_bayesian_merger.mapping import OPTIMIZABLE_HYPERPARAMETERS

import sd_mecha

logger = logging.getLogger(__name__)


class Bounds:
    @staticmethod
    def set_block_bounds(lb: float = 0.0, ub: float = 1.0) -> Tuple[float, float]:
        return lb, ub

    @staticmethod
    def create_default_bounds(cfg: DictConfig) -> Dict:
        """Creates the default bounds for hyperparameters based on component configuration."""
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        default_hypers = mecha_merge_method.get_hyper_names()
        optimizable_params = OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, default_hypers)
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()

        bounds = {}
        for component_config in cfg.optimization_guide.components:
            component_name = component_config.name
            if component_config.optimize == "all":
                for block_id in model_arch.user_keys():
                    if f"_{component_name}_block_" in block_id:
                        for param_name in optimizable_params:
                            if param_name not in volatile_hypers:
                                bounds[f"{block_id}_{param_name}"] = (0.0, 1.0)
            elif component_config.optimize == "selected":
                for block_id in component_config.selected_blocks:
                    for param_name in optimizable_params:
                        if param_name not in volatile_hypers:
                            bounds[f"{block_id}_{param_name}"] = (0.0, 1.0)
            elif component_config.optimize == "grouped":
                for i, group in enumerate(component_config.groups):
                    for block_id in group:
                        for param_name in optimizable_params:
                            if param_name not in volatile_hypers:
                                bounds[f"{block_id}_{param_name}_group_{i}"] = (0.0, 1.0)
            elif component_config.optimize == "group-all":
                for param_name in optimizable_params:
                    if param_name not in volatile_hypers:
                        bounds[f"{cfg.model_arch}_{component_name}_default_{param_name}"] = (0.0, 1.0)

        # Sort bounds based on component order in guide.yaml
        component_order = [c.name for c in cfg.optimization_guide.components]
        bounds = dict(sorted(bounds.items(), key=lambda item: (component_order.index(item[0].split("_")[1]) if len(item[0].split("_")) > 1 and item[0].split("_")[1] in component_order else len(component_order), sd_mecha.hypers.natural_sort_key(item[0]))))  # Sort by component order first, then natural order within each component

        return bounds

    @staticmethod
    def validate_custom_bounds(custom_bounds: Dict[str, Union[List[float], List[int], ListConfig]]) -> Dict:  # Add ListConfig to type hint
        """Validates the custom_bounds dictionary."""
        if custom_bounds is None:
            return {}

        for param_name, bound in custom_bounds.items():
            if isinstance(bound, (list, ListConfig)):
                if len(bound) == 2:
                    if all(isinstance(v, (int, float)) for v in bound):  # Handle both float and int range bounds
                        if bound[0] > bound[1]:
                            raise ValueError(
                                f"Invalid range bound for '{param_name}': {bound}. Lower bound cannot be greater than upper bound.")
                    elif all(v in [0, 1] for v in bound):  # Handle binary bounds
                        pass  # Valid binary bound
                    else:
                        raise ValueError(
                            f"Invalid bound for '{param_name}': {bound}. Range bounds must contain floats or binary bounds must be integers 0 and 1.")
                else:
                    raise ValueError(
                        f"Invalid bound for '{param_name}': {bound}. Must contain either two floats (range) or one integer (binary).")

            else:
                raise ValueError(f"Invalid bound for '{param_name}': {bound}. Bounds must be lists.")

        return custom_bounds

    @staticmethod
    def freeze_bounds(bounds: Dict[str, Union[Tuple[float, float], float]],
                        frozen: Dict[str, Optional[float]] = None) -> Dict:
        """Removes bounds for frozen parameters."""
        frozen = frozen or {}
        return {b: r for b, r in bounds.items() if b not in frozen}

    @staticmethod
    def get_bounds(
            frozen_params: Dict[str, Optional[float]] = None,
            custom_ranges: Dict[str, Union[Tuple[float, float], float]] = None,
            custom_bounds: Dict[str, Union[List[float], List[int], ListConfig, int, float]] = None,
            cfg=None,
    ) -> Dict:
        """Gets the final bounds after applying custom bounds and freezing."""

        frozen_params = frozen_params or {}
        custom_ranges = custom_ranges or DictConfig({})
        custom_bounds = custom_bounds or {}

        bounds = Bounds.create_default_bounds(cfg)

        # Apply custom_bounds
        for param_name, custom_bound in custom_bounds.items():
            if param_name in bounds:
                bounds[param_name] = custom_bound

        logger.debug("Input Parameters:")
        logger.debug(f"Frozen Params: {frozen_params}")
        logger.debug(f"Custom Ranges: {custom_ranges}")
        logger.debug(f"Custom Bounds: {custom_bounds}")

        not_frozen_bounds = Bounds.freeze_bounds(bounds, frozen_params)

        logger.debug(f"Bounds After Default Bounds: {bounds}")
        return not_frozen_bounds  # Return not_frozen_bounds directly

    @staticmethod
    def assemble_params(
            params: Dict,
            frozen: Dict,
            cfg: DictConfig,
    ) -> Dict[str, float]:

        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        default_hypers = mecha_merge_method.get_hyper_names()
        optimizable_params = OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, default_hypers)
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()

        assembled_params = {}
        for component_config in cfg.optimization_guide.components:  # Iterate directly over the list
            component_name = component_config.name
            if component_config.optimize == "all":
                for block_id in model_arch.user_keys():
                    if f"_{component_name}_block_" in block_id:
                        for param_name in optimizable_params:
                            if param_name not in volatile_hypers:
                                key = f"{block_id}_{param_name}"
                                assembled_params[key] = params.get(key, (frozen.get(key, 0.0) if frozen else 0.0))
            elif component_config.optimize == "selected":
                for block_id in component_config.selected_blocks:
                    for param_name in optimizable_params:
                        if param_name not in volatile_hypers:
                            key = f"{block_id}_{param_name}"
                            assembled_params[key] = params.get(key, frozen.get(key, 0.0))
            elif component_config.optimize == "grouped":  # Handle "grouped" separately
                for i, group in enumerate(component_config.groups):
                    for block_id in group:
                        for param_name in optimizable_params:
                            if param_name not in volatile_hypers:
                                assembled_params[f"{block_id}_{param_name}_group_{i}"] = (0.0, 1.0)
            elif component_config.optimize == "group-all":  # Handle "group-all" separately
                for param_name in optimizable_params:
                    if param_name not in volatile_hypers:
                        assembled_params[f"{cfg.model_arch}_{component_name}_default_{param_name}"] = (0.0, 1.0)

        # Sort assembled_params based on component order in guide.yaml
        component_order = [c.name for c in cfg.optimization_guide.components]
        assembled_params = dict(sorted(assembled_params.items(), key=lambda item: (component_order.index(item[0].split("_")[1]) if len(item[0].split("_")) > 1 and item[0].split("_")[1] in component_order else len(component_order), sd_mecha.hypers.natural_sort_key(item[0]))))  # Sort by component order first, then natural order within each component

        return assembled_params
