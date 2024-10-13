import logging
import sd_mecha

from typing import Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig, ListConfig
from sd_interim_bayesian_merger.mapping import OPTIMIZABLE_HYPERPARAMETERS

logger = logging.getLogger(__name__)


class Bounds:
    @staticmethod
    def set_block_bounds(lb: float = 0.0, ub: float = 1.0) -> Tuple[float, float]:
        return lb, ub

    @staticmethod
    def create_default_bounds(cfg: DictConfig) -> Dict[str, Tuple[float, float]]:
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        optimizable_params = OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, mecha_merge_method.get_hyper_names())
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()

        bounds = {}
        for component_config in cfg.optimization_guide.components:
            component_name = component_config.name
            optimization_strategy = component_config.optimize

            if optimization_strategy == "all":
                bounds.update(
                    Bounds._create_bounds_for_all(model_arch, component_name, optimizable_params, volatile_hypers))
            elif optimization_strategy == "selected":
                bounds.update(Bounds._create_bounds_for_selected(component_config, optimizable_params, volatile_hypers))
            elif optimization_strategy == "grouped":
                bounds.update(Bounds._create_bounds_for_grouped(component_config, optimizable_params, volatile_hypers))
            elif optimization_strategy == "group-all":
                bounds.update(
                    Bounds._create_bounds_for_group_all(cfg, component_name, optimizable_params, volatile_hypers))
            elif optimization_strategy == "none":
                logger.info(f"Optimization disabled for component '{component_name}'.")
            else:
                raise ValueError(f"Invalid optimization strategy: {optimization_strategy}")

        return bounds

    @staticmethod
    def _create_bounds_for_all(model_arch, component_name, optimizable_params, volatile_hypers):
        component_bounds = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                # Create bound for component default
                component_bounds[f"{model_arch.identifier}_{component_name}_default_{param_name}"] = (0.0, 1.0)

                # Create bounds for block-specific overrides
                for block_id in model_arch.user_keys():
                    if f"_{component_name}_block_" in block_id:
                        key = f"{block_id}_{param_name}"
                        component_bounds[key] = (0.0, 1.0)
        return component_bounds

    @staticmethod
    def _create_bounds_for_selected(component_config, optimizable_params, volatile_hypers):
        component_bounds = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                # Create bound for component default (assuming it's desired for "selected" strategy)
                component_bounds[f"{component_config.name}_default_{param_name}"] = (0.0, 1.0)

                # Create bounds for selected blocks
                for block_id in component_config.selected_blocks:
                    key = f"{block_id}_{param_name}"
                    component_bounds[key] = (0.0, 1.0)
        return component_bounds

    @staticmethod
    def _create_bounds_for_grouped(component_config, optimizable_params, volatile_hypers):
        component_bounds = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                # Create bound for component default (assuming it's desired for "grouped" strategy)
                component_bounds[f"{component_config.name}_default_{param_name}"] = (0.0, 1.0)

                # Create bounds for each group
                for i, group in enumerate(component_config.groups):
                    group_name = "-".join([f"{block}_{param_name}" for block in group])
                    component_bounds[group_name] = (0.0, 1.0)
        return component_bounds

    @staticmethod
    def _create_bounds_for_group_all(cfg, component_name, optimizable_params, volatile_hypers):
        component_bounds = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                key = f"{cfg.model_arch}_{component_name}_default_{param_name}"
                component_bounds[key] = (0.0, 1.0)
        return component_bounds

    @staticmethod
    def validate_custom_bounds(custom_bounds: Dict[str, Union[List[float], List[int], ListConfig, int, float]]) -> Dict:
        """Validates the custom_bounds dictionary."""
        if custom_bounds is None:  # Add a check for None
            return {}

        for param_name, bound in custom_bounds.items():
            if isinstance(bound, (list, ListConfig)):
                if len(bound) == 2:  # Range or binary bound
                    if all(isinstance(v, (int, float)) for v in bound):
                        if bound[0] > bound[1]:
                            raise ValueError(
                                f"Invalid range bound for '{param_name}': {bound}. Lower bound cannot be greater than upper bound.")
                    elif all(v in [0, 1] for v in bound):
                        pass  # Valid binary bound
                    else:
                        raise ValueError(
                            f"Invalid bound for '{param_name}': {bound}. Range bounds must contain floats or binary bounds must be integers 0 and 1.")
                else:
                    raise ValueError(
                        f"Invalid bound for '{param_name}': {bound}. Must contain two elements for range or binary bounds.")
            elif isinstance(bound, (int, float)):
                pass  # Valid single value
            else:
                raise ValueError(
                    f"Invalid bound for '{param_name}': {bound}. Bounds must be lists, tuples, integers, or floats.")
        return custom_bounds

    @staticmethod
    def freeze_bounds(bounds: Dict[str, Union[Tuple[float, float], int, float]],
                      frozen: Dict[str, Optional[float]]) -> Dict:  # Remove default value
        """Removes bounds for frozen parameters."""
        return {b: r for b, r in bounds.items() if b not in frozen}

    @staticmethod
    def get_bounds(
            frozen_params: Dict[str, Optional[float]],
            custom_ranges: Dict[str, Union[Tuple[float, float], float]],
            custom_bounds: Dict[str, Union[List[float], List[int], ListConfig, int, float]],
            cfg=None,
    ) -> Dict[str, Union[Tuple[float, float], float]]:
        """Gets the final bounds after applying custom bounds, freezing and grouping."""

        bounds = Bounds.create_default_bounds(cfg)

        logger.debug("Input Parameters:")
        logger.debug(f"Frozen Params: {frozen_params}")
        logger.debug(f"Custom Ranges: {custom_ranges}")
        logger.debug(f"Custom Bounds: {custom_bounds}")

        # Apply custom_bounds
        for param_name, bound in custom_bounds.items():
            if isinstance(bound, (tuple, list)):  # Apply range or binary bounds
                bounds[param_name] = bound
            elif isinstance(bound, (int, float)):  # Apply single value bounds
                bounds[param_name] = (bound, bound)  # Create a tuple for single values
            else:
                raise ValueError(f"Invalid custom bound for '{param_name}': {bound}")

        bounds = Bounds.freeze_bounds(bounds, frozen_params)  # Freeze bounds directly

        logger.debug(f"Final Bounds: {bounds}")
        return bounds  # Return the final bounds

        # Validate and apply custom bounds
#        if cfg.optimization_guide.get("custom_bounds") is not None:  # Check for None
#            custom_bounds = Bounds.validate_custom_bounds(cfg.optimization_guide.custom_bounds)

    @staticmethod
    def assemble_params(
            params: Dict,
            cfg: DictConfig
    ) -> Dict[str, Dict[str, float]]:
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        optimizable_params = OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, mecha_merge_method.get_hyper_names())
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()
        frozen_params = cfg.optimization_guide.get("frozen_params", {})

        assembled_params = {}
        for component_config in cfg.optimization_guide.components:
            component_name = component_config.name
            optimization_strategy = component_config.optimize

            if optimization_strategy == "all":
                assembled_params.update(Bounds._assemble_params_for_all(
                    model_arch, component_name, optimizable_params, volatile_hypers, params, frozen_params
                ))
            elif optimization_strategy == "selected":
                assembled_params.update(Bounds._assemble_params_for_selected(
                    component_config, optimizable_params, volatile_hypers, params, frozen_params
                ))
            elif optimization_strategy == "grouped":
                assembled_params.update(Bounds._assemble_params_for_grouped(
                    component_config, optimizable_params, volatile_hypers, params, frozen_params
                ))
            elif optimization_strategy == "group-all":
                assembled_params.update(Bounds._assemble_params_for_group_all(
                    cfg, component_name, optimizable_params, volatile_hypers, params, frozen_params
                ))
            elif optimization_strategy == "none":
                logger.info(f"Optimization disabled for component '{component_name}' in assemble_params.")
            else:
                raise ValueError(f"Invalid optimization strategy: {optimization_strategy}")

        return assembled_params

    @staticmethod
    def _assemble_params_for_all(model_arch, component_name, optimizable_params, volatile_hypers, params, frozen):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                # Create an entry for the parameter with component default
                component_params[param_name] = {
                    f"{model_arch.identifier}_{component_name}_default": params.get(
                        f"{model_arch.identifier}_{component_name}_default_{param_name}",
                        frozen.get(f"{model_arch.identifier}_{component_name}_default_{param_name}", 0.0)
                    )
                }

                # Add block-specific overrides
                for block_id in model_arch.user_keys():
                    if f"_{component_name}_block_" in block_id:
                        key = f"{block_id}_{param_name}"
                        component_params[param_name][block_id] = params.get(key, frozen.get(key, 0.0))
        return component_params

    @staticmethod
    def _assemble_params_for_selected(component_config, optimizable_params, volatile_hypers, params, frozen):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                # Create an entry for the parameter with component default
                component_params[param_name] = {
                    f"{component_config.name}_default": params.get(
                        f"{component_config.name}_default_{param_name}",
                        frozen.get(f"{component_config.name}_default_{param_name}", 0.0)
                    )
                }

                # Add block-specific overrides
                for block_id in component_config.selected_blocks:
                    key = f"{block_id}_{param_name}"
                    component_params[param_name][block_id] = params.get(key, frozen.get(key, 0.0))
        return component_params

    @staticmethod
    def _assemble_params_for_grouped(component_config, optimizable_params, volatile_hypers, params, frozen):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                # Create an entry for the parameter
                component_params[param_name] = {
                    f"{component_config.name}_default": params.get(
                        f"{component_config.name}_default_{param_name}",
                        frozen.get(f"{component_config.name}_default_{param_name}", 0.0)
                    )
                }
                for i, group in enumerate(component_config.groups):
                    group_name = "-".join([f"{block}_{param_name}" for block in group])
                    group_value = params.get(group_name, frozen.get(group_name, 0.0))
                    for block_id in group:
                        component_params[param_name][block_id] = group_value

        return component_params

    @staticmethod
    def _assemble_params_for_group_all(cfg, component_name, optimizable_params, volatile_hypers, params, frozen):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                key = f"{cfg.model_arch}_{component_name}_default_{param_name}"
                component_params[param_name] = {
                    f"{cfg.model_arch}_{component_name}_default": params.get(key, frozen.get(key, 0.0))
                }
        return component_params