import logging
import sd_mecha

from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, ListConfig

from sd_interim_bayesian_merger import utils

logger = logging.getLogger(__name__)


class Bounds:
    @staticmethod
    def set_block_bounds(lb: float = 0.0, ub: float = 1.0) -> Tuple[float, float]:
        return lb, ub

    @staticmethod
    def create_default_bounds(cfg: DictConfig) -> Dict[str, Tuple[float, float]]:
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()

        if cfg.recipe_optimization.enabled:
            # Extract hyperparameters from target nodes
            extracted_hypers = utils.get_target_nodes(
                cfg.recipe_optimization.recipe_path,
                cfg.recipe_optimization.target_nodes
            )

            # Get merge method from first target node
            first_target = cfg.recipe_optimization.target_nodes
            if isinstance(first_target, list):
                first_target = first_target[0]
            merge_mode = extracted_hypers[first_target]['merge_method']

            # Filter through optimizable parameters
            optimizable_params = utils.OPTIMIZABLE_HYPERPARAMETERS.get(
                merge_mode,
                extracted_hypers[first_target]['hypers'].keys()
            )
        else:
            optimizable_params = utils.OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, mecha_merge_method.get_hyper_names())

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
                bounds.update(Bounds._create_bounds_for_grouped(component_config, optimizable_params, volatile_hypers, model_arch))
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
        for block_id in model_arch.user_keys():
            if f"_{component_name}_block_" in block_id:
                for param_name in optimizable_params:
                    if param_name not in volatile_hypers:
                        key = f"{block_id}_{param_name}"
                        component_bounds[key] = (0.0, 1.0)
        return component_bounds

    @staticmethod
    def _create_bounds_for_selected(component_config, optimizable_params, volatile_hypers):
        component_bounds = {}
        for block_id in component_config.selected_blocks:
            for param_name in optimizable_params:
                if param_name not in volatile_hypers:
                    key = f"{block_id}_{param_name}"
                    component_bounds[key] = (0.0, 1.0)
        return component_bounds

    @staticmethod
    def _create_bounds_for_grouped(component_config, optimizable_params, volatile_hypers, model_arch):
        component_bounds = {}
        is_integer_group = isinstance(component_config.groups, int)

        if is_integer_group:
            # Get all blocks for automatic grouping
            block_ids = [block_id for block_id in model_arch.user_keys() if
                         f"_{component_config.name}_block_" in block_id]
            num_groups = component_config.groups
            group_size = max(1, len(block_ids) // num_groups)
            grouped_blocks = [block_ids[i:i + group_size] for i in range(0, len(block_ids), group_size)]
        else:
            # Use only the explicitly specified blocks
            grouped_blocks = component_config.groups

        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                for i, group in enumerate(grouped_blocks):
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
    def get_bounds(
            custom_ranges: Dict[str, Union[Tuple[float, float], float]],
            custom_bounds: Dict[str, Union[List[float], List[int], ListConfig, int, float]],
            cfg=None,
    ) -> Dict[str, Union[Tuple[float, float], float]]:
        """Gets the final bounds after applying custom bounds, freezing and grouping."""

        bounds = Bounds.create_default_bounds(cfg)
        logger.debug("Input Parameters:")
        logger.debug(f"Custom Ranges: {custom_ranges}")
        logger.debug(f"Custom Bounds: {custom_bounds}")

        # Validate custom_bounds
        validated_custom_bounds = Bounds.validate_custom_bounds(custom_bounds)

        # Apply custom_bounds
        for custom_param, custom_bound in validated_custom_bounds.items():
            matching_keys = [key for key in bounds.keys() if custom_param in key]

            if matching_keys:
                for key in matching_keys:
                    bounds[key] = tuple(custom_bound)
                logger.info(
                    f"Applied custom bound {custom_bound} to {len(matching_keys)} keys containing '{custom_param}'")
            else:
                logger.warning(f"No matching keys found for custom bound '{custom_param}'. Skipping.")

        # Log bounds for each hyperparameter on a single line
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()
        component_order = [c.name for c in cfg.optimization_guide.components]

        if cfg.recipe_optimization.enabled:
            # Extract hyperparameters from target nodes
            extracted_hypers = utils.get_target_nodes(
                cfg.recipe_optimization.recipe_path,
                cfg.recipe_optimization.target_nodes
            )

            # Get merge method from first target node
            first_target = cfg.recipe_optimization.target_nodes
            if isinstance(first_target, list):
                first_target = first_target[0]
            merge_mode = extracted_hypers[first_target]['merge_method']

            # Filter through optimizable parameters
            optimizable_params = utils.OPTIMIZABLE_HYPERPARAMETERS.get(
                merge_mode,
                extracted_hypers[first_target]['hypers'].keys()
            )
        else:
            optimizable_params = utils.OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, mecha_merge_method.get_hyper_names())

        for param_name in optimizable_params:  # Iterate over optimizable params
            if param_name not in volatile_hypers:
                param_bounds = [
                    f"{key}: {value}"
                    for key, value in  sorted(bounds.items(), key=lambda item: utils.custom_sort_key(item[0], component_order))
                    if f"_{param_name}" in key
                ]
                logger.info(f"Bounds for {param_name}: {', '.join(param_bounds)}")

        return bounds  # Return the final bounds

    @staticmethod
    def assemble_params(
            params: Dict,
            cfg: DictConfig
    ) -> Dict[str, float]:
        """Assembles hyperparameters for each component based on the optimization strategy."""

        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()

        if cfg.recipe_optimization.enabled:
            # Extract hyperparameters from target nodes
            extracted_hypers = utils.get_target_nodes(
                cfg.recipe_optimization.recipe_path,
                cfg.recipe_optimization.target_nodes
            )

            # Get merge method from first target node
            first_target = cfg.recipe_optimization.target_nodes
            if isinstance(first_target, list):
                first_target = first_target[0]
            merge_mode = extracted_hypers[first_target]['merge_method']

            # Filter through optimizable parameters
            optimizable_params = utils.OPTIMIZABLE_HYPERPARAMETERS.get(
                merge_mode,
                extracted_hypers[first_target]['hypers'].keys()
            )
        else:
            optimizable_params = utils.OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, mecha_merge_method.get_hyper_names())

        assembled_params = {}
        for component_config in cfg.optimization_guide.components:
            component_name = component_config.name
            optimization_strategy = component_config.optimize
            groups = component_config.get("groups", [])  # Get groups, default to empty list
            selected_blocks = component_config.get("selected_blocks", [])  # Get selected_blocks, default to empty list

            if optimization_strategy == "all":
                for param_name, param_values in Bounds._assemble_params_for_all(
                        model_arch, component_name, optimizable_params, volatile_hypers, params
                ).items():
                    assembled_params[param_name] = assembled_params.get(param_name, {})
                    assembled_params[param_name].update(param_values)
            elif optimization_strategy == "selected":
                if not selected_blocks:
                    raise ValueError(f"No 'selected_blocks' specified for component '{component_name}'")
                for param_name, param_values in Bounds._assemble_params_for_selected(
                        component_config, optimizable_params, volatile_hypers, params
                        # Pass component_config instead of selected_blocks
                ).items():
                    assembled_params[param_name] = assembled_params.get(param_name, {})
                    assembled_params[param_name].update(param_values)
            elif optimization_strategy == "grouped":
                for param_name, param_values in Bounds._assemble_params_for_grouped(
                        component_config, optimizable_params, volatile_hypers, params, cfg
                ).items():
                    assembled_params[param_name] = assembled_params.get(param_name, {})
                    assembled_params[param_name].update(param_values)
            elif optimization_strategy == "group-all":
                for param_name, param_values in Bounds._assemble_params_for_group_all(
                        cfg, component_name, optimizable_params, volatile_hypers, params
                ).items():
                    assembled_params[param_name] = assembled_params.get(param_name, {})
                    assembled_params[param_name].update(param_values)
            elif optimization_strategy == "none":
                logger.info(f"Optimization disabled for component '{component_name}' in assemble_params.")
            else:
                raise ValueError(f"Invalid optimization strategy: {optimization_strategy}")

        # Sort the assembled parameters within each component
        component_order = [c.name for c in cfg.optimization_guide.components]
        for component_name, component_params in assembled_params.items():
            sorted_component_params = dict(sorted(component_params.items(), key=lambda item: utils.custom_sort_key(item[0], component_order)))
            assembled_params[component_name] = sorted_component_params

        return assembled_params

    @staticmethod
    def _assemble_params_for_all(model_arch, component_name, optimizable_params, volatile_hypers, params):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                component_params[param_name] = {}
                for block_id in model_arch.user_keys():
                    if f"_{component_name}_block_" in block_id:
                        key = f"{block_id}_{param_name}"
                        component_params[param_name][block_id] = params.get(key, 0.0)
        return component_params

    @staticmethod
    def _assemble_params_for_selected(component_config, optimizable_params, volatile_hypers, params):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                component_params[param_name] = {}
                # Iterate through selected blocks correctly
                for block_id in component_config["selected_blocks"]:
                    key = f"{block_id}_{param_name}"
                    component_params[param_name][block_id] = params.get(key, 0.0)
        return component_params

    @staticmethod
    def _assemble_params_for_grouped(component_config, optimizable_params, volatile_hypers, params, model_arch):
        component_params = {}
        is_integer_group = isinstance(component_config.groups, int)

        if is_integer_group:
            # Get all blocks for automatic grouping
            block_ids = [block_id for block_id in model_arch.user_keys() if
                         f"_{component_config.name}_block_" in block_id]
            num_groups = component_config.groups
            group_size = max(1, len(block_ids) // num_groups)
            grouped_blocks = [block_ids[i:i + group_size] for i in range(0, len(block_ids), group_size)]
        else:
            # Use only the explicitly specified blocks
            grouped_blocks = component_config.groups

        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                component_params[param_name] = {}
                for group in grouped_blocks:
                    group_name = "-".join([f"{block}_{param_name}" for block in group])
                    group_value = params.get(group_name, 0.0)
                    for block_id in group:
                        component_params[param_name][block_id] = group_value

        return component_params

    @staticmethod
    def _assemble_params_for_group_all(cfg, component_name, optimizable_params, volatile_hypers, params):
        component_params = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                key = f"{cfg.model_arch}_{component_name}_default_{param_name}"
                component_params[param_name] = {
                    f"{cfg.model_arch}_{component_name}_default": params.get(key, 0.0)
                }
        return component_params
