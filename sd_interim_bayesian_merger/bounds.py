import logging

from typing import Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig, OmegaConf, ListConfig
from sd_interim_bayesian_merger.mapping import OPTIMIZABLE_HYPERPARAMETERS

import sd_mecha

logger = logging.getLogger(__name__)


class Bounds:
    @staticmethod
    def set_block_bounds(lb: float = 0.0, ub: float = 1.0) -> Tuple[float, float]:
        return lb, ub

    @staticmethod
    def create_default_bounds(custom_ranges: Dict[str, Union[Tuple[float, float], float]] = None, cfg=None) -> Dict:
        """Creates the default bounds for block-level hyperparameters."""
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        unet_block_identifiers = [key for key in model_arch.user_keys() if "_unet_block_" in key]
        block_count = len(unet_block_identifiers)

        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        default_hypers = mecha_merge_method.get_hyper_names()
        optimizable_params = OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, default_hypers)
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()

        bounds = {}
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:
                for i in range(block_count):
                    key = f"block_{i}_{param_name}"
                    bounds[key] = (0.0, 1.0)
                bounds[f"base_{param_name}"] = (0.0, 1.0)

        if cfg.optimizer.guided_optimization:
            bounds.update(OmegaConf.to_object(custom_ranges))

        return bounds

    @staticmethod
    def validate_custom_bounds(custom_bounds: Dict[str, Union[List[float], List[int], ListConfig]]) -> Dict:  # Add ListConfig to type hint
        """Validates the custom_bounds dictionary."""
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
    def apply_custom_bounds(bounds: Dict, custom_bounds: Dict) -> Dict:
        """Applies custom bounds to the hyperparameter space."""
        for param_name, bound in custom_bounds.items():
            if isinstance(bound, list):  # Handle binary bounds
                bounds[param_name] = bound  # Replace with binary bounds directly
            elif isinstance(bound, tuple):  # Handle range bounds
                bounds[param_name] = bound  # Replace with custom range
            else:
                raise ValueError(f"Invalid custom bound for '{param_name}': {bound}")
        return bounds

    @staticmethod
    def freeze_bounds(bounds: Dict[str, Union[Tuple[float, float], float]],
                        frozen: Dict[str, Optional[float]] = None) -> Dict:
        """Removes bounds for frozen parameters."""
        frozen = frozen or {}
        return {b: r for b, r in bounds.items() if b not in frozen}

    @staticmethod
    def group_bounds(bounds: Dict[str, Union[Tuple[float, float], float]],
                        groups: List[List[str]] = None) -> Dict[str, Union[Tuple[float, float], float]]:
        """Groups bounds according to the provided groups."""
        if groups is None:
            groups = []
        for group in groups:
            if not group:
                continue
            group_name = "-".join(group)
            group_range = next((bounds[b] for b in group if b in bounds), None)

            if group_range is not None:
                bounds[group_name] = group_range
                for b in group:
                    bounds.pop(b, None)

        return bounds

    @staticmethod
    def get_bounds(
            frozen_params: Dict[str, Optional[float]] = None,
            custom_ranges: Dict[str, Union[Tuple[float, float], float]] = None,
            groups: List[List[str]] = None,
            cfg=None,
    ) -> Dict:
        """Gets the final bounds after applying custom bounds, freezing and grouping."""

        frozen_params = frozen_params or {}
        custom_ranges = custom_ranges or DictConfig({})
        groups = groups or []
        bounds = Bounds.create_default_bounds(custom_ranges, cfg)

        # Validate and apply custom bounds if guided optimization is enabled
        if cfg.optimizer.guided_optimization:
            custom_bounds = Bounds.validate_custom_bounds(cfg.optimization_guide.custom_bounds)
        else:
            custom_bounds = {}

        logger.debug("Input Parameters:")
        logger.debug(f"Frozen Params: {frozen_params}")
        logger.debug(f"Custom Ranges: {custom_ranges}")
        logger.debug(f"Custom Bounds: {custom_bounds}")
        logger.debug(f"Groups: {groups}")

        # Apply custom_bounds
        for param_name, custom_bound in custom_bounds.items():
            for key in bounds:
                if param_name in key:
                    bounds[key] = custom_bound

        not_frozen_bounds = Bounds.freeze_bounds(bounds, frozen_params)
        grouped_bounds = Bounds.group_bounds(not_frozen_bounds, groups)

        logger.debug(f"Bounds After Default Bounds: {bounds}")
        return Bounds.freeze_groups(grouped_bounds, groups, frozen_params)

    @staticmethod
    def freeze_groups(bounds: Dict[str, Union[Tuple[float, float], float]],
                        groups: List[List[str]],
                        frozen: Dict[str, Optional[float]]) -> Dict[str, Union[Tuple[float, float], float]]:
        """Removes grouped bounds if any member is frozen."""
        for group in groups:
            group_name = "-".join(group)
            if group_name in bounds and any(b in frozen for b in group):
                del bounds[group_name]
        return bounds

    @staticmethod
    def get_value(params: Dict[str, float], block_name: str,
                  frozen: Dict[str, Optional[float]], groups: List[List[str]]) -> float:
        """Retrieves the value for a specific block, considering groups and frozen params."""
        if block_name in params:
            return params[block_name]

        if groups:
            for group in groups:
                if block_name in group:
                    group_name = "-".join(group)
                    if group_name in params:
                        return params[group_name]
                    elif block_name in frozen:
                        return frozen.get(block_name, 0.0)  # Handle potential KeyError

        return frozen.get(block_name, 0.0)  # Handle potential KeyError

    @staticmethod
    def assemble_params(
            params: Dict,
            frozen: Dict,
            groups: List[List[str]],
            cfg,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:

        unet_block_identifiers = [
            key for key in sd_mecha.extensions.model_arch.resolve(cfg.model_arch).user_keys()
            if "_unet_block_" in key
        ]

        # Sort unet_block_identifiers using natural_sort_key
        unet_block_identifiers.sort(key=sd_mecha.hypers.natural_sort_key)

        block_count = len(unet_block_identifiers)
        if frozen is None:
            frozen = {}
        if groups is None:
            groups = []

        weights_list = {}
        base_values = {}

        # Get the expected number of parameters from the merging method
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        default_hypers = mecha_merge_method.get_hyper_names()  # Get all hyperparameter names
        optimizable_params = OPTIMIZABLE_HYPERPARAMETERS.get(cfg.merge_mode, default_hypers)  # Get optimizable parameters from the mapping
        volatile_hypers = mecha_merge_method.get_volatile_hyper_names()  # Get volatile hyperparameters

        # Iterate over optimizable parameter names
        for param_name in optimizable_params:
            if param_name not in volatile_hypers:  # Exclude volatile hyperparameters
                # Initialize weights_list for the current parameter
                weights_list[param_name] = {}
                for i in range(block_count):
                    key = f"block_{i}_{param_name}"
                    weights_list[param_name][unet_block_identifiers[i]] = Bounds.get_value(params, key, frozen, groups)

                base_name = f"base_{param_name}"
                base_values[base_name] = Bounds.get_value(params, base_name, frozen, groups)

        assert len(weights_list) == len(optimizable_params) - len(set(optimizable_params) & volatile_hypers)
        print(f"Assembled Weights List: {weights_list}")
        print(f"Assembled Base Values: {base_values}")
        return weights_list, base_values
