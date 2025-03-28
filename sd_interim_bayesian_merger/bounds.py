# bounds.py - Version 1.0
import logging
import sd_mecha
import fnmatch  # Import for wildcard matching
import torch

from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


class ParameterHandler:
    """Handles parameter bounds creation and assembly, unifying the logic."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_config = sd_mecha.infer_model_configs(
            [sd_mecha.model(model) for model in cfg.model_paths] # type: ignore
        )[0] # Get the model config
        self.merge_method = sd_mecha.extensions.merge_methods.resolve(cfg.merge_mode)
        self.param_names = self._get_optimizable_parameters()


    def _get_optimizable_parameters(self) -> List[str]:
        """Gets the names of optimizable parameters from the merge method."""
        # Access the merge method's parameter information.  This replaces
        # utils.OPTIMIZABLE_HYPERPARAMETERS and the volatile_hypers logic.
        param_names = []
        input_types = self.merge_method.get_input_types().as_dict()
        for name, param_type in input_types.items():
            # We optimize parameters that are Tensors or StateDicts, and don't have default values
            if isinstance(param_type, type) and issubclass(param_type, (sd_mecha.extensions.merge_methods.StateDict, torch.Tensor)):
                param_names.append(name)
        return param_names

    def get_bounds(
        self,
        custom_bounds: Dict[str, Union[List[float], List[int], ListConfig, int, float]],
    ) -> Dict[str, Union[Tuple[float, float], float]]:
        """Gets the final bounds after applying custom bounds."""

        bounds = self.create_parameter_bounds()
        logger.debug(f"Initial Parameter Bounds: {bounds}")

        # Validate and apply custom_bounds
        validated_custom_bounds = self.validate_custom_bounds(custom_bounds)
        for param_name, custom_bound in validated_custom_bounds.items():
            if param_name in bounds:
                bounds[param_name] = tuple(custom_bound)  # Directly use the custom bound
                logger.info(f"Applied custom bound {custom_bound} to parameter '{param_name}'")
            else:
                logger.warning(
                    f"Custom bound provided for unknown parameter '{param_name}'. Skipping."
                )

        # Log bounds for each hyperparameter
        for param_name in self.param_names:
            if param_name in bounds:
                logger.info(f"Bounds for {param_name}: {bounds[param_name]}")

        return bounds


    def create_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Creates parameter bounds based on the optimization guide."""
        bounds = {}
        for component_config in self.cfg.optimization_guide.components:
            component_name = component_config.name
            optimization_strategy = component_config.strategy

            if optimization_strategy == "all":
                bounds.update(
                    self._handle_params(component_name, bounds=(0.0, 1.0))
                )
            elif optimization_strategy == "selected":
                if not component_config.keys: # Changed from selected_blocks
                    raise ValueError(f"No 'keys' specified for component '{component_name}'")
                for key in component_config.keys: # Changed from selected_blocks
                    bounds.update(
                        self._handle_params(component_name, block_id=key, bounds=(0.0, 1.0))
                    )
            elif optimization_strategy == "grouped":
                bounds.update(
                    self._handle_params_for_grouped(component_config, bounds=(0.0,1.0))
                )
            elif optimization_strategy == "single":
                bounds[f"{component_name}_single_param"] = (0.0, 1.0)
            elif optimization_strategy == "none":
                logger.info(f"Optimization disabled for component '{component_name}'.")
            # TODO: Handle layer adjustments
            # elif optimization_strategy == "layer_adjustments":
            #     bounds.update(
            #         self._handle_params_for_layer_adjustments(component_config)
            #     )
            else:
                raise ValueError(f"Invalid optimization strategy: {optimization_strategy}")
        return bounds

    def _handle_params(self, component_name: str, block_id: str = None, group_name: str = None, bounds: Tuple[float, float] = None) -> Dict[str, Tuple[float, float]]:
        """Handles parameter processing for 'all', 'selected', and 'group-all' strategies."""
        import fnmatch  # Import for wildcard matching

        component_bounds = {}

        for param_name in self.param_names:
            if bounds is not None:  # Creating bounds
                if block_id:  # 'selected' strategy
                    # Check if block_id is a wildcard pattern
                    if "*" in block_id:
                        # Get all keys in the component
                        for key in self.model_config.components[component_name].keys:
                            # Match the key against the pattern
                            if fnmatch.fnmatch(key, block_id):
                                component_bounds[f"{key}_{param_name}"] = bounds
                    else:
                        # Not a wildcard, use the block_id directly
                        key = f"{block_id}_{param_name}"
                        component_bounds[key] = bounds
                elif group_name:
                    component_bounds[group_name] = bounds
                else: # all
                    for key in self.model_config.components[component_name].keys:
                        component_bounds[f"{key}_{param_name}"] = bounds
        return component_bounds

    def _handle_params_for_grouped(self, component_config, bounds: Tuple[float, float] = None) -> Dict[str, Tuple[float, float]]:
        """Handles parameter processing for the 'grouped' strategy."""
        component_bounds = {}

        # Get all blocks for automatic grouping
        block_ids = [block_id for block_id in self.model_config.components[component_config.name].keys]
        grouped_blocks = component_config.groups

        for param_name in self.param_names:
            for i, group in enumerate(grouped_blocks):
                group_name = "-".join([f"{block['keys']}_{param_name}" for block in group])
                component_bounds[group_name] = bounds
        return component_bounds

    @staticmethod
    def validate_custom_bounds(custom_bounds: Dict[str, Union[List[float], List[int], int, float]]) -> Dict[
        str, Union[Tuple[float, float], float, int]]:
        """Validates the custom_bounds dictionary."""
        if custom_bounds is None:
            return {}

        validated_bounds: Dict[str, Union[Tuple[float, float], float, int]] = {}  # <- Type hint

        for param_name, bound in custom_bounds.items():
            if isinstance(bound, list):
                if len(bound) == 2:
                    if all(isinstance(v, (int, float)) for v in bound):
                        if bound[0] > bound[1]:
                            raise ValueError(
                                f"Invalid range bound for '{param_name}': {bound}. Lower bound cannot be greater than upper bound.")
                        validated_bounds[param_name] = (
                        float(bound[0]), float(bound[1]))  # Explicitly create a Tuple[float, float]
                    elif all(v in [0, 1] for v in bound):
                        validated_bounds[param_name] = (float(bound[0]), float(bound[1]))  # Also here
                    else:
                        raise ValueError(...)
                elif all(isinstance(v, (int, float)) for v in bound):
                    # categorical, we don't need to do anything
                    validated_bounds[param_name] = bound  # type: ignore
                else:
                    raise ValueError(...)
            elif isinstance(bound, (int, float)):
                validated_bounds[param_name] = bound  # Keep single values as-is
            else:
                raise ValueError(...)
        return validated_bounds
