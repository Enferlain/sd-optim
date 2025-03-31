# bounds.py - Version 1.1 - Added custom_block_config_id support

import logging
from pathlib import Path

import sd_mecha
import fnmatch
import torch
import inspect # Needed to inspect parameter types

from typing import Dict, List, Tuple, Union, Optional
from omegaconf import DictConfig, ListConfig
from sd_mecha.extensions import model_configs, merge_methods, merge_spaces # Added imports
from sd_mecha.recipe_nodes import ModelRecipeNode

from sd_optim import utils

logger = logging.getLogger(__name__)


class ParameterHandler:
    """Handles parameter bounds creation, aware of custom block configs."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Infer the base model config from input paths
        try:
            # Create temporary ModelRecipeNodes just for inference
            temp_model_nodes: List[ModelRecipeNode] = [sd_mecha.model(p) for p in cfg.model_paths]
            # Need context to infer configs if paths are relative
            # Ignore type error for open_input_dicts accepting List[RecipeNode]
            with sd_mecha.open_input_dicts(temp_model_nodes, [self.models_dir]): # type: ignore[arg-type]
                 # Ignore type error for infer_model_configs accepting List[RecipeNode]
                 # The function iterates keys from the loaded state_dict within the context
                 inferred_configs = sd_mecha.infer_model_configs(temp_model_nodes) # type: ignore[arg-type]
                 if not inferred_configs:
                      raise ValueError("sd_mecha.infer_model_configs returned no matching configurations.")
                 self.base_model_config = inferred_configs[0]
                 logger.info(f"Inferred base ModelConfig: {self.base_model_config.identifier}")
        except Exception as e:
             logger.error(f"Failed to infer base ModelConfig from model_paths: {e}", exc_info=True)
             raise ValueError("Could not determine base ModelConfig.") from e

        # Resolve the primary merge method
        self.merge_method = utils.resolve_merge_method(cfg.merge_method)

        # Get the custom block config if specified
        self.custom_block_config_id = self.cfg.optimization_guide.get("custom_block_config_id", None)
        self.custom_block_config = None
        if self.custom_block_config_id:
            try:
                self.custom_block_config = model_configs.resolve(self.custom_block_config_id)
                logger.info(f"Using custom block ModelConfig: {self.custom_block_config_id}")
            except ValueError as e:
                logger.error(f"Could not resolve custom_block_config_id '{self.custom_block_config_id}': {e}. Make sure it's registered.")
                # Decide how to handle this: error out or proceed without custom blocks?
                raise ValueError(f"Invalid custom_block_config_id: {self.custom_block_config_id}") from e

        # Get list of optimizable parameters (names only)
        self.param_names = self._get_optimizable_parameter_names()

    def _is_block_level_parameter(self, param_name: str) -> bool:
        """
        Heuristic to determine if a merge method parameter likely accepts block-level weights.
        Checks if the parameter is hinted as StateDict or Tensor.
        """
        try:
            param_info = self.merge_method.get_params().as_dict().get(param_name)
            if param_info:
                # Check if the type hint is StateDict or Tensor (or potentially float/int if broadcasting is intended)
                type_hint = param_info.interface
                origin = getattr(type_hint, '__origin__', None) or type_hint
                if issubclass(origin, (merge_methods.StateDict, torch.Tensor)):
                     # Could add more checks, e.g., exclude parameters named 'base', 'model', etc.
                    return True
        except Exception as e:
             logger.warning(f"Could not inspect type for parameter '{param_name}': {e}")
        return False

    def _get_model_config_for_parameter(self, param_name: str) -> model_configs.ModelConfig:
        """Determines the ModelConfig to use for generating bounds for a specific parameter."""
        if self.custom_block_config and self._is_block_level_parameter(param_name):
            # Use the custom block config if specified and the parameter seems block-level
            return self.custom_block_config
        else:
            # Otherwise, try to get the config from the parameter's type hint
            try:
                 param_data = self.merge_method.get_params().as_dict().get(param_name)
                 if param_data and param_data.model_config:
                     # Ensure it's resolved if it's an ID string
                     return model_configs.resolve(param_data.model_config) if isinstance(param_data.model_config, str) else param_data.model_config
            except Exception as e:
                 logger.warning(f"Could not get model_config from type hint for '{param_name}': {e}. Using base config.")
            # Fallback to the inferred base model config
            return self.base_model_config


    def _get_optimizable_parameter_names(self) -> List[str]:
        """Gets the names of optimizable parameters from the merge method."""
        param_names = []
        try:
            param_details = self.merge_method.get_params().as_dict()
            param_defaults = self.merge_method.get_default_args().as_dict()

            for name, data in param_details.items():
                # Only include parameters that don't have a default value
                # (Parameters with defaults are usually fixed settings, not optimized variables)
                 is_positional = isinstance(name, int)
                 param_actual_name = self.merge_method.get_param_names().as_dict().get(name) if is_positional else name

                 if name not in param_defaults and param_actual_name not in param_defaults:
                     # Check if it's a type we typically optimize (Tensor/StateDict)
                     # This might need adjustment based on merge methods used
                     type_hint = data.interface
                     origin = getattr(type_hint, '__origin__', None) or type_hint
                     if issubclass(origin, (merge_methods.StateDict, torch.Tensor, float, int)): # Allow float/int optimization too
                          if param_actual_name: # Ensure we have a valid name
                            param_names.append(param_actual_name)

        except Exception as e:
             logger.error(f"Failed to get optimizable parameters for method '{self.merge_method.identifier}': {e}", exc_info=True)
             # Decide: return empty list or raise error?
             # raise ValueError(f"Could not determine optimizable parameters.") from e
        logger.debug(f"Optimizable parameters for '{self.merge_method.identifier}': {param_names}")
        return param_names

    def get_bounds(
        self,
        custom_bounds_config: Dict[str, Union[List[float], List[int], int, float]],
    ) -> Dict[str, Union[Tuple[float, float], float, int, List]]: # Allow List for categorical
        """Gets the final bounds after applying custom bounds."""

        # Create bounds based on guide, considering parameter-specific configs
        bounds = self.create_parameter_bounds()
        logger.debug(f"Initial Parameter Bounds (Pre-Customization): {bounds}")

        # Validate and apply custom_bounds from the config
        validated_custom_bounds = self.validate_custom_bounds(custom_bounds_config)

        final_bounds: Dict[str, Union[Tuple[float, float], float, int, List]] = {}

        # Apply specific bounds first
        for param_name in bounds:
            if param_name in validated_custom_bounds:
                bound_val = validated_custom_bounds[param_name]
                # Handle different bound types from validation
                if isinstance(bound_val, tuple) and len(bound_val) == 2:
                    final_bounds[param_name] = bound_val # Range
                elif isinstance(bound_val, (int, float)):
                    final_bounds[param_name] = bound_val # Fixed value
                elif isinstance(bound_val, list):
                     final_bounds[param_name] = bound_val # Categorical List
                logger.info(f"Applied specific custom bound {bound_val} to '{param_name}'")
            else:
                 # No specific bound, check for general bounds (e.g., 'alpha')
                 base_param_name = param_name.split('_')[-1] # Heuristic: get the base name like 'alpha'
                 if base_param_name in validated_custom_bounds:
                     bound_val = validated_custom_bounds[base_param_name]
                     if isinstance(bound_val, tuple) and len(bound_val) == 2:
                         final_bounds[param_name] = bound_val # Range
                     elif isinstance(bound_val, (int, float)):
                         final_bounds[param_name] = bound_val # Fixed value
                     elif isinstance(bound_val, list):
                         final_bounds[param_name] = bound_val # Categorical List
                     logger.info(f"Applied general custom bound '{base_param_name}'={bound_val} to '{param_name}'")
                 else:
                     # Use default bounds if no custom bound found
                     final_bounds[param_name] = bounds[param_name]


        # Log final bounds
        logger.info("--- Final Optimization Bounds ---")
        for name, bound in final_bounds.items():
            logger.info(f"{name}: {bound}")
        logger.info("---------------------------------")

        return final_bounds

    def create_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Creates parameter bounds based on the optimization guide."""
        bounds = {}
        # Iterate through each optimizable parameter of the merge method
        for param_name in self.param_names:
            # Determine the correct ModelConfig for THIS parameter
            param_model_config = self._get_model_config_for_parameter(param_name)

            # Now, apply strategies based on guide.yaml using param_model_config
            for component_config in self.cfg.optimization_guide.components:
                component_name = component_config.name

                # Check if the component exists in the parameter's relevant config
                if component_name not in param_model_config.components:
                    # This might happen if e.g. guide has 'unet' but param_config is 'sdxl-optim_blocks'
                    # which only has a 'blocks' component. Skip or warn.
                    # logger.debug(f"Component '{component_name}' not found in config '{param_model_config.identifier}' for param '{param_name}'. Skipping.")
                    continue

                strategy = component_config.strategy

                if strategy == "all":
                    bounds.update(
                        self._handle_strategy_all(param_name, component_name, param_model_config)
                    )
                elif strategy == "select":
                    bounds.update(
                        self._handle_strategy_select(param_name, component_name, component_config.keys, param_model_config)
                    )
                elif strategy == "group":
                    bounds.update(
                        self._handle_strategy_group(param_name, component_config.groups)
                    )
                elif strategy == "single":
                    bounds.update(
                        self._handle_strategy_single(param_name, component_name)
                    )
                elif strategy == "none":
                    pass # Do nothing for this component and this parameter
                # TODO: Handle layer adjustments if needed for specific parameters
                else:
                    raise ValueError(f"Invalid optimization strategy: {strategy}")
        return bounds

    # --- Strategy Handlers ---
    # These now take param_name and the relevant model_config

    def _handle_strategy_all(self, param_name: str, component_name: str, config: model_configs.ModelConfig) -> Dict[str, Tuple[float, float]]:
        """Handles 'all' strategy for a specific parameter."""
        component_bounds = {}
        default_bounds = (0.0, 1.0) # Default bounds
        try:
            for key in config.components[component_name].keys:
                 # Generate name: key_paramname (e.g., IN00_alpha)
                generated_param_name = f"{key}_{param_name}"
                component_bounds[generated_param_name] = default_bounds
        except KeyError:
             logger.warning(f"Component '{component_name}' not found in config '{config.identifier}' while processing param '{param_name}'.")
        return component_bounds

    def _handle_strategy_select(self, param_name: str, component_name: str, patterns: List[str], config: model_configs.ModelConfig) -> Dict[str, Tuple[float, float]]:
        """Handles 'select' strategy with wildcards for a specific parameter."""
        component_bounds = {}
        default_bounds = (0.0, 1.0)
        try:
             component_keys = config.components[component_name].keys
             for pattern in patterns:
                 matched_any = False
                 for key in component_keys:
                     if fnmatch.fnmatch(key, pattern):
                         generated_param_name = f"{key}_{param_name}"
                         component_bounds[generated_param_name] = default_bounds
                         matched_any = True
                 if not matched_any:
                      logger.warning(f"Pattern '{pattern}' in 'select' strategy did not match any keys in component '{component_name}' of config '{config.identifier}' for param '{param_name}'.")
        except KeyError:
              logger.warning(f"Component '{component_name}' not found in config '{config.identifier}' while processing param '{param_name}'.")
        return component_bounds

    def _handle_strategy_group(self, param_name: str, groups: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Handles 'group' strategy for a specific parameter."""
        component_bounds = {}
        default_bounds = (0.0, 1.0)
        for group in groups:
            group_name = group.get("name")
            if not group_name:
                logger.warning(f"Group definition missing 'name', skipping: {group}")
                continue
             # Generate name: groupname_paramname (e.g., my_group_1_alpha)
            generated_param_name = f"{group_name}_{param_name}"
            component_bounds[generated_param_name] = default_bounds
        return component_bounds

    def _handle_strategy_single(self, param_name: str, component_name: str) -> Dict[str, Tuple[float, float]]:
        """Handles 'single' strategy for a specific parameter."""
        default_bounds = (0.0, 1.0)
        # Generate name: componentname_single_param_paramname (e.g., unet_single_param_alpha)
        generated_param_name = f"{component_name}_single_param_{param_name}"
        return {generated_param_name: default_bounds}

    # --- Validation ---
    @staticmethod
    def validate_custom_bounds(custom_bounds: Dict[str, Union[List[float], List[int], int, float]]) -> Dict[str, Union[Tuple[float, float], float, int, List]]:
        """Validates the custom_bounds dictionary and returns typed bounds."""
        if custom_bounds is None:
            return {}

        validated_bounds: Dict[str, Union[Tuple[float, float], float, int, List]] = {}

        for param_name, bound in custom_bounds.items():
            if isinstance(bound, list):
                if len(bound) == 2:
                    if all(isinstance(v, (int, float)) for v in bound):
                        # Range or Binary
                        val1, val2 = float(bound[0]), float(bound[1])
                        if val1 > val2:
                            raise ValueError(f"Invalid range bound for '{param_name}': {bound}. Lower > Upper.")
                        validated_bounds[param_name] = (val1, val2) # Store as tuple[float, float]
                    else:
                         raise ValueError(f"Invalid range/binary bound '{param_name}': {bound}. Must contain numbers.")
                elif all(isinstance(v, (int, float)) for v in bound):
                    # Categorical List
                    # Convert all to float or int for consistency if needed, or keep mixed
                    validated_bounds[param_name] = [float(v) if isinstance(v, float) else int(v) for v in bound]
                else:
                    raise ValueError(f"Invalid list bound for '{param_name}': {bound}. Must be range [min,max] or categorical list [v1, v2,...].")
            elif isinstance(bound, (int, float)):
                # Fixed value
                validated_bounds[param_name] = float(bound) if isinstance(bound, float) else int(bound)
            else:
                raise ValueError(f"Invalid bound type for '{param_name}': {bound}. Must be list, int, or float.")
        return validated_bounds