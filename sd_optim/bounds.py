# bounds.py - Version 1.1 - Added custom_block_config_id support

import logging
from pathlib import Path

import sd_mecha
import fnmatch
import torch
import inspect # Needed to inspect parameter types

from typing import Dict, List, Tuple, Union, Optional, Any
from omegaconf import DictConfig, ListConfig
from sd_mecha.extensions import model_configs, merge_methods, merge_spaces # Added imports
from sd_mecha.extensions.merge_methods import StateDict, ParameterData # For type checking
from sd_optim import utils

logger = logging.getLogger(__name__)


class ParameterHandler:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.models_dir = Path(self.cfg.model_paths[0]).resolve().parent # Assuming model_paths[0] exists
        try:
            # Infer base config (needs context) - simplified version
            temp_model_nodes = [sd_mecha.model(p) for p in cfg.model_paths]
            with sd_mecha.open_input_dicts(temp_model_nodes, [self.models_dir]): # type: ignore[arg-type]
                inferred_configs = sd_mecha.infer_model_configs(temp_model_nodes) # type: ignore[arg-type]
                if not inferred_configs: raise ValueError("No base configs inferred.")
                self.base_model_config = inferred_configs[0]
                logger.info(f"Inferred base ModelConfig: {self.base_model_config.identifier}")
        except Exception as e:
             logger.error(f"Failed to infer base ModelConfig: {e}", exc_info=True)
             raise ValueError("Could not determine base ModelConfig.") from e

        self.merge_method = utils.resolve_merge_method(cfg.merge_method) # Assumes utils.resolve_merge_method exists

        # Load custom block config if ID is set
        self.custom_block_config_id = self.cfg.optimization_guide.get("custom_block_config_id")
        self.custom_block_config = None
        if self.custom_block_config_id:
            try:
                # Resolve using sd_mecha's registry (assuming it's registered by setup_custom_blocks)
                self.custom_block_config = model_configs.resolve(self.custom_block_config_id)
                logger.info(f"Successfully loaded custom block ModelConfig: {self.custom_block_config_id}")
            except ValueError as e:
                logger.error(f"Could not resolve custom_block_config_id '{self.custom_block_config_id}': {e}. Ensure it's defined and registered before ParameterHandler is initialized.")
                raise # Stop if the specified custom config can't be loaded

        self.param_names = self._get_optimizable_parameter_names() # Base names like 'alpha'

    def _get_optimizable_parameter_names(self) -> List[str]:
        """Gets the names of likely optimizable parameters from the merge method."""
        optimizable_names = []
        if not self.merge_method:
            logger.error("Merge method not resolved in ParameterHandler. Cannot get optimizable parameters.")
            return []

        try:
            # Get keyword-only parameters and their default values
            kwarg_params: Dict[str, ParameterData] = self.merge_method.get_params().kwargs
            kwarg_defaults: Dict[str, Any] = self.merge_method.get_default_args().kwargs

            logger.debug(f"Inspecting merge method '{self.merge_method.identifier}' for optimizable parameters...")
            logger.debug(f"  Keyword Params: {list(kwarg_params.keys())}")
            logger.debug(f"  Keyword Defaults: {list(kwarg_defaults.keys())}")

            for name, param_data in kwarg_params.items():
                # --- Condition 1: Parameter should NOT have a default value ---
                # Parameters with defaults are usually fixed settings, not variables for optimization.
                if name in kwarg_defaults:
                    logger.debug(f"  Skipping '{name}': Has a default value ({kwarg_defaults[name]}).")
                    continue

                # --- Condition 2 (Optional): Filter by expected type ---
                # We might only want to optimize numerical parameters (float/int) or maybe Tensors.
                # Let's allow float/int for now, as these are common optimization targets.
                expected_type = param_data.interface
                origin_type = getattr(expected_type, '__origin__', None) or expected_type
                # Allow basic numerical types or Tensors/StateDicts (though optimizing Tensor/SD directly is complex)
                # Note: sd_mecha.Parameter wraps the type, so access it via .interface
                if not issubclass(origin_type, (float, int, torch.Tensor, StateDict)):
                     logger.debug(f"  Skipping '{name}': Type hint '{expected_type}' is not typically optimized (float, int, Tensor, StateDict).")
                     continue

                # If conditions pass, add the base name to the list
                optimizable_names.append(name)
                logger.debug(f"  Identified '{name}' as potentially optimizable.")

        except Exception as e:
             logger.error(f"Failed to get optimizable parameters for method '{self.merge_method.identifier}': {e}", exc_info=True)
             # Decide: return empty list or raise error?
             # raise ValueError(f"Could not determine optimizable parameters for {self.merge_method.identifier}") from e

        if not optimizable_names:
             logger.warning(f"No optimizable keyword parameters (without defaults, numerical/Tensor type) found for merge method '{self.merge_method.identifier}'. Check method signature and guide.yaml.")

        logger.info(f"Optimizable parameter base names for '{self.merge_method.identifier}': {optimizable_names}")
        return optimizable_names

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
        """V1.1: Creates parameter bounds based on the optimization guide, using custom blocks where applicable."""
        bounds: Dict[str, Tuple[float, float]] = {}
        optimizable_base_params = self.param_names # Base names like 'alpha', 'beta'
        use_custom_blocks = bool(self.custom_block_config)

        if not self.cfg.optimization_guide.get("components"):
             logger.warning("No 'components' defined in optimization_guide. No bounds will be generated.")
             return bounds

        for component_config_from_guide in self.cfg.optimization_guide.components:
            guide_component_name = component_config_from_guide.name
            strategy = component_config_from_guide.strategy
            logger.debug(f"Processing guide component: '{guide_component_name}', strategy: '{strategy}'")

            # --- Decide which config provides the keys/blocks for this guide component ---
            config_to_iterate: Optional[model_configs.ModelConfig] = None
            iteration_target_is_blocks: bool = False

            if use_custom_blocks and self.custom_block_config and guide_component_name in self.custom_block_config.components:
                # Use custom blocks if available and guide targets a component within it
                config_to_iterate = self.custom_block_config
                iteration_target_is_blocks = True
                logger.debug(f"-> Iterating custom blocks from '{config_to_iterate.identifier}'")
            elif guide_component_name in self.base_model_config.components:
                # Fallback to base config keys
                config_to_iterate = self.base_model_config
                iteration_target_is_blocks = False
                logger.debug(f"-> Iterating base keys from '{config_to_iterate.identifier}'")
            else:
                logger.warning(f"Guide component '{guide_component_name}' not found in custom block config ('{self.custom_block_config_id}') or base config ('{self.base_model_config.identifier}'). Skipping.")
                continue

            # --- Get the list of block names or model keys ---
            try:
                 # Get the actual keys/block names for the targeted component within the chosen config
                 keys_or_blocks_to_process = list(config_to_iterate.components[guide_component_name].keys.keys())
                 if not keys_or_blocks_to_process:
                      logger.warning(f"Component '{guide_component_name}' in config '{config_to_iterate.identifier}' has no keys/blocks defined. Skipping.")
                      continue
                 # logger.debug(f"   Items to process for '{guide_component_name}': {keys_or_blocks_to_process[:5]}...") # Log first few
            except KeyError:
                 logger.warning(f"Component '{guide_component_name}' structure error in config '{config_to_iterate.identifier}'. Skipping.")
                 continue

            # --- Apply strategy for each optimizable base parameter ---
            for base_param_name in optimizable_base_params:
                default_bounds = (0.0, 1.0) # Define default here

                if strategy == "all":
                    for key_or_block in keys_or_blocks_to_process:
                        generated_param_name = f"{key_or_block}_{base_param_name}"
                        bounds[generated_param_name] = default_bounds
                elif strategy == "select":
                    patterns = component_config_from_guide.get("keys", [])
                    if not patterns: logger.warning(f"'select' strategy for '{guide_component_name}' has no 'keys' defined."); continue
                    for pattern in patterns:
                        for key_or_block in keys_or_blocks_to_process:
                            if fnmatch.fnmatch(key_or_block, pattern):
                                generated_param_name = f"{key_or_block}_{base_param_name}"
                                bounds[generated_param_name] = default_bounds
                elif strategy == "group":
                    groups = component_config_from_guide.get("groups", [])
                    if not groups: logger.warning(f"'group' strategy for '{guide_component_name}' has no 'groups' defined."); continue
                    for group in groups:
                        group_name = group.get("name")
                        group_patterns = group.get("keys", [])
                        if not group_name or not group_patterns: logger.warning(f"Invalid group definition in '{guide_component_name}'."); continue

                        # Check if any key/block in this component matches the group patterns
                        match_found_in_component = any(
                            fnmatch.fnmatch(k_or_b, pattern)
                            for pattern in group_patterns
                            for k_or_b in keys_or_blocks_to_process
                        )

                        if match_found_in_component:
                            generated_param_name = f"{group_name}_{base_param_name}"
                            # Only add if not already added by another group targeting the same component
                            if generated_param_name not in bounds:
                                 bounds[generated_param_name] = default_bounds
                        # else: logger.debug(f"Group '{group_name}' patterns didn't match any items in component '{guide_component_name}' of config '{config_to_iterate.identifier}'.")

                elif strategy == "single":
                    generated_param_name = f"{guide_component_name}_single_param_{base_param_name}"
                    bounds[generated_param_name] = default_bounds
                elif strategy == "none":
                    logger.debug(f"Strategy 'none' for component '{guide_component_name}', skipping parameter '{base_param_name}'.")
                    pass # Explicitly do nothing
                else:
                    logger.warning(f"Unsupported strategy '{strategy}' for component '{guide_component_name}'. Skipping.")

        logger.info(f"Generated {len(bounds)} parameter bounds based on optimization guide.")
        # Optionally log the first few bounds for debugging:
        # if bounds: logger.debug(f"Example bounds: {dict(list(bounds.items())[:5])}")
        return bounds

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