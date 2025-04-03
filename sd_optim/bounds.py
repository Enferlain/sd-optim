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


# Define a type alias for clarity (optional)
BoundsInfo = Dict[str, Dict[str, Any]]

class ParameterHandler:
    # V1.1 - Corrected base config inference logic
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # --- Infer Base Model Config ---
        self.base_model_config: Optional[model_configs.ModelConfig] = None
        if not cfg.model_paths:
            # Cannot proceed without models if merge mode requires base config knowledge
            logger.error("Configuration error: 'model_paths' is empty. Cannot infer base model config.")
            raise ValueError("'model_paths' cannot be empty.")

        # Use the first model path as representative for inferring the base structure
        representative_model_path_str = cfg.model_paths[0]
        logger.info(f"Inferring base ModelConfig structure from: {representative_model_path_str}")
        try:
            # Resolve the models_dir based on the representative path
            self.models_dir = Path(representative_model_path_str).resolve().parent
            if not self.models_dir.is_dir():
                 raise FileNotFoundError(f"Directory for base model not found: {self.models_dir}")

            # Create a SINGLE node for the representative model
            rep_model_node = sd_mecha.model(representative_model_path_str)

            # Use open_input_dicts on the SINGLE node to load its header/keys
            with sd_mecha.open_input_dicts(rep_model_node, [self.models_dir]):
                # Infer configs based on the keys of this single loaded model node
                # rep_model_node.state_dict should be populated within the context
                if rep_model_node.state_dict:
                    # Pass the keys directly to infer_model_configs
                    inferred_configs = sd_mecha.infer_model_configs(rep_model_node.state_dict.keys())
                    if inferred_configs:
                        self.base_model_config = inferred_configs[0] # Take the best match
                        logger.info(f"Inferred base ModelConfig: {self.base_model_config.identifier}")
                    else:
                        logger.error(f"sd_mecha could not infer any matching ModelConfig for {representative_model_path_str}.")
                        raise ValueError(f"Could not infer ModelConfig for {representative_model_path_str}.")
                else:
                    logger.error(f"Failed to load state_dict keys for {representative_model_path_str} within open_input_dicts.")
                    raise ValueError(f"Failed to load dictionary for {representative_model_path_str}.")

        except Exception as e:
             logger.error(f"Failed to infer base ModelConfig from '{representative_model_path_str}': {e}", exc_info=True)
             raise ValueError("Could not determine base ModelConfig.") from e
        # --- End Base Config Inference ---

        # Resolve merge method
        self.merge_method = utils.resolve_merge_method(cfg.merge_method)

        # Load custom block config if ID is set
        self.custom_block_config_id = self.cfg.optimization_guide.get("custom_block_config_id")
        self.custom_block_config = None
        if self.custom_block_config_id:
            # --- DEBUGGING STEP ---
            logger.info(f"==> DEBUG: Attempting to resolve custom_block_config_id: '{self.custom_block_config_id}'")
            try:
                all_registered_configs = [c.identifier for c in model_configs.get_all()]
                logger.info(f"==> DEBUG: sd-mecha registry contains: {all_registered_configs}")
                if self.custom_block_config_id in all_registered_configs:
                    logger.info(
                        f"==> DEBUG: Config '{self.custom_block_config_id}' IS FOUND in registry before resolve().")
                else:
                    logger.error(
                        f"==> DEBUG: Config '{self.custom_block_config_id}' IS NOT FOUND in registry before resolve()!")
                    # Optional: Log registry content more verbosely
                    logger.debug(f"==> DEBUG: Base Registry: {[c.identifier for c in model_configs.get_all_base()]}")
                    logger.debug(f"==> DEBUG: Aux Registry: {[c.identifier for c in model_configs.get_all_aux()]}")

            except Exception as debug_e:
                logger.error(f"==> DEBUG: Error occurred during registry check: {debug_e}")
            # --- END DEBUGGING STEP ---
            try:
                # This is the line that fails
                self.custom_block_config = model_configs.resolve(self.custom_block_config_id)
                logger.info(f"Successfully loaded custom block ModelConfig: {self.custom_block_config_id}")
            except ValueError as e:
                logger.error(f"Could not resolve custom_block_config_id '{self.custom_block_config_id}': {e}.")
                # Re-raise the error with context
                raise ValueError(
                    f"Invalid custom_block_config_id: {self.custom_block_config_id}. Check registration timing and spelling.") from e

        # Get optimizable base parameter names
        self.param_names = self._get_optimizable_parameter_names()

    # V1.2 - Inspects MergeMethod object, filters kwargs without defaults and by type
    def _get_optimizable_parameter_names(self) -> List[str]:
        """Gets the names of likely optimizable parameters from the merge method."""
        optimizable_names = []
        if not self.merge_method:
            logger.error("Merge method not resolved. Cannot get optimizable parameters.")
            return []
        try:
            kwarg_params: Dict[str, merge_methods.ParameterData] = self.merge_method.get_params().kwargs
            kwarg_defaults: Dict[str, Any] = self.merge_method.get_default_args().kwargs
            for name, param_data in kwarg_params.items():
                if name in kwarg_defaults: continue # Skip params with defaults
                # Check type hint - allow float/int for optimization
                expected_type = param_data.interface
                origin_type = getattr(expected_type, '__origin__', None) or expected_type
                if not issubclass(origin_type, (float, int)): # Only optimize float/int directly for now
                     # logger.debug(f"Skipping param '{name}': Type hint '{expected_type}' not float or int.")
                     continue
                optimizable_names.append(name)
        except Exception as e:
             logger.error(f"Failed get optimizable parameters for '{self.merge_method.identifier}': {e}", exc_info=True)
        if not optimizable_names: logger.warning(f"No optimizable keyword parameters found for '{self.merge_method.identifier}'.")
        logger.info(f"Optimizable base parameter names: {optimizable_names}")
        return optimizable_names

    # V1.3 - Generates richer metadata including strategy, target_type, items_covered etc.
    def create_parameter_bounds_metadata(self) -> BoundsInfo:
        """Creates parameter bounds AND metadata based on the optimization guide."""
        params_info: BoundsInfo = {} # Store the richer info here
        optimizable_base_params = self.param_names

        if not self.cfg.optimization_guide.get("components"):
             logger.warning("No 'components' defined in optimization_guide. No bounds generated.")
             return params_info

        for component_config_from_guide in self.cfg.optimization_guide.components:
            guide_component_name = component_config_from_guide.name
            strategy = component_config_from_guide.strategy
            if strategy == "none": continue # Skip 'none' strategies early

            # --- Determine Target Type (Block vs Key) and Config ---
            config_to_iterate: Optional[model_configs.ModelConfig] = None
            target_is_blocks: bool = False
            patterns_in_guide = []
            if strategy in ["select", "group"]:
                 patterns_in_guide = component_config_from_guide.get("keys", []) if strategy == "select" else \
                                    [p for group in component_config_from_guide.get("groups", []) for p in group.get("keys", [])]
            looks_like_block_target = False
            if self.custom_block_config and guide_component_name in self.custom_block_config.components:
                 if patterns_in_guide: looks_like_block_target = not (patterns_in_guide[0].__contains__(".") and patterns_in_guide[0].lower() == patterns_in_guide[0])
                 else: looks_like_block_target = True

            if looks_like_block_target: config_to_iterate = self.custom_block_config; target_is_blocks = True
            elif self.base_model_config and guide_component_name in self.base_model_config.components: config_to_iterate = self.base_model_config; target_is_blocks = False
            else: logger.warning(f"Comp. '{guide_component_name}' not found. Skipping."); continue

            target_type_str = "block" if target_is_blocks else "key"

            # --- Get the list of block names or model keys ---
            try: items_in_component = list(config_to_iterate.components[guide_component_name].keys.keys())
            except KeyError: logger.warning(f"Comp. '{guide_component_name}' structure error. Skipping."); continue
            if not items_in_component: logger.warning(f"Comp. '{guide_component_name}' empty. Skipping."); continue

            # --- Apply strategy for each optimizable base parameter ---
            for base_param_name in optimizable_base_params:
                default_bounds_tuple = (0.0, 1.0) # Default bounds

                # --- Base metadata common to all strategies for this param/component ---
                base_metadata = {
                    "strategy": strategy,
                    "target_type": target_type_str,
                    "base_param": base_param_name,
                    "component_name": guide_component_name, # Store component name context
                    # Bounds will be added below or by get_bounds
                }

                if strategy == "all":
                     for item_name in items_in_component:
                         generated_param_name = f"{item_name}_{base_param_name}"
                         if generated_param_name not in params_info: # Avoid overwriting if multiple strategies target same item
                             params_info[generated_param_name] = {
                                 **base_metadata,
                                 "item_name": item_name,
                                 "bounds": default_bounds_tuple # Add default bounds here
                             }
                             logger.debug(f"  Generated info (all): {generated_param_name}")

                elif strategy == "select":
                     patterns = component_config_from_guide.get("keys", [])
                     if not patterns: continue
                     for pattern in patterns:
                         for item_name in items_in_component:
                             if fnmatch.fnmatch(item_name, pattern):
                                 generated_param_name = f"{item_name}_{base_param_name}"
                                 if generated_param_name not in params_info:
                                     params_info[generated_param_name] = {
                                         **base_metadata,
                                         "item_name": item_name,
                                         "bounds": default_bounds_tuple
                                     }
                                     logger.debug(f"  Generated info (select): {generated_param_name}")

                elif strategy == "group":
                     groups = component_config_from_guide.get("groups", [])
                     if not groups: continue
                     for group in groups:
                         group_name = group.get("name")
                         group_patterns = group.get("keys", [])
                         if not group_name or not group_patterns: continue

                         items_covered_by_group = [
                             item for pattern in group_patterns for item in items_in_component if fnmatch.fnmatch(item, pattern)
                         ]

                         if items_covered_by_group:
                             generated_param_name = f"{group_name}_{base_param_name}"
                             if generated_param_name not in params_info:
                                 params_info[generated_param_name] = {
                                     **base_metadata,
                                     "group_name": group_name,
                                     "items_covered": items_covered_by_group, # Store covered items
                                     "bounds": default_bounds_tuple
                                 }
                                 logger.debug(f"  Generated info (group): {generated_param_name} covering {len(items_covered_by_group)} items.")
                         # else: logger.debug(f"Group '{group_name}' patterns matched no items in '{guide_component_name}'.")

                elif strategy == "single":
                     generated_param_name = f"{guide_component_name}_default_{base_param_name}"
                     if generated_param_name not in params_info:
                         params_info[generated_param_name] = {
                             **base_metadata,
                             "items_covered": items_in_component, # Store all component items
                             "bounds": default_bounds_tuple
                         }
                         logger.debug(f"  Generated info (single): {generated_param_name} covering {len(items_in_component)} items.")

        logger.info(f"Generated metadata for {len(params_info)} optimization parameters.")
        return params_info

    # V1.2 - Calls metadata generator and applies custom bounds to the 'bounds' key
    def get_bounds(
        self,
        custom_bounds_config: Optional[Dict[str, Union[List[float], List[int], int, float]]] = None
    ) -> BoundsInfo: # Return the enhanced BoundsInfo type
        """Gets the final bounds info after applying custom bounds overrides."""
        # Generate parameter info including default bounds and metadata
        params_info = self.create_parameter_bounds_metadata()

        # Validate the user's custom bounds input
        validated_custom_bounds = self.validate_custom_bounds(custom_bounds_config or {})

        # Apply custom bounds overrides to the generated info
        for param_name, info in params_info.items():
            if param_name in validated_custom_bounds:
                # Apply specific custom bound
                info['bounds'] = validated_custom_bounds[param_name]
                logger.debug(f"Applied specific custom bound to '{param_name}': {info['bounds']}")
            else:
                 # Check for general override (e.g., custom bound for 'alpha')
                 base_param_name = info.get('base_param')
                 if base_param_name and base_param_name in validated_custom_bounds:
                     info['bounds'] = validated_custom_bounds[base_param_name]
                     logger.debug(f"Applied general custom bound '{base_param_name}' to '{param_name}': {info['bounds']}")
            # If neither specific nor general override exists, the default from create_parameter_bounds_metadata remains

        # Add any fixed parameters defined only in custom_bounds (not generated by strategies)
        for custom_param, custom_bound_value in validated_custom_bounds.items():
             if custom_param not in params_info:
                  # Check it's not just a general override key
                  is_general_override_key = custom_param in self.param_names
                  if not is_general_override_key:
                       # Treat as a fixed, non-optimized parameter maybe? Or add with default metadata?
                       # For now, just add it with minimal metadata. Optimizer might ignore it if not asked.
                       params_info[custom_param] = {
                            'bounds': custom_bound_value, # Could be range, fixed, or categorical
                            'strategy': 'custom_fixed', # Indicate origin
                            'target_type': 'unknown',
                            'base_param': custom_param,
                       }
                       logger.debug(f"Added custom bound directly (fixed?): '{custom_param}': {custom_bound_value}")


        logger.info(f"--- Final {len(params_info)} Optimization Parameter Details ---")
        items_to_log = list(params_info.items())
        log_limit = 100
        if len(items_to_log) > log_limit * 2:
             for name, info in items_to_log[:log_limit]: logger.info(f"{name}: {info}")
             logger.info("...")
             for name, info in items_to_log[-log_limit:]: logger.info(f"{name}: {info}")
        else:
             for name, info in items_to_log: logger.info(f"{name}: {info}")
        logger.info("----------------------------------------------------")

        return params_info

    # V1.1 - Updated validation logic from bounds.py
    @staticmethod
    def validate_custom_bounds(
        custom_bounds: Optional[Dict[str, Union[List[float], List[int], int, float]]]
    ) -> Dict[str, Union[Tuple[float, float], float, int, List]]:
        """Validates the custom_bounds dictionary and returns typed bounds."""
        if custom_bounds is None:
            return {}
        validated_bounds: Dict[str, Union[Tuple[float, float], float, int, List]] = {}
        for param_name, bound in custom_bounds.items():
            try:
                if isinstance(bound, (list, ListConfig)): # Check omegaconf list too
                    bound_list = list(bound) # Convert if ListConfig
                    if len(bound_list) == 2 and all(isinstance(v, (int, float)) for v in bound_list):
                        val1, val2 = float(bound_list[0]), float(bound_list[1])
                        if val1 > val2: raise ValueError("Lower bound > Upper bound")
                        validated_bounds[param_name] = (val1, val2) # Range
                    elif all(isinstance(v, (int, float)) for v in bound_list):
                        # Categorical: Keep as list, ensure numeric
                        validated_bounds[param_name] = [float(v) if isinstance(v, float) else int(v) for v in bound_list]
                    else: raise ValueError("List must be [min, max] or categorical [v1, v2,...] of numbers.")
                elif isinstance(bound, (int, float)):
                    validated_bounds[param_name] = float(bound) if isinstance(bound, float) else int(bound) # Fixed
                else: raise ValueError("Bound type must be list, int, or float.")
            except ValueError as e:
                logger.error(f"Invalid custom bound configuration for '{param_name}': {bound}. Error: {e}. Skipping this bound.")
            except Exception as e_gen:
                 logger.error(f"Unexpected error validating custom bound for '{param_name}': {bound}. Error: {e_gen}. Skipping.")
        return validated_bounds