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
    # V1.2 - Reads optimize_params from guide, removed auto-detection
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # --- Infer Base Model Config ---
        # (Keep the corrected inference logic using the first model)
        self.base_model_config: Optional[model_configs.ModelConfig] = None
        if not cfg.model_paths: raise ValueError("'model_paths' cannot be empty.")
        representative_model_path_str = cfg.model_paths[0]
        logger.info(f"Inferring base ModelConfig from: {representative_model_path_str}")
        try:
            self.models_dir = Path(representative_model_path_str).resolve().parent
            if not self.models_dir.is_dir(): raise FileNotFoundError(f"Dir not found: {self.models_dir}")
            rep_model_node = sd_mecha.model(representative_model_path_str)
            with sd_mecha.open_input_dicts(rep_model_node, [self.models_dir]):
                if rep_model_node.state_dict:
                    inferred_configs = sd_mecha.infer_model_configs(rep_model_node.state_dict.keys())
                    if inferred_configs: self.base_model_config = inferred_configs[0]; logger.info(f"Inferred base ModelConfig: {self.base_model_config.identifier}")
                    else: raise ValueError(f"Could not infer ModelConfig for {representative_model_path_str}.")
                else: raise ValueError(f"Failed to load dictionary for {representative_model_path_str}.")
        except Exception as e:
             logger.error(f"Failed to infer base ModelConfig: {e}", exc_info=True)
             raise ValueError("Could not determine base ModelConfig.") from e

        # Resolve merge method
        self.merge_method = utils.resolve_merge_method(cfg.merge_method)

        # Load custom block config if ID is set
        self.custom_block_config_id = self.cfg.optimization_guide.get("custom_block_config_id")
        self.custom_block_config = None
        if self.custom_block_config_id:
            try:
                self.custom_block_config = model_configs.resolve(self.custom_block_config_id)
                logger.info(f"Successfully loaded custom block ModelConfig: {self.custom_block_config_id}")
            except ValueError as e:
                logger.error(f"Could not resolve custom_block_config_id '{self.custom_block_config_id}': {e}.")
                raise ValueError(f"Invalid custom_block_config_id: {self.custom_block_config_id}") from e

        # --- REMOVED self.param_names = self._get_optimizable_parameter_names() ---

    # --- REMOVED _get_optimizable_parameter_names method ---

    # V1.4 - Reads optimize_params list from guide component config
    def create_parameter_bounds_metadata(self) -> BoundsInfo:
        """Creates parameter bounds AND metadata based on the optimization guide."""
        params_info: BoundsInfo = {}

        guide_components = self.cfg.optimization_guide.get("components", [])
        if not guide_components:
             logger.warning("No 'components' defined in optimization_guide. No bounds generated.")
             return params_info

        for component_config_from_guide in guide_components:
            guide_component_name = component_config_from_guide.get("name")
            strategy = component_config_from_guide.get("strategy")
            # --- Get the list of base parameters to optimize for THIS component/strategy ---
            optimizable_base_params = component_config_from_guide.get("optimize_params", [])

            if not guide_component_name or not strategy:
                 logger.warning(f"Skipping guide entry due to missing 'name' or 'strategy': {component_config_from_guide}")
                 continue
            if strategy == "none": continue # Skip 'none' strategies

            if not optimizable_base_params:
                 logger.warning(f"Skipping strategy '{strategy}' for component '{guide_component_name}': Missing 'optimize_params' list in guide.yaml.")
                 continue

            # --- Determine Target Type (Block vs Key) and Config ---
            config_to_iterate: Optional[model_configs.ModelConfig] = None
            target_is_blocks: bool = False
            patterns_in_guide = [] # Check patterns used in select/group to help determine target
            if strategy in ["select", "group"]:
                 patterns_in_guide = component_config_from_guide.get("keys", []) if strategy == "select" else \
                                    [p for group in component_config_from_guide.get("groups", []) for p in group.get("keys", [])]

            looks_like_block_target = False
            if self.custom_block_config and guide_component_name in self.custom_block_config.components:
                 # Assume block target if guide targets component in custom config,
                 # UNLESS patterns clearly look like base model keys.
                 if patterns_in_guide: looks_like_block_target = not (patterns_in_guide[0].__contains__(".") and patterns_in_guide[0].lower() == patterns_in_guide[0])
                 else: looks_like_block_target = True # All/Single targets blocks if component in custom cfg

            if looks_like_block_target: config_to_iterate = self.custom_block_config; target_is_blocks = True
            elif self.base_model_config and guide_component_name in self.base_model_config.components: config_to_iterate = self.base_model_config; target_is_blocks = False
            else: logger.warning(f"Comp. '{guide_component_name}' not found. Skipping."); continue

            target_type_str = "block" if target_is_blocks else "key"
            logger.debug(f"Guide: Comp='{guide_component_name}', Strat='{strategy}', Params={optimizable_base_params}, Target='{target_type_str}'")

            # --- Get list of block names or model keys for the target component ---
            try: items_in_component = list(config_to_iterate.components[guide_component_name].keys.keys())
            except KeyError: logger.warning(f"Comp. '{guide_component_name}' structure error. Skipping."); continue
            if not items_in_component: logger.warning(f"Comp. '{guide_component_name}' empty. Skipping."); continue

            # --- Apply strategy for EACH specified optimizable base parameter ---
            for base_param_name in optimizable_base_params:
                default_bounds_tuple = (0.0, 1.0)
                base_metadata = {"strategy": strategy, "target_type": target_type_str, "base_param": base_param_name, "component_name": guide_component_name}

                if strategy == "all":
                    for item_name in items_in_component:
                        generated_param_name = f"{item_name}_{base_param_name}"
                        if generated_param_name not in params_info:
                            params_info[generated_param_name] = {**base_metadata, "item_name": item_name, "bounds": default_bounds_tuple}
                elif strategy == "select":
                    patterns = component_config_from_guide.get("keys", [])
                    if not patterns: continue
                    for pattern in patterns:
                        for item_name in items_in_component:
                            if fnmatch.fnmatch(item_name, pattern):
                                generated_param_name = f"{item_name}_{base_param_name}"
                                if generated_param_name not in params_info:
                                    params_info[generated_param_name] = {**base_metadata, "item_name": item_name, "bounds": default_bounds_tuple}
                elif strategy == "group":
                    groups = component_config_from_guide.get("groups", [])
                    if not groups: continue
                    for group in groups:
                        group_name = group.get("name"); group_patterns = group.get("keys", [])
                        if not group_name or not group_patterns: continue
                        items_covered_by_group = [item for pattern in group_patterns for item in items_in_component if fnmatch.fnmatch(item, pattern)]
                        if items_covered_by_group:
                            generated_param_name = f"{group_name}_{base_param_name}"
                            if generated_param_name not in params_info:
                                params_info[generated_param_name] = {**base_metadata, "group_name": group_name, "items_covered": items_covered_by_group, "bounds": default_bounds_tuple}
                elif strategy == "single":
                    generated_param_name = f"{guide_component_name}_default_{base_param_name}"
                    if generated_param_name not in params_info:
                        params_info[generated_param_name] = {**base_metadata, "items_covered": items_in_component, "bounds": default_bounds_tuple}
                else: logger.warning(f"Unsupported strategy '{strategy}' for '{guide_component_name}'.")

        logger.info(f"Generated metadata for {len(params_info)} optimization parameters based on guide.")
        return params_info

    # V1.3 - Uses create_parameter_bounds_metadata
    def get_bounds(
        self,
        custom_bounds_config: Optional[Dict[str, Union[List[float], List[int], int, float]]] = None
    ) -> BoundsInfo:
        """Gets the final parameter info dict after applying custom bounds overrides."""
        params_info = self.create_parameter_bounds_metadata()
        validated_custom_bounds = self.validate_custom_bounds(custom_bounds_config or {})

        # Apply custom bounds overrides
        for param_name, info in params_info.items():
            if param_name in validated_custom_bounds:
                info['bounds'] = validated_custom_bounds[param_name]
            else: # Check for general override by base_param name
                 base_param_name = info.get('base_param')
                 # Check if the custom bound key EXACTLY matches a base parameter name
                 if base_param_name and base_param_name in validated_custom_bounds:
                     info['bounds'] = validated_custom_bounds[base_param_name]
                     # Log general override application here if desired

        # Add fixed parameters defined only in custom_bounds
        all_possible_base_params = {p_name for info in params_info.values() for p_name in info.get('base_param', [])} # Collect all base params used
        for custom_param, custom_bound_value in validated_custom_bounds.items():
             if custom_param not in params_info and custom_param not in all_possible_base_params:
                  # Add as a custom/fixed parameter if not already present and not a base override key
                  params_info[custom_param] = {
                       'bounds': custom_bound_value, 'strategy': 'custom_fixed',
                       'target_type': 'unknown', 'base_param': custom_param,
                  }

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