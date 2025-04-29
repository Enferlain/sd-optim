import logging
from pathlib import Path

import sd_mecha
import fnmatch
import torch
import inspect # Needed to inspect parameter types

from typing import Dict, List, Tuple, Union, Optional, Any
from omegaconf import DictConfig, ListConfig, OmegaConf
from sd_mecha.extensions import model_configs, merge_methods, merge_spaces # Added imports
from sd_mecha.extensions.merge_methods import StateDict, ParameterData # For type checking
from sd_mecha.recipe_nodes import RecipeNode, ModelRecipeNode

from sd_optim import utils

logger = logging.getLogger(__name__)


# Define a type alias for clarity (optional)
BoundsInfo = Dict[str, Dict[str, Any]]

class ParameterHandler:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if not cfg.model_paths: raise ValueError("'model_paths' cannot be empty.")

        self.base_model_config: Optional[model_configs.ModelConfig] = None
        representative_model_path_str = cfg.model_paths[0]
        logger.info(f"Merger: Inferring base ModelConfig from: {representative_model_path_str}")
        try:
            self.models_dir = Path(representative_model_path_str).resolve().parent
            if not self.models_dir.is_dir(): raise FileNotFoundError(f"Merger: Dir not found: {self.models_dir}")
            logger.info(f"Merger: Determined models directory: {self.models_dir}")

            rep_model_node: RecipeNode = sd_mecha.model(representative_model_path_str)
            with sd_mecha.open_input_dicts(rep_model_node, [self.models_dir]):
                if isinstance(rep_model_node, ModelRecipeNode) and rep_model_node.state_dict:
                    # --- MODIFICATION START ---
                    # infer_model_configs now returns List[Set[ModelConfig]]
                    inferred_sets = sd_mecha.infer_model_configs(rep_model_node.state_dict.keys())
                    if inferred_sets:
                         # Get the set with the highest affinity (first element)
                         best_set = inferred_sets[0]
                         # If there's only one config in the best set, use it
                         if len(best_set) == 1:
                              self.base_model_config = next(iter(best_set))
                              logger.info(f"Merger: Inferred base ModelConfig: {self.base_model_config.identifier}")
                         else:
                              # Handle ambiguity if multiple configs match equally well
                              config_names = {c.identifier for c in best_set}
                              logger.warning(f"Merger: Ambiguous base ModelConfig inferred for {representative_model_path_str}. Possible matches: {config_names}. Picking first one arbitrarily.")
                              # You might want more sophisticated logic here, e.g., check against a preferred list
                              self.base_model_config = next(iter(best_set)) # Pick first for now
                    # --- MODIFICATION END ---
                    else: raise ValueError(f"Merger: Cannot infer ModelConfig for {representative_model_path_str}")
                else: raise ValueError(f"Merger: Cannot load dictionary for {representative_model_path_str}")
        except Exception as e:
             logger.error(f"Merger: Error inferring base config or models_dir: {e}", exc_info=True)
             raise ValueError("Merger could not determine base ModelConfig or models directory.") from e

        # Load Custom Config remains the same...
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

    # V1.5: Supports nested strategies per component + Conflict Detection
    def create_parameter_bounds_metadata(self) -> BoundsInfo:
        """
        Creates parameter bounds AND metadata based on the optimization guide,
        supporting multiple strategies per component via a 'strategies' list
        and performing conflict detection.
        """
        params_info: BoundsInfo = {}
        # --- Track item assignments to detect conflicts ---
        # Key: (base_param_name, item_name), Value: "strategy_type:id" (e.g., "select", "group:unet_early")
        assigned_items: Dict[Tuple[str, str], str] = {}

        guide_components = self.cfg.optimization_guide.get("components", [])
        if not guide_components or not isinstance(guide_components, (list, ListConfig)):
             logger.warning("No 'components' list found or invalid format in optimization_guide. No bounds generated.")
             return {} # Return empty dict

        # --- Outer loop: Iterate through defined components ---
        for component_index, component_config_from_guide_raw in enumerate(guide_components):
            # Ensure it's a dictionary-like structure
            if not isinstance(component_config_from_guide_raw, (dict, DictConfig)):
                 logger.warning(f"Skipping component entry at index {component_index}: Not a dictionary.")
                 continue
            # Convert to plain dict if OmegaConf for easier .get usage with defaults
            component_config_from_guide = OmegaConf.to_container(component_config_from_guide_raw, resolve=True) \
                                           if isinstance(component_config_from_guide_raw, DictConfig) else component_config_from_guide_raw

            guide_component_name = component_config_from_guide.get("name")
            if not guide_component_name:
                 logger.warning(f"Skipping component entry at index {component_index} due to missing 'name'.")
                 continue

            # Get component-level optimize_params (acts as default for strategies)
            component_optimize_params = component_config_from_guide.get("optimize_params", [])
            if not isinstance(component_optimize_params, list):
                logger.warning(f"Invalid 'optimize_params' format for component '{guide_component_name}', must be a list. Using empty list.")
                component_optimize_params = []

            # --- Check for the new 'strategies' list ---
            strategies_list_raw = component_config_from_guide.get("strategies")
            if not strategies_list_raw or not isinstance(strategies_list_raw, list):
                logger.warning(f"Component '{guide_component_name}' is missing a valid 'strategies' list. Skipping this component.")
                continue

            # --- Determine Target Type ---
            config_to_iterate: Optional[model_configs.ModelConfig] = None
            target_is_blocks: bool = False
            # --- MODIFICATION: Use method calls ---
            if self.custom_block_config and guide_component_name in self.custom_block_config.components():
                 config_to_iterate = self.custom_block_config
                 target_is_blocks = True
            elif self.base_model_config and guide_component_name in self.base_model_config.components():
                 config_to_iterate = self.base_model_config
                 target_is_blocks = False
            # --- END MODIFICATION ---
            else:
                 logger.warning(f"Component '{guide_component_name}' not found in known configs. Skipping component.")
                 continue

            target_type_str = "block" if target_is_blocks else "key"

            # --- Get list of block/key items ---
            try:
                # --- MODIFICATION: Use method calls ---
                if guide_component_name not in config_to_iterate.components():
                     raise KeyError(f"Component name '{guide_component_name}' not found within config '{config_to_iterate.identifier}'")
                # Access the ModelComponent object first, then call its keys() method
                component_obj = config_to_iterate.components()[guide_component_name]
                items_in_component = list(component_obj.keys().keys())
                # --- END MODIFICATION ---
            except (KeyError, AttributeError) as e_struct:
                 logger.warning(f"Error accessing items for component '{guide_component_name}' using config '{config_to_iterate.identifier}': {e_struct}. Skipping component.")
                 continue
            if not items_in_component:
                logger.warning(f"Component '{guide_component_name}' has no items in config '{config_to_iterate.identifier}'. Skipping component.")
                continue

            # --- Inner loop: Iterate through strategies defined for this component ---
            processed_groups_in_component = set() # Track groups processed within this component run
            for strategy_index, strategy_config_raw in enumerate(strategies_list_raw):
                if not isinstance(strategy_config_raw, (dict, DictConfig)):
                    logger.warning(f"Invalid strategy entry format at index {strategy_index} in '{guide_component_name}'. Skipping.")
                    continue
                strategy_config = OmegaConf.to_container(strategy_config_raw, resolve=True) \
                                 if isinstance(strategy_config_raw, DictConfig) else strategy_config_raw

                strategy_type = strategy_config.get("type")
                if not strategy_type or strategy_type not in ["all", "select", "group", "single", "none"]:
                    logger.warning(f"Missing or invalid strategy 'type' at index {strategy_index} in '{guide_component_name}'. Skipping.")
                    continue
                if strategy_type == "none": continue

                # Determine optimize_params for this specific strategy
                current_optimize_params = strategy_config.get("optimize_params", component_optimize_params)
                if not isinstance(current_optimize_params, list):
                    logger.warning(f"Invalid 'optimize_params' for strategy '{strategy_type}' (index {strategy_index}) in '{guide_component_name}'. Using component default or skipping.")
                    current_optimize_params = component_optimize_params # Fallback to component default if invalid format
                if not current_optimize_params:
                    logger.debug(f"No 'optimize_params' specified for strategy '{strategy_type}' (index {strategy_index}) in '{guide_component_name}'. Skipping strategy.")
                    continue

                logger.debug(f"Processing Component='{guide_component_name}', Strategy='{strategy_type}', Params={current_optimize_params}, Target='{target_type_str}'")

                # --- Apply strategy for EACH base_param for THIS strategy ---
                for base_param_name in current_optimize_params:
                    default_bounds_tuple = (0.0, 1.0)
                    base_metadata = {"strategy": strategy_type, "target_type": target_type_str, "base_param": base_param_name, "component_name": guide_component_name}

                    # --- Logic based on strategy_type ---
                    if strategy_type == "all":
                        strategy_identifier = "all"
                        for item_name in items_in_component:
                            assignment_key = (base_param_name, item_name)
                            if assignment_key in assigned_items:
                                logger.error(f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}'. Skipping.")
                                continue
                            assigned_items[assignment_key] = strategy_identifier
                            generated_param_name = f"{item_name}_{base_param_name}" # Keep unique name
                            params_info[generated_param_name] = {**base_metadata, "item_name": item_name, "bounds": default_bounds_tuple}

                    elif strategy_type == "select":
                        patterns = strategy_config.get("keys", [])
                        if not patterns or not isinstance(patterns, list):
                             logger.warning(f"'select' strategy needs a valid 'keys' list in '{guide_component_name}'. Skipping for '{base_param_name}'.")
                             continue
                        strategy_identifier = "select"
                        items_matched_in_strategy = 0
                        for pattern in patterns:
                            for item_name in items_in_component:
                                if fnmatch.fnmatch(item_name, pattern):
                                    assignment_key = (base_param_name, item_name)
                                    if assignment_key in assigned_items:
                                        logger.error(f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}' pattern '{pattern}'. Skipping.")
                                        continue
                                    assigned_items[assignment_key] = f"{strategy_identifier}:{pattern}" # Store pattern for info
                                    generated_param_name = f"{item_name}_{base_param_name}" # Keep unique name
                                    params_info[generated_param_name] = {**base_metadata, "item_name": item_name, "bounds": default_bounds_tuple}
                                    items_matched_in_strategy += 1
                        if items_matched_in_strategy == 0:
                             logger.warning(f"'select' strategy patterns {patterns} did not match any items in component '{guide_component_name}' for param '{base_param_name}'.")

                    elif strategy_type == "group":
                        groups = strategy_config.get("groups", [])
                        if not groups or not isinstance(groups, list):
                             logger.warning(f"'group' strategy needs a valid 'groups' list in '{guide_component_name}'. Skipping for '{base_param_name}'.")
                             continue
                        for group_index, group_raw in enumerate(groups):
                            if not isinstance(group_raw, (dict, DictConfig)): logger.warning(f"Invalid group format at index {group_index} in '{guide_component_name}'. Skipping."); continue
                            group = OmegaConf.to_container(group_raw, resolve=True) if isinstance(group_raw, DictConfig) else group_raw

                            group_name = group.get("name"); group_patterns = group.get("keys", [])
                            if not group_name or not isinstance(group_patterns, list): logger.warning(f"Invalid group format (missing name or keys list) for group at index {group_index} in '{guide_component_name}'. Skipping group."); continue

                            strategy_identifier = f"group:{group_name}"
                            items_covered_by_group = []
                            conflict_found_for_group = False
                            # Check conflicts FIRST for all items in the group
                            temp_items_for_group = []
                            for pattern in group_patterns:
                                for item_name in items_in_component:
                                    if fnmatch.fnmatch(item_name, pattern):
                                         temp_items_for_group.append(item_name)
                                         assignment_key = (base_param_name, item_name)
                                         if assignment_key in assigned_items:
                                             logger.error(f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}'. Group will be skipped.")
                                             conflict_found_for_group = True
                                             break # No need to check other items in this pattern
                                if conflict_found_for_group: break # No need to check other patterns
                            if conflict_found_for_group: continue # Skip this group entirely if conflict

                            # If no conflicts, add the group parameter and mark items assigned
                            if temp_items_for_group:
                                generated_param_name = f"{group_name}_{base_param_name}"
                                # Add group param info only if not already added by another param in this strategy run
                                # if group_name not in processed_groups_in_component: # <<< DELETE THIS LINE
                                params_info[generated_param_name] = { # <<< INDENT THIS BLOCK BACK
                                    **base_metadata,
                                    "strategy": "group", # Ensure strategy is set correctly
                                    "group_name": group_name,
                                    "items_covered": list(set(temp_items_for_group)), # Use unique list
                                    "bounds": default_bounds_tuple
                                }
                                # processed_groups_in_component.add(group_name)  # <<< DELETE THIS LINE

                                # Mark all items as assigned by this group
                                for item_name in temp_items_for_group:
                                    assigned_items[(base_param_name, item_name)] = strategy_identifier
                            else:
                                 logger.warning(f"Group '{group_name}' patterns {group_patterns} did not match any items in component '{guide_component_name}' for param '{base_param_name}'.")

                    elif strategy_type == "single":
                        # --- ADD LOGGING ---
                        logger.debug(f"PARAMETER_HANDLER: Processing single strategy for base_param: '{base_param_name}' in component '{guide_component_name}'")
                        # --- END LOGGING ---

                        group_name = f"{guide_component_name}_single"
                        strategy_identifier = f"single:{group_name}"
                        items_covered_by_single = []
                        conflict_found_for_single = False

                        # Check conflicts FIRST
                        if not items_in_component: # Add check if component item list is empty
                             logger.warning(f"PARAMETER_HANDLER: No items found for component '{guide_component_name}' when processing '{base_param_name}'. Skipping single strategy.")
                             continue

                        for item_name in items_in_component:
                             assignment_key = (base_param_name, item_name)
                             if assignment_key in assigned_items:
                                 logger.error(f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}'. Skipping.")
                                 conflict_found_for_single = True
                                 break
                        if conflict_found_for_single: continue

                        # If no conflicts, add the single parameter and mark items
                        generated_param_name = f"{group_name}_{base_param_name}"
                        # --- ADD LOGGING ---
                        logger.debug(f"PARAMETER_HANDLER: Generating single param: '{generated_param_name}'")
                        # --- END LOGGING ---
                        params_info[generated_param_name] = {
                             **base_metadata,
                             "strategy": "single",
                             "group_name": group_name,
                             "items_covered": items_in_component,
                             "bounds": default_bounds_tuple
                         }

                        # Mark all items as assigned
                        for item_name in items_in_component:
                            assigned_items[(base_param_name, item_name)] = strategy_identifier
                        # --- ADD LOGGING ---
                        logger.debug(f"PARAMETER_HANDLER: Marked {len(items_in_component)} items as assigned for '{generated_param_name}'")
                        # --- END LOGGING ---

        logger.info(f"Generated metadata for {len(params_info)} optimization parameters based on guide.")
        return params_info

    # V1.6: Apply custom bounds by specific name OR base name
    def get_bounds(
        self,
        custom_bounds_config: Optional[Dict[str, Union[List[float], List[int], int, float]]] = None
    ) -> Tuple[BoundsInfo, Dict[str, Union[Tuple[float, float], float, int, List]]]:
        # Step 1: Generate base param_info from strategies (as before)
        params_info: BoundsInfo = self.create_parameter_bounds_metadata()
        if not params_info: return {}, {}

        # Step 2: Apply custom bounds overrides
        validated_custom_bounds = self.validate_custom_bounds(custom_bounds_config or {})
        updated_params_count = 0

        # Create lookup by base_param for efficiency
        base_param_map = {}
        for param_name, info in params_info.items():
            base_param = info.get('base_param')
            if base_param:
                base_param_map.setdefault(base_param, []).append(param_name)

        for custom_key, custom_value in validated_custom_bounds.items():
            found_match = False
            # --- PRIORITY 1: Check for EXACT optimizer parameter name match ---
            if custom_key in params_info:
                original_bounds = params_info[custom_key].get('bounds')
                params_info[custom_key]['bounds'] = custom_value # Override specific param
                logger.info(f"  Overrode bounds for specific param '{custom_key}' from {original_bounds} to {custom_value} via custom_bounds.")
                updated_params_count += 1
                found_match = True
            # --- PRIORITY 2: Check for BASE parameter name match ---
            elif custom_key in base_param_map:
                # Apply to all parameters sharing this base name ONLY IF not overridden specifically
                for param_name in base_param_map[custom_key]:
                    # Check if this specific param wasn't already overridden by exact name match
                    if param_name not in validated_custom_bounds:
                         original_bounds = params_info[param_name].get('bounds')
                         params_info[param_name]['bounds'] = custom_value # Update bounds
                         logger.debug(f"  Updated bounds for '{param_name}' (base: {custom_key}) from {original_bounds} to {custom_value} via custom_bounds base match.")
                         updated_params_count += 1 # Count updates even if value is same
                found_match = True # Mark base param as handled

            if not found_match:
                 logger.debug(f"Custom bound key '{custom_key}' did not match any generated parameter name or base_param. It may be used as a fixed keyword argument if applicable.")

        # Step 3: Extract bounds for the optimizer (remains the same)
        optimizer_pbounds = {
            param_name: info['bounds']
            for param_name, info in params_info.items()
            if 'bounds' in info
        }

        logger.info(f"--- Final {len(params_info)} Optimization Parameter Details (Bounds Updated: {updated_params_count}) ---")
        items_to_log = list(params_info.items())
        log_limit = 100
        if len(items_to_log) > log_limit * 2:
             for name, info in items_to_log[:log_limit]: logger.info(f"{name}: {info}")
             logger.info("...")
             for name, info in items_to_log[-log_limit:]: logger.info(f"{name}: {info}")
        else:
             for name, info in items_to_log: logger.info(f"{name}: {info}")
        logger.info("----------------------------------------------------")


        # Return the full metadata (with updated bounds) AND the specific bounds for the optimizer
        return params_info, optimizer_pbounds

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