import logging
import re
from pathlib import Path

import sd_mecha
import fnmatch
import torch
import inspect  # Needed to inspect parameter types

from typing import Dict, List, Tuple, Union, Optional, Any
from omegaconf import DictConfig, ListConfig, OmegaConf
from sd_mecha.extensions import model_configs, merge_methods, merge_spaces  # Added imports
from sd_mecha.extensions.merge_methods import StateDict, ParameterData  # For type checking
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
                            logger.warning(
                                f"Merger: Ambiguous base ModelConfig inferred for {representative_model_path_str}. Possible matches: {config_names}. Picking first one arbitrarily.")
                            # You might want more sophisticated logic here, e.g., check against a preferred list
                            self.base_model_config = next(iter(best_set))  # Pick first for now
                    # --- MODIFICATION END ---
                    else:
                        raise ValueError(f"Merger: Cannot infer ModelConfig for {representative_model_path_str}")
                else:
                    raise ValueError(f"Merger: Cannot load dictionary for {representative_model_path_str}")
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

    # V1.6: Refactor and target type fix
    def create_parameter_bounds_metadata(self) -> BoundsInfo:
        """
        Creates parameter bounds AND metadata based on the optimization guide,
        supporting multiple strategies per component via a 'strategies' list
        and performing conflict detection.
        """
        params_info: BoundsInfo = {}
        assigned_items: Dict[Tuple[str, str], str] = {}

        guide_components = self.cfg.optimization_guide.get("components", [])
        if not guide_components or not isinstance(guide_components, (list, ListConfig)):
            logger.warning("No 'components' list found or invalid format in optimization_guide. No bounds generated.")
            return {}

        # Process each component
        for component_index, component_config_raw in enumerate(guide_components):
            component_params = self._process_component(component_index, component_config_raw, assigned_items)
            params_info.update(component_params)

        logger.info(f"Generated metadata for {len(params_info)} optimization parameters based on guide.")
        return params_info

    def _process_component(self, component_index: int, component_config_raw: Any,
                           assigned_items: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
        """Process a single component and all its strategies"""
        # Validate component structure
        if not isinstance(component_config_raw, (dict, DictConfig)):
            logger.warning(f"Skipping component entry at index {component_index}: Not a dictionary.")
            return {}

        # Convert to plain dict for easier access
        component_config = OmegaConf.to_container(component_config_raw, resolve=True) \
            if isinstance(component_config_raw, DictConfig) else component_config_raw

        guide_component_name = component_config.get("name")
        if not guide_component_name:
            logger.warning(f"Skipping component entry at index {component_index} due to missing 'name'.")
            return {}

        # Get component-level optimize_params
        component_optimize_params = component_config.get("optimize_params", [])
        if not isinstance(component_optimize_params, list):
            logger.warning(
                f"Invalid 'optimize_params' format for component '{guide_component_name}', must be a list. Using empty list.")
            component_optimize_params = []

        # Check for strategies list
        strategies_list_raw = component_config.get("strategies")
        if not strategies_list_raw or not isinstance(strategies_list_raw, list):
            logger.warning(
                f"Component '{guide_component_name}' is missing a valid 'strategies' list. Skipping this component.")
            return {}

        # Process all strategies for this component
        params_info = {}
        for strategy_index, strategy_config_raw in enumerate(strategies_list_raw):
            strategy_params = self._process_strategy(
                guide_component_name, component_optimize_params,
                strategy_index, strategy_config_raw, assigned_items
            )
            params_info.update(strategy_params)

        return params_info

    def _process_strategy(self, guide_component_name: str, component_optimize_params: List[str],
                          strategy_index: int, strategy_config_raw: Any,
                          assigned_items: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
        """Process a single strategy"""
        # Validate strategy structure
        if not isinstance(strategy_config_raw, (dict, DictConfig)):
            logger.warning(
                f"Invalid strategy entry format at index {strategy_index} in '{guide_component_name}'. Skipping.")
            return {}

        strategy_config = OmegaConf.to_container(strategy_config_raw, resolve=True) \
            if isinstance(strategy_config_raw, DictConfig) else strategy_config_raw

        strategy_type = strategy_config.get("type")
        if not strategy_type or strategy_type not in ["all", "select", "group", "single", "none"]:
            logger.warning(
                f"Missing or invalid strategy 'type' at index {strategy_index} in '{guide_component_name}'. Skipping.")
            return {}

        if strategy_type == "none":
            return {}

        # Determine target config and type
        config_to_iterate, target_is_blocks, target_type_str = self._determine_target_config(
            guide_component_name, strategy_config
        )
        if not config_to_iterate:
            return {}

        # Get component items
        items_in_component = self._get_component_items(config_to_iterate, guide_component_name)
        if not items_in_component:
            logger.warning(
                f"Component '{guide_component_name}' has no items in config '{config_to_iterate.identifier}'. Skipping.")
            return {}

        # Determine optimize_params for this strategy
        current_optimize_params = strategy_config.get("optimize_params", component_optimize_params)
        if not isinstance(current_optimize_params, list):
            logger.warning(
                f"Invalid 'optimize_params' for strategy '{strategy_type}' (index {strategy_index}) in '{guide_component_name}'. Using component default or skipping.")
            current_optimize_params = component_optimize_params

        if not current_optimize_params:
            logger.debug(
                f"No 'optimize_params' specified for strategy '{strategy_type}' (index {strategy_index}) in '{guide_component_name}'. Skipping strategy.")
            return {}

        logger.debug(
            f"Processing Component='{guide_component_name}', Strategy='{strategy_type}', Params={current_optimize_params}, Target='{target_type_str}' (Config: {config_to_iterate.identifier})")

        # Create base metadata
        base_metadata = {
            "strategy": strategy_type,
            "target_type": target_type_str,
            "component_name": guide_component_name
        }

        # Process strategy based on type
        if strategy_type == "all":
            return self._process_all_strategy(
                current_optimize_params, items_in_component, base_metadata,
                strategy_config, guide_component_name, assigned_items
            )
        elif strategy_type == "select":
            return self._process_select_strategy(
                current_optimize_params, items_in_component, base_metadata,
                strategy_config, guide_component_name, assigned_items
            )
        elif strategy_type == "group":
            return self._process_group_strategy(
                current_optimize_params, items_in_component, base_metadata,
                strategy_config, guide_component_name, assigned_items
            )
        elif strategy_type == "single":
            return self._process_single_strategy(
                current_optimize_params, items_in_component, base_metadata,
                strategy_config, guide_component_name, assigned_items
            )
        else:
            # This shouldn't happen due to earlier validation, but just in case
            logger.warning(f"Unknown strategy type '{strategy_type}' - this shouldn't happen!")
            return {}

    def _determine_target_config(self, guide_component_name: str, strategy_config: Dict[str, Any]) -> Tuple[
        Optional[Any], bool, str]:
        """Determine which config to use and target type based on strategy specification"""
        strategy_target_type = strategy_config.get("target_type")

        if strategy_target_type == "block":
            if self.custom_block_config and guide_component_name in self.custom_block_config.components():
                return self.custom_block_config, True, "block"
            else:
                logger.warning(
                    f"Strategy specifies target_type='block' but component '{guide_component_name}' not found in custom_block_config. Skipping strategy.")
                return None, False, ""

        elif strategy_target_type == "key":
            if self.base_model_config and guide_component_name in self.base_model_config.components():
                return self.base_model_config, False, "key"
            else:
                logger.warning(
                    f"Strategy specifies target_type='key' but component '{guide_component_name}' not found in base_model_config. Skipping strategy.")
                return None, False, ""

        else:
            # Original logic - auto-determine based on component availability
            if self.custom_block_config and guide_component_name in self.custom_block_config.components():
                return self.custom_block_config, True, "block"
            elif self.base_model_config and guide_component_name in self.base_model_config.components():
                return self.base_model_config, False, "key"
            else:
                logger.warning(f"Component '{guide_component_name}' not found in known configs. Skipping component.")
                return None, False, ""

    def _get_component_items(self, config_to_iterate: Any, guide_component_name: str) -> List[str]:
        """Get list of items (blocks/keys) for a component"""
        try:
            if guide_component_name not in config_to_iterate.components():
                raise KeyError(
                    f"Component name '{guide_component_name}' not found within config '{config_to_iterate.identifier}'")

            component_obj = config_to_iterate.components()[guide_component_name]
            return list(component_obj.keys().keys())

        except (KeyError, AttributeError) as e_struct:
            logger.warning(
                f"Error accessing items for component '{guide_component_name}' using config '{config_to_iterate.identifier}': {e_struct}. Skipping component.")
            return []

    def _process_all_strategy(self, current_optimize_params: List[str], items_in_component: List[str],
                              base_metadata: Dict[str, Any], strategy_config: Dict[str, Any],
                              guide_component_name: str, assigned_items: Dict[Tuple[str, str], str]) -> Dict[str, Any]:
        """Process 'all' strategy type"""
        params_info = {}
        default_bounds_tuple = (0.0, 1.0)
        strategy_identifier = "all"

        for base_param_name in current_optimize_params:
            for item_name in items_in_component:
                assignment_key = (base_param_name, item_name)
                if assignment_key in assigned_items:
                    logger.error(
                        f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}'. Skipping.")
                    continue

                assigned_items[assignment_key] = strategy_identifier
                generated_param_name = f"{item_name}_{base_param_name}"
                params_info[generated_param_name] = {
                    **base_metadata,
                    "item_name": item_name,
                    "base_param": base_param_name,
                    "bounds": default_bounds_tuple
                }

        return params_info

    def _process_select_strategy(self, current_optimize_params: List[str], items_in_component: List[str],
                                 base_metadata: Dict[str, Any], strategy_config: Dict[str, Any],
                                 guide_component_name: str, assigned_items: Dict[Tuple[str, str], str]) -> Dict[
        str, Any]:
        """Process 'select' strategy type"""
        params_info = {}
        default_bounds_tuple = (0.0, 1.0)
        patterns = strategy_config.get("keys", [])

        if not patterns or not isinstance(patterns, list):
            logger.warning(
                f"'select' strategy needs a valid 'keys' list in '{guide_component_name}'. Skipping for all params.")
            return params_info

        strategy_identifier = "select"

        for base_param_name in current_optimize_params:
            items_matched_in_strategy = 0
            for pattern in patterns:
                for item_name in items_in_component:
                    if fnmatch.fnmatch(item_name, pattern):
                        assignment_key = (base_param_name, item_name)
                        if assignment_key in assigned_items:
                            logger.error(
                                f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}' pattern '{pattern}'. Skipping.")
                            continue

                        assigned_items[assignment_key] = f"{strategy_identifier}:{pattern}"
                        generated_param_name = f"{item_name}_{base_param_name}"
                        params_info[generated_param_name] = {
                            **base_metadata,
                            "item_name": item_name,
                            "base_param": base_param_name,
                            "bounds": default_bounds_tuple
                        }
                        items_matched_in_strategy += 1

            if items_matched_in_strategy == 0:
                logger.warning(
                    f"'select' strategy patterns {patterns} did not match any items in component '{guide_component_name}' for param '{base_param_name}'.")

        return params_info

    def _process_group_strategy(self, current_optimize_params: List[str], items_in_component: List[str],
                                base_metadata: Dict[str, Any], strategy_config: Dict[str, Any],
                                guide_component_name: str, assigned_items: Dict[Tuple[str, str], str]) -> Dict[
        str, Any]:
        """Process 'group' strategy type"""
        params_info = {}
        default_bounds_tuple = (0.0, 1.0)
        groups = strategy_config.get("groups", [])

        if not groups or not isinstance(groups, list):
            logger.warning(
                f"'group' strategy needs a valid 'groups' list in '{guide_component_name}'. Skipping for all params.")
            return params_info

        for base_param_name in current_optimize_params:
            logger.debug(
                f"PARAMETER_HANDLER: Processing group strategy for base_param: '{base_param_name}' in component '{guide_component_name}'")

            for group_index, group_raw in enumerate(groups):
                if not isinstance(group_raw, (dict, DictConfig)):
                    logger.warning(
                        f"Invalid group format at index {group_index} in '{guide_component_name}'. Skipping.")
                    continue

                group = OmegaConf.to_container(group_raw, resolve=True) if isinstance(group_raw,
                                                                                      DictConfig) else group_raw

                group_name = group.get("name")
                group_patterns = group.get("keys", [])

                if not group_name or not isinstance(group_patterns, list):
                    logger.warning(
                        f"Invalid group format (missing name or keys list) for group at index {group_index} in '{guide_component_name}'. Skipping group.")
                    continue

                strategy_identifier = f"group:{group_name}"

                # Check conflicts FIRST for all items in the group
                temp_items_for_group = []
                conflict_found_for_group = False

                for pattern in group_patterns:
                    for item_name in items_in_component:
                        if fnmatch.fnmatch(item_name, pattern):
                            temp_items_for_group.append(item_name)
                            assignment_key = (base_param_name, item_name)
                            if assignment_key in assigned_items:
                                logger.error(
                                    f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}'. Group will be skipped.")
                                conflict_found_for_group = True
                                break
                    if conflict_found_for_group:
                        break

                if conflict_found_for_group:
                    continue

                # If no conflicts, add the group parameter and mark items assigned
                if temp_items_for_group:
                    generated_param_name = f"{group_name}_{base_param_name}"
                    logger.debug(
                        f"PARAMETER_HANDLER: Generating group param: '{generated_param_name}' covering {len(temp_items_for_group)} items")  # ← ADD THIS
                    params_info[generated_param_name] = {
                        **base_metadata,
                        "strategy": "group",  # Ensure strategy is set correctly
                        "group_name": group_name,
                        "base_param": base_param_name,
                        "items_covered": list(set(temp_items_for_group)),  # Use unique list
                        "bounds": default_bounds_tuple
                    }

                    # Mark all items as assigned by this group
                    for item_name in temp_items_for_group:
                        assigned_items[(base_param_name, item_name)] = strategy_identifier
                    logger.debug(
                        f"PARAMETER_HANDLER: Marked {len(temp_items_for_group)} items as assigned for '{generated_param_name}'")  # ← ADD THIS
                else:
                    logger.warning(
                        f"Group '{group_name}' patterns {group_patterns} did not match any items in component '{guide_component_name}' for param '{base_param_name}'.")

        return params_info

    def _process_single_strategy(self, current_optimize_params: List[str], items_in_component: List[str],
                                 base_metadata: Dict[str, Any], strategy_config: Dict[str, Any],
                                 guide_component_name: str, assigned_items: Dict[Tuple[str, str], str]) -> Dict[
        str, Any]:
        """Process 'single' strategy type"""
        params_info = {}
        default_bounds_tuple = (0.0, 1.0)

        for base_param_name in current_optimize_params:
            logger.debug(
                f"PARAMETER_HANDLER: Processing single strategy for base_param: '{base_param_name}' in component '{guide_component_name}'")

            group_name = f"{guide_component_name}_single"
            strategy_identifier = f"single:{group_name}"
            conflict_found_for_single = False

            # Check conflicts FIRST
            if not items_in_component:
                logger.warning(
                    f"PARAMETER_HANDLER: No items found for component '{guide_component_name}' when processing '{base_param_name}'. Skipping single strategy.")
                continue

            for item_name in items_in_component:
                assignment_key = (base_param_name, item_name)
                if assignment_key in assigned_items:
                    logger.error(
                        f"Conflict for item '{item_name}' (param '{base_param_name}')! Assigned by '{assigned_items[assignment_key]}', cannot assign by '{strategy_identifier}'. Skipping.")
                    conflict_found_for_single = True
                    break

            if conflict_found_for_single:
                continue

            # If no conflicts, add the single parameter and mark items
            generated_param_name = f"{group_name}_{base_param_name}"
            logger.debug(f"PARAMETER_HANDLER: Generating single param: '{generated_param_name}'")

            params_info[generated_param_name] = {
                **base_metadata,
                "strategy": "single",
                "group_name": group_name,
                "base_param": base_param_name,
                "items_covered": items_in_component,
                "bounds": default_bounds_tuple
            }

            # Mark all items as assigned
            for item_name in items_in_component:
                assigned_items[(base_param_name, item_name)] = strategy_identifier

            logger.debug(
                f"PARAMETER_HANDLER: Marked {len(items_in_component)} items as assigned for '{generated_param_name}'")

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
                params_info[custom_key]['bounds'] = custom_value  # Override specific param
                logger.info(
                    f"  Overrode bounds for specific param '{custom_key}' from {original_bounds} to {custom_value} via custom_bounds.")
                updated_params_count += 1
                found_match = True
            # --- PRIORITY 2: Check for BASE parameter name match ---
            elif custom_key in base_param_map:
                # Apply to all parameters sharing this base name ONLY IF not overridden specifically
                for param_name in base_param_map[custom_key]:
                    # Check if this specific param wasn't already overridden by exact name match
                    if param_name not in validated_custom_bounds:
                        original_bounds = params_info[param_name].get('bounds')
                        params_info[param_name]['bounds'] = custom_value  # Update bounds
                        logger.debug(
                            f"  Updated bounds for '{param_name}' (base: {custom_key}) from {original_bounds} to {custom_value} via custom_bounds base match.")
                        updated_params_count += 1  # Count updates even if value is same
                found_match = True  # Mark base param as handled

            if not found_match:
                logger.debug(
                    f"Custom bound key '{custom_key}' did not match any generated parameter name or base_param. It may be used as a fixed keyword argument if applicable.")

        # Step 3: Extract bounds for the optimizer (remains the same)
        optimizer_pbounds = {
            param_name: info['bounds']
            for param_name, info in params_info.items()
            if 'bounds' in info
        }

        logger.info(
            f"--- Final {len(params_info)} Optimization Parameter Details (Bounds Updated: {updated_params_count}) ---")
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

    # V1.2 - Better validation, more types and conflict warnings
    @staticmethod
    def validate_custom_bounds(
            custom_bounds: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if custom_bounds is None: return {}

        validated_bounds: Dict[str, Any] = {}
        for param_name, bound_config in custom_bounds.items():
            try:
                # Case 0: strings as tuple
                if isinstance(bound_config, str):
                    # This regex matches "(num1, num2)" and captures the numbers
                    match = re.match(r'^\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)$', bound_config.strip())
                    if match:
                        # Extract the two numbers as strings
                        num1_str, num2_str = match.groups()
                        # Convert to int if possible, otherwise float, to preserve type
                        try:
                            num1 = int(num1_str)
                        except ValueError:
                            num1 = float(num1_str)
                        try:
                            num2 = int(num2_str)
                        except ValueError:
                            num2 = float(num2_str)
                        # Overwrite bound_config with the parsed tuple
                        bound_config = (num1, num2)
                        logger.debug(f"Successfully parsed string '{param_name}' into tuple: {bound_config}")
                    else:
                        # If it's a string but doesn't match, it's an error.
                        raise ValueError(f"String value '{bound_config}' is not a valid tuple format '(min, max)'.")

                # Case 1: Rich dictionary format (for ranges with options)
                if isinstance(bound_config, (dict, DictConfig)):
                    if "range" not in bound_config:
                        raise ValueError("Rich bound format requires a 'range' key.")

                    validated_param = {}
                    range_val = list(bound_config["range"])
                    if len(range_val) != 2: raise ValueError("'range' must have 2 values [min, max].")
                    validated_param["range"] = (float(range_val[0]), float(range_val[1]))

                    if "log" in bound_config: validated_param["log"] = bool(bound_config["log"])
                    if "step" in bound_config: validated_param["step"] = float(bound_config["step"])

                    # --- ADDED: Edge Case Validation ---
                    if validated_param.get("log") and validated_param.get("step"):
                        logger.warning(
                            f"For parameter '{param_name}', both 'log' and 'step' are specified. "
                            f"Optuna will prioritize 'log' sampling over the step interval."
                        )
                    # --- END OF ADDITION ---

                    validated_bounds[param_name] = validated_param

                # Case 2: List format (always for categorical)
                elif isinstance(bound_config, (list, ListConfig)):
                    validated_bounds[param_name] = list(bound_config)

                # Case 3: Tuple format (simple continuous range)
                elif isinstance(bound_config, tuple):
                    if len(bound_config) != 2: raise ValueError("Range tuple must have 2 values (min, max).")
                    validated_bounds[param_name] = bound_config

                # Case 4: Fixed value
                elif isinstance(bound_config, (int, float)):
                    validated_bounds[param_name] = bound_config

                else:
                    raise ValueError(
                        "Bound must be a tuple (range), list (categorical), dict (advanced), int, or float.")

            except Exception as e:
                logger.error(f"Invalid custom bound for '{param_name}': {bound_config}. Error: {e}. Skipping.")

        return validated_bounds
