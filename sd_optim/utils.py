# bounds.py - Version 1.1 - Added custom_block_config_id support and custom_blocks support

import functools
import gc
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import sys
import textwrap
from fnmatch import fnmatch

import torch
import safetensors.torch
import sd_mecha
import logging
import ast
import torch
import yaml
import threading

from pathlib import Path
from typing import List, Tuple, TypeVar, Dict, Set, Union, Any, ClassVar, Optional, MutableMapping, Mapping
from dataclasses import field, dataclass
from torch import Tensor

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
from pynput import keyboard
from copy import deepcopy

from sd_optim.merge_methods import MergeMethods
from sd_mecha.extensions import model_configs, merge_methods
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict, T, MergeMethod
from sd_mecha.streaming import StateDictKeyError
from sd_mecha.extensions.model_configs import ModelConfigImpl, KeyMetadata # Need these for creation
from omegaconf import DictConfig, OmegaConf # If reading from Hydra config

logger = logging.getLogger(__name__)


########################################
### --- Custom model config code --- ###
########################################
# --- Re-use or define the regexes needed for mapping ---
# (Copied from sd-mecha's internal block converters for consistency)
re_inp = re.compile(r"\.input_blocks\.(\d+)\.")  # Matches .input_blocks.0. to .input_blocks.8.
re_mid = re.compile(r"\.middle_block\.(\d+)\.")  # Matches .middle_block.0.
re_out = re.compile(r"\.output_blocks\.(\d+)\.") # Matches .output_blocks.0. to .output_blocks.8.

# Define T if not already imported from merge_methods
# T = TypeVar("T")

@merge_method(
    identifier="convert_sdxl_optim_blocks_to_sdxl_sgm", # Unique ID for this conversion
    is_conversion=True # Mark this as a conversion function for sd_mecha.convert
)
def convert_sdxl_optim_blocks_to_sdxl_sgm(
    # Input: A StateDict containing block names and their values (e.g., alpha values)
    # The model_config MUST match the identifier of our registered custom block config
    blocks_dict: Parameter(StateDict[T], model_config="sdxl-optim_blocks"),
    # kwargs will contain 'key', the name of the TARGET key (from sdxl-sgm)
    **kwargs,
) -> Return(T, model_config="sdxl-sgm"): # Output: The value for the target key, config is sdxl-sgm
    """
    Maps an sdxl-sgm key back to a block name defined in sdxl-optim_blocks
    and returns the corresponding value from the blocks_dict.
    """
    target_key = kwargs["key"]
    block_name = None # Initialize block name

    # --- Mapping Logic ---
    # This needs to correctly categorize every possible key from sdxl-sgm
    # back into one of the block names defined in sdxl-optim_blocks.yaml

    # 1. UNET Component Keys (prefix: model.diffusion_model.)
    if target_key.startswith("model.diffusion_model."):
        if ".time_embed" in target_key:
            block_name = "UNET_TIME_EMBED"
        elif ".out." in target_key: # Final output convs
            block_name = "UNET_OUT"
        elif m := re_inp.search(target_key):
            block_num = int(m.group(1))
            if 0 <= block_num <= 8:
                block_name = f"UNET_IN{block_num:02d}"
        elif re_mid.search(target_key):
            block_name = "UNET_M00"
        elif m := re_out.search(target_key):
            block_num = int(m.group(1))
            if 0 <= block_num <= 8:
                block_name = f"UNET_OUT{block_num:02d}"

        # Fallback for UNET keys not caught above
        if block_name is None:
            block_name = "UNET_ELSE"

    # 2. CLIP-L (Text Encoder 1) Keys (prefix: conditioner.embedders.0.transformer.text_model.)
    elif target_key.startswith("conditioner.embedders.0.transformer.text_model."):
        if ".embeddings." in target_key:
            block_name = "CLIP_L_EMBEDDING"
        elif ".final_layer_norm." in target_key:
            block_name = "CLIP_L_FINAL_NORM"
        elif ".encoder.layers." in target_key:
            try:
                # Extract layer number
                layer_num_match = re.search(r"\.layers\.(\d+)\.", target_key)
                if layer_num_match:
                    layer_num = int(layer_num_match.group(1))
                    if 0 <= layer_num <= 11:
                         block_name = f"CLIP_L_IN{layer_num:02d}"
            except (ValueError, IndexError):
                 pass # Fallback below handles errors

        # Fallback for CLIP-L keys
        if block_name is None:
            block_name = "CLIP_L_ELSE"

    # 3. CLIP-G (Text Encoder 2) Keys (prefix: conditioner.embedders.1.model.)
    elif target_key.startswith("conditioner.embedders.1.model."):
        if ".token_embedding." in target_key or ".positional_embedding" in target_key:
             block_name = "CLIP_G_EMBEDDING"
        elif ".text_projection" in target_key:
             block_name = "CLIP_G_TEXT_PROJECTION"
        elif ".ln_final." in target_key:
             block_name = "CLIP_G_LN_FINAL"
        elif ".transformer.resblocks." in target_key or ".encoder.layers." in target_key: # Handle different naming conventions
            try:
                # Extract layer number (might be resblocks.X or layers.X)
                layer_num_match = re.search(r"\.(?:resblocks|layers)\.(\d+)\.", target_key)
                if layer_num_match:
                    layer_num = int(layer_num_match.group(1))
                    if 0 <= layer_num <= 31: # Assuming 0-31 blocks for CLIP-G
                         block_name = f"CLIP_G_IN{layer_num:02d}"
            except (ValueError, IndexError):
                 pass # Fallback below

        # Fallback for CLIP-G keys
        if block_name is None:
            block_name = "CLIP_G_ELSE"

    # 4. VAE Keys (prefix: first_stage_model.)
    elif target_key.startswith("first_stage_model."):
        # For simplicity, map all VAE keys to one block.
        # Could be refined with more complex regex if needed.
        block_name = "VAE_ALL"

    # --- Error Handling & Return ---
    if block_name is None:
        # This should ideally not happen if the logic above is exhaustive
        logger.error(f"Could not map target key '{target_key}' to any block in 'sdxl-optim_blocks'. Check conversion logic.")
        # Decide what to return: raise error, return a default (e.g., from *_ELSE?), etc.
        # Returning value from UNET_ELSE as a safe default, but this indicates a logic gap.
        block_name = "UNET_ELSE" # Or another appropriate fallback
        # raise ValueError(f"Unhandled key in conversion: {target_key}")

    try:
        # Retrieve the value (e.g., alpha) associated with the determined block name
        value = blocks_dict[block_name]
        # logger.debug(f"Key '{target_key}' mapped to block '{block_name}', returning value {value}")
        return value
    except KeyError:
        # This means the block_name determined above doesn't exist as a key in the input blocks_dict
        # This indicates an inconsistency between the block config YAML and this conversion function.
        logger.error(f"Block name '{block_name}' (determined for key '{target_key}') not found in input blocks_dict. Check block config YAML and conversion function.")
        # Raise error or return a sensible default
        raise StateDictKeyError(f"Block '{block_name}' not found in the input dictionary for conversion.")
    except Exception as e:
         logger.error(f"Unexpected error during conversion for key '{target_key}' mapped to block '{block_name}': {e}", exc_info=True)
         raise

# --- Need to ensure this function gets registered ---
# If utils.py is imported early (e.g., in sd_optim.py), the decorator handles it.
# If generating dynamically, you'd use exec() and then manually register or ensure import.


#############################
### --- Custom blocks --- ###
#############################
# Placeholder for the dynamic module where we'll exec the converter
_DYNAMIC_CONVERTER_MODULE_NAME = "sd_optim_dynamic_converters"

def setup_custom_blocks(cfg: DictConfig):
    """
    Parses custom block definitions, generates and registers
    corresponding sd-mecha ModelConfigs and conversion functions.
    """
    custom_block_defs = cfg.optimization_guide.get("custom_block_configs")
    if not custom_block_defs:
        logger.info("No 'custom_block_configs' section found in optimization_guide. Skipping setup.")
        return

    if not isinstance(custom_block_defs, (list, ListConfig)):
        logger.error("'custom_block_configs' must be a list.")
        raise TypeError("'custom_block_configs' must be a list.")

    logger.info(f"Found {len(custom_block_defs)} custom block configuration(s) to process.")

    # Create a dummy module to hold dynamically generated functions
    dynamic_module_spec = importlib.util.spec_from_loader(_DYNAMIC_CONVERTER_MODULE_NAME, loader=None)
    if dynamic_module_spec is None:
         # Fallback for some environments? Or raise error.
         logger.warning(f"Could not create spec for dynamic module '{_DYNAMIC_CONVERTER_MODULE_NAME}'. Registration might fail.")
         dynamic_module = type(sys)(_DYNAMIC_CONVERTER_MODULE_NAME) # Create basic module object
    else:
         dynamic_module = importlib.util.module_from_spec(dynamic_module_spec)

    # Add necessary imports to the dynamic module's namespace
    dynamic_module.sd_mecha = sd_mecha
    dynamic_module.logging = logging
    dynamic_module.fnmatch = fnmatch
    dynamic_module.re = re
    dynamic_module.Parameter = Parameter
    dynamic_module.Return = Return
    dynamic_module.StateDict = StateDict
    dynamic_module.StateDictKeyError = sd_mecha.streaming.StateDictKeyError
    dynamic_module.T = T # Make sure T is defined or imported appropriately
    dynamic_module.merge_method = merge_method # Pass the decorator itself

    # Make the logger available within the generated code's scope if needed
    dynamic_module.logger = logging.getLogger(f"{__name__}.dynamic_converters")


    for i, block_def in enumerate(custom_block_defs):
        block_config_id = None # Initialize at the start of the iteration's scope
        try:
            target_config_id = block_def.get("target_config_id")
            block_config_id = block_def.get("block_config_id")
            block_definitions = block_def.get("blocks")

            if not all([target_config_id, block_config_id, block_definitions]):
                logger.error(f"Custom block definition #{i+1} is missing required keys ('target_config_id', 'block_config_id', 'blocks'). Skipping.")
                continue

            logger.info(f"Processing custom block config '{block_config_id}' targeting '{target_config_id}'...")

            # --- 1. Generate and Register ModelConfig ---
            generated_config = generate_block_model_config(block_config_id, block_definitions)
            model_configs.register(generated_config)
            logger.info(f"Registered ModelConfig: {block_config_id}")

            # --- 2. Generate and Register Conversion Function ---
            converter_name = f"convert_{block_config_id.replace('-', '_')}_to_{target_config_id.replace('-', '_')}"
            converter_code = generate_conversion_function_code(
                converter_name=converter_name,
                block_config_id=block_config_id,
                target_config_id=target_config_id,
                block_definitions=block_definitions
            )

            # Execute in the dummy module's namespace
            exec(converter_code, dynamic_module.__dict__)
            logger.info(f"Defined conversion function: {converter_name}")

            # Retrieve the function object (it's now an attribute of the dummy module)
            # converter_func = getattr(dynamic_module, converter_name)
            # Registration happens via the @merge_method decorator included in the generated code string

        except Exception as e:
            # Now block_config_id is guaranteed to exist (might be None or the ID)
            error_id_info = f"'{block_config_id}'" if block_config_id else "(ID assignment failed or missing)"
            logger.error(f"Failed to process custom block definition #{i+1} ({error_id_info}): {e}", exc_info=True)
            # raise # Optionally re-raise to halt execution

def generate_block_model_config(config_id: str, block_defs: List[Dict]) -> model_configs.ModelConfig:
    """Generates an sd_mecha ModelConfig object from block definitions."""
    components = {'blocks': {}} # Use a single 'blocks' component for simplicity
    for block in block_defs:
        name = block.get("name")
        if not name: continue
        # Blocks map to single floats in this config
        components['blocks'][name] = {'shape': [], 'dtype': 'float32'}

    # Validate that a fallback block (e.g., '*_ELSE' or '*') exists if needed
    has_fallback = any(p == "*" for block in block_defs for p in block.get("patterns",[])) or \
                   any(n.endswith("_ELSE") for block in block_defs for n in block.get("name",""))

    if not has_fallback and block_defs:
         logger.warning(f"Custom block config '{config_id}' might be missing a fallback block (e.g., name ending in '_ELSE' or pattern '*') to catch unmapped keys.")

    config_dict = {
        "identifier": config_id,
        "components": components
    }
    return model_configs.ModelConfigImpl(**config_dict) # Use the concrete implementation

def generate_conversion_function_code(converter_name: str, block_config_id: str, target_config_id: str, block_definitions: List[Dict]) -> str:
    """Generates the Python source code string for the conversion function."""

    # Header and imports within the generated code's scope
    code = f"""
import sd_mecha # Access via passed module object
import logging
import fnmatch
import re
from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, T
from sd_mecha.streaming import StateDictKeyError

# Use logger passed into the dynamic module's scope
logger = logging.getLogger("{__name__}.dynamic_converters.{converter_name}")

@sd_mecha.extensions.merge_methods.merge_method( # Use decorator via sd_mecha module reference
    identifier="{converter_name}",
    is_conversion=True
)
def {converter_name}(
    blocks_dict: Parameter(StateDict[T], model_config="{block_config_id}"),
    **kwargs,
) -> Return(T, model_config="{target_config_id}"):

    target_key = kwargs["key"]
    block_name = None
    # logger.debug(f"Converter '{converter_name}' called for target key: {{target_key}}")

"""
    # Build the if/elif chain for pattern matching
    fallback_block = None
    for block in block_definitions:
        name = block.get("name")
        patterns = block.get("patterns")
        if not name or not patterns:
            continue

        # Check if this is the fallback block (must be last)
        is_fallback = "*" in patterns
        if is_fallback:
             if fallback_block is not None:
                 logger.warning(f"Multiple fallback patterns ('*') found in '{block_config_id}'. Using last one: '{name}'")
             fallback_block = name
             continue # Add fallback check at the end

        # Use fnmatch for wildcards, consider regex for more complex patterns
        # Need to escape regex special chars if using fnmatch patterns directly with re
        conditions = []
        for p in patterns:
             # Basic check if regex might be intended, otherwise use fnmatch
             if any(c in p for c in r"[]().+?^${}"): # Simple check for regex chars
                 # Use re.fullmatch (anchored match)
                 conditions.append(f're.fullmatch(r"{p}", target_key)') # Pass 're' module
             else:
                 # Use fnmatch
                 conditions.append(f'fnmatch.fnmatch(target_key, "{p}")') # Pass 'fnmatch' module

        condition_str = " or ".join(conditions)
        code += f"""
    if {condition_str}:
        block_name = "{name}"
"""

    # Add the fallback block condition at the end
    if fallback_block:
        code += f"""
    if block_name is None: # If no specific pattern matched, use fallback
        block_name = "{fallback_block}"
"""
    else:
         # No fallback defined, log error if no match
         code += f"""
    if block_name is None:
        logger.error(f"Converter '{converter_name}': Key '{{target_key}}' did not match any defined block patterns and no fallback ('*') was specified.")
        # Option 1: Raise an error
        # raise ValueError(f"Unhandled key in conversion '{converter_name}': {{target_key}}")
        # Option 2: Return a default (e.g., 0.0) - potentially dangerous
        logger.warning(f"Returning default value (e.g., 0.0 or neutral) for unmapped key '{{target_key}}'. Define a fallback pattern ('*') to handle this explicitly.")
        # Attempt to return neutral element (0 for float/tensor) - requires type T awareness
        return 0.0 # Simplistic default, might need refinement based on T
"""

    # Add the final lookup and return part
    code += f"""
    # logger.debug(f"Key '{{target_key}}' mapped to block '{{block_name}}'")
    try:
        return blocks_dict[block_name]
    except KeyError:
        logger.error(f"Block name '{{block_name}}' (determined for key '{{target_key}}') not found in input blocks_dict for converter '{converter_name}'. Check block config YAML and guide.yaml definitions.")
        raise StateDictKeyError(f"Block '{{block_name}}' not found in the input dictionary for conversion.")
    except Exception as e:
        logger.error(f"Unexpected error during conversion for key '{{target_key}}' mapped to block '{{block_name}}': {{e}}", exc_info=True)
        raise
"""
    return code


##############################
### --- Method resolve --- ###
##############################
def resolve_merge_method(merge_mode_name: str) -> MergeMethod:
    """Resolves merge method from sd-mecha built-ins or local MergeMethods class."""
    try:
        # Try resolving built-in method
        merge_func = sd_mecha.extensions.merge_methods.resolve(merge_mode_name)
        logger.debug(f"Resolved merge method '{merge_mode_name}' from sd-mecha built-ins.")
        return merge_func
    except ValueError:
        # If not found, try getting from our custom class
        logger.debug(f"'{merge_mode_name}' not found in sd-mecha built-ins, checking local MergeMethods...")
        if hasattr(MergeMethods, merge_mode_name):
            merge_func = getattr(MergeMethods, merge_mode_name)
            # IMPORTANT: Assumes methods in MergeMethods are decorated with @merge_method
            if isinstance(merge_func, MergeMethod):
                logger.debug(f"Resolved merge method '{merge_mode_name}' from local MergeMethods.")
                return merge_func
            else:
                # Attempt to manually wrap if not decorated (less ideal)
                try:
                    wrapped_func = sd_mecha.merge_method(merge_func, identifier=merge_mode_name, register=False)
                    logger.warning(f"Manually wrapping local method '{merge_mode_name}'. Decorate with @sd_mecha.merge_method for proper registration.")
                    return wrapped_func
                except Exception as wrap_e:
                    raise ValueError(f"Local method '{merge_mode_name}' exists but is not a valid sd-mecha MergeMethod and couldn't be wrapped: {wrap_e}")
        else:
            raise ValueError(f"Merge method '{merge_mode_name}' not found in sd-mecha built-ins or local MergeMethods.")


### ------------- old functions that haven't been updated to new mecha yet ------------ ###

### Recipe optimization
def get_target_nodes(recipe_path: Union[str, pathlib.Path], target_nodes: Union[str, List[str]]) -> Dict[
    str, Dict[str, Any]]:
    """
    Extract hyperparameters from specified target nodes in the recipe.

    Args:
        recipe_path: Path to the recipe file
        target_nodes: Target node(s) specified as '&N' or ['&N', '&M', ...]

    Returns:
        Dict mapping node references to their hyperparameters
    """
    if isinstance(target_nodes, str):
        target_nodes = [target_nodes]

    recipe = sd_mecha.deserialize(recipe_path)
    node_map = _build_node_map(recipe)

    extracted_hypers = {}
    for target in target_nodes:
        node_index = int(target.strip('&'))
        if node_index in node_map:
            node = node_map[node_index]
            if isinstance(node, MergeRecipeNode):
                extracted_hypers[target] = {
                    'merge_method': node.merge_method.get_name(),
                    'hypers': node.hypers
                }

    return extracted_hypers

def update_recipe(recipe: RecipeNode, target_nodes: Union[str, List[str]], assembled_params: Dict[str, Any]) -> RecipeNode:
    """
    Update recipe with new hyperparameters, inserting dict nodes as needed.

    Args:
        recipe: The recipe to modify
        target_nodes: Target node(s) to update
        assembled_params: New parameter values to apply

    Returns:
        Modified recipe with updated hyperparameters
    """
    if isinstance(target_nodes, str):
        target_nodes = [target_nodes]

    # Convert recipe to text form for manipulation
    recipe_lines = sd_mecha.serialize(recipe).split('\n')

    # Prepare new dict lines from assembled_params
    new_dicts = []
    for param_set in assembled_params.values():
        dict_line = _create_dict_line(param_set)
        new_dicts.append(dict_line)

    # Insert new dict lines at the top
    recipe_lines = new_dicts + recipe_lines

    # Determine the increment value (number of new dicts inserted)
    increment = len(new_dicts)

    # Update all references in existing lines
    recipe_lines = _increment_node_refs(recipe_lines, increment)

    # Create a mapping from param_key to new dict index
    param_key_to_new_dict_index = {key: idx for idx, key in enumerate(assembled_params.keys())}

    # Handle each target node
    for target in target_nodes:
        node_index = int(target.strip('&')) + increment  # Adjust for new dicts
        if node_index >= len(recipe_lines):
            continue  # Skip if the node index is out of range after increment

        target_line = recipe_lines[node_index]

        if not target_line.startswith('merge'):
            continue

        # Update parameters in the merge line to reference new dicts
        parts = target_line.split()
        for param_key in assembled_params.keys():
            # Find the parameter position in the merge line
            for i, part in enumerate(parts):
                if part.startswith(f'{param_key}='):
                    # Replace the existing reference with the new dict reference
                    parts[i] = f'{param_key}=&{param_key_to_new_dict_index[param_key]}'
                    break
            else:
                # If parameter doesn't exist, append it
                parts.append(f'{param_key}=&{param_key_to_new_dict_index[param_key]}')

        # Update the target line
        recipe_lines[node_index] = ' '.join(parts)

    # Convert back to recipe object
    modified_recipe = sd_mecha.deserialize(recipe_lines)
    return modified_recipe

def _build_node_map(recipe: RecipeNode) -> Dict[int, RecipeNode]:
    """Build a map of node indices to their corresponding RecipeNode objects."""
    lines = sd_mecha.serialize(recipe).split('\n')
    node_map = {}

    current_recipe = []
    for i, line in enumerate(lines):
        current_recipe.append(line)
        node = sd_mecha.deserialize(current_recipe)
        node_map[i] = node

    return node_map

def _parse_dict_line(line: str) -> Dict[str, Any]:
    """Parse a dict line into a dictionary of parameter values."""
    parts = line.split()
    result = {}
    for part in parts[1:]:  # Skip 'dict' command
        if '=' in part:
            key, value = part.split('=', 1)
            # Convert string value to appropriate type
            try:
                if value.replace('.', '').replace('e-', '').replace('e+', '').isdigit():
                    value = float(value)
            except ValueError:
                pass  # Keep as string if conversion fails
            result[key] = value
    return result

def _increment_refs_in_line(line: str, increment: int) -> str:
    """Increment all node references in a single line by a specified amount."""
    parts = line.split()
    for i, part in enumerate(parts):
        if part.startswith('&') and part[1:].isdigit():
            ref_num = int(part[1:])
            parts[i] = f'&{ref_num + increment}'
        elif '=' in part:
            key, value = part.split('=', 1)
            if value.startswith('&') and value[1:].isdigit():
                ref_num = int(value[1:])
                parts[i] = f'{key}=&{ref_num + increment}'
    return ' '.join(parts)

def _create_dict_line(params: Dict[str, Any]) -> str:
    """Create a dict line from a dictionary of parameters."""
    param_strs = []
    for key, value in params.items():
        if isinstance(value, (int, float)):
            param_strs.append(f"{key}={value}")
        else:
            param_strs.append(f'{key}="{value}"')
    return 'dict ' + ' '.join(param_strs)

def _extract_param_value(line: str, param: str) -> Optional[str]:
    """Extract the value of a parameter from a merge line."""
    parts = line.split()
    for part in parts:
        if part.startswith(f'{param}='):
            return part.split('=', 1)[1]
    return None

def _update_dict_line(line: str, updates: Dict[str, Any]) -> str:
    """Update parameters in a dict line."""
    parts = line.split()
    params = {p.split('=')[0]: p.split('=')[1] for p in parts[1:]}
    params.update(updates)
    return 'dict ' + ' '.join(f'{k}={v}' for k, v in params.items())

def _replace_param_value(line: str, param: str, new_value: str) -> str:
    """Replace the value of a parameter in a merge line."""
    parts = line.split()
    found = False
    for i, part in enumerate(parts):
        if part.startswith(f'{param}='):
            parts[i] = f'{param}={new_value}'
            found = True
            break
    if not found:
        parts.append(f'{param}={new_value}')
    return ' '.join(parts)

def _increment_node_refs(lines: List[str], increment: int) -> List[str]:
    """Increment all node references in the recipe by a specified amount."""
    updated_lines = []
    for line in lines:
        parts = line.split()
        for i, part in enumerate(parts):
            if '=' in part:
                key, value = part.split('=', 1)
                if value.startswith('&'):
                    ref_num = int(value[1:])
                    parts[i] = f'{key}=&{ref_num + increment}'
            elif part.startswith('&'):
                ref_num = int(part[1:])
                parts[i] = f'&{ref_num + increment}'
        updated_lines.append(' '.join(parts))
    return updated_lines

def traverse_recipe(node: RecipeNode) -> List[RecipeNode]:
    """Traverses the recipe tree and yields all nodes."""
    yield node
    if isinstance(node, MergeRecipeNode):
        for model in node.models:
            yield from traverse_recipe(model)

def get_model_names_from_recipe(recipe: RecipeNode) -> List[str]:
    """Extracts model names from a recipe."""
    model_names = []
    for node in traverse_recipe(recipe):
        if isinstance(node, ModelRecipeNode):
            # Extract model name from path, handling potential None
            model_name = Path(node.path).stem if node.path else "unknown_model"
            model_names.append(model_name)
    return model_names

def get_merge_method(recipe_path: Union[str, Path], target_node: str) -> str:
    """Extract the merge_method from the target node in a recipe."""
    target_nodes = [target_node]
    extracted_hypers = get_target_nodes(recipe_path, target_nodes)  # Use existing get_target_nodes
    return extracted_hypers[target_node]['merge_method']


### Custom sorting function that uses component order from config ###
def custom_sort_key(key, component_order):
    parts = key.split("_")

    # Check if the key matches the expected format for layer adjustments
    if parts[0] in ["detail1", "detail2", "detail3", "contrast", "brightness", "col1", "col2", "col3"]:
        # Assign a high priority to layer_adjustments by setting component_index to -1
        component_index = -1
        block_key = parts[0]  # The parameter name itself is the block key
    else:
        # Handle the original format for other keys
        if len(parts) > 2 and parts[1] in component_order:
            component_index = component_order.index(parts[1])
            block_key = "_".join(parts[2:])
        else:
            # Default sorting if component not found or key is too short
            component_index = len(component_order)  # Ensure these keys are sorted last
            block_key = key

    return component_index, sd_mecha.hypers.natural_sort_key(block_key)


### Cache for recipes (does it work tho?) ###
class CacheInjectorVisitor(RecipeVisitor):
    def __init__(self, cache: Optional[Dict[str, Dict[str, Tensor]]]):
        self.cache = cache

    def visit_model(self, node: ModelRecipeNode, *args, **kwargs) -> ModelRecipeNode:
        # Return the original ModelRecipeNode without modification
        return node

    def visit_parameter(self, node: ParameterRecipeNode, *args, **kwargs) -> ParameterRecipeNode:
        return node

    def visit_merge(self, node: MergeRecipeNode, *args, **kwargs) -> MergeRecipeNode:
        # Recursively visit and reconstruct child nodes
        new_models = [model.accept(self, *args, **kwargs) for model in node.models]

        # Check if the merge method supports caching
        merge_method_signature = inspect.signature(node.merge_method)

        # Create a copy of volatile_hypers and add cache if supported
        new_volatile_hypers = node.volatile_hypers.copy()
        if 'cache' in merge_method_signature.parameters:
            new_volatile_hypers['cache'] = self.cache

        # Reconstruct the merge node with modified volatile_hypers
        return MergeRecipeNode(
            node.merge_method,
            *new_models,
            hypers=node.hypers,
            volatile_hypers=new_volatile_hypers,
            device=node.device,
            dtype=node.dtype,
        )


### Recipe merger patch for key merging ###
class ModelsListVisitor(RecipeVisitor):
    def __init__(self):
        self.models = []  # Stores individual ModelRecipeNode objects

    def visit_model(self, node: ModelRecipeNode):
        self.models.append(node)  # Add single model node

    def visit_merge(self, node: MergeRecipeNode):
        # Recursively visit child models in merge nodes
        for model in node.models:
            model.accept(self)

    def visit_parameter(self, node: ParameterRecipeNode):
        pass  # Ignore parameters


def patch_recipe_merger(merge_keys_config: Dict[str, Any], cfg: DictConfig):
    """Patches the RecipeMerger.merge_and_save method to enable selective merging."""

    original_merge_and_save = RecipeMerger.merge_and_save

    @functools.wraps(original_merge_and_save)
    def patched_merge_and_save(
        self, recipe: sd_mecha.extensions.merge_method.RecipeNodeOrPath, *,
        output: MutableMapping[str, torch.Tensor] | pathlib.Path | str = "merge",
        fallback_model: Optional[sd_mecha.Mapping[str, torch.Tensor] | sd_mecha.recipe_nodes.ModelRecipeNode | pathlib.Path | str] = None,
        save_device: Optional[str] = "cpu",
        save_dtype: Optional[torch.dtype] = torch.float16,
        threads: Optional[int] = None,
        total_buffer_size: int = 2**28,
        strict_weight_space: bool = True
    ):
        # Early exit if merge keys are not enabled
        if not merge_keys_config or not merge_keys_config.get("enabled", False):
            print("Merge keys not enabled, using default merge_and_save")
            return original_merge_and_save(self, recipe, output=output, fallback_model=fallback_model, save_device=save_device, save_dtype=save_dtype, threads=threads, total_buffer_size=total_buffer_size)

        merge_keys = merge_keys_config.get("keys", [])
        default_behavior = merge_keys_config.get("default_behavior", "keep_a")

        # Resolve the recipe path to a node
        recipe = sd_mecha.extensions.merge_method.path_to_node(recipe)
        if recipe.merge_space != sd_mecha.recipe_nodes.MergeSpace.BASE:
            raise ValueError(f"recipe should be in model merge space, not {str(recipe.merge_space).split('.')[-1]}")

        # Resolve the fallback model path to a node
        if isinstance(fallback_model, (str, pathlib.Path)):
            fallback_model = sd_mecha.extensions.merge_method.path_to_node(fallback_model)
        elif not isinstance(fallback_model, (sd_mecha.recipe_nodes.ModelRecipeNode, Mapping, type(None))):
            raise ValueError(f"fallback_model should be a simple model or None, not {type(fallback_model)}")

        # Clear model paths cache
        sd_mecha.extensions.merge_method.clear_model_paths_cache()

        # Determine fallback model node
        fallback_model_index = merge_keys_config.get("fallback_model_index", 0)
        models_visitor = ModelsListVisitor()
        recipe.accept(models_visitor)
        models_in_recipe = models_visitor.models

        # Validate fallback_model_index
        if fallback_model_index >= len(models_in_recipe):
            raise ValueError(
                f"fallback_model_index {fallback_model_index} is out of range. "
                f"Recipe only contains {len(models_in_recipe)} models: "
                f"{[model.path for model in models_in_recipe]}"
            )

        fallback_model_node = models_in_recipe[fallback_model_index] if len(models_in_recipe) > 0 else None

        # Determine if fallback model is external
        fallback_is_external = fallback_model_node is not None and fallback_model_node not in recipe

        # ==============================================
        # Calculate total_files_open based on the models involved
        # ==============================================
        total_files_open = (
            len(models_in_recipe) +
            int(isinstance(output, (str, pathlib.Path))) +
            int(fallback_is_external)
        )

        buffer_size_per_file = total_buffer_size // max(total_files_open, 1)  # Prevent division by zero
        threads = threads or total_files_open

        # ==============================================
        # Load state_dicts with calculated buffer size
        # ==============================================
        load_input_dicts_visitor = sd_mecha.recipe_merger.LoadInputDictsVisitor(
            self._RecipeMerger__base_dirs,
            buffer_size_per_file,
        )
        recipe.accept(load_input_dicts_visitor)

        # Load fallback model's state_dict if it's external
        if fallback_is_external:
            fallback_model_node.accept(load_input_dicts_visitor)
            fallback_model = fallback_model_node.state_dict
        elif fallback_model_node:
            fallback_model = fallback_model_node.state_dict
        else:
            fallback_model = None

        # ==============================================
        # Rest of selective merging logic
        # ==============================================

        # Get model configuration
        model_config = recipe.accept(sd_mecha.model_detection.DetermineConfigVisitor())

        # Normalize output
        output = self._RecipeMerger__normalize_output_to_dict(
            output,
            model_config.get_minimal_dummy_header(),
            model_config.get_keys_to_merge(),
            recipe_serializer.serialize(recipe),
            buffer_size_per_file // threads,
            save_dtype,
        )

        # ==============================================
        # Moved key filtering logic outside the loop
        # ==============================================
        def _should_merge_key(key: str, merge_keys: List[str]) -> bool:
            """Checks if a key should be merged using regex for wildcard patterns."""
            # Check exclusion first
            for pattern in merge_keys:
                if pattern.startswith("!"):
                    exclusion_pattern = pattern[1:].strip()
                    # Convert wildcard to regex
                    regex_pattern = exclusion_pattern.replace(".", r"\.").replace("*", ".*")
                    if re.fullmatch(regex_pattern, key):
                        return False  # Skip merge

            # Check inclusion
            for pattern in merge_keys:
                if not pattern.startswith("!"):
                    # Convert wildcard to regex
                    regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
                    if re.fullmatch(regex_pattern, key):
                        return True  # Merge

            return False  # Default: skip merge

        thread_local_data = threading.local()
        progress = self._RecipeMerger__tqdm(total=len(model_config.keys()), desc="Merging recipe")
        with sd_mecha.recipe_merger.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in model_config.keys():
                if not _should_merge_key(key, merge_keys):
                    if default_behavior == "zero":
                        print(f"Skipping merge for key: {key}, setting to zero")
                        shape = model_config.get_shape(key)
                        if shape is not None:
                            output[key] = torch.zeros(shape, dtype=save_dtype, device=save_device)
                        else:
                            logger.warning(f"Shape not found for key: {key}. Skipping.")
                        progress.update()
                        continue
                    elif default_behavior == "keep_a":
                        print(f"Skipping merge for key: {key}, keeping original value from model")
                        try:
                            if fallback_model is not None:
                                output[key] = fallback_model[key].to(device=save_device, dtype=save_dtype)
                            else:
                                output[key] = recipe.models[0].state_dict[key].to(device=save_device,
                                                                                  dtype=save_dtype)
                        except KeyError:
                            logging.warning(f"Key {key} not found in any model. Skipping.")
                        progress.update()
                        continue

                key_merger = model_config.get_key_merger(key, recipe, fallback_model, self._RecipeMerger__default_device, self._RecipeMerger__default_dtype)
                key_merger = self._RecipeMerger__track_output(key_merger, output, key, save_dtype, save_device)
                key_merger = self._RecipeMerger__track_progress(key_merger, key, model_config.get_shape(key), progress)
                key_merger = self._RecipeMerger__wrap_thread_context(key_merger, thread_local_data)
                futures.append(executor.submit(key_merger))

            for future in sd_mecha.recipe_merger.as_completed(futures):
                if future.exception() is not None:
                    for future_to_cancel in futures:
                        future_to_cancel.cancel()
                    raise future.exception()
                future.result()

        progress.close()
        if isinstance(output, sd_mecha.recipe_merger.OutSafetensorsDict):
            output.close()
        recipe.accept(sd_mecha.recipe_merger.CloseInputDictsVisitor())

        gc.collect()
        torch.cuda.empty_cache()

    RecipeMerger.merge_and_save = patched_merge_and_save


### Save merge method code ###
@dataclass
class ImportInfo:
    """Store information about imports."""
    module: str
    names: Set[str] = field(default_factory=set)  # For 'from' imports
    alias: str = None  # For regular imports with 'as'
    is_from_import: bool = False

    def to_source(self) -> str:
        """Convert the import back to source code."""
        if self.is_from_import:
            names_str = ", ".join(sorted(self.names))
            return f"from {self.module} import {names_str}"
        else:
            if self.alias:
                return f"import {self.module} as {self.alias}"
            return f"import {self.module}"


class CodeAnalysisVisitor(ast.NodeVisitor):
    """AST visitor that finds method calls and imports within a function."""

    def __init__(self, class_methods: List[str]):
        self.class_methods = class_methods
        self.called_methods = set()
        self.imports: Dict[str, ImportInfo] = {}
        self.used_names = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a function call node in the AST."""
        # Track method calls
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in ('self', 'cls'):
                method_name = node.func.attr
                if method_name in self.class_methods:
                    self.called_methods.add(method_name)

        # Track all names used in function calls
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            self.used_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.used_names.add(node.func.value.id)

    def visit_Name(self, node: ast.Name) -> None:
        """Visit a name node to track used variables."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit an import node."""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                alias=alias.asname
            )
            self.imports[alias.asname or alias.name] = import_info
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit a from-import node."""
        if node.module is None:  # Handle "from . import x"
            return

        import_info = ImportInfo(
            module=node.module,
            is_from_import=True
        )
        for alias in node.names:
            import_info.names.add(alias.asname or alias.name)

        # Use the module name as key for from-imports
        self.imports[node.module] = import_info
        self.generic_visit(node)


class MergeMethodCodeSaver:
    """A class to handle saving merge method code with caching to prevent duplicate saves."""

    _saved_methods: ClassVar[Set[str]] = set()

    @classmethod
    def analyze_code(cls, source: str, class_methods: List[str]) -> Tuple[Set[str], Dict[str, ImportInfo]]:
        """Analyze source code for dependencies and imports."""
        tree = ast.parse(source)
        visitor = CodeAnalysisVisitor(class_methods)
        visitor.visit(tree)
        return visitor.called_methods, visitor.imports

    @classmethod
    def get_method_dependencies(cls, method: Any, class_obj: Any) -> Tuple[Set[str], Dict[str, ImportInfo]]:
        """Recursively get all function dependencies and their imports from the method's source code."""
        all_dependencies = set()
        all_imports: Dict[str, ImportInfo] = {}

        try:
            source = inspect.getsource(method)
            all_methods = inspect.getmembers(class_obj, predicate=inspect.isfunction)
            method_names = [name for name, _ in all_methods]

            # Get direct dependencies and imports
            called_methods, imports = cls.analyze_code(source, method_names)
            all_imports.update(imports)

            # Recursively analyze dependencies
            for called_method_name in called_methods:
                if called_method_name != method.__name__:
                    all_dependencies.add(called_method_name)
                    called_method = getattr(class_obj, called_method_name)
                    dep_methods, dep_imports = cls.get_method_dependencies(called_method, class_obj)
                    all_dependencies.update(dep_methods)
                    all_imports.update(dep_imports)

            return all_dependencies, all_imports
        except (TypeError, OSError) as e:
            logger.warning(f"Could not analyze dependencies for {method.__name__}: {e}")
            return set(), {}

    @classmethod
    def get_full_method_source(cls, method_name: str, class_obj: Any, visited: Set[str] = None) -> str:
        """Get the source code of the method and all its dependencies."""
        if visited is None:
            visited = set()
        if method_name in visited:
            return ""

        visited.add(method_name)
        try:
            method = getattr(class_obj, method_name)
        except AttributeError as e:
            logger.error(f"Method {method_name} not found in class: {e}")
            return f"# Error: Method {method_name} not found in class"

        try:
            # Get the source and analyze dependencies
            source = inspect.getsource(method)
            dependencies, imports = cls.get_method_dependencies(method, class_obj)

            # Build the complete source code starting with imports
            full_source = []

            # Add imports section
            if imports:
                full_source.extend([
                    "# Required imports",
                    *[imp_info.to_source() for imp_info in imports.values()],
                    "\n"
                ])

            # Add main method
            full_source.extend([
                f"# {'-' * 20} Main merge method {'-' * 20}",
                source
            ])

            # Add dependencies if any
            if dependencies:
                full_source.extend([
                    f"\n# {'-' * 20} Dependencies {'-' * 20}"
                ])
                for dep_name in dependencies:
                    if dep_name not in visited:
                        dep_method = getattr(class_obj, dep_name)
                        full_source.extend([
                            f"\n# Dependency: {dep_name}",
                            inspect.getsource(dep_method)
                        ])
                        visited.add(dep_name)

            return "\n".join(full_source)
        except (TypeError, OSError) as e:
            logger.error(f"Failed to get source code for {method_name}: {e}")
            return f"# Error getting source code for {method_name}: {str(e)}"

    @classmethod
    def save_merge_method_code(cls, merge_method: str, model_path: Path, class_obj: Any) -> None:
        """Save the merge method code and its dependencies to a file if not already saved."""
        if merge_method in cls._saved_methods:
            logger.debug(f"Merge method {merge_method} already saved in this run, skipping...")
            return

        try:
            if not hasattr(class_obj, merge_method):
                raise AttributeError(f"Method '{merge_method}' not found in class {class_obj.__name__}")

            # Determine log directory
            log_dir = Path(os.getcwd()) if "HydraConfig" not in globals() else Path(
                HydraConfig.get().runtime.output_dir)
            merge_code_dir = log_dir / "merge_methods"
            os.makedirs(merge_code_dir, exist_ok=True)

            # Get the complete source code including dependencies
            full_source = cls.get_full_method_source(merge_method, class_obj)
            if not full_source.strip():
                logger.warning(f"Source code for merge method '{merge_method}' is empty.")
                return

            # Dedent and clean the source code to ensure consistent formatting
            full_source_cleaned = textwrap.dedent(full_source)

            # Save the merge method code
            iteration_file_name = model_path.stem
            code_file_path = merge_code_dir / f"{iteration_file_name}_merge_method.py"

            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Merge method: {merge_method}\n")
                f.write(f"# Used in merge: {iteration_file_name}\n")
                f.write("# This file includes the main merge method and all its dependencies\n\n")
                f.write(full_source_cleaned)

            logger.info(f"Saved merge method code to {code_file_path}")
            cls._saved_methods.add(merge_method)

        except Exception as e:
            logger.error(f"Failed to save merge method code: {e}")
            raise


### Add keys to models
def add_extra_keys(
    model_path: Path
) -> None:
    """Loads a model, adds 'v_pred' and 'ztsnr' keys with empty tensors to its state dictionary, and saves it.

    Args:
        model_path: Path to the merged model file.
        cfg: The project configuration.
    """
    state_dict = safetensors.torch.load_file(model_path)
    state_dict["v_pred"] = torch.tensor([])
    state_dict["ztsnr"] = torch.tensor([])
    logger.info("Added 'v_pred' and 'ztsnr' keys to state_dict.")
    safetensors.torch.save_file(state_dict, model_path)
    logger.info(f"Saved model with additional keys to: {model_path}")


### Layer tuning
# --- Constants for Color Adjustments ---
COLS = [[-1, 1 / 3, 2 / 3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

# --- Layer Mapping for Adjustments ---
LAYER_MAPPING = {
    0: "model.diffusion_model.input_blocks.0.0.weight",
    1: "model.diffusion_model.input_blocks.0.0.bias",
    2: "model.diffusion_model.out.0.weight",
    3: "model.diffusion_model.out.0.bias",
    4: "model.diffusion_model.out.2.weight",
    5: "model.diffusion_model.out.2.bias",
}

# --- Helper Functions ---
def colorcalc(cols, isxl):
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i, x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

def fineman(fine, isxl):
    if isinstance(fine, str) and fine.find(",") != -1:
        tmp = [t.strip() for t in fine.split(",")]
        fines = [0.0] * 8
        for i, f in enumerate(tmp[0:8]):
            try:
                fines[i] = float(f)
            except ValueError:
                print(f"Warning: Could not convert '{f}' to float. Using 0.0 instead.")
                fines[i] = 0.0
        fine = fines
    elif not isinstance(fine, list):
        print("Error: Invalid input type for 'fine'. Expected a comma-separated string or a list.")
        return None

    fine = [
        1 - fine[0] * 0.01,
        1 + fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1 + fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3] * 0.02] + colorcalc(fine[4:8], isxl)
    ]
    return fine

def weighttoxl(weights):
    """
    Possibly converts weights to SDXL format by removing elements 9 to 11 and adding a zero at the end.
    """
    if len(weights) >= 22:
        weights = weights[:9] + weights[12:22] + [0]
    return weights

def modify_state_dict(state_dict: Dict, adjustments: Dict, is_xl_model: bool) -> Dict:
    """Modifies the state_dict based on the given adjustments."""

    fine_adjustments = fineman(",".join(map(str, adjustments.values())), is_xl_model)

    if fine_adjustments is None:
        raise ValueError("Error: Invalid 'fine' string format for fineman function.")

    modified_state_dict = state_dict.copy()

    if is_xl_model:
        fine_adjustments = weighttoxl(fine_adjustments)

    for index, layer_name in LAYER_MAPPING.items():
        if layer_name in state_dict:
            if index < 5:
                modified_state_dict[layer_name] = state_dict[layer_name] * fine_adjustments[index]
            else:
                modified_state_dict[layer_name] = state_dict[layer_name] + torch.tensor(
                    fine_adjustments[index], dtype=state_dict[layer_name].dtype, device=state_dict[layer_name].device
                )
        else:
            print(f"Warning: Layer '{layer_name}' not found in the state_dict.")

    return modified_state_dict


# Hotkey behavior
HOTKEY_SWITCH_MANUAL = keyboard.Key.ctrl, 'm'  # Ctrl+M for manual scoring
HOTKEY_SWITCH_AUTO = keyboard.Key.ctrl, 'a'  # Ctrl+A for automatic scoring
# ... other hotkeys ...


# Hotkey Listener Class
class HotkeyListener:
    def __init__(self, scoring_mode):
        self.scoring_mode = scoring_mode
        self.listener = keyboard.Listener(on_press=self.on_press)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            return False  # Stop listener
        try:
            if key == HOTKEY_SWITCH_MANUAL[1] and all(k in keyboard._pressed_events for k in HOTKEY_SWITCH_MANUAL[0]):
                self.scoring_mode.value = "manual"  # Assuming scoring_mode is a shared variable
                print("Switching to manual scoring mode!")
            elif key == HOTKEY_SWITCH_AUTO[1] and all(k in keyboard._pressed_events for k in HOTKEY_SWITCH_AUTO[0]):
                self.scoring_mode.value = "automatic"
                print("Switching to automatic scoring mode!")
        except AttributeError:
            pass




### Other Utility Functions (e.g., for early stopping, etc.)
# ...





def get_summary_images(log_file: Path, imgs_dir: Path, top_iterations: int) -> List[Tuple[str, float, Path]]:
    """Parses the log file, identifies top-scoring iterations, and selects images for summary."""
    try:
        with open(log_file, "r") as f:
            log_data = [json.loads(line) for line in f]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading log file: {e}")
        return []  # Return empty list if error occurs

    # Sort iterations by target score in descending order and select top iterations
    sorted_iterations = sorted(log_data, key=lambda x: x["target"], reverse=True)[:top_iterations]

    summary_images = []
    for iteration_data in sorted_iterations:
        iteration_num = len(summary_images)  # Use current index as iteration_num

        # Create payload -> [images] dict
        payload_images = {}
        for file_name in os.listdir(imgs_dir):
            if file_name.startswith(f"{iteration_num:03}-"):
                parts = file_name[:-4].split("-")  # ignore .png
                image_index = parts[1]
                payload = "-".join(parts[2:-1])
                score = parts[-1]

                # Put the file into payload group based on it's index
                payload_images.setdefault(payload, []).append((image_index, score, Path(imgs_dir, file_name)))

        # Select best image per payload
        for i, image_set in enumerate(payload_images.values()):
            # Find image path with highest score
            highest_scoring_image = max(image_set, key=lambda x: x[1])
            summary_images.append(
                (f"iter {iteration_num:03} - {i}", float(highest_scoring_image[1]), highest_scoring_image[2]))

    return summary_images


def update_log_scores(log_file: Path, summary_images, new_scores):
    """Updates the log file with the new average scores from the interactive rescoring."""

    try:
        with open(log_file, 'r+') as f:  # Open for both reading and writing
            log_data = [json.loads(line) for line in f]  # Load existing log data



            # Update scores for the corresponding iterations
            # TODO: Handle offset based on what iteration it starts on?
            for i in range(len(summary_images)):  # Loop through summary_images to get indices
                # Update score based on the index
                log_data[i]['target'] = new_scores[i]  # Update directly with a single value, not a list


            f.seek(0)  # Go to the beginning of the file
            json.dump(log_data, f, indent=4)  # Write the updated data
            f.truncate()  # Remove any remaining old data
    except Exception as e:
        logger.error(f"Error updating log file: {e}")