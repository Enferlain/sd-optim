# bounds.py - Version 1.1 - Added custom_block_config_id support and custom_blocks support

import functools
import gc
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import pkgutil
import re
import sys
import textwrap
import argparse
import optuna
import torch
import safetensors.torch
import sd_mecha
import logging
import ast
import torch
import yaml
import threading

from fnmatch import fnmatch
from pathlib import Path
from typing import List, Tuple, TypeVar, Dict, Set, Union, Any, ClassVar, Optional, MutableMapping, Mapping
from dataclasses import field, dataclass


from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
from pynput import keyboard
from copy import deepcopy

from sd_optim.merge_methods import MergeMethods
from sd_mecha import recipe_nodes
from sd_mecha.extensions import model_configs, merge_methods
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict, T, MergeMethod
from sd_mecha.streaming import StateDictKeyError
from sd_mecha.extensions.model_configs import ModelConfigImpl, KeyMetadata  # Need these for creation
from sd_mecha.recipe_nodes import RecipeNode, MergeRecipeNode, RecipeVisitor, LiteralRecipeNode, ModelRecipeNode
from omegaconf import DictConfig, OmegaConf  # If reading from Hydra config

logger = logging.getLogger(__name__)


# Define T if not already imported from merge_methods


#######################################
### --- Config loading function --- ###
#######################################
# V1.0 - Scans directory, parses YAML, registers configs
def load_and_register_custom_configs(config_dir: Path):
    """Scans a directory for YAML files, parses them as ModelConfig, and registers them."""
    logger.info(f"Scanning for custom ModelConfigs in: {config_dir}")
    registered_count = 0
    if not config_dir.is_dir():
        logger.warning(f"Custom config directory not found: {config_dir}. Skipping registration.")
        return

    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    for filepath in config_dir.glob("*.yaml"):
        try:
            logger.debug(f"  Loading config file: {filepath.name}")
            with open(filepath, 'r', encoding='utf-8') as f:
                yaml_data = yaml.load(f, Loader=Loader)
                if not isinstance(yaml_data, dict) or "identifier" not in yaml_data:
                    logger.warning(f"    Skipping {filepath.name}: Invalid format or missing 'identifier'.")
                    continue

                # Use ModelConfigImpl to parse the structure
                config_obj = ModelConfigImpl(**yaml_data)
                config_id = config_obj.identifier

                # Register using register_aux for user-defined configs
                model_configs.register_aux(config_obj)
                logger.info(f"  Successfully registered AUX ModelConfig: '{config_id}' from {filepath.name}")
                registered_count += 1
        except yaml.YAMLError as e_yaml:
            logger.error(f"  Error parsing YAML file {filepath.name}: {e_yaml}", exc_info=True)
        except TypeError as e_type:
            logger.error(f"  Error constructing ModelConfig from {filepath.name} (likely structure mismatch): {e_type}",
                         exc_info=True)
        except ValueError as e_val:
            logger.error(f"  Error registering ModelConfig from {filepath.name} (likely duplicate ID): {e_val}",
                         exc_info=True)
        except Exception as e_other:
            logger.error(f"  Unexpected error processing {filepath.name}: {e_other}", exc_info=True)

    logger.info(f"Finished custom config scan. Registered {registered_count} config(s).")


###########################################
### --- Conversion loading function --- ###
###########################################
# V1.0 - Scans directory, imports modules to trigger decorator registration
def load_and_register_custom_conversion(conversion_dir: Path):
    """Scans a directory for Python files and imports them to register merge methods."""
    logger.info(f"Scanning for custom Conversion/MergeMethods in: {conversion_dir}")
    registered_count = 0
    if not conversion_dir.is_dir():
        logger.warning(f"Custom conversion directory not found: {conversion_dir}. Skipping registration.")
        return

    # Add the conversion directory to the Python path temporarily to allow direct imports
    sys.path.insert(0, str(conversion_dir.parent.resolve()))  # Add parent directory

    try:
        for module_info in pkgutil.iter_modules([str(conversion_dir)]):
            module_name = module_info.name
            if module_name.startswith('_'):  # Skip private/utility modules
                continue
            try:
                logger.debug(f"  Importing conversion module: {module_name}")
                # Perform the import - this triggers the @merge_method decorators inside
                # Need to construct the full import path relative to something in sys.path
                # Assuming conversion_dir is like 'sd_optim/model_configs'
                import_path = f"{conversion_dir.parent.name}.{conversion_dir.name}.{module_name}"
                importlib.import_module(import_path)
                logger.info(f"  Successfully imported and potentially registered methods from: {module_name}.py")
                registered_count += 1  # Count modules imported, not methods registered
            except ImportError as e_imp:
                logger.error(f"  Error importing module {module_name}.py: {e_imp}", exc_info=True)
            except Exception as e_other:
                logger.error(f"  Unexpected error importing/registering from {module_name}.py: {e_other}",
                             exc_info=True)
    finally:
        # Clean up sys.path
        if str(conversion_dir.parent.resolve()) in sys.path:
            sys.path.pop(0)

    logger.info(f"Finished custom conversion scan. Imported {registered_count} module(s).")


##############################
### --- Method resolve --- ###
##############################
def resolve_merge_method(merge_method_name: str) -> merge_methods.MergeMethod:
    """
    Resolves merge method, prioritizing our local MergeMethods class before
    checking the sd-mecha global registry to ensure correct log attribution.
    """

    # --- Step 1: Check our own local MergeMethods class FIRST ---
    # This is the crucial step to correctly identify our custom methods.
    if hasattr(MergeMethods, merge_method_name):
        merge_func = getattr(MergeMethods, merge_method_name)

        # Check if it's already a decorated sd-mecha method object
        if isinstance(merge_func, merge_methods.MergeMethod):
            logger.debug(f"Resolved merge method '{merge_method_name}' from local merge_methods.py.")
            return merge_func
        else:
            # This is a fallback for safety, in case we forget a decorator.
            # It attempts to wrap the raw function into a temporary MergeMethod object.
            try:
                wrapped_func = sd_mecha.merge_method(merge_func, identifier=merge_method_name, register=False)
                logger.warning(
                    f"Manually wrapping local method '{merge_method_name}'. Decorate with @merge_method for proper registration.")
                return wrapped_func
            except Exception as wrap_e:
                # If it exists but can't be wrapped, it's a critical error in our code.
                logger.error(
                    f"FATAL: Local method '{merge_method_name}' exists but is not a valid sd-mecha MergeMethod and couldn't be wrapped: {wrap_e}")
                os.abort()

    # --- Step 2: If not found locally, check the sd-mecha global registry ---
    # This will now only find true built-in methods, since we checked our local
    # ones (which are also registered here) in Step 1.
    try:
        merge_func = sd_mecha.extensions.merge_methods.resolve(merge_method_name)
        logger.debug(f"Resolved merge method '{merge_method_name}' from sd-mecha built-ins.")
        return merge_func
    except ValueError:
        # --- Step 3: If it's not in our class and not in sd-mecha's registry, it doesn't exist. ---
        logger.error(
            f"FATAL: Merge method '{merge_method_name}' not found in local MergeMethods or in sd-mecha's registry.")
        os.abort()


###########################
### Recipe Optimization ###
###########################
def serialize_nodes_for_rewrite(
        nodes_dict: Dict[str, sd_mecha.recipe_nodes.RecipeNode]
) -> Tuple[List[str], Dict[str, int]]:
    """
    Takes a dictionary of named RecipeNode objects and serializes them into
    a list of .mecha string lines and a map of name to final line index.
    This is a generic utility for preparing nodes for rewriting.
    """
    all_new_lines = []
    param_to_final_idx = {}
    current_offset = 0

    # Sort for deterministic output
    for param_name, node in sorted(nodes_dict.items()):
        serialized_text = sd_mecha.serialize(node)
        node_lines = serialized_text.strip().split('\n')[1:]

        def shift_ref(match):
            original_idx = int(match.group(1))
            return f"&{original_idx + current_offset}"

        shifted_lines = [re.sub(r'&(\d+)', shift_ref, line) for line in node_lines]

        all_new_lines.extend(shifted_lines)
        param_to_final_idx[param_name] = current_offset + len(node_lines) - 1
        current_offset += len(node_lines)

    return all_new_lines, param_to_final_idx


def rewrite_recipe_text(
        original_recipe_text: str,
        target_node_idx: int,
        new_node_strings: List[str],
        param_to_final_idx: Dict[str, int]
) -> str:
    """
    Rewrites a .mecha recipe text by prepending new nodes and patching a target line.
    This function is a PURE text manipulator with the corrected "shift, then patch" logic.
    """
    original_lines = original_recipe_text.strip().split('\n')[1:]
    num_new_nodes = len(new_node_strings)

    # Helper function to perform the shift
    def shift_old_ref(match):
        original_idx = int(match.group(1))
        return f"&{original_idx + num_new_nodes}"

    # --- The New, Corrected Logic ---
    rewritten_lines = []
    for i, line in enumerate(original_lines):
        # Clean the line of comments and whitespace first
        line_base = line.split('#', 1)[0].strip()
        if not line_base: continue

        # STEP 1: Always perform the global reference shift first.
        # This turns old references like `&14` into `&26`.
        shifted_line = re.sub(r'&(\d+)', shift_old_ref, line_base)

        # STEP 2: NOW, check if this is the target line we need to patch.
        if i == target_node_idx:
            # We take the *already shifted* line and patch in our new, correct, and final parameter references.
            # These new references (e.g., `&5`, `&11`) will NOT be shifted again.
            line_to_append = shifted_line
            for param_name, new_node_index in param_to_final_idx.items():
                pattern = re.compile(f"({re.escape(param_name)}=)([^ ]+)")
                line_to_append = pattern.sub(f"\\1&{new_node_index}", line_to_append)
        else:
            # If it's not the target line, the globally shifted version is all we need.
            line_to_append = shifted_line

        rewritten_lines.append(line_to_append)

    logger.info("Successfully shifted original recipe references and patched target line.")

    # --- Assembly step remains the same ---
    final_recipe_text = (
            "version 0.1.0\n" +
            "\n".join(new_node_strings) + "\n" +
            "\n".join(rewritten_lines)
    )
    return final_recipe_text


class ModelVisitor(recipe_nodes.RecipeVisitor):
    """A simple visitor to find all ModelRecipeNodes in a graph."""

    def __init__(self):
        self.models: List[recipe_nodes.ModelRecipeNode] = []
        # Memoization to prevent visiting the same node multiple times in complex graphs
        self.visited: Set[recipe_nodes.RecipeNode] = set()

    # REMOVED the incorrect generic .visit() method.
    # We will rely on the default dispatching mechanism.

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        # We only add a model if we haven't seen its object reference before.
        if node not in self.visited:
            self.models.append(node)
            self.visited.add(node)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        # We only process a merge node if we haven't seen it before.
        if node in self.visited:
            return
        self.visited.add(node)

        # CORRECT TRAVERSAL:
        # Politely ask each child node to accept this visitor, which continues the process.
        for arg in node.args:
            arg.accept(self)
        for kwarg in node.kwargs.values():
            kwarg.accept(self)

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        # Literals don't contain models, so we just mark as visited and stop.
        if node in self.visited:
            return
        self.visited.add(node)


def get_info_from_target_node(root_node: recipe_nodes.RecipeNode, target_node_ref: str) -> Dict[str, Any]:
    """
    Analyzes a recipe graph to extract the merge method name and input model names
    for a specific target node.
    """

    # Helper to find a node by its serialized representation's last line
    def find_node_by_ref(start_node, ref_str):
        target_line_num = int(ref_str.strip('&'))

        # We can find the node by traversing and checking its line number during serialization
        # This is complex, a simpler way is to build a map first.
        # Let's reuse the helper from the merger.
        def get_all_nodes(node_to_serialize):
            text = sd_mecha.serialize(node_to_serialize)
            lines = text.strip().split('\n')
            node_map = {}
            for i in range(1, len(lines)):
                # This is inefficient but robust for finding the node object by index
                node_map[i - 1] = sd_mecha.deserialize(lines[:i + 1])
            return node_map

        node_map = get_all_nodes(start_node)
        return node_map.get(target_line_num)

    target_node = find_node_by_ref(root_node, target_node_ref)

    if not isinstance(target_node, recipe_nodes.MergeRecipeNode):
        return {}

    # Find all base model nodes that are ancestors of this target node
    model_visitor = ModelVisitor()
    target_node.accept(model_visitor)

    model_names = [model.path for model in model_visitor.models]

    return {
        "method_name": target_node.merge_method.identifier,
        "model_names": model_names,
    }


#######################
### Save Artifacts  ###
#######################
def save_merge_artifacts(
        cfg: DictConfig,
        merger: 'Merger',  # fake warning
        final_recipe_node: recipe_nodes.RecipeNode,
        model_path: Path,
        iteration: int
):
    """
    Creates a single, self-contained, and executable Python script that can
    reproduce a merge iteration. It intelligently gathers all necessary components
    from the current run's configuration and recipe graph.
    """
    try:
        scripts_dir = Path(HydraConfig.get().runtime.output_dir) / "merge_artifacts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_file_path = scripts_dir / f"{model_path.stem}_run.py"

        # --- Step 1: Get the Main Merge Method & Its Code ---
        main_method_name = get_main_method_name(cfg, final_recipe_node)
        method_imports, methods_code = get_source_code_for_methods({main_method_name})

        # --- Step 2: Find All Converters Used in the Recipe ---
        converter_names = find_used_converters(final_recipe_node)
        converter_imports, converters_code = get_source_code_for_methods(converter_names)

        # --- Step 3: Find the Custom Config Used ---
        custom_config_name = cfg.optimization_guide.get("custom_block_config_id")
        yaml_content = get_yaml_content(custom_config_name, cfg.configs_dir)

        # --- Step 4: Get Fallback Model Info ---
        fallback_path_str = get_fallback_model_path_str(cfg, merger)

        # --- Step 5: Transpile the Recipe to Python ---
        recipe_python_code = MechaToPythonConverter(sd_mecha.serialize(final_recipe_node)).convert()

        # --- Step 6: Assemble the Final Script ---
        all_imports = method_imports | converter_imports
        final_script_content = build_reproducible_script(
            cfg=cfg,
            merger=merger,
            output_filename=model_path.name,
            iteration=iteration,
            transpiled_recipe=recipe_python_code,
            yaml_name=custom_config_name,
            yaml_content=yaml_content,
            all_imports=all_imports,
            methods_code_block=methods_code,
            converters_code_block=converters_code,
            fallback_model_path_str=fallback_path_str
        )

        # --- Step 7: Write to File ---
        script_file_path.write_text(final_script_content, encoding="utf-8")
        logger.info(f"Saved reproducible script to: {script_file_path}")

    except Exception as e:
        logger.error(f"Failed to save runnable script: {e}", exc_info=True)


def get_main_method_name(cfg: DictConfig, root_node: recipe_nodes.RecipeNode) -> str:
    """Gets the name of the primary merge method being optimized."""
    if cfg.optimization_mode == "merge":
        return cfg.merge_method
    elif cfg.optimization_mode == "recipe":
        # Find the node targeted by the optimizer and get its method name
        target_ref = cfg.recipe_optimization.get("target_nodes")
        if target_ref:
            # This helper finds the node in the graph and returns its method identifier
            info = get_info_from_target_node(root_node, target_ref)
            return info.get("method_name", "unknown_recipe_method")
    return "unknown" # Default for layer_adjust or other modes


class ConverterFinder(recipe_nodes.RecipeVisitor):
    """A targeted visitor to find only the identifiers of CUSTOM conversion methods."""

    def __init__(self):
        self.converter_names: Set[str] = set()
        self.visited: Set[recipe_nodes.RecipeNode] = set()
        self.known_converters = sd_mecha.extensions.merge_methods.get_all_converters()

        # --- ADDITION: We need to know where sd-mecha lives ---
        try:
            self.sd_mecha_path = Path(inspect.getfile(sd_mecha)).parent.resolve()
        except TypeError:
            # Fallback if sd-mecha path can't be found
            self.sd_mecha_path = None
            logger.warning("Could not determine sd-mecha library path. Converter filtering might be inaccurate.")

    def visit(self, node):
        if node not in self.visited:
            self.visited.add(node)
            node.accept(self)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        method_obj = node.merge_method

        # Check if this method is in sd-mecha's list of known converters
        if method_obj in self.known_converters:
            # It's a converter. Now, where is it from?
            try:
                # Get the absolute path of the file the function was defined in
                unwrapped_func = inspect.unwrap(method_obj)
                source_file_path = Path(inspect.getfile(unwrapped_func)).resolve()

                # If we have a valid sd-mecha path, check if the source file is inside it.
                # If it's NOT, then it must be one of ours!
                if self.sd_mecha_path and self.sd_mecha_path not in source_file_path.parents:
                    self.converter_names.add(method_obj.identifier)
                # If we couldn't find the sd-mecha path, we fall back to a simpler check.
                # This is less robust but better than nothing.
                elif self.sd_mecha_path is None and 'sd_mecha' not in str(source_file_path):
                    self.converter_names.add(method_obj.identifier)

            except (TypeError, OSError):
                # Could not get the file path for this function.
                # It's likely a dynamically generated or C-based function.
                # It's safest to assume it's a built-in and ignore it.
                pass

        # Continue traversal for the rest of the recipe
        for arg in node.args: self.visit(arg)
        for kwarg in node.kwargs.values(): self.visit(kwarg)

    # visit_model and visit_literal can remain the same
    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        pass

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        if isinstance(node.value, dict):
            for v in node.value.values():
                if isinstance(v, recipe_nodes.RecipeNode): self.visit(v)


def find_used_converters(root_node: recipe_nodes.RecipeNode) -> Set[str]:
    """Traverses a recipe graph to find all unique conversion methods used."""
    finder = ConverterFinder()
    finder.visit(root_node)
    return finder.converter_names


def get_yaml_content(config_name: Optional[str], configs_dir_path: str) -> Optional[str]:
    """Reads the content of a specific YAML file."""
    if not config_name: return None
    config_file = Path(configs_dir_path) / f"{config_name}.yaml"
    if config_file.is_file():
        return config_file.read_text(encoding="utf-8")
    logger.warning(f"Could not find source for custom config: {config_name}")
    return None


def get_fallback_model_path_str(
        cfg: DictConfig,
        merger: 'Merger'  # fake warn ing
) -> str:
    """Determines the fallback model path string for the script."""
    fallback_index = cfg.get("fallback_model_index", -1)
    if fallback_index is not None and fallback_index != -1 and fallback_index < len(merger.models):
        fallback_node = merger.models[fallback_index]
        return f'"{fallback_node.path}"'
    return "None"


def _format_and_deduplicate_imports(imports: Set[str]) -> str:
    """
    Takes a set of import statement strings and cleans them up,
    merging 'from ... import ...' statements, handling aliasing,
    and removing duplicates and broken relative imports.
    """
    direct_imports = {}  # e.g., {'torch': 'torch', 'numpy': 'np'}
    from_imports = {}  # e.g., {'torch': {'Tensor': None, 'nn': 'nn'}, 'pathlib': {'Path': None}}

    for imp_line in sorted(list(imports)):  # Sort for consistent processing order
        imp_line = imp_line.strip()

        try:
            # Use Python's own AST parser to understand the import line
            tree = ast.parse(imp_line)
            node = tree.body[0]

            if isinstance(node, ast.Import):
                for alias in node.names:
                    # e.g., import numpy as np -> direct_imports['numpy'] = 'np'
                    direct_imports[alias.name] = alias.asname or alias.name

            elif isinstance(node, ast.ImportFrom):
                # Ignore broken relative imports like 'from . import ...'
                if node.level > 0:  # node.level > 0 indicates a relative import (., ..)
                    logger.warning(f"Skipping relative import, it cannot be made portable: '{imp_line}'")
                    continue

                module_name = node.module
                if module_name not in from_imports:
                    from_imports[module_name] = {}

                for alias in node.names:
                    # e.g., from torch import Tensor as T -> from_imports['torch']['Tensor'] = 'T'
                    from_imports[module_name][alias.name] = alias.asname

        except (SyntaxError, IndexError):
            logger.warning(f"Could not parse import line: '{imp_line}'. Skipping.")
            continue

    # --- Now, we rebuild the import block from our structured data ---

    # First, handle cases where a module is both directly imported and has 'from' imports
    # e.g., 'import torch' and 'from torch import Tensor'. We should merge them.
    for module in list(direct_imports.keys()):
        if module in from_imports:
            # We have 'from torch import ...', so the 'import torch' is redundant.
            del direct_imports[module]

    # Rebuild the final, clean import lines
    final_import_lines = []

    # Direct imports (e.g., import re, import numpy as np)
    for module, alias in sorted(direct_imports.items()):
        if module == alias:
            final_import_lines.append(f"import {module}")
        else:
            final_import_lines.append(f"import {module} as {alias}")

    # From imports (e.g., from pathlib import Path)
    for module, names in sorted(from_imports.items()):
        name_parts = []
        for name, alias in sorted(names.items()):
            if name == alias or alias is None:
                name_parts.append(name)
            else:
                name_parts.append(f"{name} as {alias}")

        # This creates a beautifully formatted, multi-line import if it's too long
        import_list_str = ", ".join(name_parts)
        line = f"from {module} import {import_list_str}"
        if len(line) > 88:  # Use a reasonable line length limit
            line = f"from {module} import (\n    " + ",\n    ".join(name_parts) + "\n)"

        final_import_lines.append(line)

    return "\n".join(final_import_lines)


def build_reproducible_script(
        cfg: DictConfig,
        merger: 'Merger',  # fake warning
        output_filename: str,
        iteration: int,
        transpiled_recipe: str,
        yaml_name: Optional[str],
        yaml_content: Optional[str],
        all_imports: Set[str],
        methods_code_block: str,
        converters_code_block: str,
        fallback_model_path_str: str
) -> str:
    """The template that writes a clean, human-readable, and executable Python script."""
    models_dir_str = str(merger.models_dir.resolve())
    merge_device = cfg.get("device", "cpu")

    # Create a small translation dictionary
    dtype_map = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }

    # Get the correct, full names for the dtypes
    # Use .get() to safely fall back to the original value if it's not in our map
    merge_dtype_name = dtype_map.get(cfg.get("merge_dtype", "float64"), "float64")
    save_dtype_name = dtype_map.get(cfg.get("save_dtype", "float16"), "float16")

    # Now we build the full torch string
    merge_dtype_str = f"torch.{merge_dtype_name}"
    save_dtype_str = f"torch.{save_dtype_name}"
    threads_value = cfg.get("threads")

    embedded_yamls = {yaml_name: yaml_content} if yaml_name and yaml_content else {}
    embedded_yamls_str = repr(embedded_yamls)
    import_block = _format_and_deduplicate_imports(all_imports)

    return f'''
# =================================================================
#  Auto-generated by sd-optim for Full Reproducibility
#  Iteration: {iteration}
#  Output Model: {output_filename}
# =================================================================

import sd_mecha
import torch
import yaml
import inspect
import textwrap
import re
import logging
from pathlib import Path
from torch import Tensor
from typing import Optional, Dict, Tuple, Set, List, Any
from sd_mecha.recipe_nodes import RecipeNode
from sd_mecha import Parameter, Return, merge_method, StateDict, recipe_nodes, extensions
from sd_mecha.extensions import merge_methods

# -----------------------------------------------------------------
#  Discovered Imports for Custom Functions
# -----------------------------------------------------------------
{import_block}

# -----------------------------------------------------------------
#  Environment Setup Function
# -----------------------------------------------------------------
def setup_custom_configs():
    """Parses and registers the embedded YAML configs with sd-mecha."""
    embedded_yamls = {embedded_yamls_str}
    if not embedded_yamls:
        print("No custom YAML configs to register.")
        return
    for name, yaml_str in embedded_yamls.items():
        if not name or not yaml_str: continue
        print(f"Registering custom config: {{name}}")
        try:
            config_data = yaml.safe_load(yaml_str)
            sd_mecha.extensions.model_configs.register_aux(
                sd_mecha.extensions.model_configs.ModelConfigImpl(**config_data)
            )
        except Exception as e:
            print(f"  ERROR: Could not register config '{{name}}': {{e}}")
            
# Run setup
setup_custom_configs()

# -----------------------------------------------------------------
#  Custom Merge Methods
# -----------------------------------------------------------------
{methods_code_block}

# -----------------------------------------------------------------
#  Custom Converters
# -----------------------------------------------------------------
{converters_code_block}

# -----------------------------------------------------------------
#  Transpiled sd-mecha Recipe
# -----------------------------------------------------------------
def get_recipe() -> RecipeNode:
    """This function contains the transpiled .mecha recipe."""
{textwrap.indent(transpiled_recipe, "    ")}
    return final_recipe

# -----------------------------------------------------------------
#  Main Execution Blocl
# -----------------------------------------------------------------
def main():
    # Configuration from the original run
    MODELS_DIR = Path(r"{models_dir_str}")
    OUTPUT_FILENAME = "{output_filename.replace(".safetensors", "_external.safetensors")}"
    MERGE_DEVICE = "{merge_device}"
    MERGE_DTYPE = {merge_dtype_str}
    SAVE_DTYPE = {save_dtype_str}
    THREADS = {threads_value}
    FALLBACK_MODEL_PATH = {fallback_model_path_str}

    # Get and execute recipe
    print("Building recipe...")
    recipe_to_run = get_recipe()
    output_path = Path(MODELS_DIR) / OUTPUT_FILENAME

    print(f"Executing merge and saving to {{output_path}}...")
    sd_mecha.merge(
        recipe=recipe_to_run,
        output=output_path,
        fallback_model=sd_mecha.model(FALLBACK_MODEL_PATH) if FALLBACK_MODEL_PATH != "None" else None,
        merge_device=MERGE_DEVICE,
        merge_dtype=MERGE_DTYPE,
        output_dtype=SAVE_DTYPE,
        threads=THREADS,
        model_dirs=[MODELS_DIR],
        check_mandatory_keys=False,
    )
    print("\\nMerge complete!")

if __name__ == "__main__":
    main()
'''


class _CodeParser(ast.NodeVisitor):
    """An AST visitor to find all names used and local methods called within a function's AST."""

    def __init__(self, local_method_names: Set[str]):
        self.used_names: Set[str] = set()
        self.called_local_methods: Set[str] = set()
        self.local_method_names = local_method_names

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        curr_node = node
        while isinstance(curr_node, ast.Attribute):
            curr_node = curr_node.value
        if isinstance(curr_node, ast.Name):
            self.used_names.add(curr_node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.local_method_names:
            self.called_local_methods.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in ('self', 'cls') and func.attr in self.local_method_names:
                self.called_local_methods.add(func.attr)
        self.generic_visit(node)


# --- And now the main function itself ---
def get_source_code_for_methods(method_names: Set[str]) -> Tuple[Set[str], str]:
    """
    Takes a set of method names, finds their source code including their
    original decorators, and discovers all necessary imports and top-level variables.
    """
    if not method_names:
        return set(), ""

    _source_cache: Dict[str, str] = {}

    all_source_code_blocks = []
    all_used_names = set()
    files_to_scan = set()
    top_level_code_to_add = set()  # To hold things like T = TypeVar(...)

    # We need a better parser that finds all names used in any context.
    class FullNameFinder(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node):
            self.names.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # This helps find the root of things like torch.nn.functional -> torch
            try:
                self.names.add(ast.unparse(node))
            except:  # fallback for complex nodes
                pass
            self.generic_visit(node)

    for method_name in sorted(list(method_names)):
        for method_name in sorted(list(method_names)):
            try:
                # Instead of using sd_mecha.resolve(), we use OUR OWN resolver.
                # This ensures we correctly find methods from our MergeMethods class.
                method_obj = resolve_merge_method(method_name)

                # The rest of the logic can now proceed safely
                source_text = inspect.getsource(method_obj.__wrapped__)
                _source_cache[method_name] = textwrap.dedent(source_text)

                source_code = _source_cache[method_name]
                all_source_code_blocks.append(source_code)

                module = inspect.getmodule(inspect.unwrap(method_obj))
                if module and hasattr(module, '__file__'):
                    files_to_scan.add(Path(module.__file__))

                parser = FullNameFinder()
                parser.visit(ast.parse(source_code))
                all_used_names.update(parser.names)

            except (ValueError, TypeError) as e:
                # Now, if resolve_merge_method fails, it will raise a SystemExit,
                # which is better than a silent failure. We can catch this if we want.
                logger.error(f"Could not get or parse source for '{method_name}': {e}")
                all_source_code_blocks.append(f"# ERROR: Could not get source for {method_name}")
            except SystemExit:
                logger.error(
                    f"FATAL: resolve_merge_method could not find '{method_name}'. Halting artifact generation for this method.")
                all_source_code_blocks.append(f"# ERROR: Could not resolve merge method '{method_name}'.")

    # Step 4: Scan the discovered files to find imports AND top-level assignments
    relevant_imports = set()
    for file_path in files_to_scan:
        try:
            file_source = file_path.read_text(encoding="utf-8")
            file_tree = ast.parse(file_source)

            for node in file_tree.body:  # Iterate top-level nodes in the file
                # Case A: It's an import statement
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check if any name provided by this import was used in our functions
                    for alias in node.names:
                        # Check full module path too, e.g., sd_mecha.recipe_nodes
                        potential_names = {alias.name, alias.asname, alias.name.split('.')[0]}
                        if not all_used_names.isdisjoint(potential_names):
                            relevant_imports.add(ast.unparse(node))
                            break

                # Case B: It's a top-level assignment (like T=..., logger=..., re_inp=...)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in all_used_names:
                            top_level_code_to_add.add(ast.unparse(node))
                            break
        except Exception as e:
            logger.error(f"Could not parse file {file_path} for imports/variables: {e}")

    # Combine the discovered code blocks
    final_code_str = "\n".join(sorted(list(top_level_code_to_add))) + "\n\n" + "\n\n".join(all_source_code_blocks)

    return relevant_imports, final_code_str


class MechaToPythonConverter:
    """
    Translates (trans-piles) a .mecha recipe string into a syntactically correct,
    standalone Python script that uses the sd-mecha API.
    """

    def __init__(self, recipe_str: str):
        self.mecha_lines = recipe_str.strip().split('\n')[1:]
        self.python_lines = []

    def convert(self) -> str:
        for i, line in enumerate(self.mecha_lines):
            var_name = f"var_{i}"
            parts = self._parse_line(line)
            command = parts[0]
            python_line = f"# Original: {line}"

            if command == "dict":
                _args, kwargs = self._extract_args_kwargs(parts[1:])

                # --- THIS IS THE ONLY CHANGE WE NEED TO MAKE ---
                # It uses a colon ':' and puts quotes around the key. That's it.
                dict_items = [f'"{k}": {self._remap_ref(v)}' for k, v in kwargs.items()]
                python_line += f"\n{var_name} = {{{', '.join(dict_items)}}}"

            elif command in ["model", "literal"]:
                # This block was already correct. We leave it alone.
                args, kwargs = self._extract_args_kwargs(parts[1:])
                if 'model_config' in kwargs:
                    kwargs['config'] = kwargs.pop('model_config')
                remapped_args = [self._remap_ref(arg) for arg in args]
                remapped_kwargs = {k: self._remap_ref(v) for k, v in kwargs.items()}
                call_args = ", ".join(remapped_args)
                if remapped_kwargs:
                    if call_args: call_args += ", "
                    call_args += ", ".join(f"{k}={v}" for k, v in remapped_kwargs.items())
                python_line += f'\n{var_name} = sd_mecha.{command}({call_args})'

            elif command == "merge":
                # This block was also correct. We leave it alone.
                method_identifier_str = parts[1]
                args, kwargs = self._extract_args_kwargs(parts[2:])
                remapped_args = [self._remap_ref(arg) for arg in args]
                remapped_kwargs = {k: self._remap_ref(v) for k, v in kwargs.items()}
                call_args = ", ".join(remapped_args)
                if remapped_kwargs:
                    if call_args: call_args += ", "
                    call_args += ", ".join(f"{k}={v}" for k, v in remapped_kwargs.items())
                python_line += f'\n{var_name} = sd_mecha.extensions.merge_methods.resolve({method_identifier_str})({call_args})'

            else:
                python_line += f'\n# SKIPPED UNKNOWN COMMAND: {line}'

            # The append at the end was the cause of the duplicate "SKIPPED" line.
            # We fix this by only appending if the command was recognized.
            if command in ["dict", "model", "literal", "merge"]:
                self.python_lines.append(python_line)
            else:  # for unknown commands
                self.python_lines.append(f"# SKIPPED UNKNOWN COMMAND: {line}")

        final_var_name = f"var_{len(self.mecha_lines) - 1}"
        self.python_lines.append(f"\n# The final recipe is held in this variable")
        self.python_lines.append(f"final_recipe = {final_var_name}")

        return "\n\n".join(self.python_lines)

    def _parse_line(self, line: str) -> List[str]:
        """A simple parser that respects quotes."""
        return re.findall(r'"[^"]*"|\S+', line)

    def _extract_args_kwargs(self, parts: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Separates a list of parts into positional and keyword arguments."""
        args = []
        kwargs = {}
        for part in parts:
            if "=" in part:
                key, val = part.split('=', 1)
                kwargs[key] = val
            else:
                args.append(part)
        return args, kwargs

    def _remap_ref(self, ref_str: str) -> str:
        """Converts a '&N' reference to a 'var_N' variable name."""
        if ref_str.startswith('&') and ref_str[1:].isdigit():
            index = int(ref_str[1:])
            return f"var_{index}"
        # Return the original string (it might be a literal number or a quoted string)
        return ref_str


#############################
### Generate optuna notes ###
#############################
# def load_config(run_dir):
#     # Logic to find and load .hydra/config.yaml or config.yaml
#     # Return the loaded config dictionary/omegaconf object
#     pass
#
# def load_optuna_study(config, run_dir):
#     # Logic to determine DB path and study name from config
#     # Call optuna.load_study
#     # Return study object
#     pass
#
# def load_bayes_results(config, run_dir):
#     # Logic to find and parse .jsonl or .pkl
#     # Return list of results
#     pass
#
# def format_markdown(config, study_or_results, run_dir):
#     # --- Extract data ---
#     run_name = config.get('run_name', 'Unknown')
#     models = config.get('model_paths', [])
#     # ... extract all needed info ...
#     best_trial = study_or_results.best_trial # Example for Optuna
#     best_score = best_trial.value
#     best_params_str = json.dumps(best_trial.params, indent=4) # Format params nicely
#     num_trials = len(study_or_results.trials)
#     # ... etc ...
#
#     # --- Build Markdown String ---
#     md = f"""
# # Notes: Optuna Run - {run_name}
#
# **Date:** {run_dir.name[:10]} # Extract from dir name? Or get current date?
#
# **Setup:**
# *   **Study Name:** {study_or_results.study_name}
# *   **Models:** {', '.join(models)}
# *   **Merge Method:** {config.get('merge_method', 'N/A')}
# *   **Scorer(s):** {config.get('scorer_method', [])}
# *   **Optimizer:** {config.get('optimizer', {})} # Display optimizer type/settings
# *   **Plan:** {config.optimizer.get('init_points', '?')} init + {config.optimizer.get('n_iters', '?')} iters
#
# **Run Status:**
# *   Completed Trials: {num_trials}
# *   Status: [TODO: Add status - Finished / Interrupted / Crashed?]
#
# **Performance Summary:**
# *   **Best Score:** {best_score:.5f} (Trial {best_trial.number})
# *   **Score Range (EDF):** [TODO: Add min/max or describe EDF shape]
# *   **Convergence Trend:** [TODO: Describe history plot]
#
# **Best Trial (#{best_trial.number}) Analysis:**
# *   **Parameters:**
#     ```json
#     {best_params_str}
#     ```
# *   **Observations:** [TODO: Add manual analysis]
#
# **Issues Encountered:**
# *   [TODO: Check logs or add manually - e.g., butterfly_merge error?]
# *   [TODO: e.g., tkinter errors?]
# *   [TODO: e.g., Background check failures?]
#
# **Potential Next Steps:**
# *   [TODO: e.g., Resume run?]
# *   [TODO: e.g., Fix issues?]
# *   [TODO: e.g., Refine search space?]
# """
#     return md
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate notes template for sd-optim run.")
#     parser.add_argument("run_dir", type=str, help="Path to the Hydra run directory.")
#     args = parser.parse_args()
#
#     run_directory = Path(args.run_dir).resolve()
#     if not run_directory.is_dir():
#         print(f"Error: Directory not found: {run_directory}")
#         exit(1)
#
#     try:
#         # Chdir might be needed if using HydraConfig relative paths
#         # os.chdir(run_directory) # Or load config relative to run_directory
#         cfg = load_config(run_directory) # Implement this
#         if cfg.optimizer.get("optuna"):
#             study = load_optuna_study(cfg, run_directory) # Implement this
#             results_obj = study
#         elif cfg.optimizer.get("bayes"):
#             results = load_bayes_results(cfg, run_directory) # Implement this
#             # Need to structure Bayes results similarly or adapt formatting
#             results_obj = results # Placeholder
#         else:
#             raise ValueError("Unknown optimizer type in config")
#
#         markdown_content = format_markdown(cfg, results_obj, run_directory)
#
#         output_path = run_directory / "run_notes_template.md"
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(markdown_content)
#         print(f"Successfully generated notes template: {output_path}")
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         # Print traceback for debugging
#         import traceback
#         traceback.print_exc()

# ### Cache for recipes (does it work tho?) ###
# class CacheInjectorVisitor(RecipeVisitor):
#     def __init__(self, cache: Optional[Dict[str, Dict[str, Tensor]]]):
#         self.cache = cache
#
#     def visit_model(self, node: ModelRecipeNode, *args, **kwargs) -> ModelRecipeNode:
#         # Return the original ModelRecipeNode without modification
#         return node
#
#     def visit_parameter(self, node: ParameterRecipeNode, *args, **kwargs) -> ParameterRecipeNode:
#         return node
#
#     def visit_merge(self, node: MergeRecipeNode, *args, **kwargs) -> MergeRecipeNode:
#         # Recursively visit and reconstruct child nodes
#         new_models = [model.accept(self, *args, **kwargs) for model in node.models]
#
#         # Check if the merge method supports caching
#         merge_method_signature = inspect.signature(node.merge_method)
#
#         # Create a copy of volatile_hypers and add cache if supported
#         new_volatile_hypers = node.volatile_hypers.copy()
#         if 'cache' in merge_method_signature.parameters:
#             new_volatile_hypers['cache'] = self.cache
#
#         # Reconstruct the merge node with modified volatile_hypers
#         return MergeRecipeNode(
#             node.merge_method,
#             *new_models,
#             hypers=node.hypers,
#             volatile_hypers=new_volatile_hypers,
#             device=node.device,
#             dtype=node.dtype,
#         )
#

####################
### Layer tuning ###
####################
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
