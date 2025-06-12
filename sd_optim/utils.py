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
class RecipePatcher(recipe_nodes.RecipeVisitor):
    """
    Traverses an existing recipe graph and builds a new one, patching in
    new logic for a specific parameter on a target node. This operates
    entirely on RecipeNode objects, avoiding text manipulation.
    """

    def __init__(self,
                 target_merge_node: recipe_nodes.MergeRecipeNode,
                 target_param_name: str,
                 new_param_node: recipe_nodes.RecipeNode):

        self.target_merge_node = target_merge_node
        self.target_param_name = target_param_name
        self.new_param_node = new_param_node
        self.memo: Dict[recipe_nodes.RecipeNode, recipe_nodes.RecipeNode] = {}

    def visit(self, node: recipe_nodes.RecipeNode) -> recipe_nodes.RecipeNode:
        if node in self.memo:
            return self.memo[node]
        new_node = node.accept(self)
        self.memo[node] = new_node
        return new_node

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode) -> recipe_nodes.RecipeNode:
        # This is correct. Terminal nodes are returned as-is.
        return node

    def visit_model(self, node: recipe_nodes.ModelRecipeNode) -> recipe_nodes.RecipeNode:
        # This is also correct.
        return node

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode) -> recipe_nodes.RecipeNode:
        if node in self.memo:
            return self.memo[node]

        # Rebuild children first by recursively visiting them.
        new_args = tuple(arg.accept(self) for arg in node.args)
        new_kwargs = {k: v.accept(self) for k, v in node.kwargs.items()}

        rebuilt_node = recipe_nodes.MergeRecipeNode(node.merge_method, new_args, new_kwargs, node.cache)

        # NOW, check if this REBUILT node is our target.
        if node is self.target_merge_node:
            logger.debug(
                f"PATCHER: Found target node {node.merge_method.identifier}. Patching param '{self.target_param_name}'.")

            # This is the final, correct patching logic.
            # We take the freshly rebuilt node and create a final version with our new parameter.
            param_info = rebuilt_node.merge_method.get_param_names()
            final_args = list(rebuilt_node.args)
            final_kwargs = rebuilt_node.kwargs.copy()

            patched = False
            if self.target_param_name in param_info.args:
                idx = param_info.args.index(self.target_param_name)
                if idx < len(final_args):
                    final_args[idx] = self.new_param_node
                    patched = True

            if not patched and self.target_param_name in param_info.kwargs:
                final_kwargs[self.target_param_name] = self.new_param_node
                patched = True

            if not patched:
                # This handles implicit defaults.
                final_kwargs[self.target_param_name] = self.new_param_node

            rebuilt_node = recipe_nodes.MergeRecipeNode(rebuilt_node.merge_method, tuple(final_args), final_kwargs,
                                                        rebuilt_node.cache)

        self.memo[node] = rebuilt_node
        return rebuilt_node


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
# THIS SEQUENCE NEEDS WORK
def save_merge_artifacts(
        cfg: DictConfig,
        merger: 'Merger',
        final_recipe_node: recipe_nodes.RecipeNode,
        model_path: Path,
        iteration: int
):
    """
    Creates a single, self-contained, and executable Python script that documents
    and can reproduce a merge iteration, including all embedded custom components.
    """
    try:
        log_dir = Path(HydraConfig.get().runtime.output_dir)
        # Using the name we agreed on!
        scripts_dir = log_dir / "merge_artifacts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_file_path = scripts_dir / f"{model_path.stem}_run.py"

        # --- Step 1: Find all custom components using our smarter visitor ---
        finder = ComponentFinder()
        final_recipe_node.accept(finder)

        # Now finder has `custom_configs`, `custom_converters`, and `custom_methods` populated.

        # --- Step 2: Bundle all the environment artifacts we found ---
        embedded_yamls = _bundle_yaml_configs(finder.custom_configs, cfg)
        embedded_converters = _bundle_python_converters(finder.custom_converters | finder.custom_methods)

        # --- Step 3: Get the source for the *main* custom merge method, if it exists ---
        # This is for the human-readable section at the top of the script.
        # We'll just grab the first one from the set for now.
        method_source_code = "# No primary custom merge method identified."
        if finder.custom_methods:
            main_method_name = next(iter(finder.custom_methods))
            method_source_code = MergeMethodCodeSaver.get_full_method_source(
                main_method_name, MergeMethods, model_path
            )
        else:
            # If no main custom method, maybe log the recipe's top-level method
            if isinstance(final_recipe_node, recipe_nodes.MergeRecipeNode):
                method_name = final_recipe_node.merge_method.identifier
                method_source_code = f"# Top-level merge method was a built-in: {method_name}"

        # --- Step 4: Transpile the recipe object to code ---
        recipe_str = sd_mecha.serialize(final_recipe_node)
        converter = MechaToPythonConverter(recipe_str)
        transpiled_recipe_body = converter.convert()

        # --- Step 5: Assemble the final script using the template ---
        # We pass all the collected artifacts to the template.
        final_script_content = _create_unified_script_template(
            cfg, merger, model_path.name, iteration,
            method_source_code, transpiled_recipe_body,
            embedded_yamls, embedded_converters
        )

        with open(script_file_path, "w", encoding="utf-8") as f:
            f.write(final_script_content)
        logger.info(f"Saved complete runnable script to: {script_file_path}")

    except Exception as e:
        logger.error(f"Failed to save runnable script: {e}", exc_info=True)


def _bundle_yaml_configs(config_names: Set[str], cfg: DictConfig) -> Dict[str, str]:
    """Finds YAML source files and reads their content."""
    embedded_yamls = {}
    configs_dir = Path(cfg.configs_dir)
    if not configs_dir.is_dir():
        logger.warning(f"Custom configs directory not found: {configs_dir}")
        return {}

    for name in config_names:
        config_file_path = configs_dir / f"{name}.yaml"
        if config_file_path.exists():
            embedded_yamls[name] = config_file_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Could not find source file for custom config: {name}")
    return embedded_yamls


def _bundle_python_converters(method_names: Set[str]) -> Dict[str, str]:
    """Finds functions in memory and extracts their source code."""
    embedded_code = {}
    for name in method_names:
        try:
            method_obj = sd_mecha.extensions.merge_methods.resolve(name)
            # We get the source of the original function, before decorators
            source_code = inspect.getsource(method_obj.__wrapped__)

            # Reconstruct the decorator to ensure it's correct
            decorator = f'@merge_method(identifier="{name}"'
            if getattr(method_obj, 'is_conversion', False):
                decorator += ', is_conversion=True'
            decorator += ')\n'

            embedded_code[name] = decorator + textwrap.dedent(source_code)
        except Exception as e:
            logger.warning(f"Could not find source for custom method/converter '{name}': {e}")
    return embedded_code


# --- This is the corrected template function ---
def _create_unified_script_template(
        cfg: DictConfig,
        merger: 'Merger', # fake warning
        output_filename: str,
        iteration: int,
        method_source: str,
        transpiled_recipe: str,
        embedded_yamls: Dict[str, str],
        embedded_converters: Dict[str, str]
) -> str:
    """The new template that correctly embeds all code and artifacts."""
    models_dir_str = str(merger.models_dir.resolve())
    merge_device = cfg.get("device", "cpu")
    merge_dtype_str = f"torch.{cfg.get('merge_dtype', 'float64')}"
    save_dtype_str = f"torch.{cfg.get('save_dtype', 'float16')}"

    # Use repr() to safely escape the strings for embedding
    embedded_yamls_str = repr(embedded_yamls)
    embedded_converters_str = repr(embedded_converters)

    # THE FIX IS HERE: We no longer wrap method_source in a """...""" docstring.
    # We just indent it so it becomes part of the script's executable code.
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
from pathlib import Path
from torch import Tensor
from sd_mecha import Parameter, Return, merge_method

# --- Configuration from the original run ---
MODELS_DIR = r"{models_dir_str}"
OUTPUT_FILENAME = "{output_filename.replace(".safetensors", "_rerun.safetensors")}"
MERGE_DEVICE = "{merge_device}"
MERGE_DTYPE = {merge_dtype_str}
SAVE_DTYPE = {save_dtype_str}

# -----------------------------------------------------------------
#  Embedded Environment Artifacts
# -----------------------------------------------------------------
EMBEDDED_YAML_CONFIGS = {embedded_yamls_str}
EMBEDDED_PYTHON_CONVERTERS = {embedded_converters_str}

# -----------------------------------------------------------------
#  Embedded Custom Merge Method Source
# -----------------------------------------------------------------
# The following code is executed to define the custom merge method in this script's scope.
{textwrap.indent(method_source, "")}

# -----------------------------------------------------------------
#  Setup Function to Recreate the Environment
# -----------------------------------------------------------------
def setup_environment():
    """Parses and registers the embedded artifacts with sd-mecha at runtime."""
    print("--- Setting up custom environment ---")

    for name, yaml_str in EMBEDDED_YAML_CONFIGS.items():
        print(f"Registering config: {{name}}")
        config_data = yaml.safe_load(yaml_str)
        sd_mecha.extensions.model_configs.register_aux(
            sd_mecha.extensions.model_configs.ModelConfigImpl(**config_data)
        )

    for name, code_str in EMBEDDED_PYTHON_CONVERTERS.items():
        print(f"Registering converter/method: {{name}}")
        exec(code_str, globals())

    print("--- Custom environment setup complete ---")

# -----------------------------------------------------------------
#  Transpiled sd-mecha Recipe
# -----------------------------------------------------------------
def get_recipe() -> sd_mecha.RecipeNode:
    """This function contains the transpiled .mecha recipe."""
{textwrap.indent(transpiled_recipe, "    ")}
    return final_recipe

# -----------------------------------------------------------------
#  Main Execution Block
# -----------------------------------------------------------------
def main():
    setup_environment()
    recipe_to_run = get_recipe()
    output_path = Path(MODELS_DIR) / OUTPUT_FILENAME

    print(f"Executing merge and saving to {{output_path}}...")
    sd_mecha.merge(
        recipe_to_run,
        output=output_path,
        merge_device=MERGE_DEVICE,
        merge_dtype=MERGE_DTYPE,
        output_dtype=SAVE_DTYPE,
        model_dirs=[MODELS_DIR],
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
        """Catches variables, functions, and modules being used."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Catches attribute access like `torch.nn` to get the root `torch`."""
        # Traverse down to the root of an attribute chain, e.g., torch.nn.functional -> torch
        curr_node = node
        while isinstance(curr_node, ast.Attribute):
            curr_node = curr_node.value
        if isinstance(curr_node, ast.Name):
            self.used_names.add(curr_node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Catches function calls to identify dependencies on other local methods."""
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.local_method_names:
            self.called_local_methods.add(func.id)
        # Also check for `self.method_name()` calls
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in ('self', 'cls') and func.attr in self.local_method_names:
                self.called_local_methods.add(func.attr)
        self.generic_visit(node)


class MergeMethodCodeSaver:
    """
    Analyzes and saves the source code of a merge method and all its local
    dependencies into a self-contained, readable Python script.
    """
    _source_cache: Dict[str, str] = {}
    _dependency_cache: Dict[str, Set[str]] = {}
    _imports_cache: Dict[str, str] = {}

    @classmethod
    def _get_method_dependencies(cls, method_name: str, class_obj: Any) -> Set[str]:
        """Recursively finds all local methods called by a given method."""
        if method_name in cls._dependency_cache:
            return cls._dependency_cache[method_name]

        all_local_methods = {name for name, func in inspect.getmembers(class_obj, inspect.isfunction)}

        try:
            method = getattr(class_obj, method_name)
            source = inspect.getsource(method)
            tree = ast.parse(textwrap.dedent(source))

            parser = _CodeParser(all_local_methods)
            parser.visit(tree)

            dependencies = parser.called_local_methods
            # Recursively get dependencies of dependencies
            recursive_deps = set(dependencies)
            for dep_name in dependencies:
                recursive_deps.update(cls._get_method_dependencies(dep_name, class_obj))

            cls._dependency_cache[method_name] = recursive_deps
            return recursive_deps
        except (TypeError, OSError, AttributeError):
            return set()

    @classmethod
    def _get_source(cls, method_name: str, class_obj: Any) -> str:
        """Gets and caches the dedented source code of a method."""
        if method_name not in cls._source_cache:
            try:
                method = getattr(class_obj, method_name)
                cls._source_cache[method_name] = textwrap.dedent(inspect.getsource(method))
            except (TypeError, OSError, AttributeError):
                cls._source_cache[method_name] = f"# Could not retrieve source for {method_name}\n"
        return cls._source_cache[method_name]


    @classmethod
    def get_full_method_source(cls, merge_method_name: str, class_obj: Any, model_path: Path) -> str:
        """
        Analyzes a method and its dependencies, returning a single string
        containing all necessary imports and source code.
        """
        if not hasattr(class_obj, merge_method_name):
            return f"# Method '{merge_method_name}' not found in {class_obj.__name__}."

        # Step 1: Find all function dependencies recursively.
        dependencies = cls._get_method_dependencies(merge_method_name, class_obj)
        all_methods_to_save = {merge_method_name} | dependencies

        # Step 2: Get the source code for all needed functions and parse their ASTs
        # to find out what names (e.g., 'torch', 'rankdata') they use.
        all_used_names = set()
        all_source_code = {}
        all_local_method_names = {name for name, _ in inspect.getmembers(class_obj, inspect.isfunction)}

        for method_name in all_methods_to_save:
            source = cls._get_source(method_name, class_obj)
            all_source_code[method_name] = source
            try:
                tree = ast.parse(source)
                parser = _CodeParser(all_local_method_names)
                parser.visit(tree)
                all_used_names.update(parser.used_names)
            except SyntaxError:
                logger.warning(f"Could not parse AST for method '{method_name}'. Import filtering may be incomplete.")

        # Step 3: Parse the entire source file to find all possible imports.
        relevant_imports = []
        try:
            source_file_path = inspect.getfile(class_obj)
            with open(source_file_path, 'r', encoding='utf-8') as f:
                file_source = f.read()
            file_tree = ast.parse(file_source)

            # Walk through the AST of the whole file...
            for node in ast.walk(file_tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    is_relevant = False
                    for alias in node.names:
                        if (alias.asname or alias.name) in all_used_names:
                            is_relevant = True
                            break
                    if is_relevant:
                        # If it's relevant, add the line of code to our list.
                        relevant_imports.append(ast.unparse(node))  # fake alert
        except Exception as e:
            logger.error(f"Could not automatically parse imports: {e}")
            relevant_imports = ["# ERROR: Could not parse imports automatically. Please add them manually."]

        # Step 4: Assemble the final script string with all the parts.
        output_lines = [
            f"# Merge Method: {merge_method_name}",
            f"# Source generated for model: {model_path.name}",
            "\n# --- Relevant Imports ---",
            *sorted(list(set(relevant_imports))),  # Use set to ensure unique imports
            "\n# --- Helper Functions (Dependencies) ---\n"
        ]

        # Add the source code for all helper functions.
        for method_name in sorted(list(dependencies)):
            output_lines.append(all_source_code[method_name])
            output_lines.append("\n")

        # Add the source code for the main method last.
        output_lines.extend([
            "# --- Main Merge Method ---\n",
            all_source_code[merge_method_name]
        ])

        return "\n".join(output_lines)


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

            # --- This logic is now much smarter ---
            if command in ["model", "literal", "dict"]:
                args, kwargs = self._extract_args_kwargs(parts[1:])
                # Remap all references in both args and kwargs
                remapped_args = [self._remap_ref(arg) for arg in args]
                remapped_kwargs = {k: self._remap_ref(v) for k, v in kwargs.items()}

                # Build the final call string
                call_args = ", ".join(remapped_args)
                if remapped_kwargs:
                    if call_args: call_args += ", "
                    call_args += ", ".join(f"{k}={v}" for k, v in remapped_kwargs.items())

                python_line += f'\n{var_name} = sd_mecha.{command}({call_args})'

            elif command == "merge":
                # Sanitize the method name by removing quotes
                method_name = parts[1].strip('"')

                args, kwargs = self._extract_args_kwargs(parts[2:])
                remapped_args = [self._remap_ref(arg) for arg in args]
                remapped_kwargs = {k: self._remap_ref(v) for k, v in kwargs.items()}

                call_args = ", ".join(remapped_args)
                if remapped_kwargs:
                    if call_args: call_args += ", "
                    call_args += ", ".join(f"{k}={v}" for k, v in remapped_kwargs.items())

                python_line += f'\n{var_name} = sd_mecha.{method_name}({call_args})'

            else:
                python_line += f'\n# SKIPPED UNKNOWN COMMAND: {line}'

            self.python_lines.append(python_line)

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


class ComponentFinder(recipe_nodes.RecipeVisitor):
    def __init__(self):
        self.custom_configs = set()
        self.custom_converters = set()
        self.custom_methods = set()  # <-- ADDED: A new set for our main methods!
        self.visited = set()

        try:
            import sd_mecha
            self.sd_mecha_path = Path(inspect.getfile(sd_mecha)).parent
        except Exception:
            self.sd_mecha_path = None

        try:
            import sd_optim
            self.our_project_path = Path(inspect.getfile(sd_optim)).parent
        except Exception:
            self.our_project_path = Path.cwd()

    def visit(self, node):
        if node in self.visited: return
        self.visited.add(node)
        node.accept(self)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        method_obj = node.merge_method

        try:
            method_file = Path(inspect.getfile(method_obj.__wrapped__)).resolve()

            # Check if the method is ours
            if self.our_project_path in method_file.parents and \
                    (self.sd_mecha_path is None or self.sd_mecha_path not in method_file.parents):

                # Now we check the flag that the @merge_method decorator sets.
                if getattr(method_obj, 'is_conversion', False):
                    logger.debug(f"Identified our custom converter: {method_obj.identifier}")
                    self.custom_converters.add(method_obj.identifier)
                else:
                    logger.debug(f"Identified our custom merge method: {method_obj.identifier}")
                    self.custom_methods.add(method_obj.identifier)

        except (TypeError, OSError):
            pass  # This is a built-in, ignore it.

        # Continue traversal
        for arg in node.args: arg.accept(self)
        for kwarg in node.kwargs.values(): kwarg.accept(self)

    # visit_model and visit_literal remain the same.
    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        # We can make this check more robust too.
        try:
            # We assume aux configs are ours.
            # This requires sd-mecha's registry to be populated.
            all_base_configs = {c.identifier for c in sd_mecha.extensions.model_configs.get_all_base()}
            if node.model_config and node.model_config.identifier not in all_base_configs:
                self.custom_configs.add(node.model_config.identifier)
        except Exception:
            # Fallback if registry isn't ready
            if node.model_config and "sd1" not in node.model_config.identifier and "sdxl" not in node.model_config.identifier:
                self.custom_configs.add(node.model_config.identifier)

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        # same logic as visit_model
        try:
            all_base_configs = {c.identifier for c in sd_mecha.extensions.model_configs.get_all_base()}
            if node.model_config and node.model_config.identifier not in all_base_configs:
                self.custom_configs.add(node.model_config.identifier)
        except Exception:
            if node.model_config and "sd1" not in node.model_config.identifier and "sdxl" not in node.model_config.identifier:
                self.custom_configs.add(node.model_config.identifier)
        if isinstance(node.value, dict):
            for v in node.value.values():
                if isinstance(v, recipe_nodes.RecipeNode):
                    v.accept(self)


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
