import inspect
import json
import os
import pathlib
import textwrap
import torch
import safetensors.torch
import sd_mecha
import logging
import ast
import torch
import yaml
import threading

from pathlib import Path
from typing import List, Tuple, Dict, Set, Union, Any, ClassVar, Optional
from dataclasses import field, dataclass

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pynput import keyboard
from copy import deepcopy

from sd_mecha import recipe_nodes, recipe_serializer
from sd_mecha.recipe_nodes import RecipeNode, MergeRecipeNode, RecipeVisitor, ModelRecipeNode, ParameterRecipeNode
from sd_mecha.hypers import Hyper

logger = logging.getLogger(__name__)


### What to pik
OPTIMIZABLE_HYPERPARAMETERS = {
    "lu_merge": ["alpha", "theta"],
    "orth_pro": ["alpha"],
    "clyb_merge": ["alpha"],
    "ties_sum_extended": ["k"],
    "streaming_ties_sum_extended": ["k"],
    "ties_sum_with_dropout": ["probability", "della_eps", "k", "rescale"],
    "slerp_norm_sign": ["alpha"],
    "polar_interpolate": ["alpha"],
    "wavelet_packet_merge": ["alpha"],
    "multi_domain_alignment": ["alpha", "beta"],
    "merge_layers": ["alpha"],
    # ... other methods and their optimizable hyperparameters
}


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


### Custom sorting function that uses component order from config ###
def custom_sort_key(key, component_order):
    # Extract the component (txt, txt2, unet, etc.) and parameter from the key
    parts = key.split("_")
    component = parts[1]  # Assumes component is the second element

    # Determine the index of the component in the order from the config
    component_index = component_order.index(component) if component in component_order else len(component_order)

    # Apply natural sorting to the rest of the key
    block_key = "_".join(parts[2:])

    return component_index, sd_mecha.hypers.natural_sort_key(block_key)


### Save mm code
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
    def save_merge_method_code(cls, merge_mode: str, model_path: Path, class_obj: Any) -> None:
        """Save the merge method code and its dependencies to a file if not already saved."""
        if merge_mode in cls._saved_methods:
            logger.debug(f"Merge method {merge_mode} already saved in this run, skipping...")
            return

        try:
            if not hasattr(class_obj, merge_mode):
                raise AttributeError(f"Method '{merge_mode}' not found in class {class_obj.__name__}")

            # Determine log directory
            log_dir = Path(os.getcwd()) if "HydraConfig" not in globals() else Path(
                HydraConfig.get().runtime.output_dir)
            merge_code_dir = log_dir / "merge_methods"
            os.makedirs(merge_code_dir, exist_ok=True)

            # Get the complete source code including dependencies
            full_source = cls.get_full_method_source(merge_mode, class_obj)
            if not full_source.strip():
                logger.warning(f"Source code for merge method '{merge_mode}' is empty.")
                return

            # Dedent and clean the source code to ensure consistent formatting
            full_source_cleaned = textwrap.dedent(full_source)

            # Save the merge method code
            iteration_file_name = model_path.stem
            code_file_path = merge_code_dir / f"{iteration_file_name}_merge_method.py"

            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Merge method: {merge_mode}\n")
                f.write(f"# Used in merge: {iteration_file_name}\n")
                f.write("# This file includes the main merge method and all its dependencies\n\n")
                f.write(full_source_cleaned)

            logger.info(f"Saved merge method code to {code_file_path}")
            cls._saved_methods.add(merge_mode)

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