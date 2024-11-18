import inspect
import json
import os
import pathlib
import sd_mecha
import logging

import threading
from pathlib import Path
from typing import List, Tuple, Dict, Set, Union, Any

from hydra.core.hydra_config import HydraConfig
from pynput import keyboard
from copy import deepcopy

from sd_mecha import recipe_nodes, recipe_serializer
from sd_mecha.recipe_nodes import RecipeNode, MergeRecipeNode, RecipeVisitor, ModelRecipeNode, ParameterRecipeNode
from sd_mecha.hypers import Hyper

logger = logging.getLogger(__name__)


### for methods that require selective optimization, ie contains on/off hypers, learning rate hypers, and such
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
class NodeCollectorVisitor(RecipeVisitor):
    """Enhanced visitor that collects nodes and provides validation methods."""

    def __init__(self):
        self.nodes_by_ref = {}
        self.current_index = 0

    def visit_model(self, node: ModelRecipeNode):
        ref = f"&{self.current_index}"
        self.nodes_by_ref[ref] = node
        self.current_index += 1

    def visit_parameter(self, node: ParameterRecipeNode):
        ref = f"&{self.current_index}"
        self.nodes_by_ref[ref] = node
        self.current_index += 1

    def visit_merge(self, node: MergeRecipeNode):
        for model in node.models:
            model.accept(self)

        ref = f"&{self.current_index}"
        self.nodes_by_ref[ref] = node
        self.current_index += 1


def get_target_nodes_hypers(recipe_path: Union[str, pathlib.Path], target_nodes: List[str]) -> Dict[str, Hyper]:
    """
    Retrieves hyperparameters from specified target nodes in a recipe.

    Args:
        recipe_path: Path to the recipe file
        target_nodes: List of node references (e.g., ["&6", "&10"])

    Returns:
        Dictionary mapping hyperparameter names to their values

    Raises:
        ValueError: If target node not found
        TypeError: If target node is not a MergeRecipeNode
    """
    recipe = recipe_serializer.deserialize(recipe_path)

    # Collect all nodes and their references
    collector = NodeCollectorVisitor()
    recipe.accept(collector)

    # Extract hypers from target nodes
    extracted_hypers = {}
    for node_ref in target_nodes:
        if node_ref not in collector.nodes_by_ref:
            raise ValueError(f"Target node '{node_ref}' not found in recipe.")

        target_node = collector.nodes_by_ref[node_ref]
        if not isinstance(target_node, MergeRecipeNode):
            raise TypeError(
                f"Target node '{node_ref}' must be a MergeRecipeNode, "
                f"not {type(target_node).__name__}"
            )

        extracted_hypers.update(target_node.hypers)

    return extracted_hypers


class RecipeUpdaterVisitor(RecipeVisitor):
    """Visitor that updates hyperparameters in specified target nodes."""

    def __init__(self, target_nodes: List[str], assembled_params: Dict[str, Dict[str, float]]):
        self.target_nodes = target_nodes
        self.assembled_params = assembled_params
        self.current_index = 0

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        self.current_index += 1
        return node

    def visit_parameter(self, node: recipe_nodes.ParameterRecipeNode):
        self.current_index += 1
        return node

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        # First process all child nodes
        updated_models = [model.accept(self) for model in node.models]

        # Check if current node needs updating
        node_ref = f"&{self.current_index}"
        self.current_index += 1

        if node_ref in self.target_nodes:
            # Create new node with updated hypers
            return MergeRecipeNode(
                merge_method=node.merge_method,
                *updated_models,
                hypers={**node.hypers, **self.assembled_params},
                volatile_hypers=node.volatile_hypers,
                device=node.device,
                dtype=node.dtype
            )

        return node


def update_recipe_with_params(
    recipe: RecipeNode,
    target_nodes: Union[str, List[str]],
    assembled_params: Dict[str, Union[float, Dict[str, float]]]
) -> RecipeNode:
    """Updates the recipe, handling all hyperparameter modification scenarios."""

    if isinstance(target_nodes, str):
        target_nodes = [target_nodes.strip()]

    collector = NodeCollectorVisitor()
    recipe.accept(collector)

    updated_recipe = deepcopy(recipe)

    class UpdatingVisitor(recipe_nodes.RecipeVisitor):
        def __init__(self, target_nodes, assembled_params, nodes_by_ref):
            self.target_nodes = target_nodes
            self.assembled_params = assembled_params
            self.current_index = 0
            self.nodes_by_ref = nodes_by_ref

        def visit_model(self, node: recipe_nodes.ModelRecipeNode):
            self.current_index += 1

        def visit_parameter(self, node: recipe_nodes.ParameterRecipeNode):
            self.current_index += 1

        def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
            node_ref = f"&{self.current_index}"
            self.current_index += 1

            if node_ref in self.target_nodes:
                for key, value in self.assembled_params.items():
                    if key in node.hypers and isinstance(node.hypers[key], dict) and isinstance(value, dict):
                        node.hypers[key].update(value)  # Merge nested dictionaries.
                    else:
                        node.hypers[key] = value  # Set values directly or for non-nested dicts.
            for model in node.models:  # Go through the tree
                model.accept(self)

    updated_recipe.accept(UpdatingVisitor(target_nodes, assembled_params, collector.nodes_by_ref))
    new_recipe_lines = recipe_serializer.serialize(updated_recipe).splitlines()

    # Handle newly created dict params
    created_dict_params = {} # New dict hyperparameters, to be inserted.

    new_recipe_lines = [f"dict {' '.join(f'{k}={v}' for k, v in dict_value.items())}" # Format new dict hyperparameters as strings, to be appended to the recipe later.
                        for dict_key, dict_value in created_dict_params.items()] + new_recipe_lines #P repend new dictionary nodes to recipe

    return recipe_serializer.deserialize("\n".join(new_recipe_lines)) # Deserialize updated recipe.


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
def get_method_dependencies(method: Any, cls: Any) -> Set[str]:
    """Recursively get all function dependencies from the method's source code."""
    dependencies = set()
    try:
        # Get the source code of the method
        source = inspect.getsource(method)

        # Get all methods from the class directly using the passed cls parameter
        all_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        method_names = [name for name, _ in all_methods]

        # Look for calls to other methods in the source code
        for name in method_names:
            if name in source and name != method.__name__:
                dependencies.add(name)
                # Recursively get dependencies of the called method
                called_method = getattr(cls, name)
                dependencies.update(get_method_dependencies(called_method, cls))
        return dependencies
    except (TypeError, OSError) as e:
        logger.warning(f"Could not analyze dependencies for {method.__name__}: {e}")
        return set()


def get_full_method_source(method_name: str, cls: Any, visited: Set[str] = None) -> str:
    """Get the source code of the method and all its dependencies."""
    if visited is None:
        visited = set()
    if method_name in visited:
        return ""

    visited.add(method_name)
    try:
        method = getattr(cls, method_name)
    except AttributeError as e:
        logger.error(f"Method {method_name} not found in class: {e}")
        return f"# Error: Method {method_name} not found in class"

    try:
        # Get the source of the main method
        source = inspect.getsource(method)

        # Get dependencies
        dependencies = get_method_dependencies(method, cls)

        # Build the complete source code
        full_source = [
            f"\n# {'-' * 20} Main merge method {'-' * 20}",
            source
        ]

        # Add dependencies if any
        if dependencies:
            full_source.extend([
                f"\n# {'-' * 20} Dependencies {'-' * 20}"
            ])
            for dep_name in dependencies:
                if dep_name not in visited:
                    dep_method = getattr(cls, dep_name)
                    full_source.extend([
                        f"\n# Dependency: {dep_name}",
                        inspect.getsource(dep_method)
                    ])
                    visited.add(dep_name)
        return "\n".join(full_source)
    except (TypeError, OSError) as e:
        logger.error(f"Failed to get source code for {method_name}: {e}")
        return f"# Error getting source code for {method_name}: {str(e)}"


def save_merge_method_code(merge_mode: str, model_path: Path, cls: Any) -> None:
    """Save the merge method code and its dependencies to a file."""
    try:
        # Verify that the method exists in the class
        if not hasattr(cls, merge_mode):
            raise AttributeError(f"Method '{merge_mode}' not found in class {cls.__name__}")

        log_dir = Path(os.getcwd()) if "HydraConfig" not in globals() else Path(HydraConfig.get().runtime.output_dir)
        merge_code_dir = log_dir / "merge_methods"
        os.makedirs(merge_code_dir, exist_ok=True)

        # Get the complete source code including dependencies
        full_source = get_full_method_source(merge_mode, cls)

        # Save the merge method code
        iteration_file_name = model_path.stem
        code_file_path = merge_code_dir / f"{iteration_file_name}_merge_method.py"

        with open(code_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Merge method: {merge_mode}\n")
            f.write(f"# Used in merge: {iteration_file_name}\n")
            f.write("# This file includes the main merge method and all its dependencies\n\n")
            f.write(full_source)

        logger.info(f"Saved merge method code to {code_file_path}")
    except Exception as e:
        logger.error(f"Failed to save merge method code: {e}")
        raise  # Re-raise the exception to see the full error trace


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