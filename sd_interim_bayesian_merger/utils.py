import json
import os
import re
import sd_mecha
import logging

import threading
from pathlib import Path
from typing import List, Tuple, Dict
from pynput import keyboard
from sd_mecha.recipe_nodes import RecipeNode, MergeRecipeNode

logger = logging.getLogger(__name__)


# for methods that require selective optimization, ie contains on/off hypers, learning rate hypers, and such
OPTIMIZABLE_HYPERPARAMETERS = {
    "lu_merge": ["alpha", "theta"],
    "orth_pro": ["alpha"],
    "clyb_merge": ["alpha"],
    "ties_sum_extended": ["k", "apply_median", "apply_stock"],
    "ties_sum_with_dropout": ["probability", "della_eps", "apply_median", "apply_stock"],
    "slerp_norm_sign": ["alpha"],
    "polar_interpolate": ["alpha"],
    "wavelet_packet_merge": ["alpha"],
    "multi_domain_alignment": ["alpha", "beta"],
    # ... other methods and their optimizable hyperparameters
}


# Custom sorting function that uses component order from config
def custom_sort_key(key, component_order):
    # Extract the component (txt, txt2, unet, etc.) and parameter from the key
    parts = key.split("_")
    component = parts[1]  # Assumes component is the second element

    # Determine the index of the component in the order from the config
    component_index = component_order.index(component) if component in component_order else len(component_order)

    # Apply natural sorting to the rest of the key
    block_key = "_".join(parts[2:])

    return component_index, sd_mecha.hypers.natural_sort_key(block_key)


def load_and_prepare_recipe(cfg) -> Dict:
    """Loads a pre-built recipe and prepares bounds for optimization."""
    recipe_path = cfg.recipe_optimization.recipe_path
    optimization_target = cfg.recipe_optimization.optimization_target
    logger.info(f"Loading recipe from: {recipe_path}")
    logger.info(f"Optimization target: {optimization_target}")

    # Load the recipe using sd-mecha's deserialize function
    with open(recipe_path, "r") as f:
        recipe = sd_mecha.deserialize(f.readlines())

    # Get default bounds for all parameters in the recipe
    from sd_interim_bayesian_merger.bounds import Bounds
    default_bounds = Bounds.create_default_bounds(cfg)  # No need to modify this further

    # Identify the merge node to optimize
    target_node = recipe
    while target_node is not None:
        for i, child in enumerate(target_node.models):
            if f"&{i}" == optimization_target:
                target_node = child
                break
        else:
            break  # Target node not found, exit the loop

    if target_node is None:
        raise ValueError(f"Invalid optimization target: {optimization_target}")

    # Extract hyperparameters from the target node
    optimizable_params = target_node.hypers

    # Generate modified bounds for the selected hyperparameters
    modified_bounds = {k: v for k, v in default_bounds.items() if k in optimizable_params}

    # Return the modified bounds
    logger.info(f"Modified Bounds: {modified_bounds}")
    return modified_bounds


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




# Other Utility Functions (e.g., for early stopping, etc.)
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