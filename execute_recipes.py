# execute_recipes.py
import sd_mecha
import torch
import logging
from pathlib import Path
from sd_optim import utils  # To load our custom stuff!
import re

# --- Configuration Section (Option 1: Hardcoded) ---
# Easier to start, harder to change without editing code
MERGE_DEVICE = "cuda"  # "cpu" or "cuda"
MERGE_DTYPE = torch.float32  # torch.float16, torch.bfloat16, torch.float32, torch.float64
OUTPUT_DTYPE = torch.bfloat16
THREADS = 4
ENABLE_CACHING = True  # Enable caching for merge operations
# Paths relative to this script's location? Or absolute? Absolute might be safer.
# Assume sd-optim structure: execute_recipes.py is at root, sd_optim/ is the package
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR  # Assuming script is at project root
DEFAULT_CUSTOM_CONFIGS_DIR = PROJECT_ROOT / "sd_optim" / "model_configs"
DEFAULT_CUSTOM_CONVERSION_DIR = PROJECT_ROOT / "sd_optim" / "model_configs"
# Directory where your models (.safetensors) are generally located
# Needed for resolving relative paths in recipes
DEFAULT_MODELS_BASE_DIR = Path("D:/stable-diffusion-webui-reforge/models/Stable-diffusion")  # Example absolute path

# --- NEW: Fallback Model Setting ---
# Set to None or "" to disable fallback
# Use path relative to DEFAULT_MODELS_BASE_DIR or an absolute path
FALLBACK_MODEL_PATH = (
    "D:/stable-diffusion-webui-reforge/models/Stable-diffusion/2182048-62.safetensors"  # <<< ADD YOUR FALLBACK MODEL PATH HERE
)

# --- Configuration Section (Option 2: Simple YAML config like 'exec_config.yaml') ---
# More flexible
# try:
#     with open("exec_config.yaml", "r") as f:
#         exec_cfg = yaml.safe_load(f)
#     MERGE_DEVICE = exec_cfg.get("merge_device", "cpu")
#     MERGE_DTYPE_STR = exec_cfg.get("merge_dtype", "fp64") # Store as string
#     # ... load other settings ...
#     # Convert dtype strings:
#     from sd_optim.merger import precision_mapping # Reuse mapping
#     MERGE_DTYPE = precision_mapping.get(MERGE_DTYPE_STR)
#     # ... handle paths ...
# except FileNotFoundError:
#     print("exec_config.yaml not found, using defaults.")
#     # Set defaults as above
# except Exception as e:
#     print(f"Error loading exec_config.yaml: {e}")
#     exit()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("execute_recipe")


def main():
    logger.info("--- Execute Mecha Recipe Utility ---")

    # 1. Load Custom Extensions (using configured paths)
    logger.info("Loading custom sd-optim extensions...")
    # Use paths defined in config section (either hardcoded or from exec_config.yaml)
    custom_configs_path = DEFAULT_CUSTOM_CONFIGS_DIR  # Or path from exec_cfg
    custom_conversion_path = DEFAULT_CUSTOM_CONVERSION_DIR  # Or path from exec_cfg
    utils.load_and_register_custom_configs(custom_configs_path)
    utils.load_and_register_custom_conversion(custom_conversion_path)
    logger.info("Custom extensions loaded.")

    # 2. Prompt for Recipe Directory
    while True:
        recipe_dir_str = input("Enter the directory containing your .mecha recipe files: ")
        recipe_dir = Path(recipe_dir_str).resolve()
        if recipe_dir.is_dir():
            break
        else:
            logger.error(f"Directory not found: {recipe_dir}")

    # 3. Find and List Recipes
    try:

        def numerical_sort_key(filename):
            # Extract the iteration number from filename like "...-it_93.mecha"
            match = re.search(r"-it_(\d+)\.mecha$", filename.name)
            if match:
                return int(match.group(1))
            return 0  # fallback for files that don't match the pattern

        recipes = sorted([f for f in recipe_dir.glob("*.mecha")], key=numerical_sort_key)
        if not recipes:
            logger.error(f"No .mecha files found in {recipe_dir}")
            return

        logger.info("Found Recipes:")
        for i, recipe_path in enumerate(recipes):
            # Extract iteration number for display
            match = re.search(r"-it_(\d+)\.mecha$", recipe_path.name)
            iter_num = match.group(1) if match else "?"
            print(f"  [{i}] {recipe_path.name} (iteration {iter_num})")

    except Exception as e:
        logger.error(f"Error reading recipes from {recipe_dir}: {e}")
        return

    # 4. Prompt for Selection
    while True:
        selection_str = input("Enter indices of recipes to merge (e.g., '0, 2-4, 6', or 'all'): ")
        selected_indices = set()
        try:
            if selection_str.lower() == "all":
                selected_indices.update(range(len(recipes)))
                break
            parts = selection_str.replace(" ", "").split(",")
            for part in parts:
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    if 0 <= start <= end < len(recipes):
                        selected_indices.update(range(start, end + 1))
                    else:
                        raise ValueError("Range out of bounds")
                else:
                    idx = int(part)
                    if 0 <= idx < len(recipes):
                        selected_indices.add(idx)
                    else:
                        raise ValueError("Index out of bounds")
            if selected_indices:
                break
            else:
                logger.warning("No valid indices selected.")
        except ValueError as e:
            logger.error(f"Invalid input: {e}. Please use numbers, commas, or ranges.")

    # 5. Prompt for Output Directory
    while True:
        output_dir_str = input("Enter the directory to save merged models: ")
        output_dir = Path(output_dir_str).resolve()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)  # Create if needed
            break
        except Exception as e:
            logger.error(f"Cannot create or access output directory {output_dir}: {e}")

    # 6. Determine Models Base Directory (from config section)
    models_base_dir = DEFAULT_MODELS_BASE_DIR  # Or path from exec_cfg
    if not models_base_dir.is_dir():
        logger.warning(f"Models base directory '{models_base_dir}' not found. Relative paths in recipes might fail.")
        effective_model_dirs = []
    else:
        effective_model_dirs = [models_base_dir]

    # 6.5 --- NEW: Prepare Fallback Model Node ---
    fallback_model_node = None
    if FALLBACK_MODEL_PATH:
        fallback_full_path = Path(FALLBACK_MODEL_PATH)
        # Try resolving relative to models base dir first
        if not fallback_full_path.is_absolute():
            resolved_fb_path = DEFAULT_MODELS_BASE_DIR / fallback_full_path
            if resolved_fb_path.exists():
                fallback_model_node = sd_mecha.model(FALLBACK_MODEL_PATH)  # Use relative path for sd_mecha
                logger.info(f"Using fallback model (relative): {FALLBACK_MODEL_PATH}")
            elif fallback_full_path.exists():  # Check if it exists directly (maybe user gave abs path?)
                fallback_model_node = sd_mecha.model(str(fallback_full_path))  # Use absolute path string
                logger.info(f"Using fallback model (absolute): {fallback_full_path}")
            else:
                logger.warning(f"Fallback model path not found: {FALLBACK_MODEL_PATH} or {resolved_fb_path}. Proceeding without fallback.")
        elif fallback_full_path.exists():  # If it was absolute path from the start
            fallback_model_node = sd_mecha.model(str(fallback_full_path))  # Use absolute path string
            logger.info(f"Using fallback model (absolute): {fallback_full_path}")
        else:
            logger.warning(f"Fallback model path not found: {FALLBACK_MODEL_PATH}. Proceeding without fallback.")
    else:
        logger.info("No fallback model specified.")
    # --- End Fallback Prep ---

    # 7. Process Selected Recipes
    # Create a shared cache for all recipes if caching is enabled
    shared_cache = {} if ENABLE_CACHING else None

    logger.info(f"\n--- Starting Merges ({len(selected_indices)} selected) ---")
    selected_recipes = sorted([recipes[i] for i in selected_indices])

    for i, recipe_path in enumerate(selected_recipes):
        logger.info(f"\nProcessing [{i + 1}/{len(selected_recipes)}]: {recipe_path.name}")
        output_filename = output_dir / f"{recipe_path.stem}_merged.safetensors"

        try:
            recipe_node = sd_mecha.deserialize_path(recipe_path)

            # Set up caching if enabled
            if ENABLE_CACHING and shared_cache is not None:
                recipe_node = recipe_node.set_cache(shared_cache)

            if isinstance(recipe_node, sd_mecha.recipe_nodes.MergeRecipeNode) and recipe_node.merge_method.identifier == "pop_lora":
                alpha_node_from_recipe = recipe_node.kwargs.get("alpha")
                rank_ratio_node_from_recipe = recipe_node.kwargs.get("rank_ratio")
                logger.info(f"Deserialized alpha_node: {type(alpha_node_from_recipe)}, {alpha_node_from_recipe}")
                logger.info(f"Deserialized rank_ratio_node: {type(rank_ratio_node_from_recipe)}, {rank_ratio_node_from_recipe}")
                # You could try to see their structure more deeply if they are MergeRecipeNodes themselves
                if isinstance(alpha_node_from_recipe, sd_mecha.recipe_nodes.MergeRecipeNode):
                    logger.info(f"  Alpha node method: {alpha_node_from_recipe.merge_method.identifier}")
                    logger.info(f"  Alpha node args: {alpha_node_from_recipe.args}")
                    logger.info(f"  Alpha node kwargs: {alpha_node_from_recipe.kwargs}")
            # --- End inspection ---

            sd_mecha.merge(
                recipe=recipe_node,
                output=output_filename,
                # --- PASS FALLBACK MODEL NODE ---
                fallback_model=fallback_model_node,  # <<< ADDED
                # --- Rest of parameters ---
                merge_device=MERGE_DEVICE,
                merge_dtype=MERGE_DTYPE,
                output_device="cpu",
                output_dtype=OUTPUT_DTYPE,
                threads=THREADS,
                model_dirs=effective_model_dirs,
                check_mandatory_keys=False,  # Keep this
            )
            logger.info(f"Successfully merged and saved: {output_filename}")

            # Log cache information if caching is enabled
            if ENABLE_CACHING and shared_cache is not None:
                logger.info(f"Cache contains {len(shared_cache)} entries")

        except Exception as e:
            logger.error(f"Failed to merge recipe {recipe_path.name}: {e}", exc_info=True)

    logger.info("\n--- All selected merges finished. ---")


if __name__ == "__main__":
    main()
