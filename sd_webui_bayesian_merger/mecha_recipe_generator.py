import re
import sd_mecha
import inspect


# Define a dictionary mapping generic greek letters to sd-mecha hyperparameter names
GREEK_LETTER_MAPPING = {
    "alpha": [],
    "beta": [],
    # Add more greek letters as needed
}

# Populate GREEK_LETTER_MAPPING once, outside the function
for merge_method_name in sd_mecha.extensions.merge_method._merge_methods_registry:
    mecha_merge_method = sd_mecha.extensions.merge_method.resolve(merge_method_name)
    params = inspect.signature(mecha_merge_method.__call__).parameters
    greek_letter_index = 0
    for name, param in params.items():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            if greek_letter_index < len(GREEK_LETTER_MAPPING):
                current_greek_letter = list(GREEK_LETTER_MAPPING.keys())[greek_letter_index]
                GREEK_LETTER_MAPPING[current_greek_letter].append(name)
                greek_letter_index += 1

def generate_mecha_recipe(base_values, weights_list, merge_method, cfg):
    """Generates an sd-mecha recipe from Bayesian Merger parameters."""

    # Create ModelRecipeNodes for the models being merged, using model_arch from cfg
    model_a = sd_mecha.recipe_nodes.ModelRecipeNode(state_dict=cfg.model_a, model_arch=cfg.model_arch)
    model_b = sd_mecha.recipe_nodes.ModelRecipeNode(state_dict=cfg.model_b, model_arch=cfg.model_arch)

    # Retrieve and filter block identifiers for the UNet component
    model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
    unet_block_identifiers = [
        key for key in model_arch.user_keys()
        if "_unet_block_" in key
    ]
    unet_block_identifiers.sort(key=sd_mecha.hypers.natural_sort_key)

    # Create a single hyperparameters dictionary
    hypers = {}
    for greek_letter in weights_list:
        for i, weight in enumerate(weights_list[greek_letter]):
            block_id = unet_block_identifiers[i]
            hypers[block_id] = weight[0]

        base_value = base_values.get(greek_letter, 0)
        hypers.update({f"{cfg.model_arch}_{component}_default": base_value for component in ["txt", "txt2"]})

    # Create the merging operation, but don't call it yet
    mecha_merge_method = sd_mecha.extensions.merge_method.resolve(merge_method)

    # Create the initial MergeRecipeNode
    merge_node = sd_mecha.recipe_nodes.MergeRecipeNode(
        mecha_merge_method,
        model_a,
        model_b,
        hypers={},  # Initially empty hyperparameters
        volatile_hypers={},
    )

    # Update the hyperparameters of the merge_node
    merge_node.hypers = {list(base_values.keys())[0]: hypers}

    # Assign merge_node to final_recipe
    final_recipe = merge_node

    # Serialize the recipe to text format
    recipe_text = sd_mecha.recipe_serializer.serialize(final_recipe)
    return recipe_text

def translate_optimiser_parameters(bases, weights):
    """Translates parameters from Bayesian Merger's optimiser.py to sd-mecha format."""

    base_values = {}
    weights_list = {}

    # Extract base values
    for greek_letter in bases:
        base_values[greek_letter] = bases[greek_letter]

    # Extract block parameters with "block_X_" prefix, grouped by greek letter
    for key, value in weights.items():
        match = re.match(r"block_(\d+)_(\w+)", key)
        if match:
            block_index = int(match.group(1))
            greek_letter = match.group(2)
            if greek_letter not in weights_list:
                weights_list[greek_letter] = []
            # Ensure the list is long enough to accommodate the current block index
            if len(weights_list[greek_letter]) <= block_index:
                weights_list[greek_letter].extend([None] * (block_index + 1 - len(weights_list[greek_letter])))
            weights_list[greek_letter][block_index] = value

    return base_values, weights_list