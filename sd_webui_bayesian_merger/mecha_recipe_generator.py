import re
import sd_mecha

def generate_mecha_recipe(base_values, weights_list, merge_method, cfg):
    """Generates an sd-mecha recipe from Bayesian Merger parameters."""

    # Dynamically create ModelRecipeNodes for all models in cfg
    models = []
    for model_key in ["model_a", "model_b", "model_c"]:
        if model_key in cfg:
            model = sd_mecha.recipe_nodes.ModelRecipeNode(state_dict=cfg[model_key], model_arch=cfg.model_arch)
            models.append(model)

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

    # Get the expected merging spaces for the merging method
    mecha_merge_method = sd_mecha.extensions.merge_method.resolve(merge_method)
    input_merge_spaces, _ = mecha_merge_method.get_input_merge_spaces()

    # Create the initial MergeRecipeNode
    merge_node = sd_mecha.recipe_nodes.MergeRecipeNode(
        mecha_merge_method,
        model_a,
        model_b,
        hypers={},  # Initially empty hyperparameters
        volatile_hypers={},
    )

    # Dynamically create deltas based on merging space requirements
    models = [model_a]
    for i, model_or_delta in enumerate(merge_node.models[1:], start=1):  # Skip model_a
        if input_merge_spaces[i] == sd_mecha.recipe_nodes.MergeSpace.DELTA:
            # Create a delta using the subtract method
            delta = sd_mecha.recipe_nodes.MergeRecipeNode(
                sd_mecha.extensions.merge_method.resolve("subtract"),
                model_or_delta,  # The model to subtract from
                model_a,           # The base model
            )
            models.append(delta)
        else:
            models.append(model_or_delta)  # Use the model directly if not a delta

    merge_node.models = models  # Update the merge node with the models and deltas

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