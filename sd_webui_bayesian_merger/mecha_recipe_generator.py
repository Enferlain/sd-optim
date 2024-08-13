import sd_mecha

def generate_mecha_recipe(base_values, weights_list, merge_method, cfg):
    """Generates an sd-mecha recipe from Bayesian Merger parameters."""

    # Dynamically create ModelRecipeNodes for all models in cfg, using sd_mecha.model
    models = []
    for model_key in ["model_a", "model_b", "model_c"]:
        if model_key in cfg:
            model = sd_mecha.model(cfg[model_key], cfg.model_arch)
            models.append(model)

    # Retrieve and filter block identifiers for the UNet component
    model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
    unet_block_identifiers = [
        key for key in model_arch.user_keys()
        if "_unet_block_" in key
    ]
    unet_block_identifiers.sort(key=sd_mecha.hypers.natural_sort_key)

    # Construct the recipe using the appropriate sd-mecha function from MergeMethods
    primary_param = list(base_values.keys())[0]
    hypers = {}
    for greek_letter in weights_list:
        for i, weight in enumerate(weights_list[greek_letter]):
            block_id = unet_block_identifiers[i]
            # Assign the weight directly, not as a list
            hypers[block_id] = weight

        base_value = base_values.get(greek_letter, 0)
        # Extract the float value from the base_value list
        hypers.update({f"{cfg.model_arch}_{component}_default": base_value[0] for component in ["txt", "txt2"]})

    final_recipe = getattr(sd_mecha, merge_method)(*models, **{primary_param: hypers})

    # Serialize the recipe to text format
    recipe_text = sd_mecha.recipe_serializer.serialize(final_recipe)
    return recipe_text