from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import sd_mecha

from omegaconf import DictConfig, ListConfig
from sd_mecha import extensions, recipe_nodes
from sd_mecha.extensions.merge_methods import MergeMethod, RecipeNodeOrValue
from sd_mecha.recipe_nodes import ModelRecipeNode

from sd_optim import utils
from sd_optim.bounds import BoundsInfo, ParameterHandler

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)


def slice_models(
    merger: Merger,
    prepared_model_nodes: list[RecipeNodeOrValue],
    merge_method: MergeMethod,
) -> list[RecipeNodeOrValue]:
    """Slice the model list to match the expected number of model-type arguments."""
    param_info = merge_method.get_param_names()
    if param_info.has_varargs():
        return prepared_model_nodes

    input_types = merge_method.get_input_types().args
    expected_num_models = 0
    for arg_type in input_types:
        origin_type = getattr(arg_type, "__origin__", arg_type)
        if isinstance(origin_type, type) and issubclass(origin_type, sd_mecha.extensions.merge_methods.StateDict):
            expected_num_models += 1

    num_provided = len(prepared_model_nodes)

    if num_provided > expected_num_models:
        logger.warning(
            "Merge method '%s' expects %s model arguments, but %s were prepared. Using the first %s.",
            merge_method.identifier,
            expected_num_models,
            num_provided,
            expected_num_models,
        )
        return prepared_model_nodes[:expected_num_models]
    if num_provided < expected_num_models:
        logger.warning(
            "Merge method '%s' expects %s model arguments, but only %s were prepared. This might lead to errors.",
            merge_method.identifier,
            expected_num_models,
            num_provided,
        )
    return prepared_model_nodes


def handle_delta_output(
    merger: Merger,
    core_recipe_node: recipe_nodes.MergeRecipeNode,
    base_model_node: ModelRecipeNode | None,
    merge_method: MergeMethod,
) -> recipe_nodes.RecipeNode:
    """Wrap the core recipe with add_difference if the output is a delta and a base model exists."""
    input_spaces_args = [node.merge_space for node in core_recipe_node.bound_args.args]
    input_spaces_kwargs = {key: node.merge_space for key, node in core_recipe_node.bound_args.kwargs.items()}

    try:
        return_data = merge_method.get_return_type().data
        output_space = return_data.merge_space
        if output_space is None:
            output_space = merge_method.default_merge_space

        if isinstance(output_space, sd_mecha.extensions.merge_spaces.MergeSpaceSymbol):
            concrete_inputs = [space for space in (*input_spaces_args, *input_spaces_kwargs.values()) if space is not None]
            output_space = next(
                (space for space in concrete_inputs if space in output_space),
                None,
            )

        if output_space is None:
            raise ValueError("Could not resolve a concrete output merge space from the merge method signature.")
    except Exception as error:
        logger.error("Could not determine output merge space for %s: %s. Assuming 'weight'.", merge_method.identifier, error)
        output_space = sd_mecha.extensions.merge_spaces.resolve("weight")

    delta_space = sd_mecha.extensions.merge_spaces.resolve("delta")

    if output_space == delta_space:
        if base_model_node:
            logger.info("Output is a delta, applying to base model.")
            add_difference_method = sd_mecha.extensions.merge_methods.resolve("add_difference")
            return add_difference_method(base_model_node, core_recipe_node, alpha=1.0)
        logger.warning(
            "Merge method '%s' outputs a delta, but no base model was selected. Returning the delta directly.",
            merge_method.identifier,
        )
    return core_recipe_node


def prepare_model_recipe_args(
    merger: Merger,
    initial_model_nodes: list[recipe_nodes.ModelRecipeNode],
    base_model_node: recipe_nodes.ModelRecipeNode | None,
    merge_method: MergeMethod,
) -> list[RecipeNodeOrValue]:
    """
    Prepare the list of model nodes to be passed as positional arguments to a merge method.

    Handles automatic LoRA conversion, delta preparation, and exclusion of the base model
    itself when a delta input is required.
    """
    prepared_nodes: list[RecipeNodeOrValue] = []
    param_info = merge_method.get_param_names()
    input_spaces = merge_method.get_input_merge_spaces()
    delta_space = extensions.merge_spaces.resolve("delta")

    conversion_target_node = base_model_node if base_model_node else (initial_model_nodes[0] if initial_model_nodes else None)

    for index, model_node in enumerate(initial_model_nodes):
        current_node: recipe_nodes.RecipeNode = model_node
        should_add_node = True
        is_lora = False
        original_path_for_logging = model_node.path

        try:
            inferred_lora_configs = list(merger._get_adapter_candidate_ids(model_node))
            if inferred_lora_configs:
                is_lora = True
                logger.info(
                    "Identified LoRA/LyCORIS: '%s' with candidate config(s) %s",
                    original_path_for_logging,
                    inferred_lora_configs,
                )
        except Exception as error:
            logger.error("Could not infer config for model %s: %s", original_path_for_logging, error)
            is_lora = False

        if is_lora:
            if not conversion_target_node:
                raise ValueError(
                    f"LoRA '{original_path_for_logging}' detected, but no base model was provided to convert it against."
                )

            logger.info("Converting LoRA '%s' to a delta...", original_path_for_logging)
            try:
                current_node = utils.convert_with_model_dirs(
                    current_node,
                    conversion_target_node,
                    model_dirs_to_add=[merger.models_dir],
                )
                logger.info("LoRA converted successfully. New node is in '%s' space.", current_node.merge_space.identifier)
            except Exception as error:
                logger.error(
                    "CRITICAL: LoRA conversion failed for %s: %s",
                    original_path_for_logging,
                    error,
                    exc_info=True,
                )
                raise

        is_delta_expected = False
        expected_space_for_arg = None
        if param_info.has_varargs() and index >= len(param_info.args):
            expected_space_for_arg = input_spaces.vararg
        elif index < len(param_info.args):
            expected_space_for_arg = input_spaces.args[index]

        if isinstance(expected_space_for_arg, set):
            is_delta_expected = delta_space in expected_space_for_arg
        elif isinstance(expected_space_for_arg, extensions.merge_spaces.MergeSpace):
            is_delta_expected = expected_space_for_arg == delta_space

        if is_delta_expected and current_node.merge_space != delta_space:
            if not base_model_node:
                raise ValueError(
                    f"Merge method '{merge_method.identifier}' requires a delta for argument {index}, but no base_model was selected."
                )

            if current_node is base_model_node:
                log_path = base_model_node.path if isinstance(base_model_node, recipe_nodes.ModelRecipeNode) else "the base model"
                logger.warning("Base model '%s' matches arg %s which expects a delta. Excluding it from arguments.", log_path, index)
                should_add_node = False
            else:
                log_path = current_node.path if isinstance(current_node, recipe_nodes.ModelRecipeNode) else "a model node"
                logger.info("Creating delta for '%s'...", log_path)
                current_node = sd_mecha.subtract(current_node, base_model_node)

        if should_add_node:
            prepared_nodes.append(current_node)

    logger.info("Finished preparing %s model arguments for merge method '%s'.", len(prepared_nodes), merge_method.identifier)
    return prepared_nodes


def prepare_param_recipe_args(
    merger: Merger,
    params: dict[str, Any],
    param_info: BoundsInfo,
    merge_method: MergeMethod,
) -> dict[str, recipe_nodes.RecipeNode]:
    """
    Prepare sd-mecha nodes for parameters based on strategies and handle fixed kwargs.

    Supports combining block and key configs using fallback merge semantics.
    """
    final_param_nodes: dict[str, recipe_nodes.RecipeNode] = {}
    block_based_values_per_param: dict[str, dict[str, Any]] = {}
    key_based_values_per_param: dict[str, dict[str, Any]] = {}
    handled_base_params = set()

    logger.debug("Parsing optimizer params using parameter info metadata...")

    for opt_param_name, info in param_info.items():
        if opt_param_name not in params:
            logger.warning("Optimizer did not provide value for parameter '%s'. Skipping.", opt_param_name)
            continue

        value = params[opt_param_name]
        strategy = info.get("strategy")
        target_type = info.get("target_type")
        base_param = info.get("base_param")
        item_name = info.get("item_name")
        items_covered = info.get("items_covered", [])

        if not base_param:
            continue

        handled_base_params.add(base_param)

        if target_type == "block":
            block_based_values_per_param.setdefault(base_param, {})
        elif target_type == "key":
            key_based_values_per_param.setdefault(base_param, {})
        else:
            logger.warning("Unknown target_type '%s' for '%s'.", target_type, opt_param_name)
            continue

        if strategy in ["all", "select"]:
            if not item_name:
                logger.warning("Missing 'item_name' for '%s' (%s).", opt_param_name, strategy)
                continue
            if target_type == "block":
                block_based_values_per_param[base_param][item_name] = value
            else:
                key_based_values_per_param[base_param][item_name] = value
        elif strategy in ["group", "single"]:
            if not items_covered:
                logger.warning("Missing 'items_covered' for '%s' (%s).", opt_param_name, strategy)
                continue
            for item in items_covered:
                if target_type == "block":
                    block_based_values_per_param[base_param][item] = value
                else:
                    key_based_values_per_param[base_param][item] = value

    target_model_node = merger._get_conversion_context_node()

    if not target_model_node:
        logger.error("Cannot prepare parameter nodes: A conversion context model is missing.")
        return {}

    all_base_params = set(block_based_values_per_param.keys()) | set(key_based_values_per_param.keys())

    for base_param in all_base_params:
        block_dict = block_based_values_per_param.get(base_param, {})
        key_dict = key_based_values_per_param.get(base_param, {})

        if block_dict and not key_dict:
            if not merger.custom_block_config:
                logger.error("Cannot make block node for '%s': custom block config missing.", base_param)
                continue
            try:
                literal_node = sd_mecha.literal(block_dict, config=merger.custom_block_config.identifier)
                converted_node = utils.convert_with_model_dirs(
                    literal_node,
                    target_model_node,
                    model_dirs_to_add=[merger.models_dir],
                )
                final_param_nodes[base_param] = converted_node
                logger.debug("Created BLOCK-ONLY node for '%s' (%s blocks).", base_param, len(block_dict))
            except Exception as error:
                logger.error("Failed creating block node for '%s': %s", base_param, error, exc_info=True)
        elif key_dict and not block_dict:
            if not merger.base_model_config:
                logger.error("Cannot create key node for '%s': base_model_config not loaded.", base_param)
                continue
            try:
                literal_node = sd_mecha.literal(key_dict, config=merger.base_model_config.identifier)
                final_param_nodes[base_param] = literal_node
                logger.debug("Created KEY-ONLY node for '%s' (%s keys).", base_param, len(key_dict))
            except Exception as error:
                logger.error("Failed creating key node for '%s': %s", base_param, error, exc_info=True)
        elif block_dict and key_dict:
            if not merger.custom_block_config:
                logger.error("Cannot make block node for '%s': custom block config missing.", base_param)
                continue
            if not merger.base_model_config:
                logger.error("Cannot create key node for '%s': base_model_config not loaded.", base_param)
                continue

            logger.debug(
                "Merging block and key values for '%s' (%s blocks + %s keys). Keys will override blocks.",
                base_param,
                len(block_dict),
                len(key_dict),
            )
            try:
                block_literal = sd_mecha.literal(block_dict, config=merger.custom_block_config.identifier)
                block_converted = utils.convert_with_model_dirs(
                    block_literal,
                    target_model_node,
                    model_dirs_to_add=[merger.models_dir],
                )
                key_literal = sd_mecha.literal(key_dict, config=merger.base_model_config.identifier)
                final_param_nodes[base_param] = key_literal | block_converted
                logger.debug("Created FALLBACK-MERGED node for '%s' (blocks + keys, keys override).", base_param)
            except Exception as error:
                logger.error("Failed creating merged node for '%s': %s", base_param, error, exc_info=True)
        else:
            logger.warning("No values found for base parameter '%s' - this shouldn't happen.", base_param)

    logger.debug("Checking for fixed keyword arguments using custom_bounds...")
    expected_kwargs = set(merge_method.get_params().kwargs.keys())

    unhandled_kwargs = expected_kwargs - handled_base_params
    logger.debug("Expected Kwargs: %s", expected_kwargs)
    logger.debug("Handled Base Params (Strategies): %s", handled_base_params)
    logger.debug("Unhandled Expected Kwargs: %s", unhandled_kwargs)

    custom_bounds_config = merger.cfg.optimization_guide.get("custom_bounds", {})
    validated_custom_bounds = ParameterHandler.validate_custom_bounds(custom_bounds_config)
    recipe_target_params: set[str] = set()
    if merger.cfg.optimization_mode == "recipe":
        target_params_raw = merger.cfg.recipe_optimization.get("target_params", [])
        if isinstance(target_params_raw, (list, ListConfig)):
            recipe_target_params = {str(name) for name in target_params_raw}

    for kwarg_name in unhandled_kwargs:
        if merger.cfg.optimization_mode == "recipe" and kwarg_name not in recipe_target_params:
            logger.info(
                "Parameter '%s' is not targeted by recipe_optimization.target_params; using value from source .mecha line (or merge-method default if absent).",
                kwarg_name,
            )
            continue

        if kwarg_name in validated_custom_bounds:
            fixed_value = validated_custom_bounds[kwarg_name]
            if isinstance(fixed_value, (list, tuple, dict, ListConfig, DictConfig)):
                raise ValueError(
                    f"custom_bounds['{kwarg_name}'] must be a scalar fixed value in this run. "
                    f"Got {type(fixed_value).__name__}: {fixed_value}. "
                    "Use a scalar value for fixed behavior. For optimization, define the parameter "
                    "through optimization_guide component strategies so it is generated in param metadata."
                )
            final_param_nodes[kwarg_name] = sd_mecha.literal(fixed_value)
            logger.info("Using fixed value for '%s' from custom_bounds: %s", kwarg_name, fixed_value)
        else:
            if merger.cfg.optimization_mode == "recipe":
                logger.info("Using value for parameter '%s' from the source .mecha file.", kwarg_name)
            else:
                logger.info(
                    "Using default value for parameter '%s' from merge method '%s'.",
                    kwarg_name,
                    merge_method.identifier,
                )

    logger.info("Prepared %s final parameter nodes for merge method '%s'.", len(final_param_nodes), merge_method.identifier)
    return final_param_nodes
