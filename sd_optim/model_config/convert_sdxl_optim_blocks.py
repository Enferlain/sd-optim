# In sd_optim/custom_converters/convert_sdxl_optim_blocks.py

import logging
import sd_mecha
import re
from typing import TypeVar
from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.streaming import StateDictKeyError

logger = logging.getLogger(__name__)
T = TypeVar("T")

# --- Regexes needed for this specific converter ---
re_inp = re.compile(r"\.input_blocks\.(\d+)\.")
re_mid = re.compile(r"\.middle_block\.(\d+)\.")
re_out = re.compile(r"\.output_blocks\.(\d+)\.")

@merge_method(
    identifier="convert_sdxl_optim_blocks_to_sdxl_sgm", # Unique ID
    is_conversion=True # Mark as conversion
)
def convert_sdxl_optim_blocks_to_sdxl_sgm(
    # Type hints use the string IDs - this works because this file is imported
    # AFTER the corresponding configs are registered by load_and_register_custom_configs
    blocks_dict: Parameter(StateDict[T], model_config="sdxl-optim_blocks"),
    **kwargs,
) -> Return(T, model_config="sdxl-sgm"):
    """
    Maps an sdxl-sgm key back to a block name defined in sdxl-optim_blocks
    and returns the corresponding value from the blocks_dict.
    (Defined statically in its own module)
    """
    target_key = kwargs.get("key")
    if target_key is None: return 0.0 # Or raise

    block_name = None
    # --- Start of specific mapping logic ---
    if target_key.startswith("model.diffusion_model."):
        if ".time_embed" in target_key: block_name = "UNET_TIME_EMBED"
        elif ".out." in target_key: block_name = "UNET_OUT_FINAL"
        elif m := re_inp.search(target_key):
            block_num = int(m.group(1)); block_name = f"UNET_IN{block_num:02d}" if 0 <= block_num <= 8 else None
        elif re_mid.search(target_key): block_name = "UNET_MID00"
        elif m := re_out.search(target_key):
             block_num = int(m.group(1)); block_name = f"UNET_OUT{block_num:02d}" if 0 <= block_num <= 8 else None
        if block_name is None: block_name = "UNET_ELSE"
    elif target_key.startswith("conditioner.embedders.0.transformer.text_model."):
        if ".embeddings." in target_key: block_name = "CLIP_L_EMBEDDING"
        elif ".final_layer_norm." in target_key: block_name = "CLIP_L_FINAL_NORM"
        elif m := re.search(r"\.layers\.(\d+)\.", target_key):
             layer_num = int(m.group(1)); block_name = f"CLIP_L_IN{layer_num:02d}" if 0 <= layer_num <= 11 else None
        if block_name is None: block_name = "CLIP_L_ELSE"
    elif target_key.startswith("conditioner.embedders.1.model."):
        if ".token_embedding." in target_key or ".positional_embedding" in target_key or ".embeddings." in target_key: block_name = "CLIP_G_EMBEDDING"
        elif ".text_projection" in target_key: block_name = "CLIP_G_TEXT_PROJECTION"
        elif ".ln_final." in target_key: block_name = "CLIP_G_LN_FINAL"
        elif m := re.search(r"\.(?:resblocks|layers)\.(\d+)\.", target_key):
             layer_num = int(m.group(1)); block_name = f"CLIP_G_IN{layer_num:02d}" if 0 <= layer_num <= 31 else None
        if block_name is None: block_name = "CLIP_G_ELSE"
    elif target_key.startswith("first_stage_model."):
        block_name = "VAE_ALL" # Assuming VAE_ALL exists in the block config YAML
    # --- End of specific mapping logic ---

    if block_name is None:
        logger.error(f"Converter could not map target key '{target_key}' to any defined block.")
        raise ValueError(f"Unhandled key in conversion: {target_key}")
    try:
        return blocks_dict[block_name]
    except KeyError:
        logger.error(f"Block name '{block_name}' (for key '{target_key}') not found in input dict. Check config 'sdxl-optim_blocks'.")
        raise StateDictKeyError(f"Block '{block_name}' not found for conversion.")
    except Exception as conv_e:
         logger.error(f"Error during conversion for key '{target_key}' (block '{block_name}'): {conv_e}", exc_info=True)
         raise
