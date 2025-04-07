# sd_optim/custom_converters/convert_sdxl_optim_blocks.py (V1.1 - Correct Fallback)

import logging
import sd_mecha
import re
from typing import TypeVar, Optional
from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.streaming import StateDictKeyError # Ensure this is imported

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Regexes needed
re_inp = re.compile(r"\.input_blocks\.(\d+)\.")
re_mid = re.compile(r"\.middle_block\.(\d+)\.")
re_out = re.compile(r"\.output_blocks\.(\d+)\.")

@merge_method(
    identifier="convert_sdxl_optim_blocks_to_sdxl_sgm",
    is_conversion=True
)
def convert_sdxl_optim_blocks_to_sdxl_sgm(
    blocks_dict: Parameter(StateDict[T], model_config="sdxl-optim_blocks"),
    **kwargs,
) -> Return(T, model_config="sdxl-sgm"):
    """
    Maps an sdxl-sgm key to a block name defined in sdxl-optim_blocks.
    If mapped, returns the corresponding value from blocks_dict.
    If the key cannot be mapped OR the block is not in blocks_dict,
    raises StateDictKeyError to trigger sd_mecha's fallback mechanism.
    """
    target_key = kwargs.get("key")
    if target_key is None:
        logger.warning("Converter called without 'key'. Signaling fallback.")
        raise StateDictKeyError("Converter called without 'key'.")

    block_name: Optional[str] = None # Use Optional typing

    # --- Attempt to map target_key to a DEFINED block name ---
    # Only include logic for prefixes corresponding to blocks defined
    # in the sdxl-optim_blocks.yaml config.
    if target_key.startswith("model.diffusion_model."):
        if ".time_embed" in target_key: block_name = "UNET_TIME_EMBED"
        elif ".label_emb" in target_key: block_name = "UNET_LABEL_EMBED" # Group label_emb here too
        elif ".out." in target_key: block_name = "UNET_OUT_FINAL"
        elif m := re_inp.search(target_key):
            block_num = int(m.group(1)); block_name = f"UNET_IN{block_num:02d}" if 0 <= block_num <= 8 else None
        elif re_mid.search(target_key): block_name = "UNET_MID00"
        elif m := re_out.search(target_key):
             block_num = int(m.group(1)); block_name = f"UNET_OUT{block_num:02d}" if 0 <= block_num <= 8 else None
        # REMOVED UNET_ELSE mapping - unmapped UNet keys will trigger fallback

    elif target_key.startswith("conditioner.embedders.0.transformer.text_model."):
        if ".embeddings." in target_key: block_name = "CLIP_L_EMBEDDING"
        elif ".final_layer_norm." in target_key: block_name = "CLIP_L_FINAL_NORM"
        elif m := re.search(r"\.layers\.(\d+)\.", target_key):
             layer_num = int(m.group(1)); block_name = f"CLIP_L_IN{layer_num:02d}" if 0 <= layer_num <= 11 else None
        # REMOVED CLIP_L_ELSE mapping

    elif target_key.startswith("conditioner.embedders.1.model."):
        if ".token_embedding." in target_key or ".positional_embedding" in target_key or ".embeddings." in target_key: block_name = "CLIP_G_EMBEDDING"
        elif ".text_projection" in target_key: block_name = "CLIP_G_TEXT_PROJECTION"
        elif ".ln_final." in target_key: block_name = "CLIP_G_LN_FINAL"
        elif m := re.search(r"\.(?:resblocks|layers)\.(\d+)\.", target_key):
             layer_num = int(m.group(1)); block_name = f"CLIP_G_IN{layer_num:02d}" if 0 <= layer_num <= 31 else None
        # REMOVED CLIP_G_ELSE mapping

    # Example: Handle VAE only IF VAE_ALL is defined in the YAML
    # (Comment this out if VAE_ALL is not in your sdxl-optim-blocks.yaml)
    elif target_key.startswith("first_stage_model."):
         block_name = "VAE_ALL"

    # --- Check if a mapping was found AND if the block exists in the input dict ---
    if block_name is not None:
        try:
            # Mapping found, try to get value from optimized blocks dict
            value = blocks_dict[block_name]
            # logger.debug(f"Key '{target_key}' maps to OPTIMIZED block '{block_name}'. Returning value.")
            return value
        except KeyError:
            # Mapping found, but block wasn't optimized (not in dict). Signal fallback.
            logger.debug(f"Key '{target_key}' maps to block '{block_name}', but block not in optimized dict. Signaling fallback.")
            raise StateDictKeyError(f"Block '{block_name}' not in optimized input dict for key '{target_key}'")
        except Exception as conv_e:
            # Catch other potential errors during lookup
            logger.error(f"Unexpected error during conversion lookup for key '{target_key}' (block '{block_name}'): {conv_e}", exc_info=True)
            raise # Re-raise other errors
    else:
        # --- NO MAPPING FOUND ---
        # target_key does not correspond to any defined block structure. Signal fallback.
        logger.debug(f"Key '{target_key}' does not map to any known block structure. Signaling fallback.")
        raise StateDictKeyError(f"Key '{target_key}' not handled by block conversion.")
