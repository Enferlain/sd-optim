# sd_optim/custom_converters/convert_sdxl_optim_blocks_sub.py
# Maps sdxl-sgm keys to granular sub-blocks in sdxl-optim_blocks_sub.yaml

import logging
import re
from typing import TypeVar
from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.streaming import StateDictKeyError

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Regexes for granular UNet blocks
re_inp = re.compile(r"\.input_blocks\.(\d+)\.(\d+)\.")
re_mid = re.compile(r"\.middle_block\.(\d+)\.")
re_out = re.compile(r"\.output_blocks\.(\d+)\.(\d+)\.")


@merge_method(identifier="convert_sdxl_optim_blocks_sub_to_sdxl_sgm", is_conversion=True)
def convert_sdxl_optim_blocks_sub_to_sdxl_sgm(
    optimized_blocks: Parameter(StateDict[T], model_config="sdxl-optim_blocks_sub"),
    **kwargs,
) -> Return(T, model_config="sdxl-sgm"):
    """
    Converter for sdxl-optim_blocks_sub to sdxl-sgm.
    Maps an sdxl-sgm target key to the corresponding granular sub-block name in sdxl-optim_blocks_sub.
    """
    target_key = kwargs.get("key")
    if target_key is None:
        raise StateDictKeyError("Converter called without 'key'.")

    block_name: str | None = None

    # --- UNet Block Mapping (Granular) ---
    if target_key.startswith("model.diffusion_model."):
        if ".time_embed" in target_key:
            block_name = "UNET_TIME_EMBED"
        elif ".label_emb" in target_key:
            block_name = "UNET_LABEL_EMBED"
        elif target_key.startswith("model.diffusion_model.out."):
            block_name = "UNET_OUT_FINAL"
        elif m := re_inp.search(target_key):
            main_idx = int(m.group(1))
            sub_idx = int(m.group(2))
            block_name = f"UNET_IN{main_idx:02d}_{sub_idx}"
        elif m := re_mid.search(target_key):
            # middle_block.0 -> MID00_0, middle_block.1 -> MID00_1, etc.
            sub_idx = int(m.group(1))
            block_name = f"UNET_MID00_{sub_idx}"
        elif m := re_out.search(target_key):
            main_idx = int(m.group(1))
            sub_idx = int(m.group(2))
            block_name = f"UNET_OUT{main_idx:02d}_{sub_idx}"

    # --- CLIP-L Block Mapping ---
    elif target_key.startswith("conditioner.embedders.0.transformer.text_model."):
        if ".text_model.embeddings." in target_key:
            block_name = "CLIP_L_EMBEDDING"
        elif ".final_layer_norm." in target_key:
            block_name = "CLIP_L_FINAL_NORM"
        elif m := re.search(r"\.layers\.(\d+)\.", target_key):
            layer_num = int(m.group(1))
            block_name = f"CLIP_L_IN{layer_num:02d}"
        else:
            block_name = "CLIP_L_ELSE"

    # --- CLIP-G Block Mapping ---
    elif target_key.startswith("conditioner.embedders.1.model."):
        if any(x in target_key for x in [".token_embedding.", ".positional_embedding", ".embeddings."]):
            block_name = "CLIP_G_EMBEDDING"
        elif ".text_projection" in target_key:
            block_name = "CLIP_G_TEXT_PROJECTION"
        elif ".ln_final." in target_key:
            block_name = "CLIP_G_LN_FINAL"
        elif m := re.search(r"\.(?:resblocks|layers)\.(\d+)\.", target_key):
            layer_num = int(m.group(1))
            block_name = f"CLIP_G_IN{layer_num:02d}"
        else:
            block_name = "CLIP_G_ELSE"

    # --- VAE Block Mapping ---
    elif target_key.startswith("first_stage_model."):
        key_suffix = target_key.split(".", 1)[1]
        if key_suffix.startswith("encoder.conv_in."):
            block_name = "VAE_ENCODER_IN"
        elif key_suffix.startswith("encoder.down."):
            block_name = "VAE_ENCODER_DOWN"
        elif key_suffix.startswith("encoder.mid."):
            block_name = "VAE_ENCODER_MID"
        elif key_suffix.startswith("encoder.norm_out.") or key_suffix.startswith("encoder.conv_out."):
            block_name = "VAE_ENCODER_OUT"
        elif key_suffix.startswith("quant_conv.") or key_suffix.startswith("post_quant_conv."):
            block_name = "VAE_QUANT"
        elif key_suffix.startswith("decoder.conv_in."):
            block_name = "VAE_DECODER_IN"
        elif key_suffix.startswith("decoder.mid."):
            block_name = "VAE_DECODER_MID"
        elif key_suffix.startswith("decoder.up."):
            block_name = "VAE_DECODER_UP"
        elif key_suffix.startswith("decoder.norm_out.") or key_suffix.startswith("decoder.conv_out."):
            block_name = "VAE_DECODER_OUT"
        else:
            block_name = "VAE_ELSE"

    if block_name:
        try:
            return optimized_blocks[block_name]
        except KeyError as err:
            raise StateDictKeyError(f"Block '{block_name}' not in optimized dict for key '{target_key}'") from err

    raise StateDictKeyError(f"Key '{target_key}' not handled by granular block conversion.")
