import logging
import re
from typing import TypeVar, cast

from sd_mecha import skip_key
from sd_mecha.extensions.merge_methods import Parameter, Return, StateDict, merge_method
from sd_mecha.keys_map import KeyMapBuilder, RealizedKeyRelation
from sd_mecha.streaming import StateDictKeyError

logger = logging.getLogger(__name__)
T = TypeVar("T")

_RE_INPUT_BLOCK = re.compile(r"\.input_blocks\.(\d+)\.")
_RE_MIDDLE_BLOCK = re.compile(r"\.middle_block\.(\d+)\.")
_RE_OUTPUT_BLOCK = re.compile(r"\.output_blocks\.(\d+)\.")
_RE_CLIP_L_LAYER = re.compile(r"\.layers\.(\d+)\.")
_RE_CLIP_G_LAYER = re.compile(r"\.(?:resblocks|layers)\.(\d+)\.")


@merge_method(identifier="convert_sdxl_optim_blocks_to_sdxl_sgm", is_conversion=True)
class convert_sdxl_optim_blocks_to_sdxl_sgm:
    @staticmethod
    def _map_target_key(target_key: str) -> str | None:
        if target_key.startswith("model.diffusion_model."):
            if ".time_embed" in target_key:
                return "UNET_TIME_EMBED"
            if ".label_emb" in target_key:
                return "UNET_LABEL_EMBED"
            if ".out." in target_key:
                return "UNET_OUT_FINAL"
            if match := _RE_INPUT_BLOCK.search(target_key):
                block_num = int(match.group(1))
                return f"UNET_IN{block_num:02d}" if 0 <= block_num <= 8 else None
            if _RE_MIDDLE_BLOCK.search(target_key):
                return "UNET_MID00"
            if match := _RE_OUTPUT_BLOCK.search(target_key):
                block_num = int(match.group(1))
                return f"UNET_OUT{block_num:02d}" if 0 <= block_num <= 8 else None
            return None

        if target_key.startswith("conditioner.embedders.0.transformer.text_model."):
            if ".embeddings." in target_key:
                return "CLIP_L_EMBEDDING"
            if ".final_layer_norm." in target_key:
                return "CLIP_L_FINAL_NORM"
            if match := _RE_CLIP_L_LAYER.search(target_key):
                layer_num = int(match.group(1))
                return f"CLIP_L_IN{layer_num:02d}" if 0 <= layer_num <= 11 else None
            return None

        if target_key.startswith("conditioner.embedders.1.model."):
            if ".token_embedding." in target_key or ".positional_embedding" in target_key or ".embeddings." in target_key:
                return "CLIP_G_EMBEDDING"
            if ".text_projection" in target_key:
                return "CLIP_G_TEXT_PROJECTION"
            if ".ln_final." in target_key:
                return "CLIP_G_LN_FINAL"
            if match := _RE_CLIP_G_LAYER.search(target_key):
                layer_num = int(match.group(1))
                return f"CLIP_G_IN{layer_num:02d}" if 0 <= layer_num <= 31 else None
            return None

        if target_key.startswith("first_stage_model."):
            key_suffix = target_key.split(".", 1)[1]
            if key_suffix.startswith("encoder.conv_in."):
                return "VAE_ENCODER_IN"
            if key_suffix.startswith("encoder.down."):
                return "VAE_ENCODER_DOWN"
            if key_suffix.startswith("encoder.mid."):
                return "VAE_ENCODER_MID"
            if key_suffix.startswith("encoder.norm_out.") or key_suffix.startswith("encoder.conv_out."):
                return "VAE_ENCODER_OUT"
            if key_suffix.startswith("quant_conv.") or key_suffix.startswith("post_quant_conv."):
                return "VAE_QUANT"
            if key_suffix.startswith("decoder.conv_in."):
                return "VAE_DECODER_IN"
            if key_suffix.startswith("decoder.mid."):
                return "VAE_DECODER_MID"
            if key_suffix.startswith("decoder.up."):
                return "VAE_DECODER_UP"
            if key_suffix.startswith("decoder.norm_out.") or key_suffix.startswith("decoder.conv_out."):
                return "VAE_DECODER_OUT"
            return "VAE_ELSE"

        return None

    @classmethod
    def map_keys(cls, builder: KeyMapBuilder) -> None:
        input_keys = set(builder.blocks_dict.keys())
        for target_key in builder.out.keys():  # noqa: SIM118 - sd_mecha exposes a callable accessor, not a dict view.
            block_name = cls._map_target_key(target_key)
            if block_name is None or block_name not in input_keys:
                continue
            builder[target_key] = builder.blocks_dict.keys[block_name]

    def __call__(
        self,
        blocks_dict: Parameter(StateDict[T], model_config="sdxl-optim_blocks"),
        **kwargs,
    ) -> Return(T, model_config="sdxl-sgm"):
        target_key = cast(str, kwargs["key"])
        key_relation = cast(RealizedKeyRelation, kwargs["key_relation"])
        block_keys = key_relation.inputs.get("blocks_dict", ())

        if not block_keys:
            logger.debug("No optimized block mapping found for key '%s'.", target_key)
            skip_key(target_key)

        block_name = block_keys[0]
        try:
            return blocks_dict[block_name]
        except (KeyError, StateDictKeyError):
            logger.debug("Optimized block '%s' missing for key '%s'.", block_name, target_key)
            skip_key(target_key)
