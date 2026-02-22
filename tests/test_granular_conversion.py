import pytest
from pathlib import Path
import yaml
import contextlib
import sd_mecha
from sd_mecha.streaming import StateDictKeyError


# Register the custom config BEFORE importing the converter that uses it
def register_test_config():
    config_path = Path(__file__).parent.parent / "sd_optim" / "model_configs" / "sdxl-optim_blocks_sub.yaml"
    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        config_obj = sd_mecha.extensions.model_configs.ModelConfigImpl(**config_data)
        with contextlib.suppress(ValueError):
            sd_mecha.extensions.model_configs.register_aux(config_obj)


register_test_config()

from sd_optim.model_configs.convert_sdxl_optim_blocks_sub import convert_sdxl_optim_blocks_sub_to_sdxl_sgm


class MockStateDict(dict):
    def __getitem__(self, key):
        if key in self:
            return f"val_{key}"
        raise KeyError(key)


@pytest.fixture
def blocks_dict():
    # provide names that are in the ACTUAL verified sdxl-optim_blocks_sub.yaml
    blocks = [
        "UNET_TIME_EMBED",
        "UNET_LABEL_EMBED",
        "UNET_OUT_FINAL",
        "UNET_IN00_0",
        "UNET_IN04_0",
        "UNET_IN04_1",
        "UNET_MID00_0",
        "UNET_MID00_1",
        "UNET_MID00_2",
        "UNET_OUT00_0",
        "UNET_OUT00_1",
        "UNET_OUT02_2",
        "UNET_OUT05_2",
        "CLIP_L_IN05",
        "CLIP_L_EMBEDDING",
        "CLIP_G_IN10",
        "CLIP_G_TEXT_PROJECTION",
        "VAE_ENCODER_DOWN",
        "VAE_DECODER_UP",
    ]
    return MockStateDict({b: f"val_{b}" for b in blocks})


@pytest.mark.parametrize(
    "target_key, expected_block",
    [
        # UNet Input Blocks
        ("model.diffusion_model.input_blocks.0.0.weight", "UNET_IN00_0"),
        ("model.diffusion_model.input_blocks.4.0.emb_layers.1.weight", "UNET_IN04_0"),
        ("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight", "UNET_IN04_1"),
        # UNet Middle Blocks
        ("model.diffusion_model.middle_block.0.weight", "UNET_MID00_0"),
        ("model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight", "UNET_MID00_1"),
        ("model.diffusion_model.middle_block.2.weight", "UNET_MID00_2"),
        # UNet Output Blocks
        ("model.diffusion_model.output_blocks.0.0.weight", "UNET_OUT00_0"),
        ("model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_out.0.weight", "UNET_OUT00_1"),
        ("model.diffusion_model.output_blocks.2.2.conv.weight", "UNET_OUT02_2"),
        ("model.diffusion_model.output_blocks.5.2.conv.weight", "UNET_OUT05_2"),
        # Global UNet
        ("model.diffusion_model.time_embed.0.weight", "UNET_TIME_EMBED"),
        ("model.diffusion_model.label_emb.0.0.weight", "UNET_LABEL_EMBED"),
        ("model.diffusion_model.out.2.weight", "UNET_OUT_FINAL"),
        # CLIP-L
        ("conditioner.embedders.0.transformer.text_model.encoder.layers.5.self_attn.q_proj.weight", "CLIP_L_IN05"),
        ("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight", "CLIP_L_EMBEDDING"),
        # CLIP-G
        ("conditioner.embedders.1.model.transformer.resblocks.10.attn.q_proj.weight", "CLIP_G_IN10"),
        ("conditioner.embedders.1.model.text_projection", "CLIP_G_TEXT_PROJECTION"),
        # VAE
        ("first_stage_model.encoder.down.0.block.0.conv1.weight", "VAE_ENCODER_DOWN"),
        ("first_stage_model.decoder.up.0.block.0.conv1.weight", "VAE_DECODER_UP"),
    ],
)
def test_conversion_mapping(blocks_dict, target_key, expected_block):
    result = convert_sdxl_optim_blocks_sub_to_sdxl_sgm.merge_key([blocks_dict], {}, target_key, None)
    assert result == f"val_{expected_block}"


def test_conversion_unhandled_key(blocks_dict):
    with pytest.raises(StateDictKeyError):
        convert_sdxl_optim_blocks_sub_to_sdxl_sgm.merge_key([blocks_dict], {}, "unknown.key.structure", None)


def test_conversion_missing_block(blocks_dict):
    # This block (UNET_IN08_0) is in YAML but not in our blocks_dict fixture
    with pytest.raises(StateDictKeyError):
        convert_sdxl_optim_blocks_sub_to_sdxl_sgm.merge_key([blocks_dict], {}, "model.diffusion_model.input_blocks.8.0.weight", None)
