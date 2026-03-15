from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import pytest
import sd_mecha
import yaml
from sd_mecha.keys_map import RealizedKeyRelation
from sd_mecha.streaming import StateDictKeyError

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _register_test_config(identifier: str) -> None:
    config_path = (
        REPO_ROOT / "sd_optim" / "builtin" / "model_configs" / f"{identifier}.yaml"
    )
    with open(config_path, encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    config_obj = sd_mecha.extensions.model_configs.ModelConfigImpl(**config_data)
    with contextlib.suppress(ValueError):
        sd_mecha.extensions.model_configs.register_aux(config_obj)


def _import_converter(module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    return importlib.import_module(module_name)


def _realized_relation(output_key: str, input_name: str, input_key: str) -> RealizedKeyRelation:
    return RealizedKeyRelation(
        outputs=(output_key,),
        inputs={input_name: (input_key,)},
    )


@pytest.fixture(autouse=True)
def _register_custom_configs() -> None:
    _register_test_config("sdxl-optim_blocks")
    _register_test_config("sdxl-optim_blocks_sub")


def test_main_converter_imports_as_class_method_and_builds_explicit_key_map() -> None:
    module = _import_converter(
        "sd_optim.builtin.model_configs.convert_sdxl_optim_blocks"
    )
    converter = module.convert_sdxl_optim_blocks_to_sdxl_sgm

    assert converter.wrapped_is_class is True

    input_config = sd_mecha.extensions.model_configs.resolve("sdxl-optim_blocks")
    output_config = sd_mecha.extensions.model_configs.resolve("sdxl-sgm")
    key_map = converter.build_key_map((input_config,), {}, output_config)

    relation = key_map["model.diffusion_model.input_blocks.3.0.op.weight"]
    assert relation.outputs == ("model.diffusion_model.input_blocks.3.0.op.weight",)
    assert relation.clauses[0].by_param == {"blocks_dict": ("UNET_IN03",)}


def test_main_converter_uses_skip_key_for_missing_optimized_block() -> None:
    module = _import_converter(
        "sd_optim.builtin.model_configs.convert_sdxl_optim_blocks"
    )
    converter = module.convert_sdxl_optim_blocks_to_sdxl_sgm
    context = converter.instantiate()

    target_key = "model.diffusion_model.output_blocks.8.0.weight"
    relation = _realized_relation(target_key, "blocks_dict", "UNET_OUT08")

    with pytest.raises(StateDictKeyError):
        converter.merge_key(
            [{"UNET_IN03": "value"}],
            {},
            target_key,
            relation,
            cache=None,
            context=context,
            output_reused=False,
        )


def test_granular_converter_imports_as_class_method_and_builds_explicit_key_map() -> None:
    module = _import_converter(
        "sd_optim.builtin.model_configs.convert_sdxl_optim_blocks_sub"
    )
    converter = module.convert_sdxl_optim_blocks_sub_to_sdxl_sgm

    assert converter.wrapped_is_class is True

    input_config = sd_mecha.extensions.model_configs.resolve("sdxl-optim_blocks_sub")
    output_config = sd_mecha.extensions.model_configs.resolve("sdxl-sgm")
    key_map = converter.build_key_map((input_config,), {}, output_config)

    relation = key_map["model.diffusion_model.output_blocks.5.2.conv.weight"]
    assert relation.outputs == ("model.diffusion_model.output_blocks.5.2.conv.weight",)
    assert relation.clauses[0].by_param == {"optimized_blocks": ("UNET_OUT05_2",)}


def test_granular_converter_returns_optimized_value_for_mapped_block() -> None:
    module = _import_converter(
        "sd_optim.builtin.model_configs.convert_sdxl_optim_blocks_sub"
    )
    converter = module.convert_sdxl_optim_blocks_sub_to_sdxl_sgm
    context = converter.instantiate()

    target_key = "conditioner.embedders.1.model.text_projection"
    relation = _realized_relation(target_key, "optimized_blocks", "CLIP_G_TEXT_PROJECTION")

    result = converter.merge_key(
        [{"CLIP_G_TEXT_PROJECTION": "projected"}],
        {},
        target_key,
        relation,
        cache=None,
        context=context,
        output_reused=False,
    )

    assert result == "projected"
