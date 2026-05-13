from pathlib import Path

import yaml

from sd_optim.guide_compiler import (
    BindingSpec,
    NamedGroupSpec,
    SelectionSpec,
    TargetSource,
    build_optimizer_bounds,
    compile_bindings,
    materialize_recipe_payloads,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "guide_compiler_expected_results.yaml"


def test_graph_compiler_recreates_expected_block_payload_excerpt() -> None:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        fixture = yaml.safe_load(handle)["delta_widen_block_excerpt"]

    source = TargetSource(
        name=fixture["source"]["name"],
        target_space=fixture["source"]["target_space"],
        items=tuple(fixture["source"]["items"]),
    )
    bindings = [
        BindingSpec(
            method_param_name=method_param_name,
            component_name="unet",
            strategy_label="all",
            selection=SelectionSpec(source_name=source.name),
            grouping="per_target",
        )
        for method_param_name in fixture["binding_params"]
    ]

    compiled = compile_bindings([source], bindings)
    payloads = materialize_recipe_payloads(fixture["sampled_values"], compiled)

    assert payloads == fixture["expected_payloads"]


def test_graph_compiler_supports_shared_and_named_group_bindings() -> None:
    source = TargetSource(
        name="unet_keys",
        target_space="key",
        items=(
            "model.diffusion_model.out.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.out.2.bias",
            "model.diffusion_model.time_embed.weight",
        ),
    )
    bindings = [
        BindingSpec(
            method_param_name="alpha",
            component_name="unet",
            strategy_label="single",
            selection=SelectionSpec(
                source_name="unet_keys",
                include=("model.diffusion_model.out.*",),
                exclude=("*.bias",),
            ),
            grouping="shared",
            group_name="decoder_surface",
            bounds=(0.0, 1.0),
        ),
        BindingSpec(
            method_param_name="rank_ratio",
            component_name="unet",
            strategy_label="group",
            selection=SelectionSpec(source_name="unet_keys"),
            grouping="named_groups",
            bounds=[0.5, 0.75, 1.0],
            named_groups=(
                NamedGroupSpec(
                    name="out",
                    include=("model.diffusion_model.out.*",),
                    exclude=("*.bias",),
                ),
                NamedGroupSpec(
                    name="time_embed",
                    include=("model.diffusion_model.time_embed.*",),
                ),
            ),
        ),
    ]

    compiled = compile_bindings([source], bindings)

    assert [binding.optimizer_param_name for binding in compiled] == [
        "decoder_surface_alpha",
        "out_rank_ratio",
        "time_embed_rank_ratio",
    ]
    assert build_optimizer_bounds(compiled) == {
        "decoder_surface_alpha": (0.0, 1.0),
        "out_rank_ratio": [0.5, 0.75, 1.0],
        "time_embed_rank_ratio": [0.5, 0.75, 1.0],
    }

    payloads = materialize_recipe_payloads(
        {
            "decoder_surface_alpha": 0.8,
            "out_rank_ratio": 0.75,
            "time_embed_rank_ratio": 1.0,
        },
        compiled,
    )

    assert payloads == {
        "alpha": {
            "model.diffusion_model.out.0.weight": 0.8,
            "model.diffusion_model.out.2.weight": 0.8,
        },
        "rank_ratio": {
            "model.diffusion_model.out.0.weight": 0.75,
            "model.diffusion_model.out.2.weight": 0.75,
            "model.diffusion_model.time_embed.weight": 1.0,
        },
    }
