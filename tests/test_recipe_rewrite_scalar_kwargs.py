from __future__ import annotations

import sd_mecha
from sd_mecha import recipe_nodes


from sd_optim.merge import recipe_rewrite


def test_recipe_rewrite_inlines_scalar_kwargs_instead_of_aliasing_refs(monkeypatch) -> None:
    new_nodes = {
        "magnitude_ratio": sd_mecha.literal({"model.diffusion_model.out.0.weight": 1.25}, config="sdxl-sgm"),
        "rank_blend": recipe_nodes.LiteralRecipeNode({"value": 0.5}),
    }

    original_serialize_recipe_text = recipe_rewrite.serialize_recipe_text

    def fake_serialize_recipe_text(node, **kwargs):
        if isinstance(node, recipe_nodes.LiteralRecipeNode) and node.value_dict == {"value": 0.5}:
            return "version 0.1.0\n"
        return original_serialize_recipe_text(node, **kwargs)

    monkeypatch.setattr(recipe_rewrite, "serialize_recipe_text", fake_serialize_recipe_text)

    new_node_strings, param_to_replacement = recipe_rewrite.serialize_nodes_for_rewrite(new_nodes)

    assert param_to_replacement["magnitude_ratio"].startswith("&")
    assert param_to_replacement["rank_blend"] == "0.5"
    assert param_to_replacement["magnitude_ratio"] != param_to_replacement["rank_blend"]

    original_recipe = "\n".join(
        [
            "version 0.1.0",
            'model "a.safetensors" model_config="sdxl-sgm" merge_space="weight"',
            'merge "delta_widen" &0 magnitude_ratio=1.0 rank_blend=0.0',
        ]
    )

    rewritten = recipe_rewrite.rewrite_recipe_text(
        original_recipe_text=original_recipe,
        target_node_idx=1,
        new_node_strings=new_node_strings,
        param_to_replacement=param_to_replacement,
    )

    assert "rank_blend=0.5" in rewritten
    assert "rank_blend=&" not in rewritten


def test_recipe_rewrite_patches_multiple_target_lines(monkeypatch) -> None:
    rewritten = recipe_rewrite.rewrite_recipe_text(
        original_recipe_text="\n".join(
            [
                "version 0.1.0",
                'model "a.safetensors" model_config="sdxl-sgm" merge_space="weight"',
                'merge "weighted_sum" &0 alpha=0.1 beta=0.2',
                'merge "weighted_sum" &0 alpha=0.3 beta=0.4',
            ]
        ),
        target_node_indices=[1, 2],
        new_node_strings=[],
        param_to_replacement={"alpha": "0.9"},
    )

    assert 'merge "weighted_sum" &0 alpha=0.9 beta=0.2' in rewritten
    assert 'merge "weighted_sum" &0 alpha=0.9 beta=0.4' in rewritten
