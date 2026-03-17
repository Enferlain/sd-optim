from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf


def test_load_validated_target_nodes_accepts_multiple_refs(monkeypatch) -> None:
    from sd_optim.merge import recipe_optimization as recipe_mod

    class _FakeParamNames:
        args = ()
        kwargs = {"alpha": object()}

    class _FakeMethod:
        identifier = "weighted_sum"

        def get_param_names(self):
            return _FakeParamNames()

    class _FakeMergeNode:
        def __init__(self) -> None:
            self.merge_method = _FakeMethod()

    merger = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "recipe_optimization": {
                    "target_nodes": ["&0", "&1"],
                    "target_params": ["alpha"],
                }
            }
        )
    )
    original_recipe_text = "\n".join(
        [
            "version 0.1.0",
            'merge "weighted_sum" alpha=0.5',
            'merge "weighted_sum" alpha=0.6',
        ]
    )

    monkeypatch.setattr(recipe_mod, "MergeRecipeNode", _FakeMergeNode)
    monkeypatch.setattr(recipe_mod.sd_mecha, "deserialize", lambda lines: _FakeMergeNode())

    targets = recipe_mod.load_validated_target_nodes(merger, original_recipe_text)

    assert [ref for ref, _ in targets] == ["&0", "&1"]
    assert all(isinstance(node, _FakeMergeNode) for _, node in targets)
