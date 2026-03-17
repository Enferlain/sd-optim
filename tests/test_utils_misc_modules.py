from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

from omegaconf import OmegaConf
import torch


def _install_pynput_stub() -> None:
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = types.SimpleNamespace(ctrl=("ctrl",), shift=("shift",), alt=("alt",), esc="esc")
    keyboard_mod._pressed_events = {}

    class _Listener:
        def __init__(self, on_press=None):
            self.on_press = on_press
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    keyboard_mod.Listener = _Listener
    pynput_mod.keyboard = keyboard_mod
    sys.modules["pynput"] = pynput_mod
    sys.modules["pynput.keyboard"] = keyboard_mod


def _import_utils_module(module_name: str):
    _install_pynput_stub()
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def test_validate_run_config_accepts_minimal_layer_adjust_setup(tmp_path) -> None:
    config = _import_utils_module("sd_optim.utils.config")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    cfg = OmegaConf.create(
        {
            "models_dir": str(models_dir),
            "optimization_mode": "layer_adjust",
            "model_paths": ["base.safetensors"],
            "merge_dtype": "fp16",
            "save_dtype": "fp16",
        }
    )

    config.validate_run_config(cfg)


def test_modify_state_dict_scales_and_offsets_known_layers() -> None:
    images = _import_utils_module("sd_optim.utils.images")

    state_dict = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.tensor(10.0),
        "model.diffusion_model.input_blocks.0.0.bias": torch.tensor(20.0),
        "model.diffusion_model.out.0.weight": torch.tensor(30.0),
        "model.diffusion_model.out.0.bias": torch.tensor(40.0),
        "model.diffusion_model.out.2.weight": torch.tensor(50.0),
        "model.diffusion_model.out.2.bias": torch.tensor(60.0),
    }
    adjustments = {f"p{i}": value for i, value in enumerate([0, 0, 0, 10, 20, 30, 40, 50])}

    modified = images.modify_state_dict(state_dict, adjustments, is_xl_model=False)

    assert torch.equal(modified["model.diffusion_model.input_blocks.0.0.weight"], torch.tensor(10.0))
    assert torch.equal(modified["model.diffusion_model.input_blocks.0.0.bias"], torch.tensor(20.0))
    assert torch.equal(modified["model.diffusion_model.out.0.weight"], torch.tensor(30.0))
    assert torch.equal(modified["model.diffusion_model.out.0.bias"], torch.tensor(40.0))
    assert torch.equal(modified["model.diffusion_model.out.2.weight"], torch.tensor(50.0))
    assert not torch.equal(modified["model.diffusion_model.out.2.bias"], torch.tensor(60.0))


def test_get_summary_images_groups_best_image_per_payload(tmp_path) -> None:
    images = _import_utils_module("sd_optim.utils.images")
    log_file = tmp_path / "scores.jsonl"
    log_file.write_text(
        "\n".join(
            [
                json.dumps({"target": 0.1}),
                json.dumps({"target": 0.9}),
            ]
        ),
        encoding="utf-8",
    )
    imgs_dir = tmp_path / "imgs"
    imgs_dir.mkdir()
    for name in [
        "000-0-payloadA-0.20.png",
        "000-1-payloadA-0.70.png",
        "000-0-payloadB-0.30.png",
    ]:
        (imgs_dir / name).write_text("img", encoding="utf-8")

    summary = images.get_summary_images(log_file, imgs_dir, top_iterations=1)

    assert len(summary) == 2
    assert {item[1] for item in summary} == {0.70, 0.30}


def test_update_log_scores_rewrites_targets_with_new_values(tmp_path) -> None:
    images = _import_utils_module("sd_optim.utils.images")
    log_file = tmp_path / "scores.jsonl"
    log_file.write_text(
        "\n".join(
            [
                json.dumps({"target": 0.1}),
                json.dumps({"target": 0.2}),
            ]
        ),
        encoding="utf-8",
    )

    images.update_log_scores(log_file, summary_images=[("iter 000 - 0", 0.1, Path("a"))], new_scores=[0.9])

    updated = json.loads(log_file.read_text(encoding="utf-8"))
    assert updated[0]["target"] == 0.9
    assert updated[1]["target"] == 0.2


def test_hotkey_listener_switches_modes_with_pressed_modifiers() -> None:
    hotkeys = _import_utils_module("sd_optim.utils.hotkeys")
    keyboard_mod = sys.modules["pynput.keyboard"]
    keyboard_mod._pressed_events = {"ctrl": True}
    scoring_mode = types.SimpleNamespace(value="automatic")

    listener = hotkeys.HotkeyListener(scoring_mode)
    listener.on_press("m")
    assert scoring_mode.value == "manual"

    listener.on_press("a")
    assert scoring_mode.value == "automatic"
