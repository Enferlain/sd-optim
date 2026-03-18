from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sd_optim.utils import conversions


def test_duplicate_custom_config_registration_is_logged_as_warning(
    monkeypatch,
    tmp_path: Path,
    caplog,
) -> None:
    config_path = tmp_path / "demo.yaml"
    config_path.write_text("identifier: demo\n", encoding="utf-8")

    monkeypatch.setattr(
        conversions,
        "ModelConfigImpl",
        lambda **yaml_data: SimpleNamespace(identifier=yaml_data["identifier"]),
    )

    def _raise_duplicate(config_obj) -> None:
        raise ValueError(f"Model {config_obj.identifier} already exists")

    monkeypatch.setattr(conversions.model_configs, "register_aux", _raise_duplicate)

    caplog.set_level("WARNING")
    conversions.load_and_register_custom_configs(tmp_path)

    assert "Skipping custom ModelConfig from demo.yaml: duplicate identifier already registered (demo)." in caplog.text
    assert not any(record.exc_info for record in caplog.records)
