from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


def test_open_image_uses_startfile_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    from sd_optim.scoring import interaction as interaction_mod

    image_path = interaction_mod.Path("C:/tmp/test.png")
    calls: list[str] = []

    monkeypatch.setattr(interaction_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(interaction_mod.os, "startfile", lambda path: calls.append(path), raising=False)
    monkeypatch.setattr(
        interaction_mod.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be used on Windows"),
    )

    interaction_mod.open_image(image_path)

    assert calls == [str(image_path)]


def test_open_image_prefers_wsl_windows_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    from sd_optim.scoring import interaction as interaction_mod

    image_path = interaction_mod.Path("/tmp/test.png")
    calls: list[list[str]] = []

    monkeypatch.setattr(interaction_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        interaction_mod.platform,
        "uname",
        lambda: SimpleNamespace(release="6.6.87.2-microsoft-standard-WSL2", version=""),
    )
    monkeypatch.setattr(
        interaction_mod.shutil,
        "which",
        lambda name: "/usr/bin/wslview" if name == "wslview" else None,
    )
    monkeypatch.setattr(
        interaction_mod.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    interaction_mod.open_image(image_path)

    assert calls == [["wslview", str(image_path)]]


def test_open_image_warns_once_when_wsl_falls_back_to_xdg_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sd_optim.scoring import interaction as interaction_mod

    image_path = interaction_mod.Path("/tmp/test.png")
    warning_state: set[str] = set()
    calls: list[list[str]] = []

    monkeypatch.setattr(interaction_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        interaction_mod.platform,
        "uname",
        lambda: SimpleNamespace(release="6.6.87.2-microsoft-standard-WSL2", version=""),
    )
    monkeypatch.setattr(interaction_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        interaction_mod.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    caplog.set_level(logging.INFO)

    interaction_mod.open_image(image_path, warning_state=warning_state)
    interaction_mod.open_image(image_path, warning_state=warning_state)

    assert calls == [["xdg-open", str(image_path)], ["xdg-open", str(image_path)]]
    wsl_fallback_logs = [
        record.message
        for record in caplog.records
        if "falling back to 'xdg-open'" in record.message.lower()
    ]
    assert len(wsl_fallback_logs) == 1


def test_get_user_score_retries_until_valid_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sd_optim.scoring import interaction as interaction_mod

    answers = iter(["not-a-number", "11", "7.5"])

    caplog.set_level(logging.WARNING)

    assert interaction_mod.get_user_score(input_fn=lambda prompt: next(answers)) == 7.5
    assert sum(1 for record in caplog.records if "Invalid input" in record.message) == 2


def test_handle_override_prompt_retries_until_valid_score(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sd_optim.scoring import interaction as interaction_mod

    answers = iter(["", "twelve", "12", "8"])

    caplog.set_level(logging.WARNING)

    assert interaction_mod.handle_override_prompt(input_fn=lambda prompt: next(answers)) == 8.0
    assert any("Score override activated!" in record.message for record in caplog.records)
