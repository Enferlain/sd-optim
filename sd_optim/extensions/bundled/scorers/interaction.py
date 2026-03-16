"""Manual scorer interaction helpers."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_InputFn = Callable[[str], str]
_WSL_FALLBACK_WARNING_KEY = "wsl_xdg_open_fallback"


def handle_override_prompt(*, input_fn: _InputFn = input) -> float:
    """Prompt the user for a manual override score."""
    logger.warning("Score override activated!")
    while True:
        fake_score_input = input_fn(
            "\tOVERRIDE: Enter the final average score for this entire iteration (0-10): "
        ).strip()
        if not fake_score_input:
            logger.warning("Input cannot be empty.")
            continue

        try:
            fake_score = float(fake_score_input)
        except ValueError:
            logger.warning("Invalid input. Please enter a number.")
            continue

        if 0 <= fake_score <= 10:
            logger.info("Using fake average score: %.4f", fake_score)
            return fake_score

        logger.warning("Invalid score. Please enter a number between 0 and 10.")


def get_user_score(*, input_fn: _InputFn = input) -> float:
    """Prompt the user for a scorer value or an override sentinel."""
    while True:
        user_input = input_fn(
            "\n\tPlease enter the score for the shown image (a number between 0 and 10)\n\t> "
        ).strip()

        if user_input == "OVERRIDE_SCORE":
            return -1.0

        try:
            score = float(user_input)
        except ValueError:
            logger.warning("Invalid input. Please enter a number between 0 and 10.")
            continue

        if 0 <= score <= 10:
            return score

        logger.warning("Invalid input. Please enter a number between 0 and 10.")


def open_image(image_path: Path, *, warning_state: set[str] | None = None) -> None:
    """Open an image in the platform default viewer."""
    system = platform.system()

    try:
        if system == "Windows":
            os.startfile(str(image_path))
            return

        if system == "Linux":
            command = _get_linux_open_command(image_path, warning_state=warning_state)
            subprocess.run(command, check=True)
            return

        if system == "Darwin":
            subprocess.run(["open", str(image_path)], check=True)
            return

        logger.warning(
            "Automatic image opening not supported on '%s'. Open manually: %s",
            system,
            image_path,
        )
    except FileNotFoundError:
        logger.error("Could not find a configured image opener. Ensure it is installed/configured.")
    except (subprocess.CalledProcessError, OSError) as error:
        logger.error("Error opening image: %s", error)
        logger.warning("Try opening the image manually: %s", image_path)


def _get_linux_open_command(
    image_path: Path,
    *,
    warning_state: set[str] | None = None,
) -> list[str]:
    if _is_wsl():
        for opener in ("wslview", "xdg-open-wsl"):
            if shutil.which(opener):
                return [opener, str(image_path)]

        _warn_once(
            warning_state,
            _WSL_FALLBACK_WARNING_KEY,
            "WSL detected without 'wslview' or 'xdg-open-wsl'; falling back to 'xdg-open'. "
            "Install one of those tools if you want files to open in the host Windows app.",
        )

    return ["xdg-open", str(image_path)]


def _is_wsl() -> bool:
    if platform.system() != "Linux":
        return False

    uname = platform.uname()
    release = uname.release.lower()
    version = getattr(uname, "version", "").lower()
    return "microsoft" in release or "microsoft" in version or bool(os.environ.get("WSL_INTEROP"))


def _warn_once(warning_state: set[str] | None, key: str, message: str) -> None:
    if warning_state is None:
        logger.info(message)
        return

    if key in warning_state:
        return

    warning_state.add(key)
    logger.info(message)


__all__ = ["get_user_score", "handle_override_prompt", "open_image"]
