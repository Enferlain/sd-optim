from __future__ import annotations

import logging

from pynput import keyboard

logger = logging.getLogger(__name__)

HOTKEY_SWITCH_MANUAL = keyboard.Key.ctrl, "m"
HOTKEY_SWITCH_AUTO = keyboard.Key.ctrl, "a"


class HotkeyListener:
    """Minimal listener for switching scoring modes via keyboard shortcuts."""

    def __init__(self, scoring_mode):
        self.scoring_mode = scoring_mode
        self.listener = keyboard.Listener(on_press=self.on_press)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            return False
        try:
            if key == HOTKEY_SWITCH_MANUAL[1] and all(
                pressed_key in keyboard._pressed_events
                for pressed_key in HOTKEY_SWITCH_MANUAL[0]
            ):
                self.scoring_mode.value = "manual"
                logger.info("Switching to manual scoring mode.")
            elif key == HOTKEY_SWITCH_AUTO[1] and all(
                pressed_key in keyboard._pressed_events
                for pressed_key in HOTKEY_SWITCH_AUTO[0]
            ):
                self.scoring_mode.value = "automatic"
                logger.info("Switching to automatic scoring mode.")
        except AttributeError:
            return None
        return None
