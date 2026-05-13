from __future__ import annotations

from hydra.core.config_store import ConfigStore


def register_sd_optim() -> None:
    """Register the shared sd-optim root config schema."""
    from sd_optim.config.dataclasses.run import SdOptimConfig

    cs = ConfigStore.instance()
    cs.store(name="sd_optim_schema", node=SdOptimConfig)


def register_all() -> None:
    """Register all active sd-optim config schemas for tests, tools, and entrypoints."""
    register_sd_optim()
