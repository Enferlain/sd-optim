from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from pathlib import Path

import yaml

from sd_mecha.extensions import model_configs
from sd_mecha.extensions.model_configs import ModelConfigImpl

logger = logging.getLogger(__name__)


def load_and_register_custom_configs(config_dir: Path) -> None:
    """Register user-defined model configs from YAML files in `config_dir`."""
    logger.info("Scanning for custom ModelConfigs in: %s", config_dir)
    registered_count = 0
    if not config_dir.is_dir():
        logger.warning("Custom config directory not found: %s. Skipping registration.", config_dir)
        return

    try:
        from yaml import CLoader as loader
    except ImportError:
        from yaml import Loader as loader

    for filepath in config_dir.glob("*.yaml"):
        try:
            logger.debug("  Loading config file: %s", filepath.name)
            with open(filepath, encoding="utf-8") as file:
                yaml_data = yaml.load(file, Loader=loader)
                if not isinstance(yaml_data, dict) or "identifier" not in yaml_data:
                    logger.warning("    Skipping %s: Invalid format or missing 'identifier'.", filepath.name)
                    continue

                config_obj = ModelConfigImpl(**yaml_data)
                config_id = config_obj.identifier
                model_configs.register_aux(config_obj)
                logger.info("  Successfully registered AUX ModelConfig: '%s' from %s", config_id, filepath.name)
                registered_count += 1
        except yaml.YAMLError as error:
            logger.error("  Error parsing YAML file %s: %s", filepath.name, error, exc_info=True)
        except TypeError as error:
            logger.error(
                "  Error constructing ModelConfig from %s (likely structure mismatch): %s",
                filepath.name,
                error,
                exc_info=True,
            )
        except ValueError as error:
            error_message = str(error)
            if "already exists" in error_message.lower():
                logger.warning(
                    "  Skipping custom ModelConfig from %s: duplicate identifier already registered (%s).",
                    filepath.name,
                    config_id,
                )
            else:
                logger.error(
                    "  Error registering ModelConfig from %s: %s",
                    filepath.name,
                    error,
                    exc_info=True,
                )
        except Exception as error:  # noqa: BLE001 - keep broad logging for dynamic plugin loading.
            logger.error("  Unexpected error processing %s: %s", filepath.name, error, exc_info=True)

    logger.info("Finished custom config scan. Registered %s config(s).", registered_count)


def load_and_register_custom_conversion(conversion_dir: Path) -> None:
    """Import public Python modules in `conversion_dir` to register merge helpers."""
    logger.info("Scanning for custom Conversion/MergeMethods in: %s", conversion_dir)
    registered_count = 0
    if not conversion_dir.is_dir():
        logger.warning("Custom conversion directory not found: %s. Skipping registration.", conversion_dir)
        return

    package_root_dir = conversion_dir.parent.resolve()
    package_name_parts = [conversion_dir.name]
    current_dir = conversion_dir.parent.resolve()
    while (current_dir / "__init__.py").exists():
        package_name_parts.append(current_dir.name)
        package_root_dir = current_dir.parent.resolve()
        current_dir = current_dir.parent.resolve()

    import_package = ".".join(reversed(package_name_parts))
    sys.path.insert(0, str(package_root_dir))

    try:
        for module_info in pkgutil.iter_modules([str(conversion_dir)]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            try:
                logger.debug("  Importing conversion module: %s", module_name)
                import_path = f"{import_package}.{module_name}"
                importlib.import_module(import_path)
                logger.info("  Successfully imported and potentially registered methods from: %s.py", module_name)
                registered_count += 1
            except ImportError as error:
                logger.error("  Error importing module %s.py: %s", module_name, error, exc_info=True)
            except Exception as error:  # noqa: BLE001 - keep broad logging for dynamic plugin loading.
                logger.error(
                    "  Unexpected error importing/registering from %s.py: %s",
                    module_name,
                    error,
                    exc_info=True,
                )
    finally:
        if str(package_root_dir) in sys.path:
            sys.path.pop(0)

    logger.info("Finished custom conversion scan. Imported %s module(s).", registered_count)
