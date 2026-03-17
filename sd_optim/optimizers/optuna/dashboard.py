from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from sd_optim.optimizers.optuna.study_manager import build_storage_uri_for_new_study, initialize_storage_dir

if TYPE_CHECKING:
    from sd_optim.optimizers.optuna.optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)


def run_dashboard_in_background(storage_uri: str, port: int) -> subprocess.Popen | None:
    """Run the Optuna dashboard as a separate process that won't block."""
    logger.info("%s", "=" * 80)
    logger.info("LAUNCHING OPTUNA DASHBOARD")
    logger.info("%s", "=" * 80)
    logger.info("Access the dashboard at: http://localhost:%s", port)
    logger.info("The dashboard will run in the background.")
    logger.info("Check sd_optim.log for dashboard process status/errors on exit.")
    logger.info("%s", "=" * 80)
    time.sleep(0.5)

    cmd = [sys.executable, "-m", "optuna_dashboard", storage_uri, "--port", str(port)]

    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW

        dashboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        time.sleep(1.0)
        return_code = dashboard_process.poll()
        if return_code is not None:
            stderr_output = ""
            stdout_output = ""
            with contextlib.suppress(Exception):
                stdout_output, stderr_output = dashboard_process.communicate(timeout=0.2)
            logger.error(
                "Optuna dashboard exited immediately (return code %s). Command: %s",
                return_code,
                " ".join(cmd),
            )
            if stderr_output:
                logger.error("Dashboard stderr: %s", stderr_output.strip())
            if stdout_output:
                logger.error("Dashboard stdout: %s", stdout_output.strip())
            logger.error("Optuna dashboard exited immediately. Check sd_optim.log for details.")
            return None

        logger.info(
            "Launched background dashboard process (PID: %s) with command: %s",
            dashboard_process.pid,
            " ".join(cmd),
        )
        return dashboard_process
    except FileNotFoundError:
        logger.error(
            "Could not launch dashboard with '%s'. Is optuna-dashboard installed in this environment?",
            " ".join(cmd),
        )
        logger.error("Failed to launch dashboard: optuna-dashboard module not found in current environment.")
        return None
    except Exception as error:
        logger.error("Failed to launch dashboard process: %s", error, exc_info=True)
        return None


def start_dashboard_for_optimizer(optimizer: OptunaOptimizer, port: int = 8080) -> subprocess.Popen | None:
    """Determine the current study database path and launch the dashboard."""
    logger.info("Preparing to launch Optuna Dashboard in background...")

    try:
        optuna_storage_dir = getattr(optimizer, "optuna_storage_dir", None)
        if optuna_storage_dir is None:
            optuna_storage_dir = initialize_storage_dir(optimizer.cfg)
            optimizer.optuna_storage_dir = optuna_storage_dir

        storage_uri, db_filename = build_storage_uri_for_new_study(optimizer.cfg, optuna_storage_dir)
        storage_path = optuna_storage_dir / db_filename
        logger.info("Determined database URI for dashboard: %s", storage_uri)
        if not storage_path.exists():
            logger.warning(
                "Target Optuna DB file %s doesn't exist yet. Dashboard might show empty study initially.",
                storage_path,
            )
    except Exception as error:
        logger.error("Failed to determine Optuna DB path for background dashboard: %s", error)
        return None

    return run_dashboard_in_background(storage_uri, port)
