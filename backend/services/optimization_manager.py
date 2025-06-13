import asyncio
import logging
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
# Assuming sd_optim is importable from the root of the project
from sd_optim.optimizer import Optimizer as BaseOptimizer # Rename to avoid conflict
from sd_optim.bayes_optimizer import BayesOptimizer
from sd_optim.optuna_optimizer import OptunaOptimizer

# Import schemas for configuration
from backend.schemas import ConfigUpdate

logger = logging.getLogger(__name__)

class OptimizationManager:
    _instance: Optional['OptimizationManager'] = None
    _is_initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self._optimization_task: Optional[asyncio.Task] = None
            self.status: Dict[str, Any] = {
                "state": "idle",
                "current_iteration": 0,
                "total_iterations": 0,
                "best_score": 0.0,
                "best_parameters": {},
                "log_messages": [], # To store recent log messages
                "progress_percentage": 0.0
            }
            self.optimizer_instance: Optional[BaseOptimizer] = None
            self.cfg: Optional[DictConfig] = None
            self._stop_event = asyncio.Event() # For stopping the optimization
            self._pause_event = asyncio.Event() # For pausing the optimization
            self._pause_event.set() # Start in a 'resumed' state (not paused)
            self._is_initialized = True
 self.active_connections: list[WebSocket] = []

    async def _run_optimization_task(self, cfg_data: ConfigUpdate):
        """
        Internal coroutine to run the optimization process.
        This will be spawned as a background task.
        """
        self.status["state"] = "initializing"
        self.status["log_messages"].append("Initializing optimization...")
        await self.send_status_update()
        logger.info("OptimizationManager: Starting internal optimization task.")

        try:
            # Convert Pydantic config to OmegaConf DictConfig
            # This is a simplified way. A real Hydra setup might involve more.
            # For now, we are directly creating a DictConfig from the Pydantic model.
            self.cfg = OmegaConf.create(cfg_data.model_dump())

            # Dynamically select optimizer
            optimizer_class = None
            if self.cfg.optimizer.get("bayes", False):
                optimizer_class = BayesOptimizer
            elif self.cfg.optimizer.get("optuna", False):
                optimizer_class = OptunaOptimizer
            
            if not optimizer_class:
                raise ValueError("No optimizer selected in configuration.")

            self.optimizer_instance = optimizer_class(self.cfg)

            if not self.optimizer_instance.validate_optimizer_config():
                raise ValueError("Optimizer configuration validation failed.")
            
            self.status["total_iterations"] = self.cfg.optimizer.init_points + self.cfg.optimizer.n_iters
            self.status["state"] = "running"
            self.status["log_messages"].append("Optimization started.")
            await self.send_status_update()
            logger.info("OptimizationManager: Running optimizer.")

            # Run the main optimization loop. This is where the actual work happens.
            await self.optimizer_instance.optimize() # This is the main blocking call to the optimizer
            await self.send_status_update() # Status should be 'finished' or 'cancelled' after optimize returns

            self.status["state"] = "postprocessing"
            logger.info("OptimizationManager: Running postprocessing.")
            await self.optimizer_instance.postprocess()

            self.status["state"] = "finished"
            self.status["log_messages"].append("Optimization finished successfully.")
            await self.send_status_update()
            logger.info("OptimizationManager: Optimization task completed.")

        except asyncio.CancelledError:
            self.status["state"] = "cancelled"
            self.status["log_messages"].append("Optimization cancelled by user.")
            await self.send_status_update()
            logger.warning("OptimizationManager: Optimization task was cancelled.")
        except ValueError as ve:
            self.status["state"] = "error"
            self.status["log_messages"].append(f"Configuration Error: {ve}")
            await self.send_status_update()
            logger.error(f"OptimizationManager: Configuration Error: {ve}", exc_info=True)
        except Exception as e:
            self.status["state"] = "error"
            self.status["log_messages"].append(f"An unexpected error occurred: {e}")
            await self.send_status_update()
            logger.error(f"OptimizationManager: An unexpected error occurred: {e}", exc_info=True)
        finally:
            # Clean up after task is done or cancelled
            self.optimizer_instance = None
            self._optimization_task = None
            self._stop_event.clear()
            self._pause_event.set() # Reset pause event to 'resumed' state for next run
            await self.send_status_update() # Send final state (finished/cancelled/error)
            logger.info("OptimizationManager: Internal optimization task cleanup complete.")

    async def start_optimization(self, cfg_data: ConfigUpdate) -> Dict[str, Any]:
        if self._optimization_task and not self._optimization_task.done():
            return {"message": "Optimization is already running or being initialized.", "status": self.status}

        self._stop_event.clear() # Ensure stop event is clear for a new run
        self._pause_event.set() # Ensure pause event is set to 'resumed' for a new run
        self.status = {
            "state": "starting",
            "current_iteration": 0,
            "total_iterations": 0,
            "best_score": 0.0,
            "best_parameters": {},
            "log_messages": ["Starting new optimization run..."],
            "progress_percentage": 0.0
        }

        self._optimization_task = asyncio.create_task(self._run_optimization_task(cfg_data))
        await self.send_status_update()
        logger.info("OptimizationManager: Optimization task spawned.")
        return {"message": "Optimization started.", "status": self.status}

    async def pause_optimization(self) -> Dict[str, Any]:
        if self._optimization_task and self.status["state"] == "running":
            self._pause_event.clear() # Set event to block the loop
 self.status["state"] = "paused" # State becomes 'paused' once event is cleared
            self.status["log_messages"].append("Attempting to pause optimization...")
            await self.send_status_update()
            logger.info("OptimizationManager: Pause requested.")
            return {"message": "Optimization pause initiated.", "status": self.status}
        return {"message": "Optimization is not running or already paused/stopped.", "status": self.status}

    async def resume_optimization(self) -> Dict[str, Any]:
        if self._optimization_task and self.status["state"] == "paused":
            self._pause_event.set() # Clear event to unblock the loop
            self.status["state"] = "running"
 await self.send_status_update()
            self.status["log_messages"].append("Resuming optimization...")
            logger.info("OptimizationManager: Resume requested.")
            return {"message": "Optimization resumed.", "status": self.status}
        return {"message": "Optimization is not paused.", "status": self.status}

    async def cancel_optimization(self) -> Dict[str, Any]:
        if self._optimization_task and not self._optimization_task.done():
            self._stop_event.set() # Signal the task to stop
            self._pause_event.set() # Also clear pause if it was set, so task can exit cleanly
            self.status["state"] = "cancelling"
            await self.send_status_update()
            self.status["log_messages"].append("Attempting to cancel optimization...")
            logger.warning("OptimizationManager: Cancel requested.")
            # Wait for the task to actually finish cancellation, with a timeout
            try:
                await asyncio.wait_for(self._optimization_task, timeout=10) # Give it 10 seconds to stop
            except asyncio.TimeoutError:
                self.status["log_messages"].append("Optimization task did not stop gracefully, forcing shutdown.")
                logger.error("OptimizationManager: Task did not stop gracefully within timeout.")
                self._optimization_task.cancel() # Force cancel
            except Exception as e:
                logger.error(f"Error during cancellation wait: {e}", exc_info=True)

            return {"message": "Optimization cancellation initiated (or completed).\n", "status": self.status}
        return {"message": "No active optimization to cancel.", "status": self.status}

    def get_status(self) -> Dict[str, Any]:
        # In a more advanced setup, you'd also get real-time stats from optimizer_instance
        # For now, this just reflects the internal state of the manager.
        return self.status


    async def connect(self, websocket: WebSocket):
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket}. Total connections: {len(self.active_connections)}")
        await self.send_status_update() # Send current status to the new client

    async def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected: {websocket}. Total connections: {len(self.active_connections)}")
        except ValueError:
            logger.warning(f"Attempted to remove non-existent WebSocket: {websocket}")

# Global instance of the OptimizationManager (Singleton pattern)
optimization_manager = OptimizationManager()