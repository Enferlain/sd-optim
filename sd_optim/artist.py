# artist.py - Version 1.0 - Optimizer Agnostic Basic Plots

import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig

# Prevent circular imports for type checking
if TYPE_CHECKING:
    from sd_optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

@dataclass
class Artist:
    """Handles plotting and visualization of optimization results."""
    optimizer: 'Optimizer' # Reference to the main optimizer instance
    # Data storage
    iterations: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    best_scores: List[float] = field(default_factory=list)
    parameters: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        logger.info(f"Artist initialized. Visualizations will be saved to: {self.vis_dir}")

    def collect_data(self, score: float, params: Dict):
        """Collects data from a completed optimization iteration."""
        try:
            iteration_num = self.optimizer.iteration # Get current iteration
            self.iterations.append(iteration_num)
            self.scores.append(score)
            self.best_scores.append(self.optimizer.best_rolling_score) # Get current best score
            self.parameters.append(params.copy()) # Store a copy of the params
            # logger.debug(f"Artist collected data for iteration {iteration_num}: Score={score:.4f}, Best={self.optimizer.best_rolling_score:.4f}")
        except Exception as e:
            logger.error(f"Error collecting data in Artist: {e}", exc_info=True)

    async def plot_convergence(self):
        """Plots the score and best score found over iterations."""
        if not self.iterations or not self.scores:
            logger.warning("No data collected for convergence plot.")
            return

        if not self.optimizer.cfg.visualizations.get("convergence_plot", False):
            logger.info("Convergence plot disabled in configuration.")
            return

        logger.info("Generating convergence plot...")
        try:
            plt.figure(figsize=(12, 6))

            # Plot individual trial scores
            plt.plot(self.iterations, self.scores, 'o-', alpha=0.7, label='Current Iteration Score', markersize=4)

            # Plot the best score found so far
            if self.best_scores:
                 plt.plot(self.iterations, self.best_scores, 'r-', linewidth=2, label='Best Score Found')

            plt.title(f'Optimization Convergence ({self.optimizer.cfg.run_name})')
            plt.xlabel('Iteration / Trial Number')
            plt.ylabel('Score')
            min_score = min(self.scores) if self.scores else 0
            max_score = max(self.scores) if self.scores else 1
            plt.ylim(max(0, min_score - (max_score-min_score)*0.1), max_score + (max_score-min_score)*0.1 ) # Dynamic Y limits
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = self.vis_dir / f"convergence_{self.optimizer.cfg.run_name}.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Convergence plot saved to {plot_path}")

        except Exception as e:
            logger.error(f"Failed to generate convergence plot: {e}", exc_info=True)

    # --- Generic Visualization Entry Point ---
    async def visualize_optimization(self):
        """Generates all enabled generic visualizations."""
        await self.plot_convergence()
        # Add calls to other generic plots here in the future

    # --- Placeholder for Optimizer-Specific Plots ---
    # These would be called directly from the specific optimizer's postprocess method

    def plot_optuna_visualizations(self, study):
        """Generates Optuna-specific visualizations."""
        if not self.optimizer.cfg.optimizer.get("optuna", False):
             return # Only run if optuna is active

        logger.info("Generating Optuna-specific plots...")
        if study is None or len(study.trials) < 2:
            logger.warning("Optuna study not available or not enough trials for visualization.")
            return

        try:
            import optuna.visualization as vis
            # Check if matplotlib is available for Optuna's backend
            if vis.is_available():
                vis_backend = vis # Use default plotly if matplotlib fails
                try:
                     # Try importing matplotlib backend specifically
                     from optuna.visualization import matplotlib as vis_mpl
                     vis_backend = vis_mpl
                     logger.info("Using Optuna's Matplotlib backend for plots.")
                except ImportError:
                     logger.warning("Optuna's Matplotlib backend not found. Install 'matplotlib'. Falling back to Plotly (might require 'kaleido' for saving).")


                plot_functions = {
                    "optimization_history": vis_backend.plot_optimization_history,
                    "param_importances": vis_backend.plot_param_importances,
                    "slice": vis_backend.plot_slice,
                    "parallel_coordinate": vis_backend.plot_parallel_coordinate,
                    # "contour": vis_backend.plot_contour, # Often requires specific param selection
                    # "edf": vis_backend.plot_edf, # Sometimes less informative
                }

                for name, plot_func in plot_functions.items():
                    try:
                        fig = plot_func(study)
                        save_path = self.vis_dir / f"optuna_{name}_{self.optimizer.cfg.run_name}.png"
                        # Saving depends on backend (matplotlib needs savefig, plotly needs write_image)
                        if hasattr(fig, 'savefig'): # Matplotlib axes/figure
                             fig.figure.savefig(save_path) # Access figure if it's Axes object
                        elif hasattr(fig, 'write_image'): # Plotly figure
                             try:
                                 fig.write_image(str(save_path))
                             except ValueError as ve:
                                  if "kaleido" in str(ve):
                                       logger.error("Failed to save Plotly plot: Kaleido engine not found. Install with 'pip install -U kaleido'")
                                  else: raise ve # Re-raise other ValueErrors
                        plt.close(fig if hasattr(fig, 'figure') else fig) # Close figure
                        logger.info(f"Saved Optuna plot: {name}")
                    except (ImportError, ValueError, TypeError, RuntimeError) as plot_err:
                         logger.warning(f"Could not generate Optuna plot '{name}': {plot_err}")

            else:
                 logger.warning("Optuna visualization module not available. Skipping Optuna plots.")

        except ImportError:
            logger.error("Optuna library not found. Cannot generate Optuna visualizations.")
        except Exception as e:
            logger.error(f"Error generating Optuna visualizations: {e}", exc_info=True)

    # Add plot_bayes_visualizations(self, bayes_optimizer_instance) later if needed