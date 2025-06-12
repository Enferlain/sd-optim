# artist.py - Version 1.1 - Use Plotly for convergence plot

import logging
# REMOVE: import matplotlib.pyplot as plt # No longer using matplotlib here
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

# ADD Plotly imports
import plotly.graph_objects as go

from hydra.core.hydra_config import HydraConfig

# Prevent circular imports for type checking
if TYPE_CHECKING:
    from sd_optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class Artist:
    """Handles plotting and visualization of optimization results."""
    optimizer: 'Optimizer'  # Reference to the main optimizer instance
    # Data storage (kept for now, mainly for BayesOpt path)
    iterations: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    best_scores: List[float] = field(default_factory=list)
    parameters: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        logger.info(f"Artist initialized. Visualizations will be saved to: {self.vis_dir}")

    # collect_data remains the same
    def collect_data(self, score: float, params: Dict):
        """Collects data from a completed optimization iteration."""
        try:
            iteration_num = self.optimizer.iteration  # Get current iteration
            self.iterations.append(iteration_num)
            self.scores.append(score)
            # Ensure best_rolling_score exists, default to score if not
            current_best = getattr(self.optimizer, 'best_rolling_score', score)
            self.best_scores.append(current_best)
            self.parameters.append(params.copy())
        except Exception as e:
            logger.error(f"Error collecting data in Artist: {e}", exc_info=True)

    # --- MODIFIED: plot_convergence uses Plotly ---
    async def plot_convergence(self):
        """Plots the score and best score found over iterations using Plotly."""
        if not self.iterations or not self.scores:
            logger.warning("Artist: No data collected for convergence plot.")
            return

        # Check config if this specific plot should be generated
        # (using optimizer's cfg as artist doesn't have direct access otherwise)
        if not self.optimizer.cfg.visualizations.get("convergence_plot", False):
            logger.info("Artist: Convergence plot disabled in configuration.")
            return

        logger.info("Artist: Generating convergence plot (using Plotly)...")
        try:
            fig = go.Figure()

            # Plot individual trial scores
            fig.add_trace(go.Scatter(
                x=self.iterations,
                y=self.scores,
                mode='lines+markers',
                name='Current Iteration Score',
                marker=dict(size=4),
                line=dict(width=1),
                opacity=0.7
            ))

            # Plot the best score found so far
            if self.best_scores:
                fig.add_trace(go.Scatter(
                    x=self.iterations,
                    y=self.best_scores,
                    mode='lines',
                    name='Best Score Found',
                    line=dict(color='red', width=2)
                ))

            # Update layout
            plot_title = f'Optimization Convergence ({getattr(self.optimizer.cfg, "run_name", "Unknown Run")})'
            fig.update_layout(
                title=plot_title,
                xaxis_title='Iteration / Trial Number',
                yaxis_title='Score',
                # yaxis_range=[min_y, max_y], # Optional: Set dynamic range if needed
                legend_title_text='Legend',
                template="plotly_white"  # Use a clean template
            )
            fig.update_yaxes(rangemode='tozero')  # Ensure y-axis starts at 0 or below min score

            # Save the plot
            plot_path = self.vis_dir / f"convergence_{getattr(self.optimizer.cfg, 'run_name', 'Unknown Run')}.png"
            try:
                fig.write_image(str(plot_path))
                logger.info(f"Artist: Convergence plot saved to {plot_path}")
            except ValueError as ve:
                if "kaleido" in str(ve).lower():
                    logger.error(
                        "Artist: Failed to save Plotly plot: Kaleido engine not found or not functional. Install with 'pip install -U kaleido'. Skipping save.")
                else:
                    logger.error(f"Artist: ValueError saving Plotly plot: {ve}. Skipping save.")
            except Exception as e_write:
                logger.error(f"Artist: Unexpected error saving Plotly plot: {e_write}. Skipping save.")

        except Exception as e:
            logger.error(f"Artist: Failed to generate convergence plot: {e}", exc_info=True)

    # --- MODIFIED: visualize_optimization calls the new plot_convergence ---
    async def visualize_optimization(self):
        """Generates all enabled generic visualizations."""
        # Only calls plot_convergence now. Add calls to other generic/custom plots here if added later.
        await self.plot_convergence()

    # --- REMOVED plot_optuna_visualizations METHOD ---
    # def plot_optuna_visualizations(self, study):
    #     # THIS METHOD IS NO LONGER NEEDED HERE - Logic moved to OptunaOptimizer.postprocess
    #     pass
