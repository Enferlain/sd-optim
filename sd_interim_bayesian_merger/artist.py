import os
import imageio
import sd_mecha

from pathlib import Path
from typing import Dict
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hydra.core.hydra_config import HydraConfig

from sd_mecha.extensions.model_arch import resolve
from sd_interim_bayesian_merger.optimizer import Optimizer


class Artist:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.cfg = optimizer.cfg
        self.data = []
        self.unet_block_identifiers = None  # Initialize the attribute

    def collect_data(self, score, params, weights_list, base_values):  # Accept the dictionaries as arguments
        """Collects data for each optimization iteration."""
        self.data.append({
            "iteration": self.optimizer.iteration,
            "score": score,
            "params": params.copy(),
            "weights_list": weights_list,
            "base_values": base_values
        })

    def _extract_visualization_data(self):
        """Extracts iteration and score data for visualizations."""
        return [d["iteration"] for d in self.data], [d["score"] for d in self.data]

    def visualize_optimization(self):
        """Creates visualizations based on user configuration."""

        # Get the Hydra log directory
        log_dir = Path(HydraConfig.get().runtime.output_dir)

        # Create the "visualizations" folder if it doesn't exist
        visualizations_dir = log_dir / "visualizations"
        os.makedirs(visualizations_dir, exist_ok=True)

        if self.cfg.visualizations.scatter_plot:
            self.plot_3d_scatter(visualizations_dir)

        if self.cfg.visualizations.unet_diagram:
            self.visualize_unet(visualizations_dir)

        if self.cfg.visualizations.convergence_plot:
            self.plot_convergence(visualizations_dir)

        if self.cfg.visualizations.heatmap:
            self.plot_parameter_heatmap(visualizations_dir)

    def plot_3d_scatter(self, visualizations_dir):
        """Creates an interactive 3D scatter plot of the optimization process using PCA."""

        # Extract data for plotting
        iterations, scores = self._extract_visualization_data()

        # ... (rest of the 3D scatter plot code)

    def visualize_unet(self, visualizations_dir: Path):
        """Creates a heatmap visualization of UNet block weights."""

        for data_point in self.data:
            self._draw_unet_heatmap(data_point["iteration"], data_point["params"], visualizations_dir)

    def _draw_unet_heatmap(self, iteration, params, visualizations_dir):
        """Creates a heatmap for the specified iteration and parameters."""

        # Access weights_list and base_values from the data point
        weights_list = self.data[iteration - 1]["weights_list"]
        base_values = self.data[iteration - 1]["base_values"]

        model_arch = resolve(self.cfg.model_arch)
        unet_block_identifiers = [key for key in model_arch.user_keys() if "_unet_block_" in key]
        unet_block_identifiers.sort(key=sd_mecha.hypers.natural_sort_key)

        # Get parameter names from the merging method's default hyperparameters
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(self.cfg.merge_mode)
        param_names = list(mecha_merge_method.get_default_hypers().keys())  # Get parameter names in the correct order

        # Add "txt" to the unet_block_identifiers for the heatmap
        unet_block_identifiers += [f"{self.cfg.model_arch}_txt_default"]

        # Create the heatmap data (2D array)
        heatmap_data = []
        for block_id in unet_block_identifiers:
            row = []
            for param_name in param_names:
                # Check if block_id corresponds to a base parameter
                if block_id.endswith("_default"):  # Use a more specific condition
                    weight = base_values.get(f"base_{param_name}", 0.0)  # Get base value, default to 0.0 if not found
                else:
                    weight = weights_list.get(param_name, {}).get(block_id, 0.0)  # Get block-specific weight

                row.append(weight)
            heatmap_data.append(row)

        # Pad the heatmap_data to make it more square
        max_dim = max(len(unet_block_identifiers), len(param_names))
        for row in heatmap_data:
            row.extend([0.0] * (max_dim - len(row)))  # Pad rows with 0.0 values
        heatmap_data.extend([[0.0] * max_dim] * (max_dim - len(heatmap_data)))  # Pad with empty rows

        # Create the heatmap using Plotly
        fig = go.Figure(data=[go.Heatmap(
            z=heatmap_data,
            x=param_names,
            y=unet_block_identifiers,
            colorscale='RdBu',  # Use a different colormap
            zmin=0, zmax=1,  # Normalize data to 0-1 range
            colorbar=dict(title="Weight Value"),
            xgap=3, ygap=1,  # Adjust cell spacing
        )])

        fig.update_layout(
            xaxis_title="Parameters",
            yaxis_title="UNet Blocks",
            title=f"UNet Weight Heatmap - Iteration {iteration}",
            xaxis=dict(showgrid=True, gridwidth=2, gridcolor='lightgray'),  # Adjust grid lines
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor='lightgray'),
            scene=dict(aspectratio=dict(x=2, y=1, z=0.5))  # Adjust aspect ratio
        )

        # Save the heatmap to a file
        fig.write_html(visualizations_dir / f"unet_heatmap_iteration_{iteration}.html")
        fig.write_image(visualizations_dir / f"unet_heatmap_iteration_{iteration}.png")

    def plot_convergence(self, visualizations_dir):
        """Creates an interactive score convergence plot."""

        iterations, scores = self._extract_visualization_data()

        # Create the scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=iterations,
            y=scores,
            mode='lines+markers',  # Display both lines and markers
            marker=dict(
                size=8,
                color='blue',        # Use a vibrant blue for the markers
                line=dict(width=2, color='darkblue')  # Outline the markers for better visibility
            ),
            line=dict(
                color='blue',        # Use a matching blue for the line
                width=2,
                shape='spline',      # Smooth the line for a more visually appealing curve
                smoothing=0.5         # Adjust smoothing factor for desired curvature
            ),
            text=[
                f"Iteration: {i}<br>Score: {s:.4f}<br>Average Block Weight: {np.mean(list(self.data[int(i) - 1]['params'].values())):.2f}"
                for i, s in zip(iterations, scores)
            ],
            hoverinfo='text'
        )])

        # Customize layout for clarity and aesthetics
        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Score",
            title="Score Convergence",
            xaxis=dict(
                showgrid=True,      # Add grid lines for better readability
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',   # Set a white background for a clean look
            paper_bgcolor='white'   # Set a white background for the entire plot area
        )

        # Save the plot to the visualizations directory
        fig.write_html(visualizations_dir / "convergence_plot.html")
        fig.write_image(visualizations_dir / "convergence_plot.png")

    def plot_parameter_heatmap(self, visualizations_dir):
        """Creates a parameter heatmap."""

        # ... (implementation from previous responses)

    def _create_heatmap(self, param_1, param_2):
        """Creates a heatmap for the specified parameters."""

        # ... (implementation from previous responses)

    def update_heatmap(self, param_1=None, param_2=None):
        """Updates the heatmap with new parameters."""

        # ... (implementation from previous responses)