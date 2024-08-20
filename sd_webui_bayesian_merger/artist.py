import os
import sd_mecha
from pathlib import Path

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hydra.core.hydra_config import HydraConfig

from sd_mecha.extensions.model_arch import resolve
from sd_webui_bayesian_merger.optimiser import Optimiser

class Artist:
    def __init__(self, optimiser: Optimiser):
        self.optimiser = optimiser
        self.cfg = optimiser.cfg
        self.data = []

    def collect_data(self, score, params):
        """Collects data for each optimization iteration."""
        self.data.append({
            "iteration": self.optimiser.iteration,
            "score": score,
            "params": params.copy()
        })

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
            self.plot_unet_heatmap(visualizations_dir)

        if self.cfg.visualizations.convergence_plot:
            self.plot_convergence(visualizations_dir)

#        if self.cfg.visualizations.heatmap:
#            self.plot_parameter_heatmap(visualizations_dir)

    def plot_3d_scatter(self, visualizations_dir):
        """Creates an interactive 3D scatter plot of the optimization process using PCA."""
        # Extract data for plotting
        iterations = [d["iteration"] for d in self.data]
        scores = [d["score"] for d in self.data]

        # Prepare parameter data for PCA
        param_data = [[d["params"][key] for key in d["params"]] for d in self.data]
        param_data = np.array(param_data).reshape(-1, 1) if len(param_data) > 0 else np.empty((0, 1))

        # Determine maximum allowable n_components
        n_samples, n_features = param_data.shape
        max_components = min(n_samples, n_features)
        n_components = min(3, max_components)  # Choose 3 or the maximum allowable, whichever is smaller

        scaler = StandardScaler()
        scaled_param_data = scaler.fit_transform(param_data)

        # Apply PCA with the determined n_components
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(scaled_param_data)

        # Extract the reduced dimensions for plotting
        x = reduced_data[:, 0]  # First principal component
        y = reduced_data[:, 1] if n_components > 1 else np.zeros_like(
            x)  # Second principal component (if available)
        z = reduced_data[:, 2] if n_components > 2 else np.zeros_like(x)  # Third principal component (if available)

        # Create the scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,  # Use the reduced x dimension
            y=y,  # Use the reduced y dimension
            z=z,  # Use the reduced z dimension
            mode='markers',
            marker=dict(
                size=8,
                color=iterations,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f"Iteration: {i}<br>Score: {s}<br>Params: {p}" for i, s, p in
                  zip(iterations, scores, [d["params"] for d in self.data])],
        )])

        fig.update_layout(scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        ), title="Optimization Landscape")

        # Assign the figure to self.scatter_fig before saving
        self.scatter_fig = fig
        self.scatter_fig.write_html(visualizations_dir / "optimization_landscape.html")
        self.scatter_fig.write_image(visualizations_dir / "optimization_landscape.png")

    def visualize_unet(self):
        """Creates a heatmap-style UNet visualization."""

        # Extract parameter names
        param_names = list(self.data[0][
                               "params"].keys()) if self.data else []  # Assuming parameter names are consistent across iterations

        # Retrieve UNet block identifiers
        model_arch = resolve(self.cfg.model_arch)
        unet_block_identifiers = [key for key in model_arch.user_keys() if "_unet_block_" in key]
        unet_block_identifiers.sort(key=sd_mecha.hypers.natural_sort_key)

        # Create heatmap data
        heatmap_data = []
        for param_name in param_names:
            row = []
            for block_id in unet_block_identifiers:
                # Get the weight for the current parameter and block
                weight = self.get_weight_for_parameter_and_block(param_name,
                                                                 block_id)  # Implement this function to retrieve weights from data
                row.append(weight)
            heatmap_data.append(row)

        # Create the heatmap using Plotly
        fig = go.Figure(data=[go.Heatmap(
            z=heatmap_data,
            x=unet_block_identifiers,
            y=param_names,
            colorscale='RdBu',  # Use a red-to-blue color scale
            colorbar=dict(title="Weight"),
        )])

        fig.update_layout(title="UNet Weight Visualization", xaxis_title="Block", yaxis_title="Parameter")
        fig.write_image(os.path.join(Path(HydraConfig.get().runtime.output_dir), "visualizations", "unet_diagram.png"))

    def plot_convergence(self, visualizations_dir):
        """Creates a score convergence plot."""

        iterations = [d["iteration"] for d in self.data]
        scores = [d["score"] for d in self.data]

        fig = go.Figure(data=[go.Scatter(
            x=iterations, y=scores,
            mode='lines+markers',
            marker=dict(size=8, color='blue'),
            text=[f"Iteration: {i}<br>Score: {s}" for i, s in zip(iterations, scores)],
            hoverinfo='text'
        )])

        fig.update_layout(xaxis_title="Iteration", yaxis_title="Score", title="Score Convergence")

        # Save the plot to the visualizations directory
        fig.write_html(visualizations_dir / "convergence_plot.html")
        fig.write_image(visualizations_dir / "convergence_plot.png")

    def plot_parameter_heatmap(self, visualizations_dir):
        """Creates a parameter heatmap."""

        # Get all unique parameter names across all iterations
        available_params = set()
        for data_point in self.data:
            available_params.update(data_point["params"].keys())

        # Convert the set to a list to maintain a consistent order
        available_params = list(available_params)

        param_1 = self.cfg.get("plot_x_param", available_params[0] if available_params else "base_alpha")
        param_2 = self.cfg.get("plot_y_param",
                               available_params[1] if len(available_params) > 1 else "sdxl_unet_block_in0")

        fig = self._create_heatmap(param_1, param_2)

        # Store the figure and initial parameters for later updates
        self.heatmap_fig = fig
        self.heatmap_param_1 = param_1
        self.heatmap_param_2 = param_2

        # Save the plot to the visualizations directory
        fig.write_html(visualizations_dir / "parameter_heatmap.html")
        fig.write_image(visualizations_dir / "parameter_heatmap.png")

    def _create_heatmap(self, param_1, param_2):
        """Creates a heatmap for the specified parameters."""

        # Extract data for the selected parameters
        param_1_values = [d["params"][param_1] for d in self.data]
        param_2_values = [d["params"][param_2] for d in self.data]
        scores = [d["score"] for d in self.data]

        # Create a grid of parameter values
        x_bins = np.linspace(min(param_1_values), max(param_1_values), num=20)
        y_bins = np.linspace(min(param_2_values), max(param_2_values), num=20)

        # Calculate average scores for each grid cell
        heatmap_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
        for i in range(len(y_bins) - 1):
            for j in range(len(x_bins) - 1):
                # Find data points within the current grid cell
                indices = np.where(
                    (param_1_values >= x_bins[j]) & (param_1_values < x_bins[j + 1]) & (param_2_values >= y_bins[i]) & (
                                param_2_values < y_bins[i + 1]))[0]
                if len(indices) > 0:
                    heatmap_data[i, j] = np.mean(np.array(scores)[indices])

        # Create the heatmap
        fig = go.Figure(data=[go.Heatmap(
            z=heatmap_data,
            x=x_bins, y=y_bins,
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title="Average Score")
        )])

        fig.update_layout(xaxis_title=param_1, yaxis_title=param_2, title="Parameter Heatmap")
        return fig