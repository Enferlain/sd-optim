import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sd_webui_bayesian_merger.optimiser import Optimiser

class Artist:
    def __init__(self, optimiser: Optimiser):
        self.optimiser = optimiser
        self.cfg = optimiser.cfg
        self.data = []  # Store optimization data (iteration, score, parameters)

    def collect_data(self, score, params):
        """Collects data for each optimization iteration."""
        self.data.append({
            "iteration": self.optimiser.iteration,
            "score": score,
            "params": params.copy()  # Store a copy of the parameters
        })

    def visualize_optimization(self):
        """Creates an interactive 3D scatter plot of the optimization process using PCA."""

        if self.cfg.visualizations.scatter_plot:
            # Extract data for plotting
            iterations = [d["iteration"] for d in self.data]
            scores = [d["score"] for d in self.data]
            param_1 = [d["params"][self.cfg.plot_x_param] for d in self.data]  # Extract values for the first parameter to plot
            param_2 = [d["params"][self.cfg.plot_y_param] for d in self.data]  # Extract values for the second parameter to plot

            # Prepare parameter data for PCA
            param_data = [[d["params"][key] for key in d["params"]] for d in self.data]  # Extract all parameter values
            scaler = StandardScaler()  # Create a scaler for standardization
            scaled_param_data = scaler.fit_transform(param_data)  # Standardize the parameter data

            # Apply PCA to reduce to 3 dimensions
            pca = PCA(n_components=3)  # Create a PCA object for 3 dimensions
            reduced_data = pca.fit_transform(scaled_param_data)  # Apply PCA

            # Extract the reduced dimensions for plotting
            x = reduced_data[:, 0]
            y = reduced_data[:, 1]
            z = reduced_data[:, 2]

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
                text=[f"Iteration: {i}<br>Score: {s}<br>Params: {p}" for i, s, p in zip(iterations, scores, [d["params"] for d in self.data])],
            )])

            fig.update_layout(scene=dict(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                zaxis_title="Principal Component 3"
            ), title="Optimization Landscape")

            self.scatter_fig.write_html("optimization_landscape.html")
            self.scatter_fig.write_image("optimization_landscape.png")

    if self.cfg.visualizations.unet_diagram:
        self.draw_unet()
        self.unet_fig.write_html("unet_diagram.html")
        self.unet_fig.write_image("unet_diagram.png")

    if self.cfg.visualizations.convergence_plot:
        self.plot_convergence()
        self.convergence_fig.write_html("convergence_plot.html")
        self.convergence_fig.write_image("convergence_plot.png")

    if self.cfg.visualizations.heatmap:
        self.plot_heatmap()
        self.heatmap_fig.write_html("parameter_heatmap.html")
        self.heatmap_fig.write_image("parameter_heatmap.png")

    def draw_unet(self):
        """Creates an interactive UNet diagram."""

        model_arch = resolve(self.cfg.model_arch)
        unet_blocks = [key for key in model_arch.user_keys() if "_unet_block_" in key]
        block_names = [block.split('_')[-1] for block in unet_blocks]  # Extract block names (e.g., "in0", "mid")

        # Create nodes for the UNet blocks
        nodes = [go.Scatter3d(
            x=[i], y=[0], z=[0],  # Position nodes in a line
            mode='markers',
            marker=dict(size=10, color='lightblue'),  # Initial color
            text=block_names[i],
            hoverinfo='text',
            name=block_names[i]  # Assign block name for linking
        ) for i in range(len(unet_blocks))]

        # Create edges for connections (simplified example, adjust based on UNet structure)
        edges = []
        for i in range(len(unet_blocks) - 1):
            edges.append(go.Scatter3d(
                x=[i, i + 1], y=[0, 0], z=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2),
                hoverinfo='none'
            ))

        # Create the figure
        fig = go.Figure(data=nodes + edges)
        fig.update_layout(scene=dict(
            xaxis_title="", yaxis_title="", zaxis_title="",
            showticklabels=False, showgrid=False,  # Hide axes
        ), showlegend=False, title="Interactive UNet Diagram", margin=dict(l=0, r=0, b=0, t=40))

        # Store the figure and block identifiers for later updates
        self.unet_fig = fig
        self.unet_block_identifiers = unet_blocks

    def update_unet_diagram(self, selected_data):
        """Updates the UNet diagram based on selected data."""

        if selected_data is None:
            return  # No data selected, keep default colors

        selected_iteration = selected_data["points"][0]["pointIndex"]
        weights = self.data[selected_iteration]["params"]

        # Update node colors based on weights
        for i, block_id in enumerate(self.unet_block_identifiers):
            weight = weights.get(block_id, 0.5)  # Get weight, default to 0.5 if not found
            color = plt.cm.viridis(weight)  # Get color from Viridis colormap
            self.unet_fig.data[i].marker.color = f'rgb({color[0] * 255}, {color[1] * 255}, {color[2] * 255})'

    def plot_convergence(self):
        """Creates an interactive score convergence plot."""

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

        # Store the figure for later updates
        self.convergence_fig = fig

    def update_convergence_plot(self, selected_data):
        """Highlights the selected iteration on the convergence plot."""

        if selected_data is None:
            return  # No data selected, keep default appearance

        selected_iteration = selected_data["points"][0]["pointIndex"]

        # Highlight the selected point
        with self.convergence_fig.batch_update():
            for i, data in enumerate(self.convergence_fig.data):
                if i == selected_iteration:
                    data.marker.color = "red"  # Highlight the selected point
                    data.marker.size = 10
                else:
                    data.marker.color = "blue"  # Reset other points to default
                    data.marker.size = 8

    def plot_heatmap(self):
        """Creates an interactive parameter heatmap."""

        # Select initial parameters for the heatmap (can be made configurable)
        param_1 = self.cfg.plot_x_param
        param_2 = self.cfg.plot_y_param

        fig = self._create_heatmap(param_1, param_2)

        # Store the figure and initial parameters for later updates
        self.heatmap_fig = fig
        self.heatmap_param_1 = param_1
        self.heatmap_param_2 = param_2

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

    def update_heatmap(self, param_1=None, param_2=None):
        """Updates the heatmap with new parameters."""

        if param_1 is not None:
            self.heatmap_param_1 = param_1
        if param_2 is not None:
            self.heatmap_param_2 = param_2

        self.heatmap_fig = self._create_heatmap(self.heatmap_param_1, self.heatmap_param_2)