from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisualizationConfig:
    convergence_plot: bool = True
    scatter_plot: bool = False
    unet_diagram: bool = False
    heatmap: bool = False
