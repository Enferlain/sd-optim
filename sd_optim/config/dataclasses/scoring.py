from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringConfig:
    scorer_method: list[str] = field(default_factory=lambda: ["manual"])
    scorer_average_type: str = "arithmetic"
    scorer_weight: dict[str, float] = field(default_factory=dict)
    scorer_filters: dict[str, Any] = field(default_factory=dict)
    scorer_lazy_load_list: list[str] = field(default_factory=list)
    scorer_default_device: str = "cpu"
    scorer_device: dict[str, str] = field(default_factory=dict)
    scorer_alt_location: dict[str, Any] | None = None
    scorer_print_individual: bool = True

    pcascorer_component: int = 1
    pcascorer_mode: str = "projection"
    pcascorer_input_type: str = "color"
    pcascorer_linearize: bool = False
    pcascorer_invert: bool = False
    pcascorer_enhancement: str = "equalize"
    pcascorer_gamma: float = 1.0

    hybridnoise_kernel_size: int = 3
    hybridnoise_noise_threshold: float = 20.0
    hybridnoise_color_tolerance: int = 30
    hpsv3_uncertainty_penalty: float = 0.5
