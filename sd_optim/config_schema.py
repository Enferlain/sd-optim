from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class RecipeOptimizationConfig:
    recipe_path: str = ""
    target_nodes: Any = ""
    target_params: list[str] = field(default_factory=list)


@dataclass
class BayesAcquisitionConfig:
    kind: str = "ucb"
    kappa: float = 3.0
    xi: float = 0.05
    kappa_decay: float = 0.98
    kappa_decay_delay: Any = "${optimizer.init_points}"


@dataclass
class BayesBoundsTransformerConfig:
    enabled: bool = False
    gamma_osc: float = 0.7
    gamma_pan: float = 1.0
    eta: float = 0.9
    minimum_window: float = 0.0


@dataclass
class BayesConfig:
    load_log_file: str | None = None
    reset_log_file: bool = False
    sampler: str = "sobol"
    acquisition_function: BayesAcquisitionConfig = field(default_factory=BayesAcquisitionConfig)
    bounds_transformer: BayesBoundsTransformerConfig = field(default_factory=BayesBoundsTransformerConfig)


@dataclass
class OptunaSamplerConfig:
    type: str = "tpe"
    multivariate: bool = True
    group: bool = True
    constant_liar: bool = True
    warn_independent_sampling: bool = True
    n_ei_candidates: int = 48
    gamma: Any = None
    prior_weight: float | None = None
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    restart_strategy: str | None = None
    sigma0: float | None = None
    popsize: int | None = None
    inc_popsize: int = -1
    use_separable_cma: bool = False
    lr_adapt: bool = False
    x0: dict[str, Any] | None = None
    consider_pruned_trials: bool = False
    with_margin: bool = False
    deterministic_objective: bool = False
    n_startup_trials: int | None = None
    qmc_type: str = "sobol"
    scramble: bool = True
    warn_asynchronous_seeding: bool = True
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    population_size: int | None = None
    mutation_prob: float | None = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    crossover: Any = None
    constraints_func: Any = None
    elite_population_selection_strategy: Any = None
    child_generation_strategy: Any = None
    after_trial_strategy: Any = None
    candidates_func: Any = None
    n_objectives: int | None = None


@dataclass
class OptunaConfig:
    storage_dir: str = "optuna_db"
    resume_from_study: str | None = None
    fork_study: bool = False
    use_pruning: bool = False
    pruner_type: str = "median"
    early_stopping: bool = False
    patience: int = 10
    min_improvement: float = 0.001
    n_jobs: int = 1
    sampler: OptunaSamplerConfig = field(default_factory=OptunaSamplerConfig)
    n_startup_trials: int | None = None
    n_warmup_steps: int = 0
    interval_steps: int = 1
    direction: str | None = None
    catch: list[Any] = field(default_factory=list)
    callbacks: list[Any] = field(default_factory=list)
    launch_dashboard: bool = True
    dashboard_port: int = 8080


@dataclass
class OptimizerConfig:
    bayes: bool = False
    optuna: bool = True
    random_state: int = -1
    init_points: int = 10
    n_iters: int = 20
    bayes_config: BayesConfig = field(default_factory=BayesConfig)
    optuna_config: OptunaConfig = field(default_factory=OptunaConfig)


@dataclass
class VisualizationConfig:
    convergence_plot: bool = True
    scatter_plot: bool = False
    unet_diagram: bool = False
    heatmap: bool = False


@dataclass
class SdOptimConfig:
    run_name: str = "${merge_method}_${scorer_method}"
    hydra: dict[str, Any] = field(default_factory=dict)
    payloads: dict[str, Any] = field(default_factory=dict)
    optimization_guide: dict[str, Any] = field(default_factory=dict)

    webui_urls: dict[str, str] = field(
        default_factory=lambda: {
            "a1111": "http://localhost:7860",
            "forge": "http://localhost:7860",
            "reforge": "http://localhost:7860",
            "comfy": "http://localhost:8188",
            "swarm": "http://localhost:7801",
        }
    )
    webui: str = "forge"
    url: str = "${webui_urls[${webui}]}"

    models_dir: str = ""
    configs_dir: str = ""
    conversion_dir: str = ""
    wildcards_dir: str = "wildcards"
    scorer_model_dir: str = ""

    model_paths: list[str] = field(default_factory=list)
    base_model_index: int = 0
    fallback_model_index: int | None = -1

    merge_method: str = "weighted_sum"
    device: str = "cuda"
    threads: int = 4
    merge_dtype: str = "fp32"
    save_dtype: str = "bf16"
    add_extra_keys: bool = False

    save_merge_artifacts: bool = True
    save_best: bool = True
    reuse_cached_results: bool = False
    reuse_scan_legacy_pngs: bool = False
    fail_on_error: bool = True

    optimization_mode: str = "merge"
    recipe_optimization: RecipeOptimizationConfig = field(default_factory=RecipeOptimizationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    batch_size: int = 1
    save_imgs: bool = True
    img_average_type: str = "arithmetic"
    generator_concurrency_limit: int = 10
    generator_keepalive_interval: int = 60
    generator_total_timeout: int | float | None = 0

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

    visualizations: VisualizationConfig = field(default_factory=VisualizationConfig)


def register_config_schemas() -> None:
    """Register dataclass-backed Hydra schemas for stable sd-optim settings."""
    cs = ConfigStore.instance()
    cs.store(name="sd_optim_schema", node=SdOptimConfig)
