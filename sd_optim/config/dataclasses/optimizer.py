from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
