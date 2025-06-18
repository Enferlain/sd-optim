from pydantic import BaseModel, Field, validator, PrivateAttr
from typing import Dict, List, Optional, Union, Any

# --- Pydantic Models for Config Data (Based on config.tmpl.yaml) ---

class HydraRunConfig(BaseModel):
    dir: str = Field(..., description="Output directory for logs, images, and merged models.")

class HydraConfig(BaseModel):
    run: HydraRunConfig
    verbose: Optional[bool] = None

class WebUIUrlsConfig(BaseModel):
    a1111: str
    forge: str
    reforge: str
    comfy: str
    swarm: str

class RecipeOptimizationConfig(BaseModel):
    recipe_path: str
    target_nodes: Union[str, List[str]]
    target_params: List[str]

class BayesAcquisitionFunctionConfig(BaseModel):
    kind: str
    kappa: float
    xi: float
    kappa_decay: float
    kappa_decay_delay: Union[int, str]

class BayesBoundsTransformerConfig(BaseModel):
    enabled: bool
    gamma_osc: float
    gamma_pan: float
    eta: float
    minimum_window: float

class BayesConfig(BaseModel):
    load_log_file: Optional[str] = None
    reset_log_file: bool = False
    sampler: str
    acquisition_function: BayesAcquisitionFunctionConfig
    bounds_transformer: BayesBoundsTransformerConfig

class OptunaSamplerConfig(BaseModel):
    type: str
    multivariate: Optional[bool] = None
    group: Optional[bool] = None
    _extra: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def __pydantic_before_validator__(cls, data: Any) -> Any:
        if isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if k in cls.model_fields}
            extra_data = {k: v for k, v in data.items() if k not in cls.model_fields}
            instance = cls(**processed_data)
            instance._extra = extra_data
            return instance
        return data
    
class OptunaConfig(BaseModel):
    storage_dir: str
    resume_study_name: Optional[str] = None
    use_pruning: bool = False
    pruner_type: str
    early_stopping: bool = False
    patience: int
    min_improvement: float
    n_jobs: int = Field(1, description="Number of parallel jobs for Optuna. EXPERIMENTAL.")
    sampler: OptunaSamplerConfig
    launch_dashboard: bool = True
    dashboard_port: int

class OptimizerConfig(BaseModel):
    bayes: bool = False
    optuna: bool = False
    random_state: int = Field(-1, description="Seed for reproducibility. -1 for random.")
    init_points: int
    n_iters: int
    bayes_config: Optional[BayesConfig] = None
    optuna_config: Optional[OptunaConfig] = None

class ImageGenerationConfig(BaseModel):
    batch_size: int
    save_imgs: bool
    img_average_type: str
    background_check: dict

class GeneratorSettingsConfig(BaseModel):
    generator_concurrency_limit: int
    generator_keepalive_interval: int
    generator_total_timeout: Optional[int] = None

class ScoringConfig(BaseModel):
    scorer_method: List[str]
    scorer_average_type: str
    scorer_weight: Optional[Dict[str, float]] = None
    scorer_default_device: str
    scorer_device: Optional[Dict[str, str]] = None
    scorer_alt_location: Optional[Dict[str, Dict[str, str]]] = None
    scorer_print_individual: bool

class VisualizationsConfig(BaseModel):
    convergence_plot: bool
    _extra: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def __pydantic_before_validator__(cls, data: Any) -> Any:
        if isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if k in cls.model_fields}
            extra_data = {k: v for k, v in data.items() if k not in cls.model_fields}
            instance = cls(**processed_data)
            instance._extra = extra_data
            return instance
        return data

class ConfigUpdate(BaseModel):
    run_name: str
    hydra: HydraConfig
    webui_urls: WebUIUrlsConfig
    webui: str
    url: str
    configs_dir: str
    conversion_dir: str
    wildcards_dir: str
    scorer_model_dir: str
    model_paths: List[str]
    base_model_index: int
    fallback_model_index: Optional[int] = Field(-1, description="-1 or null to disable.")
    merge_method: str
    device: str
    threads: int
    merge_dtype: str
    save_dtype: str
    add_extra_keys: bool
    save_merge_artifacts: bool
    save_best: bool
    optimization_mode: str
    recipe_optimization: Optional[RecipeOptimizationConfig] = None
    optimizer: OptimizerConfig
    batch_size: int
    save_imgs: bool
    img_average_type: str
    background_check: dict
    generator_concurrency_limit: int
    generator_keepalive_interval: int
    generator_total_timeout: Optional[int] = None
    scorer_method: List[str]
    scorer_average_type: str
    scorer_weight: Optional[Dict[str, float]] = None
    scorer_default_device: str
    scorer_device: Optional[Dict[str, str]] = None
    scorer_alt_location: Optional[Dict[str, Dict[str, str]]] = None
    scorer_print_individual: bool
    visualizations: VisualizationsConfig
    _extra: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def __pydantic_before_validator__(cls, data: Any) -> Any:
        if isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if k in cls.model_fields}
            extra_data = {k: v for k, v in data.items() if k not in cls.model_fields}
            instance = cls(**processed_data)
            instance._extra = extra_data
            return instance
        return data

# --- Pydantic Models for Optimization Guide Data ---

class GroupStrategy(BaseModel):
    name: str
    keys: list[str]

class Strategy(BaseModel):
    type: str
    optimize_params: Optional[list[str]] = None
    target_type: Optional[str] = None
    _extra: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def __pydantic_before_validator__(cls, data: Any) -> Any:
        if isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if k in cls.model_fields}
            extra_data = {k: v for k, v in data.items() if k not in cls.model_fields}
            instance = cls(**processed_data)
            instance._extra = extra_data
            return instance
        return data

class GroupStrategyModel(Strategy):
    groups: List[GroupStrategy]
    keys: Optional[List[str]] = Field(default_factory=list)

    @validator('groups')
    def check_groups_for_group_strategy(cls, v, values):
        if values.get('type') == 'group' and not v:
            raise ValueError("'group' strategy must have a 'groups' list")
        return v

    @validator('keys', always=True)
    def check_keys_for_select_or_single_strategy(cls, v, values):
        if values.get('type') in ['select', 'single'] and not v:
            raise ValueError("'select' or 'single' strategy must have a 'keys' list")
        return v

class Component(BaseModel):
    name: str
    optimize_params: Optional[list[str]] = None
    strategies: List[Union[GroupStrategyModel, Strategy]]


class OptimizationGuideUpdate(BaseModel):
    optimization_targets: dict = {}
    custom_block_config_id: Optional[str] = None
    components: list[Component] = Field(default_factory=list)
    custom_bounds: Optional[Dict[str, Union[list[float], float, int, list[int]]]] = None

# --- Pydantic Models for Cargo and Payload Data ---

class PayloadData(BaseModel):
    score_weight: Optional[float] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    subseed: Optional[int] = None
    subseed_strength: Optional[float] = None
    seed_resize_from_h: Optional[int] = None
    seed_resize_from_w: Optional[int] = None
    enable_hr: Optional[bool] = None
    denoising_strength: Optional[float] = None
    firstphase_width: Optional[int] = None
    firstphase_height: Optional[int] = None
    hr_scale: Optional[float] = None
    hr_upscaler: Optional[str] = None
    hr_second_pass_steps: Optional[int] = None
    hr_resize_x: Optional[int] = None
    hr_resize_y: Optional[int] = None
    styles: Optional[List[str]] = None
    restore_faces: Optional[bool] = None
    tiling: Optional[bool] = None
    eta: Optional[float] = None
    s_churn: Optional[float] = None
    s_tmax: Optional[float] = None
    s_tmin: Optional[float] = None
    s_noise: Optional[float] = None
    vpred_enabled: Optional[bool] = None
    extension_name: Optional[str] = None
    _extra: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def __pydantic_before_validator__(cls, data: Any) -> Any:
        if isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if k in cls.model_fields}
            extra_data = {k: v for k, v in data.items() if k not in cls.model_fields}
            instance = cls(**processed_data)
            instance._extra = extra_data
            return instance
        return data

class CargoData(BaseModel):
    defaults: List[str] = Field(default_factory=list)
    score_weight: Optional[float] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    n_iter: Optional[int] = None
    batch_size: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sampler_name: Optional[str] = None
    sampler_index: Optional[str] = None
    seed: Optional[int] = None
    subseed: Optional[int] = None
    subseed_strength: Optional[float] = None
    seed_resize_from_h: Optional[int] = None
    seed_resize_from_w: Optional[int] = None
    enable_hr: Optional[bool] = None
    denoising_strength: Optional[float] = None
    firstphase_width: Optional[int] = None
    firstphase_height: Optional[int] = None
    hr_scale: Optional[float] = None
    hr_upscaler: Optional[str] = None
    hr_second_pass_steps: Optional[int] = None
    hr_resize_x: Optional[int] = None
    hr_resize_y: Optional[int] = None
    styles: Optional[List[str]] = None
    restore_faces: Optional[bool] = None
    tiling: Optional[bool] = None
    eta: Optional[float] = None
    s_churn: Optional[float] = None
    s_tmax: Optional[float] = None
    s_tmin: Optional[float] = None
    s_noise: Optional[float] = None
    vpred_enabled: Optional[bool] = None
    extension_name: Optional[str] = None
    _extra: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def __pydantic_before_validator__(cls, data: Any) -> Any:
        if isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if k in cls.model_fields}
            extra_data = {k: v for k, v in data.items() if k not in cls.model_fields}
            instance = cls(**processed_data)
            instance._extra = extra_data
            return instance
        return data

class PayloadUpdate(BaseModel):
    """Schema for updating an existing payload file with new content."""
    content: Dict[str, Any] = Field(..., description="The full YAML content of the payload as a dictionary.")

class PayloadCreate(PayloadUpdate):
    """Schema for creating a new payload. Inherits content and adds filename."""
    filename: str = Field(..., description="The name of the new payload file (without .yaml extension).")