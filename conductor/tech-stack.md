# Technology Stack - sd-optim

## Core
- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch
- **State Dictionary Backend:** `sd-mecha` (utilizing Safetensors and NumPy)

## Optimization & Data
- **Optimization Engine:** Optuna (including `optuna-dashboard`, `optuna-fast-fanova`, and `cmaes` for advanced sampling)
- **Configuration:** Hydra & OmegaConf (with PyYAML for robust, schema-validated configuration management)
- **Data Visualization:** Plotly & Kaleido (for interactive and static optimization tracking)

## Infrastructure & Integration
- **Asynchronous Operations:** `aiohttp` (for non-blocking API communication and I/O)
- **WebUI Integration:** Supports image generation and feedback via APIs from:
    - Stable Diffusion WebUI (A1111 / Forge / ReForge)
    - ComfyUI
- **Storage:** Optuna DB (SQLite/RDB) and YAML-based state/log files
