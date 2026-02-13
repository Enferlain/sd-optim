# sd-optim: State Dictionary Optimization Framework

An opinionated framework for optimizing operations on state dictionaries, particularly focused on Stable Diffusion model merging, leveraging Bayesian Optimization or Optuna and the [`sd-mecha`](https://github.com/ljleb/sd-mecha) library.

**(Formerly sd-interim-bayesian-merger)**

This project aims to provide a flexible and powerful tool for finding optimal parameters for `sd-mecha` merge methods or other state dictionary manipulations based on image generation and scoring feedback.

---

**Note:** This project is under active development and might undergo significant changes. While usability is a goal, stability is not always guaranteed on the main branch. A UI is also being worked on in spare time to provide a more user-friendly interface in the future.

---

## Key Features

*   **Powerful Merging Backend:** Uses [`sd-mecha`](https://github.com/ljleb/sd-mecha) for efficient, low-memory state dictionary operations.
*   **Flexible Optimization:**
    *   Supports both **Bayesian Optimization** (via `bayesian-optimization` - untested) and **Optuna** for hyperparameter search.
    *   Optimize parameters for built-in or custom `sd-mecha` merge methods.
    *   Optimize hyperparameters within existing `.mecha` recipes (`optimization_mode: recipe`).
    *   Optimize layer adjustments (`optimization_mode: layer_adjust`). *(not implemented yet)*
*   **Granular Control:** Use `conf/optimization_guide.yaml` to define *which* parameters to optimize using flexible strategies:
    *   `all`: Optimize parameters for every key in a component.
    *   `select`: Optimize parameters for specific keys or wildcard patterns.
    *   `group`: Optimize shared parameters for defined groups of keys/blocks.
    *   `single`: Optimize a single shared parameter for an entire component.
    *   `none`: Exclude a component from optimization.
*   **Custom Block Definitions:** Define custom block groupings via configuration and utility scripts for targeted optimization (see Wiki).
*   **Multiple Scorers:** Utilizes various image scoring models (Aesthetic, CLIP, BLIP, HPSv3, ImageReward, PickScore, CityAesthetics, etc.) to guide optimization. See [[Scoring]] wiki page.
*   **WebUI Integration:** Designed to run alongside a running instance of A1111, Forge, SwarmUI (comfy, reforge forks planned) via their APIs for image generation.
*   **Asynchronous Workflow:** Generates and scores images concurrently for better efficiency.

## Getting Started

1.  **Prerequisites:** Python 3.11-3.13, `uv`, Git, and a running instance of a supported WebUI (e.g., Comfy tested, A1111, Forge are untested) with its API enabled.
2.  **Installation:** Clone this repository into your WebUI's `extensions` folder:
    ```bash
    git clone -b refactor https://github.com/enferlain/sd-optim.git sd-optim
    ```
    *(Replace URL)*
    Then install dependencies with `uv`:
    ```bash
    cd sd-optim
    uv lock
    uv sync
    ```
    Install runtime extras based on your setup. Example (CUDA 13.0 + torch 2.10 + Optuna + rembg GPU backend + scorer extras):
    ```bash
    uv sync --extra torch-cu130 --extra torch-v210 --extra merge --extra optuna --extra onnx-gpu --extra scorer-hybridnoise --extra scorer-backgroundblackness --extra scorer-textureclean
    ```
3.  **Configuration:** Copy `.tmpl.yaml` files in `conf/` to `.yaml` and edit them (especially `config.yaml` and `optimization_guide/guide.yaml`) to match your paths and desired settings.
4.  **Run:** Launch your WebUI with the API enabled. Then, from the `sd-optim` directory, run:
    ```bash
    uv run python sd_optim.py
    ```

### Lockfile and Linux/Windows compatibility

- `uv.lock` is cross-platform and includes environment markers/wheels for multiple OSes.
- Linux users can use the same lockfile and run `uv sync --frozen` with their chosen extras.
- If you run lock checks with a local cache dir:
  - PowerShell: ``$env:UV_CACHE_DIR=".uv-cache"; uv lock --check``
  - Bash: ``UV_CACHE_DIR=.uv-cache uv lock --check``

**For detailed setup, configuration options, and usage guides, please see the [[Project Wiki]](https://github.com/enferlain/sd-optim/wiki).**

## Planned Features

*   Integration with more WebUIs (ComfyUI, Reforge).
*   More advanced visualization options.
*   Hotkey support for interaction (scoring mode switching, early stopping).
*   Dynamic adjustment of runtime parameters (batch size, payloads).
*   Expanded scoring options (perceptual similarity, character consistency).
*   Potential integration of other optimization libraries (e.g., Hyperactive).
*   A nice webui/app is the ultimate goal.

## Acknowledgements

*   Based on the original concept by [s1dlx](https://github.com/s1dlx).
*   Relies heavily on the fantastic [`sd-mecha`](https://github.com/ljleb/sd-mecha) library by [ljleb](https://github.com/ljleb).
*   Inspired by and utilizes concepts/code from various community projects (SuperMerger, sd-meh, etc.).
*   Scoring models from multiple creators (LAION, OpenAI, Salesforce, THUDM, yuvalkirstain, etc.).

---

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Enferlain/sd-optim)
