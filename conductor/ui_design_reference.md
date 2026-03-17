# sd-optim UI Design Reference

> A concept and data map for anyone building mock-ups or feature sketches.
> Goal: read this document, sketch UI wireframes, without needing to read the Python source code.

---

## 1. What sd-optim Does (30-Second Version)

sd-optim finds the best merge parameters for Stable Diffusion models automatically.

```
Loop (N trials):
  1. Optimizer suggests parameters (e.g., alpha=0.6, norm_weight=0.3)
  2. Merger builds a combined model from those parameters
  3. Generator creates images using that model (via A1111/Forge/ComfyUI)
  4. Scorers evaluate image quality → produce a score
  5. Optimizer learns from the score, suggests better parameters next time
```

Each trial is a complete cycle. A **study** is a collection of trials. A study can be resumed or forked.

---

## 2. The Five Core Concepts

### 2.1 Models

**What the user sees:** A list of model files (`.safetensors`).

| Property                     | Description                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ |
| `model_paths`                | List of input models to merge (2+ typically)                                   |
| `base_model_index`           | Which model is the "base" for delta methods                                    |
| `fallback_model_index`       | Model used for missing keys                                                    |
| `merge_method`               | The algorithm used to combine them (e.g., `weighted_sum`, `pop_lora`, `slerp`) |
| `merge_dtype` / `save_dtype` | Precision settings for computation and output                                  |

Models can be full checkpoints or LoRAs. LoRAs are automatically detected and converted to deltas.

**UI implications:** Model selection is a list builder — pick files, assign roles (base, fallback, input). The merge method is a dropdown with ~20+ registered methods, each with different parameter signatures.

---

### 2.2 The Optimization Guide (Parameter Space)

This is the most complex subsystem. It defines **what gets optimized** and **within what range**.

#### Architecture

The guide has three layers:

```
Component (e.g., "unet", "diffuser", "text_encoder")
  └── Strategy (how to group parameters)
       └── Bound (what range each parameter explores)
```

#### Components

A **component** maps to a named section of the model's architecture. Component names come from **ModelConfig** files — they are NOT hardcoded. Different model architectures (SDXL, SD1.5, Flux) expose different components.

Current SDXL configs define components like: `unet`, `diffuser`, `text_encoder_1`, `text_encoder_2`.

#### Strategies (How Parameters Are Grouped)

Each component has one or more **strategies** that determine the granularity of optimization:

| Strategy | What It Does                                           | Creates Parameters Named...                    |
| -------- | ------------------------------------------------------ | ---------------------------------------------- |
| `all`    | One parameter per item (block or key) in the component | `UNET_IN04_alpha`, `UNET_MID00_alpha`, etc.    |
| `select` | Like `all`, but only for explicitly listed items       | Only the named blocks, e.g., `UNET_IN07_alpha` |
| `group`  | Matches keys by glob pattern, shares one parameter     | `norm_res_alpha` (covers all `*.norm*` keys)   |
| `single` | One parameter for the entire component                 | `diffuser_single_alpha`                        |

**Key insight for UI:** Strategy type determines the interaction model:

- `select` = the user picks items from a list
- `group` = the user writes (or is suggested) a glob pattern
- `all` / `single` = automatic, no item-level selection needed

#### Target Types

Each strategy operates on either:

- **`block`** — coarse-grained blocks (e.g., `UNET_IN04`, `UNET_OUT08`). Defined in custom block config YAMLs.
- **`key`** — fine-grained individual weight tensors (e.g., `model.diffusion_model.input_blocks.4.1.norm.weight`). From the base model config.

If unspecified, the system auto-detects: it prefers custom block configs if available, falls back to base key configs.

#### Optimize Params

Each strategy declares which **parameter types** to tune. Common types:

- `alpha` — primary merge weight
- `rank_ratio` — LoRA rank ratio
- Custom per-method params (e.g., `outlier_tolerance`, `magnitude_ratio`)

These are defined at the component or strategy level and inherited downward.

#### Bounds

Bounds define the search space for each parameter:

| Shape       | Format            | Example           | Meaning                 |
| ----------- | ----------------- | ----------------- | ----------------------- |
| Continuous  | `(min, max)`      | `(0.0, 1.0)`      | Any float in range      |
| Categorical | `[val, val, ...]` | `[0.0, 0.5, 1.0]` | Only these exact values |
| Locked      | `value`           | `1.0`             | Fixed, not optimized    |

#### Custom Bounds Override (2-Priority Cascade)

After the strategy system generates default bounds, **custom bounds** can override them:

1. **Priority 1 — Exact name match:** If custom bounds key matches the full parameter name (e.g., `UNET_OUT06_alpha`), it overrides that specific parameter.
2. **Priority 2 — Base param match:** If custom bounds key matches a base parameter name (e.g., `alpha`), it overrides ALL parameters sharing that base name — unless they already have a Priority 1 override.

**UI implications:** This cascade must be visible. The user needs to see:

- Which parameters exist (generated from strategies)
- What bounds each has (default from strategy)
- Whether any override applies, and from which level

#### Dependencies

Parameters can depend on each other:

```yaml
dependencies:
  - parent: alpha
    child: rank_ratio
    condition: "!= 0"
    default: 1.0
```

This means: "only optimize `rank_ratio` for a block if its `alpha` is not zero." The dependency maps per-item — if `UNET_IN07_alpha` is 0, then `UNET_IN07_rank_ratio` uses its default.

---

### 2.3 Payloads (Image Generation Recipes)

**Payloads** define how test images are generated. They are NOT prompts — they're complete generation configurations.

#### Structure

```
Cargo file (e.g., cargo_comfy.yaml)
├── Defaults (shared settings)
│   ├── width: 1024
│   ├── height: 1024
│   ├── steps: 28
│   ├── cfg_scale: 7
│   └── seed: 218
└── Cases (individual payloads, each is a dict)
    ├── portrait_1:
    │   ├── prompt: "1girl, standing, ..."
    │   └── negative_prompt: "..."
    ├── landscape_1:
    │   └── prompt: "scenery, ..."
    └── action_1:
        └── prompt: "dynamic pose, ..."
```

- Defaults apply to all cases unless overridden.
- Each case is one image configuration.
- Prompts support **wildcards** (`__character__` → random line from `wildcards/character.txt`).
- Different cargo files exist per backend: `cargo_a1111.yaml`, `cargo_comfy.yaml`, `cargo_forge.yaml`.

**UI implications:** Payload editing is essentially a form builder. The defaults panel sets base values; individual cases override them. Wildcard resolution should be previewable. The cargo file selector is a dropdown tied to the chosen WebUI backend.

---

### 2.4 Scorers (Image Evaluation)

Scorers are AI models that evaluate generated images and produce a numerical quality score.

#### Available Scorers

| ID                    | Type             | Notes                            |
| --------------------- | ---------------- | -------------------------------- |
| `cityaes`             | Aesthetic        | Anime-tuned aesthetic predictor  |
| `laion`, `chad`       | Aesthetic        | General aesthetic predictors     |
| `hpsv3`               | Human preference | Human preference score model     |
| `pick`                | Human preference | Pick-a-Pic preference model      |
| `imagereward`         | Reward           | Image reward model               |
| `clip`, `blip`        | Alignment        | Text-image alignment             |
| `lumidinov3`          | Anatomy          | Anatomy flaw detector            |
| `simplequality`       | Quality          | Simple quality predictor         |
| `pcascorer`           | Texture analysis | PCA-based texture noise detector |
| `hybridnoise`         | Noise detection  | Hybrid noise artifact detector   |
| `textureclean`        | Texture analysis | Texture cleanliness scorer       |
| `backgroundblackness` | Composition      | Background purity checker        |
| `manual`              | Human            | User assigns score manually      |

#### Configuration Per Scorer

| Setting                 | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `scorer_method`         | List of active scorers                                 |
| `scorer_weight`         | Weight per scorer in the final average                 |
| `scorer_average_type`   | How to combine: `arithmetic`, `geometric`, `quadratic` |
| `scorer_filters`        | Per-scorer exclusion of specific payloads              |
| `scorer_lazy_load_list` | Large scorers loaded on-demand to save VRAM            |
| `scorer_default_device` | CPU or CUDA                                            |
| `scorer_device`         | Per-scorer device override                             |

#### Filtering

Scorers can **exclude** specific payload cases. For example, `backgroundblackness` may not apply to landscape prompts. Filters use payload case names.

**UI implications:**

- Scorer selection is a toggleable list with weight sliders
- Per-scorer settings (device, filters, lazy loading) expand on click
- The effective influence (weight × historical variance) could be visualized
- Preview: show how changing weights would change historical scores

---

### 2.5 The Optimizer (Search Engine)

Two optimizer backends are supported:

| Backend      | Key Settings                                                                               | Best For                          |
| ------------ | ------------------------------------------------------------------------------------------ | --------------------------------- |
| **Optuna**   | Sampler type (TPE, CMA-ES, GP, QMC, Grid, BoTorch), pruning, early stopping, parallel jobs | Most use cases, advanced features |
| **BayesOpt** | Acquisition function (UCB, EI, POI), domain reduction, initial sampling                    | Simpler setups                    |

#### Optuna-Specific Concepts

- **Study:** A named collection of trials, persisted to SQLite. Can be resumed or forked.
- **Sampler:** The algorithm that picks parameters. CMA-ES (`cmaes`) is the default.
- **Pruning:** Early-stop bad trials before they finish generating all images.
- **Dashboard:** A built-in web dashboard (optuna-dashboard) that can be auto-launched.

#### Common Settings

| Setting             | Description                                  |
| ------------------- | -------------------------------------------- |
| `init_points`       | Exploration trials (random/quasi-random)     |
| `n_iters`           | Exploitation trials (guided by model)        |
| `random_state`      | Seed for reproducibility                     |
| `resume_from_study` | Name of a previous study to continue         |
| `fork_study`        | Copy parent trials into a new study (branch) |

---

## 3. The Model Config System

Model configs are YAML files that define the internal structure of a model architecture. They are loaded via `sd-mecha` and determine what components, blocks, and keys are available.

#### Currently Available Configs (SDXL)

| File                              | Purpose                                    |
| --------------------------------- | ------------------------------------------ |
| `sdxl-optim_blocks.yaml`          | Primary block definitions for optimization |
| `sdxl-attn_blocks.yaml`           | Attention-focused block grouping           |
| `sdxl-resolution_blocks.yaml`     | Resolution-based block grouping            |
| `sdxl-stage_blocks.yaml`          | Stage-based grouping                       |
| `sdxl-type_simple_blocks.yaml`    | Simple type-based grouping                 |
| `sdxl-type_structure_blocks.yaml` | Structural type-based grouping             |
| `sdxl-lumi_blocks.yaml`           | Lumi-specific block config                 |
| `sdxl-sgm.yaml`                   | SGM base config                            |

**UI implications:** The config file determines what items appear in the parameter space. Changing the custom block config changes the available blocks. The UI should dynamically populate available components and items from the loaded config.

---

## 4. Optimization Modes

| Mode           | What Gets Optimized                                      | Input                                  |
| -------------- | -------------------------------------------------------- | -------------------------------------- |
| `merge`        | Merge parameters (alpha, rank_ratio, etc.) per block/key | Two or more models                     |
| `recipe`       | Parameters within an existing `.mecha` recipe file       | A recipe file + target node references |
| `layer_adjust` | Direct weight adjustments per layer                      | A single model                         |

Each mode uses the same scorer/generator infrastructure but differs in how parameters are applied.

---

## 5. Data Flow and Outputs

### Per Trial

Each trial produces:

- **Images** (1+ per payload case), saved with metadata:
  - Generation parameters (prompt, seed, dimensions, etc.)
  - Merge parameters (the trial's suggested values)
  - A deterministic SHA256 hash (for deduplication/reuse)
  - Individual scorer results
- **Scores** per scorer per image
- **Aggregated score** (weighted average across scorers and images)

### Per Study

- **SQLite database** (Optuna) or **JSON log** (BayesOpt) containing all trial data
- **Visualizations** generated on completion:
  - Optimization history plot
  - Parameter importance plot
  - Slice plots
  - Parallel coordinate plots
- **Best model** saved separately if `save_best: True`
- **Run manifest** (`run_manifest.json`) with fingerprints for image reuse

### File Locations

```
logs/
  2026-02-12_03-00-00_pop_lora_cityaes/    ← Hydra output dir (per run)
    ├── sd_optim.log                         ← Full log
    ├── run_manifest.json                    ← Image hash manifest
    ├── .hydra/                              ← Resolved config snapshot
    │   ├── config.yaml
    │   └── overrides.yaml
    ├── images/                              ← Generated images with metadata
    │   ├── 0.8521_portrait_1_it0_0.png
    │   └── ...
    ├── visualizations/                      ← Optuna plots
    │   ├── optuna_optimization_history_*.png
    │   └── optuna_param_importances_*.png
    └── trial_logs.jsonl                     ← Per-trial results
```

---

## 6. WebUI Backend Adapters

The system talks to a running Stable Diffusion instance via HTTP API:

| Backend | URL Format              | Adapter Class    | Notes              |
| ------- | ----------------------- | ---------------- | ------------------ |
| A1111   | `http://localhost:7860` | `A1111Adapter`   | txt2img API        |
| Forge   | `http://localhost:7860` | `A1111Adapter`   | Same API as A1111  |
| reForge | `http://localhost:7860` | `A1111Adapter`   | Same API as A1111  |
| ComfyUI | `http://localhost:8188` | `ComfyUIAdapter` | Workflow-based API |
| SwarmUI | `http://localhost:7801` | Not implemented  | Planned            |

Each adapter handles:

- Model loading/unloading on the backend
- Translating flat payload dicts into backend-specific API formats
- Streaming generated images back

**UI implications:** The backend selector determines which cargo format to use (different payload schemas per backend).

---

## 7. User Interaction Points (What the UI Needs to Surface)

### Setup Phase (Before Running)

| What                      | Current Interface                 | UI Opportunity                      |
| ------------------------- | --------------------------------- | ----------------------------------- |
| Select models             | Edit `model_paths` list in YAML   | File picker with drag-and-drop      |
| Choose merge method       | Edit `merge_method` in YAML       | Dropdown with method documentation  |
| Configure parameter space | Edit `guide.yaml` (complex)       | **Canvas-based strategy builder**   |
| Set up payloads           | Edit cargo YAML files             | Form builder with wildcard preview  |
| Configure scorers         | Edit scorer lists/weights in YAML | Toggle list with weight sliders     |
| Choose optimizer          | Edit optimizer section in YAML    | Settings panel with preset profiles |

### During Run

| What             | Current Interface                            | UI Opportunity                       |
| ---------------- | -------------------------------------------- | ------------------------------------ |
| Monitor progress | Terminal logs + optuna-dashboard             | Live score chart, image gallery, ETA |
| View images      | File browser                                 | Side-by-side comparison viewer       |
| Interrupt/resume | Ctrl+C, then re-run with `resume_from_study` | Pause/resume button                  |

### After Run

| What                 | Current Interface         | UI Opportunity                         |
| -------------------- | ------------------------- | -------------------------------------- |
| Review results       | Read logs, view plot PNGs | Interactive dashboard with filtering   |
| Compare trials       | Manual inspection         | Trial diff viewer                      |
| Parameter importance | Static PNG from Optuna    | Interactive importance chart           |
| Export best model    | File is already saved     | One-click deploy to WebUI              |
| Fork study           | Edit config and re-run    | "Branch from here" button on any trial |

---

## 8. Key Design Constraints

1. **Model-agnostic:** The UI must not hardcode any architecture. Components, blocks, and keys come from the loaded `ModelConfig`. SDXL has a UNet; future models (Flux, etc.) won't.

2. **Strategy composability:** Multiple strategies can apply to the same component. The conflict detection system prevents double-assignment of items, but the UI should prevent conflicts proactively rather than showing errors after the fact.

3. **Scale:** A typical SDXL optimization can have 10-100+ parameters. The UI must handle this without becoming overwhelming — progressive disclosure is essential.

4. **Deterministic reproducibility:** Seeds, hashes, and manifests ensure exact reproducibility. The UI should make it easy to trace any image back to its exact parameter configuration.

5. **Multiple backends:** The same optimization can target different WebUI backends. Payload formats differ but the parameter space doesn't.

6. **Long-running:** A single study can run for hours or days. The UI must handle connection drops, resumption, and incremental updates gracefully.
