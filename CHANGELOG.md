# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-03-15

### Added

- Added focused `sd-mecha` 1.1.x regression tests for:
  - class-based converter compatibility
  - fallback logging and key planning behavior
  - graph/model-dir compatibility helpers
  - moved SVD helper import compatibility
  - merge/convert wrapper compatibility against the 1.1.x runtime signatures

### Changed

- Updated custom block converters to work with sd-mecha 1.1.3
- Reorganized bundled repository assets so builtin scorers now live under `sd_optim/builtin/scorers/`, builtin model configs under `sd_optim/builtin/model_configs/`, and non-runtime debug artifacts live under `tools/` and `assets/`
- Split builtin scorer registry data and lazy class resolution into `sd_optim/builtin/scorers/registry.py` so `sd_optim.scorer` no longer eagerly imports every scorer module at import time
- Moved builtin scorer implementation modules into `sd_optim/builtin/scorers/models/` so the scorer package root only contains package/registry code
- Extracted scorer asset/download helpers into `sd_optim/builtin/scorers/assets.py` and scorer factory/loading helpers into `sd_optim/builtin/scorers/loading.py` so `sd_optim.scorer` can focus on runtime orchestration
- Extracted manual scorer prompt/image-opening helpers into `sd_optim/builtin/scorers/interaction.py` and updated `sd_optim.scorer` to delegate to that focused support module
- Updated the manual scoring runtime to save stable preview images into the scorer `imgs/` directory, open them through the normalized platform opener path, and record manual scorer output in `last_scorer_results`
- Restored the merger's recipe-level fallback wrapper for per-key DEBUG fallback logging under sd-mecha 1.1.x
- Upgraded the merger fallback wrapper to a class-based method that mirrors `sd_mecha.fallback` key planning and emits a visible INFO log on the first actual fallback hit
- Replaced removed `sd_mecha.open_input_dicts` / `sd_mecha.infer_model_configs` usage with `open_graph`-based compatibility helpers for optimizer and merger model inspection
- Replaced active runtime `sd_mecha.merge(..., model_dirs=..., check_mandatory_keys=...)` and `sd_mecha.convert(..., model_dirs=...)` call patterns with helper wrappers that use the new `model_dirs` registry and `strict_mandatory_keys`
- Replaced removed recipe-node `.set_cache(...)` usage with merge-time cache maps built from the recipe graph
- Updated active `MergeRecipeNode` handling to use `bound_args` instead of removed `.args` / `.kwargs`, and avoided eager `sd_mecha.add_difference(...)` wrapper logic that expects finalized merge-space metadata
- Added a shared non-finalizing recipe serialization helper for artifact generation and recipe saving under `sd-mecha` 1.1.x
- Updated saved recipe artifacts to serialize the finalized fallback-wrapped execution graph instead of the raw pre-finalized recipe
- Refined saved recipe artifacts to keep logical merge structure by dropping runtime output-cast wrappers and restoring relative model paths when possible
- Centralized adapter config detection so LoRA/LyCORIS checks reuse cached sd-mecha config inference instead of repeated substring-based scans
- Added an `sd_optim.svd` compatibility shim so runtime merge helpers still import after the helper move into `sd_optim/builtin/merge_methods/svd.py`
- Made `sd_optim.builtin.merge_methods` the explicit package surface for bundled SVD helpers and updated runtime/config defaults to point at the builtin layout

### Fixed

- Fixed a startup regression where `sd_optim.optimizer` imported `sd_optim.trial_scorer_summary` but the module was missing from the package, causing `sd_optim.py` to fail before optimizer initialization
- Fixed scorer-summary aggregation call sites after the helper moved to a keyword-only signature, which had been causing runs to fail immediately after the first scored trial
- Fixed a universal reuse regression where the image hash ignored merge/generation setup, allowing different merge methods or recipe setups to be treated as cache hits when params and payloads happened to match
- Fixed merge-method invocation for positional-parameter methods like `weighted_sum` by ensuring tensor-valued parameters such as `alpha` are not counted as extra model inputs
- Fixed fail-on-error handling so real trial crashes stop the optimization by default again, while explicit `fail_on_error: false` still allows continue-on-error behavior
- Fixed Optuna postprocess recap to avoid raising a second error when all completed trials have failed and no best trial exists yet
- Fixed startup and model-inspection crashes against `sd-mecha` 1.1.x caused by removed top-level APIs such as `open_input_dicts` and `infer_model_configs`
- Fixed stale package-local SVD helper imports after the helper implementation moved to `sd_optim/builtin/merge_methods/svd.py`
- Fixed builtin-layout import fallout after package moves by updating scorer registry module paths, converter discovery imports, and default builtin config locations
- Fixed scorer registry eager-import overhead by resolving builtin scorer classes lazily through the extracted registry module
- Fixed scorer package sprawl by separating registry/package files from concrete scorer implementation modules
- Fixed scorer runtime sprawl by moving path/download and model-loading concerns into dedicated builtin helper modules
- Fixed scorer interaction platform handling by using `os.startfile(...)` on Windows and preferring `wslview` / `xdg-open-wsl` before falling back to `xdg-open` under WSL
- Fixed manual scoring behavior so it no longer depends on transient `PIL.Image.show()` temp-file behavior and now reports manual scorer results consistently with automatic scorers
- Fixed fallback visibility during merging by making real fallback hits visible at `INFO` level and per-key fallback usage visible at `DEBUG`
- Fixed merge-mode and recipe-mode cache wiring against `sd-mecha` 1.1.x by passing explicit node-to-cache mappings into `sd_mecha.merge(...)`
- Fixed delta-output wrapping and recipe traversal against the 1.1.x node API by using `bound_args` and direct merge-method recipe construction
- Fixed recipe artifact generation and runnable-script export against the 1.1.x literal-node API by traversing `LiteralRecipeNode.value_dict`
- Fixed `.mecha` recipe saving for relative model paths by avoiding unnecessary graph finalization during serialization
- Fixed saved recipe artifacts showing unresolved `null` metadata by finalizing the same model-dir-aware execution graph that `sd_mecha.merge(...)` uses
- Fixed saved recipe artifacts leaking finalized absolute model paths and runtime `cast` wrapper nodes into the human-facing `.mecha` output
- Fixed repeated adapter validation work by caching inferred model-config candidates for LoRA/LyCORIS checks across the merger flow

## [Unreleased] - 2026-02-22

### Added

- Added custom Hydra logging profiles:
  - `conf/hydra/job_logging/sd_optim.yaml`
  - `conf/hydra/hydra_logging/sd_optim.yaml`
- Added a focused regression test `tests/test_logging_print_cleanup.py` to enforce:
  - source location logging format (`filename:lineno`)
  - no `print()` calls in core runtime modules

### Changed

- Migrated runtime console output from `print()` to structured logger calls across core execution paths, including:
  - scoring (`sd_optim/scorer.py`)
  - optimization loop (`sd_optim/optimizer.py`)
  - dashboard launcher (`sd_optim/optuna_optimizer.py`)
  - merge/runtime helpers and model runtime modules
- Enabled Hydra logging overrides in `conf/config.yaml` so logs include module/script name and source line numbers.

### Fixed

- Fixed a Hydra startup regression caused by defaults ordering (`override hydra/*` entries must be at the end of the `defaults` list in `conf/config.yaml`).
- Resolved a run-breaking failure mode where stdout `print()` calls raised `OSError: [Errno 22] Invalid argument` mid-run, which previously collapsed scorer outputs to `0.0`.

## [Unreleased] - 2026-02-11

### Added

- Created `analyze_importance.py`: A robust hyperparameter importance analysis tool supporting both Optuna `.db` files and `.jsonl` trial logs.
- Added `IMPORTANCE_GUIDE.md`: A comprehensive guide on using the analysis tool, including "high-score" vs "low-score" analysis strategies.
- Added `--maximize` flag to `analyze_importance.py` to correctly analyze the best performing trials in maximization studies.

### Fixed

- Resolved a discrepancy between script-generated importance and Optuna Dashboard visualizations by identifying and working around a "Direction Inversion" bug in the dashboard's evaluator logic.
- Improved handling of conditional search spaces in importance analysis using a new `subspace` mode.

## [Unreleased] - 2026-02-05

### Added

- New scorer for textures `sd_optim/models/TextureScorer.py` 

### Changed

### Removed

### Fixed

- Fixed a regression in `sd_optim/prompter.py` where non-dictionary shared settings (like `workflow_json`) were being ignored when defined inside a cargo configuration.
- Removed the redundant break and added a guard to ensure only the first yielded image is processed in `optimizer.py`.

## [1.1.1] - 2023-03-05
