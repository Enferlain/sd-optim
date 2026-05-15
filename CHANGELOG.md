# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-05-14

### Added

- Added a practical migration note in `conductor/tracks/post_reorg_cleanup_20260503/legacy_to_graph_native_guide_migration.md` explaining how legacy strategy-style guides map into the graph-native authored model, including worked examples and current transitional limitations
- Added `sd_optim/guide_runtime.py` as the first graph-native runtime bundle surface, with focused tests covering graph-authored optimizer bounds, runtime summary metadata, and direct block/key payload materialization from compiled bindings
- Added merge-owned helper modules for recipe rewriting, recipe graph inspection, and merge runtime orchestration in `sd_optim/merge/recipe_rewrite.py`, `sd_optim/merge/recipe_inspection.py`, and `sd_optim/merge/runtime.py`

### Changed

- Tightened graph-native guide semantics so one `group` node now always means one incoming set sharing one value; multiple grouped values must be expressed as multiple explicit branches instead of one node with internal subgroup fanout
- Aligned the saved graph-shape draft and `sd_optim/guide_nodes.py` around the same build-centered branch model for `all`, `group`, and `exclude`
- Wired the recipe stack and optimizer startup to recognize graph runtime bundles directly: `recipe_builder`, `merger`, and `recipe_optimization` now accept compiled graph runtime data without `BoundsInfo`, and optimizer startup now treats `optimization_guide.graph` as an explicit graph-authored runtime path
- Made graph mode explicitly reject legacy-only guide features instead of half-applying them: `custom_bounds` is rejected during graph startup/setup, while Optuna dependency mapping now resolves `optimization_guide.dependencies` from graph runtime bindings without calling `ParameterHandler`
- Added regression coverage for graph-generated optimizer parameters flowing through the merge trial pipeline into `Merger.merge()` as a `GraphRuntimeBundle`
- Split the old mixed `sd_optim/utils/artifacts.py` surface into merge-owned modules, moving runnable merge-artifact export to `sd_optim/merge/reproducible_artifacts.py` and keeping recipe rewrite helpers out of the generic utils package
- Reduced `sd_optim/merger.py` to a thinner public entrypoint by moving merge-iteration orchestration into `sd_optim/merge/runtime.py`
- Replaced remaining stable merge-side config `.get(...)` access with direct structured reads in touched modules such as `sd_optim/merge/artifacts.py`, `sd_optim/merge/reproducible_artifacts.py`, and `sd_optim/merge/recipe_builder.py`
- Refreshed the active post-reorg conductor notes so the documented cleanup queue matches the current package layout and completed refactor slices

### Fixed

- Fixed graph-guide semantic drift by rejecting `data.groups` on graph `group` nodes and updating runtime tests to cover multiple grouped values through separate branches
- Fixed graph merge preparation to reject graph-authored method parameters that are not valid keyword parameters for the selected merge method before sd-mecha recipe construction
- Fixed merge-side error boundaries by narrowing a first cleanup slice of broad `except Exception` handling in `sd_optim/merge/model_selection.py` and `sd_optim/merge/layer_adjust.py` to explicit model-config inference, checkpoint load, state-dict mutation, and artifact save failures

## [Unreleased] - 2026-05-12

### Added

- Added the first graph-native guide compiler path in `sd_optim/guide_nodes.py`, compiling build-centered authored branches such as `source -> type -> param -> build`, `source -> selection -> type -> domain -> param -> build`, and `source -> selection -> type(exclude) -> build` into optimizer-visible bindings
- Added graph-native guide tests covering selected block rules, disabled parked branches, excludes, shared groups, named groups, domains, and payload materialization

## [Unreleased] - 2026-05-09

### Added

- Added a graph-backed guide compiler core in `sd_optim/guide_compiler.py` with focused regression coverage for:
  - compiled optimizer-visible bindings
  - payload materialization from sampled values
  - reduced golden payload expectations derived from recorded run artifacts
- Added a legacy-guide adapter in `sd_optim/guide_legacy.py` so the current optimization guide format can compile through the new graph-backed path while preserving current behavior
- Added parity analysis scripts and reports for:
  - bounds metadata generation
  - recipe payload materialization
- Added a runtime config toggle `reuse_cached_results` to disable cross-run image-result reuse while preserving run-manifest and artifact writing
- Added regression coverage to preserve `-it_<n>` suffixes in long truncated artifact filenames

### Changed

- Extracted cache reuse classification and cached full/partial-hit handling out of `sd_optim/core/optimizer_runtime.py` into `sd_optim/core/optimizer_runtime_cache.py` so the runtime file is more focused on trial orchestration
- Switched `ParameterHandler.create_parameter_bounds_metadata()` to source legacy guide metadata from the graph-backed compiler path while keeping the existing outer validation and summary behavior
- Switched recipe payload assembly to use graph-backed legacy-guide payload materialization instead of rebuilding payload dicts ad hoc from `param_info`
- Tightened the recipe payload path so `prepare_param_recipe_args()` now honors the supplied `param_info` contract directly instead of silently recompiling guide payloads from config
- Added design notes and a worked example to anchor future guide redesign around authored intent, compiled runtime expansion, and a future node-based UI
- Updated `.gitignore` policy so workflow payload JSONs, workspace files, and local UV setup notes stay untracked while `cargo_comfy.yaml` remains trackable
- Updated guide-design notes to treat parked `name: null` guide fragments as intentionally inactive placeholders rather than active runtime semantics

### Fixed

- Fixed cross-run cache control so reruns can force fresh merge/generate/score behavior without relying on cache state
- Fixed long artifact stem truncation so saved `.mecha` recipes and reproducible merge scripts preserve the trailing iteration marker instead of silently dropping `-it_<n>`
- Fixed a graph-adapter regression where legacy `select` and `group` rules that match nothing would abort guide compilation instead of warning and being skipped
- Fixed recipe payload assembly drift by materializing from already-compiled legacy bounds metadata when callers pass narrowed or precomputed `param_info`

## [Unreleased] - 2026-03-18

### Added

- Added focused regression coverage for:
  - recipe-mode multi-target node rewriting
  - guide/parameter-space startup summary logging
  - duplicate custom config registration logging
  - Optuna optimizer bounds handoff logging

### Changed

- Updated recipe optimization so `recipe_optimization.target_nodes` works consistently as either a single recipe ref or a list of refs across validation, runtime rewriting, and artifact helpers
- Refreshed config and optimization guide templates to better match current runtime behavior and use simpler user-facing wording
- Reworked startup logging around guide/bounds setup so runs now show a compact parameter-space summary at `INFO`, keep the full generated parameter list at `DEBUG`, and reduce duplicate parameter-count logging
- Softened duplicate bundled custom ModelConfig registration into a handled warning instead of a traceback-heavy startup error when an identifier is already registered

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
- Reorganized bundled repository assets so shipped scorers now live under `sd_optim/extensions/bundled/scorers/`, shipped model configs under `sd_optim/extensions/bundled/model_configs/`, and non-runtime debug artifacts live under `tools/` and `assets/`
- Made bundled merge-method resolution lazy so configuring one bundled method only imports the module(s) that advertise that method name, instead of importing the whole moved merge-method surface at startup
- Split merge runtime helper concerns into `sd_optim/merge/`, moving fallback handling, artifact writing, and recipe-building helpers out of `sd_optim/merger.py` while keeping `Merger` as the stable orchestration entrypoint
- Split merge model-selection and execution helpers further into `sd_optim/merge/model_selection.py` and `sd_optim/merge/execution.py` so `sd_optim/merger.py` can keep shrinking toward orchestration-only logic
- Split merge model-node creation into `sd_optim/merge/model_nodes.py` and added focused tests to lock in relative-path handling for recipe model nodes
- Turned `sd_optim.utils` into a package, extracted recipe/model-dir helpers into `sd_optim/utils/recipes.py`, and kept the legacy `sd_optim.utils` helper surface re-exported for compatibility
- Extracted custom config/conversion registration helpers into `sd_optim/utils/conversions.py` and kept the legacy `sd_optim.utils` loader entrypoints re-exported for compatibility
- Extracted recipe rewrite and reproducible merge-artifact helpers into `sd_optim/utils/artifacts.py` and kept the legacy `sd_optim.utils` helper surface re-exported for compatibility
- Extracted config validation, layer-adjust/image-summary helpers, and hotkey support into `sd_optim/utils/config.py`, `sd_optim/utils/images.py`, and `sd_optim/utils/hotkeys.py` while keeping the legacy `sd_optim.utils` helper surface re-exported for compatibility
- Extracted merge-method index/discovery helpers into `sd_optim/utils/methods.py` while keeping the legacy `sd_optim.utils` resolver surface and monkeypatch hooks compatible
- Reworked `sd_optim.utils` into a lazy compatibility shim so importing the package root no longer eagerly imports every extracted helper module
- Switched internal runtime and tests over to direct `sd_optim.utils.*` imports and shrank `sd_optim/utils/__init__.py` to a minimal package marker instead of keeping a broad package-root helper surface
- Extracted scorer runtime setup/manual-preview/score-loop helpers into `sd_optim/scoring/runtime.py` and updated `sd_optim/scorer.py` to import concrete scorer support modules directly
- Extracted recipe-mode orchestration and layer-adjust execution helpers out of `sd_optim/merger.py` into `sd_optim/merge/recipe_optimization.py` and `sd_optim/merge/layer_adjust.py`
- Started the `sd_optim/core/` package by extracting optimizer cache and reuse fingerprint helpers into `sd_optim/core/optimizer_cache.py` while keeping the legacy `sd_optim.optimizer` helper surface compatible
- Extracted universal-reuse manifest/cache I/O and image-path helpers into `sd_optim/core/optimizer_cache_io.py`, and updated optimizer/Optuna/test call sites to use the new core modules directly instead of repo-local compatibility aliases
- Moved the shared optimizer orchestration into `sd_optim/core/optimizer_base.py`, made `Optimizer` a real `ABC`, removed the old `sd_optim/optimizer.py` module, and updated in-repo Optuna/Bayes/artist/test imports to the new core path directly
- Extracted optimizer best-model promotion and best-log helpers into `sd_optim/core/optimizer_artifacts.py`, and updated the base optimizer to use those artifact helpers directly
- Extracted the remaining shared optimizer trial/runtime flow into `sd_optim/core/optimizer_runtime.py`, moving generation/scoring orchestration out of `sd_optim/core/optimizer_base.py`
- Split the Optuna runtime into focused `sd_optim/optimizers/optuna/` modules for sampler setup, study/storage lifecycle, objective logic, reporting, dashboard launch, and JSONL trial logging, and reduced `sd_optim/optuna_optimizer.py` to a thin entrypoint
- Backfilled focused Optuna helper coverage with dedicated tests for objective/reporting/study-manager/support modules, bringing the extracted Optuna helper package over the repo-reorg coverage target
- Switched internal runtime and regression tests over to direct optimizer and bundled merge-helper imports while keeping the package-root and legacy merge-method shims available for external callers
- Moved the concrete `OptunaOptimizer` class into `sd_optim/optimizers/optuna/optimizer.py`, updated in-repo call sites to that packaged path, and removed the old `sd_optim/optuna_optimizer.py` module
- Reduced `sd_optim/merger.py` to public orchestration code and updated merge helpers/tests to use direct `sd_optim/merge/*` helper calls instead of preserving private one-line forwarding methods
- Moved the concrete `BayesOptimizer` class into `sd_optim/optimizers/bayes/optimizer.py`, split Bayes-specific resume/sampling/reporting helpers into a packaged `sd_optim/optimizers/bayes/` namespace, fixed the stale `reset_log_file` config lookup, and removed the old `sd_optim/bayes_optimizer.py` module
- Made Optuna sampler setup bounds-aware so CMA-ES now warns more clearly on categorical-heavy guides, explains `warn_independent_sampling` in practice, and no longer relies on docs that implied TPE could not optimize continuous ranges
- Removed the leftover `sd_optim.merge_methods` and `sd_optim.svd_ties_sum_extended` shim modules now that bundled merge helpers are used directly from their packaged runtime paths
- Moved shared SVD utilities into `sd_optim/extensions/bundled/merge_methods/svd.py`, moved trial scorer summary support into `sd_optim/core/trial_scorer_summary.py`, and removed the old package-root helper modules
- Split bundled scorer registry data and lazy class resolution into `sd_optim/extensions/bundled/scorers/registry.py` so `sd_optim.scorer` no longer eagerly imports every scorer module at import time
- Moved bundled scorer implementation modules into `sd_optim/extensions/bundled/scorers/models/` so the scorer package root only contains package/registry code
- Extracted scorer asset/download helpers into `sd_optim/extensions/bundled/scorers/assets.py` and scorer factory/loading helpers into `sd_optim/extensions/bundled/scorers/loading.py` so `sd_optim.scorer` can focus on runtime orchestration
- Extracted manual scorer prompt/image-opening helpers into `sd_optim/extensions/bundled/scorers/interaction.py` and updated `sd_optim.scorer` to delegate to that focused support module
- Moved scorer runtime/support code into `sd_optim/scoring/` so `sd_optim/extensions/bundled/scorers/` now only contains bundled implementation modules
- Reduced `sd_optim/scorer.py` to orchestration-only flow, renamed the general scoring manager from `AestheticScorer` to `Scorer`, and updated optimizer runtime/tests to call shared scoring helpers directly instead of routing through scorer wrapper methods
- Aligned the top-level `sd_optim.py` entry script with the packaged layout by switching to direct conversion-loader imports and replacing type-name string branching with explicit optimizer-kind flow control
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
- Added an `sd_optim.svd` compatibility shim so runtime merge helpers still import after the helper move into `sd_optim/extensions/bundled/merge_methods/svd.py`
- Made `sd_optim.extensions.bundled.merge_methods` the explicit package surface for bundled SVD helpers and updated runtime/config defaults to point at the packaged layout

### Fixed

- Fixed the legacy `sd_optim.merge_methods` compatibility module so `MergeMethods` now forwards to the packaged bundled SVD ties-sum helpers instead of failing at import time
- Fixed bundled merge-method startup coupling so unrelated broken moved modules no longer block resolution of the specific bundled method selected in config
- Fixed the bundled merge-method package surface to re-export SVD helpers from a standalone `sd_optim.svd` helper module, avoiding eager import cycles through the moved bundled `svd.py`
- Fixed post-flattening merge-method helper functions by removing stray top-level `@staticmethod` decorators that would otherwise turn helpers into descriptor objects instead of normal callables
- Fixed a startup regression where `sd_optim.optimizer` imported `sd_optim.trial_scorer_summary` but the module was missing from the package, causing `sd_optim.py` to fail before optimizer initialization
- Fixed scorer-summary aggregation call sites after the helper moved to a keyword-only signature, which had been causing runs to fail immediately after the first scored trial
- Fixed a universal reuse regression where the image hash ignored merge/generation setup, allowing different merge methods or recipe setups to be treated as cache hits when params and payloads happened to match
- Fixed merge-method invocation for positional-parameter methods like `weighted_sum` by ensuring tensor-valued parameters such as `alpha` are not counted as extra model inputs
- Fixed fail-on-error handling so real trial crashes stop the optimization by default again, while explicit `fail_on_error: false` still allows continue-on-error behavior
- Fixed Optuna postprocess recap to avoid raising a second error when all completed trials have failed and no best trial exists yet
- Fixed Optuna sampler/config drift by correcting the QMC `warn_asynchronous_seeding` key and validating pruning from `optimizer.optuna_config.use_pruning`
- Fixed startup and model-inspection crashes against `sd-mecha` 1.1.x caused by removed top-level APIs such as `open_input_dicts` and `infer_model_configs`
- Fixed stale package-local SVD helper imports after the helper implementation moved to `sd_optim/extensions/bundled/merge_methods/svd.py`
- Fixed full test collection after the packaged layout move by bootstrapping the repo root in `tests/conftest.py` and restoring the `sd_optim.svd_ties_sum_extended` module surface
- Fixed builtin-layout import fallout after package moves by updating scorer registry module paths, converter discovery imports, and default builtin config locations
- Fixed scorer registry eager-import overhead by resolving bundled scorer classes lazily through the extracted registry module
- Fixed scorer package sprawl by separating registry/package files from concrete scorer implementation modules
- Fixed scorer runtime sprawl by moving path/download and model-loading concerns into dedicated bundled helper modules
- Fixed scorer interaction platform handling by using `os.startfile(...)` on Windows and preferring `wslview` / `xdg-open-wsl` before falling back to `xdg-open` under WSL
- Fixed manual scoring behavior so it no longer depends on transient `PIL.Image.show()` temp-file behavior and now reports manual scorer results consistently with automatic scorers
- Fixed fallback visibility during merging by making real fallback hits visible at `INFO` level and per-key fallback usage visible at `DEBUG`
- Fixed stale regression tests after the packaged layout move by updating moved scorer paths, refreshing Optuna test stubs for `fail_on_error_enabled`, making the optional texture scorer integration opt-in, and removing wavelet shim tests for deleted helpers
- Fixed merge-mode and recipe-mode cache wiring against `sd-mecha` 1.1.x by passing explicit node-to-cache mappings into `sd_mecha.merge(...)`
- Fixed delta-output wrapping and recipe traversal against the 1.1.x node API by using `bound_args` and direct merge-method recipe construction
- Fixed recipe artifact generation and runnable-script export against the 1.1.x literal-node API by traversing `LiteralRecipeNode.value_dict`
- Fixed `.mecha` recipe saving for relative model paths by avoiding unnecessary graph finalization during serialization
- Fixed saved recipe artifacts showing unresolved `null` metadata by finalizing the same model-dir-aware execution graph that `sd_mecha.merge(...)` uses
- Fixed saved recipe artifacts leaking finalized absolute model paths and runtime `cast` wrapper nodes into the human-facing `.mecha` output
- Fixed repeated adapter validation work by caching inferred model-config candidates for LoRA/LyCORIS checks across the merger flow
- Fixed fixed-arity merge-method preprocessing so unused extra configured models no longer trigger adapter detection/conversion work before being sliced away

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
