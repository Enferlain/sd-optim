# Repo Reorganization & Optimizer Refactor Plan (2026-02-10)

## Goals
- Reduce file size and complexity by splitting large modules (notably `sd_optim/optimizer.py` and `sd_optim/merge_methods.py`).
- Establish clear package boundaries (core, optimizers, merge methods, scoring, generation, utils).
- Preserve current entrypoints and integrations via thin wrappers.
- Improve testability and future refactor safety.
- Modernize Optuna integration (`4.5.0` -> `4.7.x`) and eliminate known config/API drift.

## Target Structure (Proposed)
```
sd-optim/
  conf/
  logs/                         # runtime outputs (Hydra already targets this)
  scripts/
    api.py                      # keep for A1111/Forge, make it a thin wrapper
  sd_optim/                     # package (or future renamed package)
    __init__.py
    cli/
      main.py                   # entry for sd_optim.py
      execute_recipes.py        # entry for execute_recipes.py
    core/
      optimizer_base.py         # was sd_optim/optimizer.py
      bounds.py                 # was sd_optim/bounds.py
      merger.py                 # was sd_optim/merger.py
      prompter.py               # was sd_optim/prompter.py
    optimizers/
      optuna.py                 # was sd_optim/optuna_optimizer.py
      bayes.py                  # was sd_optim/bayes_optimizer.py
    generation/
      generator.py              # was sd_optim/generator.py
      adapters/
        a1111.py                # from sd_optim/gen_adapters.py
        comfy.py                # from sd_optim/gen_adapters.py
    merge/
      __init__.py
      methods/
        __init__.py
        linear.py
        geometric.py
        wavelets.py
        decomposition.py
        experimental.py
    scoring/
      manager.py                # was sd_optim/scorer.py
      registry.py               # MODEL_DATA and loader setup
      models/                   # was sd_optim/models/*.py (scorers)
    visualization/
      artist.py                 # was sd_optim/artist.py
    utils/
      config.py
      recipes.py
      conversions.py
      artifacts.py
      images.py
      hotkeys.py
  tests/
  tools/                        # manual scripts, diagnostics, scratch
  assets/
    test_images/
```

## Optuna Modernization & Split Plan (Added 2026-02-12)

### Scope
- Update runtime Optuna version in project environment and requirements.
- Fix known Optuna integration issues found in `sd_optim/optuna_optimizer.py`.
- Split `sd_optim/optuna_optimizer.py` into focused modules with no behavior regressions.

### Known Issues To Address
- [ ] QMC arg typo: `warn_asyncronous_seeding` should be `warn_asynchronous_seeding`.
- [ ] Validation path bug: `validate_optimizer_config()` checks `optimizer.use_pruning` instead of `optimizer.optuna_config.use_pruning`.
- [x] Deprecated CMA knobs still exposed (`restart_strategy`, `inc_popsize`) without explicit policy for Optuna 4.7 behavior. (done 2026-02-14; uncommitted)
- [ ] Current binary/categorical-heavy search spaces are using CMA-ES by default in many runs, which is a poor fit.
- [ ] Investigate + document CMA-ES independent-sampling fallback (dynamic search space / categorical dists) and why it can cause identical suggestions across runs even when trial scores differ. (notes added 2026-02-14)

### Investigation Notes (Added 2026-02-14): CMA-ES fallback + identical suggestions
- Optuna CMA-ES does not support `CategoricalDistribution` and dislikes dynamic/dimension-changing search spaces; it falls back to independent sampling via `RandomSampler` for affected parameters, emitting warnings like: “The parameter `X` in Trial#N is sampled independently… because dynamic search space and `CategoricalDistribution` are not supported…”.
- In `sd-optim`, many params become categorical unintentionally because `custom_bounds` list values (e.g. `[0.0, 1.0]`, `[0.5, 1.0]`) are treated as categorical choices (per repo rule); tuples (e.g. `(0.0, 1.0)`) are continuous ranges.
- When many params are independently sampled (and a fixed sampler seed is used), Optuna can “replay” the same suggested parameter sequence across runs even if trial values/scores differ (i.e., CMA-ES has little/no effect on the actual sampling for those params).
- Observed example (2026-02-14): runs `logs/2026-02-14_03-46-28_pop_lora_[...]` and `logs/2026-02-14_07-07-35_pop_lora_[...]` produced identical suggested-parameter sequences at least through Trial#58 (including post-startup Trial#56/57), despite different trial scores, with `n_startup_trials=55` and `seed=218`.
- The per-parameter independent-sampling warnings are typically emitted by Optuna’s own logger to console/stderr; they may not appear inside `logs/**/sd_optim.log` unless stdout/stderr is captured into a file.
- Actionable direction:
  - categorical/binary-heavy spaces -> use `TPESampler` (or otherwise stop using CMA-ES),
  - mostly-continuous fixed spaces -> CMA-ES is fine, but ensure bounds are tuples and avoid dynamic/conditional params,
  - if fallback is intended but warnings are too noisy -> set `warn_independent_sampling=False` in `CmaEsSampler` options.

### Proposed Module Split (`sd_optim/optimizers/optuna/`)
```
sd_optim/optimizers/optuna/
  __init__.py
  runner.py            # optimize() orchestration, thread handoff, callback wiring
  sampler_factory.py   # _configure_sampler()
  study_manager.py     # create/resume/fork storage + study attrs
  objective.py         # objective function and error handling
  suggest.py           # parameter suggestion + dependency handling
  trial_logger.py      # jsonl trial logger helper + callback payload
  importance.py        # post-run importance analysis helper
  compatibility.py     # version-specific feature gates/deprecation policy
```

### Milestone A: Safety Baseline
- [ ] Capture current baseline with fixed seed and short run (20-50 trials): sampler, best score, median score, wall time.
- [ ] Add/refresh tests around sampler config parsing and suggestion semantics (categorical/list vs tuple/range).
- [ ] Add regression test for resume/fork behavior and scorer mismatch guard.
- [ ] Track post-split follow-up coverage in `tests/REORG_TEST_NOTES.md`.

### Milestone B: Optuna Version Update
- [ ] Update dependency pin to Optuna `4.7.x` (requirements + environment setup notes).
- [ ] Run compatibility smoke test for `tpe`, `cmaes`, and `qmc` sampler initialization paths.
- [ ] Document version bump in changelog with migration notes.

### Milestone C: Correctness Fixes
- [ ] Fix QMC typo (`warn_asynchronous_seeding`) and add unit test asserting kwargs map.
- [ ] Fix pruning config validation path and add unit test for both enabled/disabled cases.
- [x] Introduce compatibility guard for deprecated CMA params: (done 2026-02-14; uncommitted)
- [x] `restart_strategy`, `inc_popsize`: warn clearly when set on Optuna >= 4.4 and explain fallback behavior. (done 2026-02-14; uncommitted)
- [x] Keep backward-compatible parsing, but annotate as deprecated in config docs. (done 2026-02-14; uncommitted)
- [x] Drop legacy Optuna version-gating for CMA-ES restarts (project pins Optuna >= 4.7, and core Optuna no longer supports these knobs). (done 2026-02-14; uncommitted)

### Milestone D: Split Refactor (No Behavior Change)
- [ ] Create `sd_optim/optimizers/optuna/` package and move logic by responsibility.
- [ ] Keep `sd_optim/optuna_optimizer.py` as thin compatibility wrapper during transition.
- [ ] Preserve public class/API surface (`OptunaOptimizer`) and existing config keys.
- [ ] Add import-path tests to ensure external callers still work.

### Milestone E: Sampler Policy for Current Guides
- [ ] Add explicit recommendation in config docs:
- [ ] Binary/categorical-heavy guides -> prefer `tpe`.
- [ ] Mostly continuous guides -> `cmaes` acceptable.
- [ ] Add optional config switch template for quick TPE/CMA profiles.
- [ ] Add log warning when CMA-ES receives many categorical params (heuristic threshold).

### Milestone F: Verification & Rollout
- [ ] Run A/B short benchmark:
- [ ] Same guide + seed, compare `tpe` vs `cmaes` for first 100 trials.
- [ ] Record: best score, top-10% mean, variance, runtime/trial.
- [ ] Run one full production-length trial set with selected sampler profile.
- [ ] If regression > predefined threshold, revert to previous sampler profile and keep split refactor only.

### Acceptance Criteria
- [ ] No regressions in run start/resume/fork, logging, and trial JSONL schema.
- [ ] Optuna `4.7.x` runs stable across `tpe`, `cmaes`, `qmc` initialization paths.
- [ ] `sd_optim/optuna_optimizer.py` reduced to thin wrapper and <200 lines.
- [ ] New optuna submodules have focused tests with >80% coverage for touched code.

## Phase 0: Discovery & Constraints
- [x] Inventory current imports and cross-module dependencies.
- [x] Identify hard external expectations (A1111/Forge API, `sd_optim.py`, `execute_recipes.py`, config paths).
- [x] Confirm which files are experiments or auto-generated and can move to `tools/`.
- [x] Decide whether to introduce a package rename alias or keep `sd_optim` stable.

## Phase 1: Scaffolding (No Behavior Change)
- [ ] Create new package subfolders with `__init__.py`.
- [ ] Preserve required external entrypoints during moves, preferring direct cutovers over temporary wrappers when feasible.
- [ ] Add compatibility imports where needed to avoid breaking external callers.

## Phase 2: Merge Methods Split
- [x] Land bundled merge-method categories under `sd_optim/extensions/bundled/merge_methods/` (`linear`, `manifold`, `transforms`, `experimental`, `svd`).
- [x] Move bundled implementations out of the old monolithic runtime surface while keeping `sd_optim.merge_methods` available as a legacy compatibility import.
- [x] Update method registration/resolution to discover and import only the requested packaged bundled module at runtime.
- [x] Add and refresh tests for method resolution and legacy compatibility surfaces.
- [x] Realign the leftover `sd_optim.merge_methods` compatibility surface with the already-packaged bundled merge-method layout.
  - Replaced the broken dangling `MergeMethods` class body in `sd_optim/merge_methods.py` with a thin compatibility shim backed by `sd_optim.extensions.bundled.merge_methods.experimental.svd_ties_sum_extended`.
  - Wired the bundled helper module back to the shim class so internal fallback counters still target the legacy `MergeMethods` surface.
  - Focused verification: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_svd_ties_sum_extended_v13.py tests/test_merge_method_resolution.py`
  - Result: `9 passed in 26.89s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/merge_methods.py tests/test_svd_ties_sum_extended_v13.py tests/test_merge_method_resolution.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/merge_methods.py sd_optim/svd_ties_sum_extended.py tests/test_svd_ties_sum_extended_v13.py`
  - Result: `Passed`
- [x] Extract merge runtime helper concerns into `sd_optim/merge/` while keeping `sd_optim.merger.Merger` stable.
  - Added `sd_optim/merge/fallback.py`, `sd_optim/merge/artifacts.py`, and `sd_optim/merge/recipe_builder.py`
  - Kept `sd_optim.merger.Merger` as the orchestration entrypoint; it now delegates artifact, fallback, and recipe-building concerns to the merge helper package
  - Re-exported `fallback_debug_logged` from `sd_optim.merger` so the existing public/test surface stays stable during the split
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_merger_fallback_debug.py tests/test_merger_model_arity.py`
  - Result: `9 passed in 29.45s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/merger.py sd_optim/merge/__init__.py sd_optim/merge/fallback.py sd_optim/merge/artifacts.py sd_optim/merge/recipe_builder.py tests/test_merger_fallback_debug.py tests/test_merger_model_arity.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/merger.py sd_optim/merge/__init__.py sd_optim/merge/fallback.py sd_optim/merge/artifacts.py sd_optim/merge/recipe_builder.py`
  - Result: success
- [x] Extract merge model-selection and execution helpers into `sd_optim/merge/`.
  - Added `sd_optim/merge/model_selection.py` for adapter detection, cached config inference, base-model selection, and conversion-context lookup
  - Added `sd_optim/merge/execution.py` for the sd-mecha merge execution path
  - Kept `Merger.merge(...)` and `Merger.recipe_optimization(...)` as orchestration entrypoints for now
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_merger_fallback_debug.py tests/test_merger_model_arity.py`
  - Result: `9 passed in 28.76s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/merger.py sd_optim/merge/__init__.py sd_optim/merge/fallback.py sd_optim/merge/artifacts.py sd_optim/merge/recipe_builder.py sd_optim/merge/model_selection.py sd_optim/merge/execution.py tests/test_merger_fallback_debug.py tests/test_merger_model_arity.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/merger.py sd_optim/merge/model_selection.py sd_optim/merge/execution.py`
  - Result: success
- [x] Extract model-node creation into `sd_optim/merge/` and add characterization coverage for its path handling.
  - Added `sd_optim/merge/model_nodes.py` and moved `_create_model_nodes()` logic there
  - Added `tests/test_merger_model_nodes.py` to lock in relative-path handling for model nodes
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_merger_fallback_debug.py tests/test_merger_model_arity.py tests/test_merger_model_nodes.py`
  - Result: `11 passed in 28.68s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/merger.py sd_optim/merge/model_nodes.py tests/test_merger_fallback_debug.py tests/test_merger_model_arity.py tests/test_merger_model_nodes.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/merger.py sd_optim/merge/model_nodes.py tests/test_merger_model_nodes.py`
  - Result: success
- [x] Make bundled merge-method loading lazy so unrelated broken modules do not crash startup.
  - `sd_optim.utils.resolve_merge_method(...)` now resolves only the requested bundled module at runtime instead of importing the whole moved merge-method surface.
  - Transition support remains in place for both old `MergeMethods` class-based modules and new top-level function modules.
  - Restored `sd_optim.svd` as a standalone helper module and made `sd_optim.extensions.bundled.merge_methods` re-export from it, so package import no longer eagerly imports the bundled `svd` merge-method module.
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_merge_method_resolution.py tests/test_merger_model_arity.py`
  - Result: `4 passed in 21.52s`
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_svd_module_compat.py`
  - Result: `2 passed in 14.61s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/svd.py sd_optim/extensions/bundled/merge_methods/__init__.py tests/test_merge_method_resolution.py tests/test_svd_module_compat.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/svd.py sd_optim/extensions/bundled/merge_methods/__init__.py sd_optim/utils.py`
  - Result: success
- [x] Remove leftover top-level `@staticmethod` decorators from flattened bundled merge-method helpers.
  - Cleaned the post-class-flattening helper functions in `experimental/svd_ties_sum_extended.py`, `experimental/butterfly_projection.py`, `experimental/rams.py`, and `experimental/merge_layers.py`.
  - This avoids turning helper functions into `staticmethod` descriptor objects after they were moved out of `MergeMethods` classes.
  - Focused syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/extensions/bundled/merge_methods/experimental/svd_ties_sum_extended.py sd_optim/extensions/bundled/merge_methods/experimental/butterfly_projection.py sd_optim/extensions/bundled/merge_methods/experimental/rams.py sd_optim/extensions/bundled/merge_methods/experimental/merge_layers.py`
  - Result: success
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_merge_method_resolution.py tests/test_svd_module_compat.py tests/test_merger_model_arity.py`
  - Result: `6 passed in 29.16s`
- [x] Fix merge-method arity handling so tensor-valued parameters like `alpha` are not treated as extra model inputs.
  - Updated `Merger._slice_models()` to count only `StateDict`-like inputs as model arguments
  - Added regression coverage in `tests/test_merger_model_arity.py`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_merger_model_arity.py tests/test_optimizer_cache_fingerprint.py tests/test_trial_scorer_summary.py`
  - Result: `5 passed in 34.81s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/merger.py sd_optim/optimizer.py tests/test_merger_model_arity.py tests/test_optimizer_cache_fingerprint.py`
  - Result: `All checks passed!`
  - Startup smoke check: `PYTHONPATH=. .venv-wsl/bin/python -c "from sd_optim import OptunaOptimizer; print('optuna import ok')"`
  - Result: `optuna import ok`

## Phase 3: Optimizer Base Refactor
- [ ] Convert base optimizer to a real ABC (`class Optimizer(ABC)`).
- [ ] Split optimizer utilities into focused modules (cache, artifacts, trials).
- [x] Extract optimizer cache/fingerprint helpers into `sd_optim/core/optimizer_cache.py` while keeping the legacy `sd_optim.optimizer` helper surface stable.
  - Added `sd_optim/core/__init__.py` and `sd_optim/core/optimizer_cache.py` as the first `core/` scaffolding for the optimizer split
  - Kept `fail_on_error_enabled`, `_compute_generation_setup_fingerprint`, and `Optimizer.calculate_image_hash(...)` working through the legacy `sd_optim.optimizer` surface
  - Added focused compatibility coverage in `tests/test_optimizer_core_cache.py`
  - Focused verification: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py`
  - Result: `13 passed in 43.60s`
  - Coverage: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest --cov=sd_optim.core.optimizer_cache --cov-report=term-missing -q tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py`
  - Result: `sd_optim/core/optimizer_cache.py` at `95%` coverage
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/optimizer.py sd_optim/core/__init__.py sd_optim/core/optimizer_cache.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/optimizer.py sd_optim/core/__init__.py sd_optim/core/optimizer_cache.py tests/test_optimizer_core_cache.py`
  - Result: success
- [x] Extract universal-reuse manifest/cache I/O into `sd_optim/core/optimizer_cache_io.py` and update optimizer call sites to use it directly.
  - Added `sd_optim/core/optimizer_cache_io.py` for history-cache loading, run-manifest writing, manifest-entry construction, and image output path generation
  - Updated `sd_optim/optimizer.py` to call the new core cache/cache-I/O helpers directly instead of keeping optimizer-local wrappers for image hash, manifest save, and image path generation
  - Updated `sd_optim/optuna_optimizer.py` and focused tests to import `fail_on_error_enabled` and cache fingerprint helpers from `sd_optim.core.optimizer_cache`
  - Added focused coverage in `tests/test_optimizer_cache_io.py` for manifest precedence, legacy PNG metadata reuse, manifest writing, path normalization, and image naming
  - Focused verification: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_optimizer_cache_io.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py`
  - Result: `23 passed in 50.81s`
  - Coverage: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest --cov=sd_optim.core.optimizer_cache_io --cov-report=term-missing -q tests/test_optimizer_cache_io.py`
  - Result: `sd_optim/core/optimizer_cache_io.py` at `88%` coverage
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/optimizer.py sd_optim/optuna_optimizer.py sd_optim/core/optimizer_cache.py sd_optim/core/optimizer_cache_io.py tests/test_optimizer_cache_io.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py`
  - Result: `All checks passed!`
  - Syntax check: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/optimizer.py sd_optim/optuna_optimizer.py sd_optim/core/optimizer_cache.py sd_optim/core/optimizer_cache_io.py tests/test_optimizer_cache_io.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py`
  - Result: success
- [ ] Keep shared orchestration in `core/optimizer_base.py`.
- [ ] Update Optuna/Bayes classes to inherit new base and pass tests.
- [x] Restore the missing trial scorer summary helper module required by optimizer startup.
  - Re-added `sd_optim/trial_scorer_summary.py` with the aggregate/payload summary helper used by `sd_optim.optimizer`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_trial_scorer_summary.py`
  - Result: `2 passed in 0.30s`
  - Startup smoke check: `PYTHONPATH=. .venv-wsl/bin/python -c "from sd_optim import OptunaOptimizer; print('optuna import ok')"`
  - Result: `optuna import ok`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/trial_scorer_summary.py tests/test_trial_scorer_summary.py`
  - Result: `All checks passed!`
- [x] Fix trial scorer summary call sites and avoid noisy Optuna postprocess failures when all trials have failed.
  - Updated `sd_optim/optimizer.py` to call `build_trial_scorer_summary(...)` with the current keyword-only signature
  - Updated `sd_optim/optuna_optimizer.py` postprocess recap to skip best-trial reporting when there are no successful completed trials yet
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_trial_scorer_summary.py tests/test_fail_on_error_policy.py`
  - Result: `6 passed in 44.96s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/optimizer.py sd_optim/optuna_optimizer.py tests/test_trial_scorer_summary.py tests/test_fail_on_error_policy.py`
  - Result: `All checks passed!`
  - Startup smoke check: `PYTHONPATH=. .venv-wsl/bin/python -c "from sd_optim import OptunaOptimizer; print('optuna import ok')"`
  - Result: `optuna import ok`
- [x] Strengthen the universal reuse/cache fingerprint so different merge setups do not collide.
  - Added a dedicated generation-setup fingerprint in `sd_optim/optimizer.py`
  - Cache keys now include merge/generation identity such as `optimization_mode`, `merge_method`, normalized input model paths, merge dtype/save dtype, WebUI, and recipe-optimization targets
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_optimizer_cache_fingerprint.py tests/test_trial_scorer_summary.py`
  - Result: `4 passed in 32.12s`
  - Startup smoke check: `PYTHONPATH=. .venv-wsl/bin/python -c "from sd_optim import OptunaOptimizer; print('optuna import ok')"`
  - Result: `optuna import ok`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/optimizer.py tests/test_optimizer_cache_fingerprint.py`
  - Result: `All checks passed!`
- [x] Make per-trial model-processing failures non-fatal by default so optimization can continue unless `fail_on_error` is explicitly enabled.
  - Corrected the temporary policy drift: fail-fast is now the default again when `fail_on_error` is unset
  - `sd_optim/optimizer.py` and `sd_optim/optuna_optimizer.py` now both stop the run on real trial errors unless `cfg.fail_on_error` is explicitly set to false
  - Added focused policy coverage in `tests/test_fail_on_error_policy.py`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_fail_on_error_policy.py`
  - Result: `4 passed in 43.31s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/optimizer.py sd_optim/optuna_optimizer.py tests/test_fail_on_error_policy.py`
  - Result: `All checks passed!`
  - Startup smoke check: `PYTHONPATH=. .venv-wsl/bin/python -c "from sd_optim import OptunaOptimizer; print('optuna import ok')"`
  - Result: `optuna import ok`

## Phase 4: Scoring Package Cleanup
- [x] Move bundled scoring model implementations under the packaged scorer namespace.
  - Updated scorer module-path registry in `sd_optim/scorer.py`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_dependency_loading.py`
  - Result: `4 passed in 119.04s`
  - Note: `ruff check sd_optim/scorer.py` still reports pre-existing SIM/F841 findings outside this refactor slice, so lint verification was kept focused on the clean touched files plus syntax compilation for `sd_optim/scorer.py`.
- [x] Separate registry/config (`MODEL_DATA`) from runtime manager.
  - Moved scorer registry data and lazy class lookup into `sd_optim/extensions/bundled/scorers/registry.py`
  - Kept `sd_optim/scorer.py` focused on scorer runtime/orchestration
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_dependency_loading.py`
  - Result: `5 passed in 112.83s`
- [x] Move test scripts and image artifacts out of package (`tools/` or `assets/`).
  - Added guardrail test: `tests/test_package_artifact_hygiene.py`
  - Moved scorer scratch scripts to `tools/scorer_debug/`
  - Moved sample scorer images to `assets/test_images/scorers/`
  - Moved archived scratch bundle to `tools/archive/random_scripts.7z`
  - Command: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_package_artifact_hygiene.py`
  - Result: `3 passed in 0.30s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_package_artifact_hygiene.py`
  - Result: `All checks passed!`
- [x] Update imports in `sd_optim/scorer.py` (or `scoring/manager.py`).
  - `sd_optim/scorer.py` now imports scorer support modules and lazy class lookup from `sd_optim.extensions.bundled.scorers`
- [x] Move bundled scorer implementation modules into a dedicated subpackage so registry/package files are not mixed with model code.
  - Moved bundled scorer modules into `sd_optim/extensions/bundled/scorers/models/`
  - Updated bundled scorer registry import paths and internal scorer-module imports
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_dependency_loading.py`
  - Result: `6 passed in 113.02s`
  - Note: `ruff` on some moved scorer implementation modules still reports pre-existing legacy findings, so lint verification stayed focused on the clean registry/test/package files plus syntax compilation for the touched moved modules.
- [x] Extract scorer asset/download helpers from `sd_optim/scorer.py` into bundled scorer support modules.
  - Added `sd_optim/extensions/bundled/scorers/assets.py`
  - `AestheticScorer.setup_evaluator_paths()`, `get_models()`, and `download_file()` now delegate to the asset helper module
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_support_modules.py tests/test_scorer_dependency_loading.py`
  - Result: `8 passed in 111.71s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_scorer_support_modules.py tests/test_scorer_dependency_loading.py sd_optim/extensions/bundled/scorers/assets.py sd_optim/extensions/bundled/scorers/loading.py`
  - Result: `All checks passed!`
- [x] Extract scorer factory/loading helpers from `sd_optim/scorer.py` into bundled scorer support modules.
  - Added `sd_optim/extensions/bundled/scorers/loading.py`
  - `AestheticScorer._build_scorer_factory()`, `_load_model()`, and `_load_all_models()` now delegate to the loading helper module
  - Syntax verification: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/scorer.py sd_optim/extensions/bundled/scorers/assets.py sd_optim/extensions/bundled/scorers/loading.py`
  - Result: `Passed`
- [x] Extract scorer manual-interaction helpers from `sd_optim/scorer.py` and modernize platform image-opening behavior.
  - Added `sd_optim/extensions/bundled/scorers/interaction.py`
  - `AestheticScorer.handle_override_prompt()`, `get_user_score()`, and `open_image()` now delegate to the focused interaction helper module
  - Replaced Windows `start` shell usage with `os.startfile(...)`
  - Updated WSL handling to prefer `wslview` / `xdg-open-wsl` when available and fall back once-per-run to `xdg-open` with an informational log
  - Removed the dead `printWSLFlag` / `wsl_instructions_printed` pattern from `sd_optim/scorer.py`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_interaction.py tests/test_scorer_support_modules.py tests/test_scorer_dependency_loading.py`
  - Result: `13 passed in 115.88s (0:01:55)`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_scorer_interaction.py tests/test_scorer_support_modules.py tests/test_scorer_dependency_loading.py sd_optim/scorer.py sd_optim/extensions/bundled/scorers/interaction.py`
  - Result: `All checks passed!`
  - Syntax verification: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/scorer.py sd_optim/extensions/bundled/scorers/interaction.py`
  - Result: `Passed`
- [x] Finish the runtime manual scoring path so it uses the normalized opener flow and records manual scorer results consistently.
  - Manual scoring now saves a real preview image under the scorer image directory instead of relying on transient `PIL.Image.show()` behavior
  - Manual previews now open via `AestheticScorer.open_image()` / bundled scorer interaction helpers
  - Manual scorer output is now stored in `last_scorer_results` just like automatic scorers
  - Added focused runtime coverage: `tests/test_scorer_manual_runtime.py`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_manual_runtime.py tests/test_scorer_interaction.py tests/test_scorer_support_modules.py tests/test_scorer_dependency_loading.py`
  - Result: `14 passed in 116.40s (0:01:56)`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_scorer_manual_runtime.py tests/test_scorer_interaction.py tests/test_scorer_support_modules.py tests/test_scorer_dependency_loading.py sd_optim/scorer.py sd_optim/extensions/bundled/scorers/interaction.py`
  - Result: `All checks passed!`
  - Syntax verification: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/scorer.py sd_optim/extensions/bundled/scorers/interaction.py`
  - Result: `Passed`

## Phase 4.5: Packaged Layout Cutover
- [x] Move bundled model configs under `sd_optim/extensions/bundled/model_configs/` and update runtime/config references.
  - Updated runtime defaults in `sd_optim.py` and `execute_recipes.py`
  - Updated dynamic converter imports in `sd_optim/utils.py`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_granular_conversion.py tests/test_svd_module_compat.py`
  - Result: `5 passed in 37.29s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_granular_conversion.py tests/test_svd_module_compat.py tests/test_scorer_dependency_loading.py sd_optim/svd.py sd_optim.py execute_recipes.py`
  - Result: `All checks passed!`
- [x] Decide and apply the final home for bundled merge-method helper modules under `sd_optim/extensions/bundled/`.
  - Exposed bundled SVD helpers from `sd_optim/extensions/bundled/merge_methods/__init__.py`
  - Updated `sd_optim/svd.py` to import through the packaged namespace instead of a raw file path
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_svd_module_compat.py`
  - Result: `2 passed in 18.79s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_svd_module_compat.py sd_optim/svd.py sd_optim/extensions/bundled/merge_methods/__init__.py`
  - Result: `All checks passed!`
- [x] Rename the shipped package assets layout from `sd_optim/builtin/...` to `sd_optim/extensions/bundled/...` and clean up fallout.
  - Goal: update runtime imports and config defaults to the new `extensions/bundled` path
  - Goal: fix tests and docs that still reference `builtin` or the intermediate `sd_optim.bundled` path
  - Goal: refresh package docstrings and track notes so the new layout reads intentionally
  - Updated scorer runtime imports to use `sd_optim.extensions.bundled.scorers.*`
  - Updated bundled scorer registry module paths to `sd_optim.extensions.bundled.scorers.models.*`
  - Updated root script defaults and config templates to point at `sd_optim/extensions/bundled/model_configs`
  - Updated focused regression tests and package docs to the `extensions/bundled` namespace
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_granular_conversion.py tests/test_svd_module_compat.py`
  - Result: `6 passed in 25.19s`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_support_modules.py tests/test_scorer_interaction.py`
  - Result: `7 passed in 0.98s`
  - Focused verification: `PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_dependency_loading.py`
  - Result: `6 passed in 110.55s (0:01:50)`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/scorer.py sd_optim/extensions/__init__.py sd_optim/extensions/bundled/__init__.py sd_optim/extensions/bundled/scorers/__init__.py sd_optim/extensions/bundled/scorers/registry.py sd_optim.py execute_recipes.py tests/test_granular_conversion.py tests/test_svd_module_compat.py tests/test_scorer_dependency_loading.py`
  - Result: `All checks passed!`
- [x] Restore full-test collection after the packaged layout move.
  - Goal: make `pytest` collection work from the repo root without requiring a manual `PYTHONPATH=.`
  - Goal: restore the moved `sd_optim.svd_ties_sum_extended` module surface used by existing tests
  - Added `tests/conftest.py` to insert the repo root into `sys.path` during pytest collection
  - Restored `sd_optim.svd_ties_sum_extended` as a compatibility module that re-exports the runtime `MergeMethods` surface without double-registering merge methods
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_comfy_adapter_prompt_injection.py`
  - Result: `1 passed in 0.79s`
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_svd_ties_sum_extended_v13.py`
  - Result: `6 passed in 21.23s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/conftest.py sd_optim/svd_ties_sum_extended.py tests/test_comfy_adapter_prompt_injection.py tests/test_fail_on_error_policy.py tests/test_svd_ties_sum_extended_v13.py`
  - Result: `All checks passed!`
  - Syntax verification: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile tests/conftest.py sd_optim/svd_ties_sum_extended.py`
  - Result: `Passed`
  - Note: `tests/test_fail_on_error_policy.py` no longer fails with `ModuleNotFoundError`, but isolated pytest execution hung in this WSL environment before producing a stable result, so I did not record it as a passing verification command in this slice.
- [x] Refresh stale tests that still assume the pre-reorg scorer and merge-helper layout.
  - Goal: keep coverage for active runtime behavior while retiring tests that only target removed compatibility helpers or manual-only integrations
  - Updated print-cleanup coverage to scan the moved scorer implementation files under `sd_optim/extensions/bundled/scorers/models/`
  - Updated Optuna sampler/dashboard tests so their optimizer stubs include the now-required `fail_on_error_enabled` helper import
  - Reworked the scalar recipe rewrite test to exercise inline literal replacement against the current `serialize_nodes_for_rewrite()` contract
  - Converted the texture scorer integration test into an explicit opt-in optional-dependency test instead of a default-suite async/manual integration
  - Removed the wavelet compatibility test file that only targeted deleted `pkg_resources` shim helpers no longer present in `sd_optim.merge_methods`
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_logging_print_cleanup.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_texture_scorer_integration.py`
  - Result: `4 passed, 1 skipped in 35.54s`
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_optuna_cma_warning_hygiene.py tests/test_optuna_dashboard_launcher.py`
  - Result: `4 passed in 17.61s`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check tests/test_logging_print_cleanup.py tests/test_optuna_cma_warning_hygiene.py tests/test_optuna_dashboard_launcher.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_texture_scorer_integration.py tests/conftest.py sd_optim/svd_ties_sum_extended.py`
  - Result: `All checks passed!`
- [x] Move scorer runtime/support code out of `sd_optim/extensions/bundled/` so bundled scorers only contain shipped implementations.
  - Goal: create a real `sd_optim/scoring/` package for registry/loading/assets/interaction support code
  - Goal: leave `sd_optim/extensions/bundled/scorers/` as the home for shipped scorer model implementations only
  - Added `sd_optim/scoring/__init__.py`, `catalog.py`, `registry.py`, `assets.py`, `loading.py`, and `interaction.py`
  - Updated `sd_optim/scorer.py` to import support code from `sd_optim.scoring.*`
  - Reduced `sd_optim/extensions/bundled/scorers/` to the bundled implementation package instead of a mixed runtime/support package
  - Updated scorer-focused tests to import the new scoring support package
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_support_modules.py tests/test_scorer_interaction.py tests/test_scorer_manual_runtime.py`
  - Result: `8 passed in 8.96s`
  - Focused verification: `CI=true .venv-wsl/bin/pytest -q -s tests/test_scorer_dependency_loading.py`
  - Result: `6 passed in 117.64s (0:01:57)`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/scorer.py sd_optim/scoring/__init__.py sd_optim/scoring/catalog.py sd_optim/scoring/registry.py sd_optim/scoring/assets.py sd_optim/scoring/loading.py sd_optim/scoring/interaction.py sd_optim/extensions/bundled/scorers/__init__.py tests/test_scorer_support_modules.py tests/test_scorer_interaction.py tests/test_scorer_dependency_loading.py`
  - Result: `All checks passed!`
  - Syntax verification: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/scorer.py sd_optim/scoring/__init__.py sd_optim/scoring/catalog.py sd_optim/scoring/registry.py sd_optim/scoring/assets.py sd_optim/scoring/loading.py sd_optim/scoring/interaction.py sd_optim/extensions/bundled/scorers/__init__.py`
  - Result: `Passed`
- [x] Extract scorer runtime setup/manual-preview/score-loop helpers from `sd_optim/scorer.py` into `sd_optim/scoring/runtime.py`.
  - Added `sd_optim/scoring/runtime.py` for rembg-session setup, image-saving/manual-preview runtime behavior, score averaging, and the main scorer execution loop.
  - Updated `sd_optim/scorer.py` to import scorer support directly from `sd_optim.scoring.assets`, `interaction`, `loading`, `registry`, and the new `runtime` module instead of routing through the scoring package root aliases.
  - Kept the public `AestheticScorer` surface stable while shrinking the class methods down to orchestration wrappers over the extracted runtime helpers.
  - Added focused runtime coverage in `tests/test_scorer_runtime_modules.py` for average calculation, manual image-saving setup, and rembg dependency enforcement.
  - Focused verification: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_scorer_runtime_modules.py tests/test_scorer_support_modules.py tests/test_scorer_interaction.py tests/test_scorer_manual_runtime.py tests/test_scorer_dependency_loading.py`
  - Result: `17 passed in 132.81s (0:02:12)`
  - Focused lint: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/scorer.py sd_optim/scoring/runtime.py tests/test_scorer_runtime_modules.py tests/test_scorer_support_modules.py tests/test_scorer_interaction.py tests/test_scorer_manual_runtime.py tests/test_scorer_dependency_loading.py`
  - Result: `All checks passed!`
  - Syntax verification: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/scorer.py sd_optim/scoring/runtime.py tests/test_scorer_runtime_modules.py tests/test_scorer_support_modules.py tests/test_scorer_interaction.py tests/test_scorer_manual_runtime.py tests/test_scorer_dependency_loading.py`
  - Result: `Passed`

## Phase 5: Utils Decomposition
- [x] Start the `sd_optim.utils` package cutover by extracting recipe/model-dir helpers into `sd_optim/utils/recipes.py` while preserving legacy `from sd_optim import utils` imports.
  - Moved `sd_optim/utils.py` to `sd_optim/utils/__init__.py` so `sd_optim.utils` is now a package and can host focused submodules.
  - Added `sd_optim/utils/recipes.py` for the sd-mecha model-dir, recipe-serialization, cache-map, and relative-path helpers.
  - Kept legacy helper access stable by re-exporting the extracted recipe helpers from `sd_optim.utils`.
  - Updated package-root detection so bundled merge-method discovery still resolves paths relative to `sd_optim/` after the module-to-package cutover.
  - Added layout regression coverage in `tests/test_utils_package_layout.py`.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_utils_package_layout.py tests/test_sd_mecha_graph_compat.py tests/test_merge_method_resolution.py`
  - Result: `15 passed in 26.51s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/utils/recipes.py tests/test_utils_package_layout.py tests/test_sd_mecha_graph_compat.py tests/test_merge_method_resolution.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/utils/__init__.py sd_optim/utils/recipes.py tests/test_utils_package_layout.py`
  - Result: `Passed`
  - Note: `ruff check --select F401 sd_optim/utils/__init__.py` still reports a pre-existing unused `yaml.CDumper` import outside this slice, so lint verification stayed focused on the new module and regression files while syntax compilation covered the package shim.
- [x] Extract custom config/conversion loading helpers into `sd_optim/utils/conversions.py` while preserving legacy `sd_optim.utils` access.
  - Added `sd_optim/utils/conversions.py` for custom model-config registration and dynamic custom conversion-module loading.
  - Kept legacy callers stable by re-exporting `load_and_register_custom_configs()` and `load_and_register_custom_conversion()` from `sd_optim.utils`.
  - Added package-layout coverage in `tests/test_utils_package_layout.py` for the new `sd_optim.utils.conversions` import path and compatibility re-exports.
  - Removed new unused-import lint fallout from `sd_optim/utils/__init__.py` after moving the loader helpers out.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_utils_package_layout.py tests/test_merge_method_resolution.py`
  - Result: `6 passed in 25.43s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/utils/conversions.py tests/test_utils_package_layout.py tests/test_merge_method_resolution.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check --select F401 sd_optim/utils/__init__.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/utils/__init__.py sd_optim/utils/conversions.py tests/test_utils_package_layout.py`
  - Result: `Passed`
- [x] Extract recipe rewrite and reproducible artifact helpers into `sd_optim/utils/artifacts.py` while preserving legacy `sd_optim.utils` access.
  - Added `sd_optim/utils/artifacts.py` for recipe text rewriting, target-node inspection, reproducible merge-script generation, converter discovery, and `.mecha` transpilation helpers.
  - Kept legacy callers stable by re-exporting the artifact helpers from `sd_optim.utils`.
  - Preserved existing test monkeypatch behavior by having the extracted artifact helpers resolve `serialize_recipe_text()` and `resolve_merge_method()` through `sd_optim.utils` at call time instead of binding those names eagerly.
  - Added package-layout coverage in `tests/test_utils_package_layout.py` for the new `sd_optim.utils.artifacts` import path and compatibility re-exports.
  - Removed new unused-import lint fallout from `sd_optim/utils/__init__.py` after moving the artifact helpers out.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_utils_package_layout.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_sd_mecha_graph_compat.py`
  - Result: `15 passed in 29.44s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/utils/artifacts.py tests/test_utils_package_layout.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_sd_mecha_graph_compat.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check --select F401 sd_optim/utils/__init__.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/utils/__init__.py sd_optim/utils/artifacts.py tests/test_utils_package_layout.py`
  - Result: `Passed`
- [x] Extract config validation, layer-adjust/image-summary helpers, and hotkey support into `sd_optim/utils/config.py`, `sd_optim/utils/images.py`, and `sd_optim/utils/hotkeys.py` while preserving legacy `sd_optim.utils` access.
  - Added `sd_optim/utils/config.py` for run-config validation and the shared precision mapping.
  - Added `sd_optim/utils/images.py` for layer-adjust helpers plus summary-image/log-score utilities.
  - Added `sd_optim/utils/hotkeys.py` for scoring-mode hotkey constants and `HotkeyListener`.
  - Kept legacy callers stable by re-exporting the extracted helpers from `sd_optim.utils`, including the old lowercase `precision_mapping` alias.
  - Added focused coverage in `tests/test_utils_misc_modules.py` for config validation, layer-adjust mutations, summary-image selection, log-score rewriting, and hotkey mode switching.
  - Expanded `tests/test_utils_package_layout.py` to cover the new `sd_optim.utils.config`, `sd_optim.utils.images`, and `sd_optim.utils.hotkeys` import paths and compatibility re-exports.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_utils_package_layout.py tests/test_utils_misc_modules.py`
  - Result: `12 passed in 30.20s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/utils/config.py sd_optim/utils/images.py sd_optim/utils/hotkeys.py tests/test_utils_package_layout.py tests/test_utils_misc_modules.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check --select F401 sd_optim/utils/__init__.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/utils/__init__.py sd_optim/utils/config.py sd_optim/utils/images.py sd_optim/utils/hotkeys.py tests/test_utils_misc_modules.py`
  - Result: `Passed`
- [x] Extract merge-method resolution/indexing helpers into `sd_optim/utils/methods.py` while preserving legacy `sd_optim.utils` monkeypatch points and compatibility re-exports.
  - Added `sd_optim/utils/methods.py` for bundled/legacy merge-method name indexing plus merge-method resolution helpers.
  - Kept the legacy `sd_optim.utils` helper surface stable by re-exporting the extracted resolver helpers from `sd_optim.utils`.
  - Preserved existing monkeypatch-based regression behavior by having the extracted resolver helpers look up patchable helper entrypoints through the already-loaded `sd_optim.utils` module at call time.
  - Expanded `tests/test_utils_package_layout.py` to cover the new `sd_optim.utils.methods` import path and compatibility re-exports.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_utils_package_layout.py tests/test_merge_method_resolution.py`
  - Result: `11 passed in 31.33s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/utils/methods.py tests/test_utils_package_layout.py tests/test_merge_method_resolution.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check --select F401 sd_optim/utils/__init__.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/utils/__init__.py sd_optim/utils/methods.py tests/test_utils_package_layout.py`
  - Result: `Passed`
- [x] Split `sd_optim/utils.py` into focused modules (`config`, `recipes`, `conversions`, `artifacts`, `images`, `hotkeys`, `methods`).
  - Completed the `sd_optim.utils` package cutover so the old flat helper module is now decomposed into focused submodules with a thin compatibility package root.
- [x] Update all import sites and keep compatibility re-exports if needed.
  - Kept active callers stable through `sd_optim.utils` compatibility re-exports while enabling direct submodule imports for the isolated helper clusters.
- [x] Add unit tests for newly isolated utility functions.
  - Added focused package-layout and module-behavior coverage for the extracted utility modules across the Phase 5 slices.
- [x] Replace the `sd_optim.utils` import-all barrel with a lazy compatibility shim.
  - Reworked `sd_optim/utils/__init__.py` so the package root no longer eagerly imports every utility submodule during import.
  - Kept the legacy `sd_optim.utils.<name>` surface working through a lazy `__getattr__` export map that resolves symbols on first access and then caches them.
  - Added a regression in `tests/test_utils_package_layout.py` to confirm that importing `sd_optim.utils` does not eagerly import helper submodules like `hotkeys` or `methods`.
- [x] Remove internal dependence on the `sd_optim.utils` package-root surface and shrink `sd_optim/utils/__init__.py` to a minimal package marker.
  - Updated optimizer, merger, merge helpers, and focused tests to import from concrete `sd_optim.utils.*` modules instead of routing through `from sd_optim import utils`.
  - Removed `sd_optim.utils` package-root re-exporting so `sd_optim/utils/__init__.py` is now just a minimal package file rather than a barrel or lazy shim.
  - Simplified `sd_optim/utils/artifacts.py`, `config.py`, and `methods.py` so they no longer reach back through `sd_optim.utils` for internal helper access.
  - Refreshed package-layout coverage to assert a minimal package root plus direct submodule imports, and updated merge/artifact regressions to patch the real target modules instead of the package root.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_utils_package_layout.py tests/test_merge_method_resolution.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_sd_mecha_graph_compat.py tests/test_utils_misc_modules.py tests/test_merger_model_arity.py`
  - Result: `24 passed in 30.43s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/optimizer.py sd_optim/merger.py sd_optim/merge/recipe_builder.py sd_optim/merge/model_selection.py sd_optim/merge/execution.py sd_optim/merge/artifacts.py sd_optim/merge/fallback.py sd_optim/utils/__init__.py sd_optim/utils/artifacts.py sd_optim/utils/config.py sd_optim/utils/methods.py tests/test_utils_package_layout.py tests/test_merge_method_resolution.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_sd_mecha_graph_compat.py tests/test_utils_misc_modules.py tests/test_logging_print_cleanup.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/optimizer.py sd_optim/merger.py sd_optim/merge/recipe_builder.py sd_optim/merge/model_selection.py sd_optim/merge/execution.py sd_optim/merge/artifacts.py sd_optim/merge/fallback.py sd_optim/utils/__init__.py sd_optim/utils/artifacts.py sd_optim/utils/config.py sd_optim/utils/methods.py tests/test_utils_package_layout.py tests/test_merge_method_resolution.py tests/test_recipe_rewrite_scalar_kwargs.py tests/test_sd_mecha_graph_compat.py tests/test_utils_misc_modules.py tests/test_logging_print_cleanup.py`
  - Result: `Passed`
- [x] Extract recipe-mode orchestration and layer-adjust execution helpers out of `sd_optim/merger.py`.
  - Added `sd_optim/merge/recipe_optimization.py` for recipe-text sanitization, target-node validation, and recipe-mode orchestration.
  - Added `sd_optim/merge/layer_adjust.py` for output/model-path resolution, checkpoint loading, SDXL detection, and layer-adjust execution.
  - Updated `sd_optim/merger.py` so recipe-mode and layer-adjust methods now delegate to the focused merge helper modules instead of carrying those workflows inline.
  - Added focused coverage in `tests/test_merger_runtime_modules.py` for recipe sanitization, recipe-target validation, layer-adjust path resolution, and SDXL detection.
  - Command: `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_merger_runtime_modules.py tests/test_merger_model_arity.py tests/test_recipe_rewrite_scalar_kwargs.py`
  - Result: `7 passed in 25.91s`
  - Command: `PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/merger.py sd_optim/merge/recipe_optimization.py sd_optim/merge/layer_adjust.py tests/test_merger_runtime_modules.py tests/test_merger_model_arity.py tests/test_recipe_rewrite_scalar_kwargs.py`
  - Result: `All checks passed!`
  - Command: `PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/merger.py sd_optim/merge/recipe_optimization.py sd_optim/merge/layer_adjust.py tests/test_merger_runtime_modules.py tests/test_merger_model_arity.py tests/test_recipe_rewrite_scalar_kwargs.py`
  - Result: `Passed`

## Phase 6: Config, Docs, and Migration Notes
- [ ] Update `conf/config.yaml` to reflect new conversion/merge method locations.
- [ ] Update README and any internal docs referencing old paths.
- [ ] Add migration notes for downstream users (path changes, deprecations).

## Phase 7: Verification & Hardening
- [ ] Run full test suite with coverage target (>80% for touched modules).
- [ ] Add regression tests for integration entrypoints.
- [ ] Validate manual workflows (merge, optimization run, API endpoints).

## Exit Criteria
- All tests pass with required coverage.
- No behavioral regressions in merge, optimize, generate, or score flows.
- Entry points and external integrations remain compatible.
- Documentation and config reflect the new structure.
