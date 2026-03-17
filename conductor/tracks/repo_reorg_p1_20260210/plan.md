# Repo Reorganization & Optimizer Refactor Plan (2026-02-10)

Detailed execution history and verification notes now live in `conductor/tracks/repo_reorg_p1_20260210/progress.md`.
This file is the short working plan.

## Goals
- Reduce file size and complexity by splitting large modules, especially the optimizer and Optuna runtime.
- Establish clear package boundaries across `core`, `merge`, `scoring`, `utils`, and packaged extensions.
- Prefer direct cutovers over compatibility layers when the repo can absorb the change safely.
- Preserve real entrypoints and runtime behavior while improving testability.
- Finish the Optuna modernization work and eliminate remaining config/API drift.

## Current Focus
- Keep the track docs/config/docs aligned with the moved package layout.
- Re-check whether Bayes-specific support code can move out of the base path as well.
- Keep moving facade-heavy runtime files toward direct orchestration code instead of one-line forwarding methods.

## Working Rules
- Follow `conductor/workflow.md`.
- Track completed implementation details in `progress.md`, not inline under every checklist item here.
- Add failing tests first for behavior changes.
- Prefer direct import updates over new compatibility wrappers unless an external entrypoint really requires one.

## Target Structure
```text
sd_optim/
  core/
    optimizer_base.py
    optimizer_cache.py
    optimizer_cache_io.py
    optimizer_artifacts.py
  merge/
  scoring/
  extensions/bundled/
  utils/
```

## Target Interaction Shapes

### Merger
- `sd_optim/merger.py` should be a small orchestration layer with a stable public API.
- Public entrypoints should remain:
  - `Merger.__init__(...)`
  - `Merger.merge(...)`
  - `Merger.recipe_optimization(...)`
  - `Merger.layer_adjust(...)`
- Those entrypoints should call focused helpers from `sd_optim/merge/*` directly.
- Private one-line forwarding methods should be treated as transitional and removed when safe.
- The intended steady state is:
  - state/config lives on `Merger`
  - flow/orchestration is readable in `merge()`
  - detailed logic lives in helper modules such as `recipe_builder.py`, `fallback.py`, `execution.py`, `artifacts.py`, and `model_selection.py`

### Optimizer / Optuna
- `sd_optim/core/optimizer_base.py` should remain a shared runtime base, not a grab bag of optimizer-specific helpers.
- Concrete optimizers should live under their packaged namespaces and keep only their true entrypoint classes near the top.
- Helper packages like `sd_optim/optimizers/optuna/` should own sampler, objective, reporting, dashboard, and study lifecycle logic directly.
- Compatibility surfaces should be kept only where external callers genuinely need them; repo-internal code should prefer direct imports.

## Optuna Modernization

### Known Issues
- [x] Fix QMC typo: `warn_asyncronous_seeding` -> `warn_asynchronous_seeding`.
- [x] Fix pruning validation path to read `optimizer.optuna_config.use_pruning`.
- [ ] Stop defaulting categorical-heavy guides toward CMA-ES where it is a poor fit.
- [ ] Document/instrument CMA-ES independent-sampling fallback behavior clearly.

### Planned Optuna Split
- [x] Create `sd_optim/optimizers/optuna/` package.
- [x] Move sampler configuration into a focused module.
- [x] Move study/storage lifecycle into a focused module.
- [x] Move objective/suggestion logic into focused modules.
- [x] Move callback/logging/importances into focused modules.
- [x] Reduce `sd_optim/optuna_optimizer.py` to a thin entrypoint.

### Acceptance Criteria
- [x] No regressions in start/resume/fork, callback logging, and trial JSONL schema.
- [x] Optuna init paths remain stable across `tpe`, `cmaes`, and `qmc`.
- [x] `sd_optim/optuna_optimizer.py` ends under 200 lines.
- [x] New Optuna submodules have focused tests with >80% coverage for touched code.

## Phase 0: Discovery & Constraints
- [x] Inventory imports and cross-module dependencies.
- [x] Identify hard external expectations and config-path constraints.
- [x] Confirm experiments/artifacts that can move out of the runtime package.
- [x] Keep `sd_optim` as the stable package name.

## Phase 1: Scaffolding (No Behavior Change)
- [ ] Create new package subfolders with `__init__.py`.
- [~] Preserve required external entrypoints during moves, preferring direct cutovers where feasible.
- [~] Add compatibility imports only where external callers genuinely require them.

## Phase 2: Merge Methods Split
- [x] Package bundled merge methods under `sd_optim/extensions/bundled/merge_methods/`.
- [x] Move merge runtime helper concerns into `sd_optim/merge/`.
- [x] Extract model selection, execution, model-node, recipe, and layer-adjust helpers.
- [x] Make bundled merge-method loading lazy.
- [x] Refresh merge-method and merge-runtime regression coverage.

## Phase 3: Optimizer Base Refactor
- [x] Convert the base optimizer to a real `ABC`.
- [x] Move shared optimizer orchestration into `sd_optim/core/optimizer_base.py`.
- [x] Update Optuna/Bayes classes to inherit the new base directly.
- [x] Extract cache/fingerprint helpers into `sd_optim/core/optimizer_cache.py`.
- [x] Extract manifest/cache I/O into `sd_optim/core/optimizer_cache_io.py`.
- [x] Extract artifact helpers into `sd_optim/core/optimizer_artifacts.py`.
- [x] Restore the missing trial scorer summary helper and fix its call sites.
- [x] Strengthen the universal reuse fingerprint.
- [x] Restore fail-fast behavior unless `fail_on_error: false` is explicitly set.
- [x] Split remaining optimizer trial/runtime helpers out of `sd_optim/core/optimizer_base.py`.
- [ ] Re-check whether Bayes-specific support code can move out of the base path as well.

## Phase 4: Scoring Package Cleanup
- [x] Move bundled scoring implementations under the packaged scorer namespace.
- [x] Separate scorer registry/config from runtime management.
- [x] Move scratch/test assets out of the runtime package.
- [x] Extract scorer asset/loading/interaction/runtime helpers into focused modules.
- [x] Move scorer support code into `sd_optim/scoring/`.

## Phase 4.5: Packaged Layout Cutover
- [x] Move bundled model configs under `sd_optim/extensions/bundled/model_configs/`.
- [x] Finalize the packaged home for bundled merge-method helpers.
- [x] Rename packaged shipped assets from `builtin` layout to `extensions/bundled`.
- [x] Restore full test collection after the layout move.
- [x] Refresh stale tests that assumed the pre-reorg layout.

## Phase 5: Utils Decomposition
- [x] Convert `sd_optim.utils` into a real package.
- [x] Extract `recipes`, `conversions`, `artifacts`, `config`, `images`, `hotkeys`, and `methods`.
- [x] Update internal callers to use direct submodule imports.
- [x] Shrink `sd_optim/utils/__init__.py` to a minimal package file.
- [x] Extract recipe-mode orchestration and layer-adjust execution helpers from `sd_optim/merger.py`.

## Phase 6: Config, Docs, and Migration Notes
- [ ] Update `conf/config.yaml` and related config/docs for the new package layout.
- [ ] Refresh README and internal docs that still point at old paths.
- [ ] Add downstream migration notes for moved modules and packaged assets.
- [ ] Keep `CHANGELOG.md` aligned with the finished cutovers.

## Phase 7: Verification & Hardening
- [ ] Run the full test suite and check coverage for touched modules.
- [ ] Add regression checks for real entrypoints and integration flows.
- [ ] Validate manual workflows for merge, optimize, generate, and API paths.

## Exit Criteria
- [ ] Touched modules have passing tests and acceptable coverage.
- [ ] No behavioral regressions in merge, optimize, generate, or score flows.
- [ ] Entry points and external integrations still work.
- [ ] Config/docs reflect the final package structure.
