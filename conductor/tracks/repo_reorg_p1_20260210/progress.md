# Repo Reorganization Progress Log

This file holds the detailed execution history for the `repo_reorg_p1_20260210` track.
The short actionable checklist lives in `plan.md`.

## Notes
- On 2026-03-17, the working plan was condensed so `plan.md` stays readable.
- Earlier detailed line-by-line history remains available in git history before that trim.
- New completed work should be summarized here instead of expanding `plan.md`.

## Phase Snapshot

### Phase 0
- Completed discovery, dependency inventory, and external-expectation mapping.

### Phase 1
- Still incomplete as a formal checklist, though several later package cutovers have already landed.

### Phase 2
- Completed the merge-method packaging and merge-runtime helper split.
- Landed lazy bundled merge-method loading and refreshed merge-runtime regression coverage.

### Phase 3
- Completed the optimizer base cutover to `sd_optim/core/optimizer_base.py`.
- Extracted cache/fingerprint helpers to `sd_optim/core/optimizer_cache.py`.
- Extracted manifest/cache I/O to `sd_optim/core/optimizer_cache_io.py`.
- Extracted artifact helpers to `sd_optim/core/optimizer_artifacts.py`.
- Updated Optuna/Bayes/artist imports to the new core base path directly.
- Extracted trial/runtime execution into `sd_optim/core/optimizer_runtime.py`.
- Split the Optuna runtime into `sd_optim/optimizers/optuna/` helper modules and reduced `sd_optim/optuna_optimizer.py` to a thin entrypoint.
- Remaining work: revisit Bayes-specific support that still lives near the shared optimizer path, and close the new-module coverage target.

### Phase 4
- Completed scoring package cleanup and support-module extraction.

### Phase 4.5
- Completed the `extensions/bundled` packaged-layout cutover and restored test collection.

### Phase 5
- Completed the `sd_optim.utils` package decomposition and direct-import cutover.

## Recent Detailed Entries

### 2026-03-17: Optimizer Cache/Fingerprint Helpers
- Added `sd_optim/core/optimizer_cache.py`.
- Focused verification:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py`
  - Result: `13 passed in 43.60s`
- Coverage:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest --cov=sd_optim.core.optimizer_cache --cov-report=term-missing -q tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py`
  - Result: `sd_optim/core/optimizer_cache.py` at `95%`

### 2026-03-17: Optimizer Cache I/O Helpers
- Added `sd_optim/core/optimizer_cache_io.py`.
- Updated optimizer and focused tests to use core cache helpers directly.
- Focused verification:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_optimizer_cache_io.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py`
  - Result: `23 passed in 50.81s`
- Coverage:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest --cov=sd_optim.core.optimizer_cache_io --cov-report=term-missing -q tests/test_optimizer_cache_io.py`
  - Result: `sd_optim/core/optimizer_cache_io.py` at `88%`

### 2026-03-17: Optimizer Base Cutover
- Moved the shared optimizer runtime from `sd_optim/optimizer.py` to `sd_optim/core/optimizer_base.py`.
- Made `Optimizer` a real `ABC`.
- Updated Optuna/Bayes/artist/test imports to the new path directly.
- Added focused import/inheritance checks in `tests/test_optimizer_base_module.py`.
- Focused verification:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_optimizer_base_module.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_io.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py`
  - Result: `26 passed in 53.81s`
- Note:
  - The Bayes inheritance test stubs the optional `bayes_opt` dependency so the import-path check stays deterministic in environments where `bayes_opt` is not installed.

### 2026-03-17: Optimizer Artifact Helpers
- Added `sd_optim/core/optimizer_artifacts.py`.
- Moved best-model promotion and `best.log` persistence behind focused helpers.
- Added `tests/test_optimizer_artifacts.py`.
- Focused verification:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_optimizer_artifacts.py tests/test_optimizer_base_module.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_io.py tests/test_optimizer_cache_fingerprint.py tests/test_fail_on_error_policy.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py`
  - Result: `30 passed in 50.31s`
- Coverage:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest --cov=sd_optim.core.optimizer_artifacts --cov-report=term-missing -q tests/test_optimizer_artifacts.py`
  - Result: `sd_optim/core/optimizer_artifacts.py` at `92%`

### 2026-03-17: Optimizer Runtime + Optuna Package Split
- Added `sd_optim/core/optimizer_runtime.py` and moved trial execution / sequential generation-scoring orchestration out of `sd_optim/core/optimizer_base.py`.
- Added focused Optuna helper modules under `sd_optim/optimizers/optuna/`:
  - `sampler_factory.py`
  - `trial_logger.py`
  - `dashboard.py`
  - `study_manager.py`
  - `objective.py`
  - `reporting.py`
- Reduced `sd_optim/optuna_optimizer.py` to a thin entrypoint class (`46` lines).
- Updated tests and direct repo call sites to use the extracted modules instead of preserving the old internal helper surface.
- Kept `sd_optim/optimizers/optuna/__init__.py` minimal so direct submodule imports do not eagerly pull visualization/scipy dependencies during collection.
- Fixed the Optuna QMC config key typo (`warn_asynchronous_seeding`) and pruning validation path (`optimizer.optuna_config.use_pruning`).
- Focused verification:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest -q -s tests/test_fail_on_error_policy.py tests/test_optuna_split_modules.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py tests/test_reorg_guardrails.py tests/test_optimizer_base_module.py tests/test_optimizer_runtime_modules.py tests/test_optimizer_artifacts.py tests/test_optimizer_core_cache.py tests/test_optimizer_cache_io.py tests/test_optimizer_cache_fingerprint.py`
  - Result: `46 passed, 4 warnings in 51.30s`
- Additional checks:
  - `PYTHONPATH=. .venv-wsl/bin/ruff check ...`
  - `PYTHONPATH=. .venv-wsl/bin/python -m py_compile ...`
  - Result: passed
- Coverage spot-check:
  - `CI=true PYTHONPATH=. .venv-wsl/bin/pytest --cov=sd_optim.optimizers.optuna.dashboard --cov=sd_optim.optimizers.optuna.objective --cov=sd_optim.optimizers.optuna.reporting --cov=sd_optim.optimizers.optuna.sampler_factory --cov=sd_optim.optimizers.optuna.study_manager --cov=sd_optim.optimizers.optuna.trial_logger --cov-report=term-missing -q tests/test_fail_on_error_policy.py tests/test_optuna_split_modules.py tests/test_optuna_dashboard_launcher.py tests/test_optuna_cma_warning_hygiene.py tests/test_reorg_guardrails.py`
  - Result: overall `37%` across those touched Optuna modules (`dashboard 80%`, `objective 42%`, `reporting 14%`, `sampler_factory 58%`, `study_manager 24%`, `trial_logger 79%`), so the coverage acceptance item remains open.
