# Track: config/guide contract cleanup (2026-03-17)

## Goals
- Align the documented guide/config contract with the runtime behavior after the repo reorg.
- Fix concrete recipe-mode `target_nodes` handling so validation and execution support the same shapes.
- Clarify recipe-mode `custom_bounds` wording so fixed-value fallback kwargs are not confused with generated optimizer params.

## Tasks
- [x] Task: Add regression tests for recipe-mode multi-target validation and rewrite behavior.
- [x] Task: Implement runtime support for `recipe_optimization.target_nodes` as either a single ref or a list of refs.
- [x] Task: Refresh config/guide template wording for strategy semantics and recipe-mode `custom_bounds`.
- [x] Task: Simplify user-facing template wording to avoid internal/runtime terminology where plain language works better.
  - Note: docs-only wording pass; no test rerun needed.
- [x] Task: Draft a present-vs-ideal logging comparison using the latest run so guide/bounds logging improvements can be reviewed before implementation.
- [x] Task: Implement a compact startup guide/parameter-space summary at `INFO` and move the full itemized parameter list to `DEBUG`.
  - Command: `timeout 90s env PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q tests/test_reorg_guardrails.py`
  - Result: `9 passed, 4 warnings in 45.88s`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/bounds.py tests/test_reorg_guardrails.py`
  - Result: `All checks passed!`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/bounds.py`
  - Result: `Passed`
- [x] Task: Trim duplicate startup logging around guide setup, custom config registration, and optimizer bounds handoff.
  - Command: `timeout 120s env PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q tests/test_reorg_guardrails.py tests/test_conversions_module.py tests/test_optuna_study_manager_module.py`
  - Result: `28 passed, 4 warnings in 45.23s`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/bounds.py sd_optim/utils/conversions.py sd_optim/optimizers/optuna/study_manager.py tests/test_reorg_guardrails.py tests/test_conversions_module.py tests/test_optuna_study_manager_module.py`
  - Result: `All checks passed!`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/bounds.py sd_optim/utils/conversions.py sd_optim/optimizers/optuna/study_manager.py`
  - Result: `Passed`
- [x] Task: Remove the last duplicate parameter-count startup line so the guide summary remains the single `INFO` count source.
  - Command: `timeout 90s env PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q tests/test_optimizer_base_module.py tests/test_reorg_guardrails.py tests/test_optuna_study_manager_module.py`
  - Result: `33 passed, 4 warnings in 43.75s`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/core/optimizer_base.py tests/test_optimizer_base_module.py`
  - Result: `All checks passed!`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/core/optimizer_base.py`
  - Result: `Passed`
- [x] Task: Update `CHANGELOG.md` with the config/guide contract cleanup and startup logging improvements.
  - Note: docs-only update; no test rerun needed.
- [x] Task: Run focused regression tests and record outcomes.
  - Command: `timeout 90s env PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q tests/test_recipe_target_nodes.py tests/test_recipe_rewrite_scalar_kwargs.py`
  - Result: `3 passed in 24.91s`
  - Command: `timeout 90s env PYTHONPATH=. CI=true .venv-wsl/bin/pytest -q tests/test_reorg_guardrails.py tests/test_optuna_objective_module.py`
  - Result: `14 passed, 4 warnings in 41.60s`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/ruff check sd_optim/utils/config.py sd_optim/merge/recipe_optimization.py sd_optim/utils/artifacts.py sd_optim/merge/artifacts.py tests/test_recipe_target_nodes.py tests/test_recipe_rewrite_scalar_kwargs.py`
  - Result: `All checks passed!`
  - Command: `timeout 60s env PYTHONPATH=. .venv-wsl/bin/python -m py_compile sd_optim/utils/config.py sd_optim/merge/recipe_optimization.py sd_optim/utils/artifacts.py sd_optim/merge/artifacts.py`
  - Result: `Passed`
