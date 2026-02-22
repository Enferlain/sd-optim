# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
