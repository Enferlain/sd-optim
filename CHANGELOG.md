# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased] - 2026-02-05

### Added

### Changed

### Removed

### Fixed

- Fixed a regression in `sd_optim/prompter.py` where non-dictionary shared settings (like `workflow_json`) were being ignored when defined inside a cargo configuration.
- Removed the redundant break and added a guard to ensure only the first yielded image is processed in `optimizer.py`.

## [1.1.1] - 2023-03-05
