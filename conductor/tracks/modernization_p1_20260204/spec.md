# Specification - Code Modernization Phase 1

## Overview

This track initiates the modernization of the `sd-optim` codebase. The primary goal is to resolve technical debt accumulated during the transition from `sd-interim-bayesian-merger` to the current `sd-mecha` based framework.

## Scope

- **Legacy Refactoring:** Identification and removal of outdated logic that contradicts the `sd-mecha` architecture.
- **Type Hinting:** Comprehensive implementation of Python type hints in core modules to improve IDE support and maintainability.
- **Configuration Standardization:** Aligning the Hydra configuration system with the project's modern standards for robustness and clarity.
- **Logging & Telemetry:** Implementing structured logging and basic performance metrics in core optimization loops.

## Target Modules

- `sd_optim/merger.py`
- `sd_optim/optimizer.py`
- `sd_optim/generator.py`
- `sd_optim/bayes_optimizer.py`

## Success Criteria

- No remaining references to legacy "interim-merger" specific hacks in core modules.
- 100% type hint coverage for function signatures in target modules.
- Hydra configuration validates successfully against updated schemas.
- Performance metrics are logged for the merge and scoring phases.
