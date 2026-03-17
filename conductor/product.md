# Product Definition - sd-optim

## Initial Concept
sd-optim is an opinionated framework for optimizing state dictionary operations, primarily focused on Stable Diffusion model merging. It leverages `sd-mecha` as a backend and employs advanced optimization techniques like Optuna and Bayesian Optimization to find ideal parameters based on image scoring feedback.

## Target Audience
The framework is designed for:
- **Stable Diffusion Model Creators:** Who need to find precise merge parameters to enhance model performance.
- **Researchers:** Experimenting with state dictionary optimization and hyperparameter tuning in generative models.

## Core Goals
- **Automated Parameter Discovery:** Streamline the process of finding optimal merge ratios and parameters to maximize output quality.
- **Modernization & Maintenance:** Transition the codebase to a robust, maintainable state by refactoring legacy logic and adopting modern Python patterns.

## Priority Features & Modernization
- **Legacy Refactoring:** Aligning internal logic with the latest `sd-mecha` architecture and purging remnants of the interim merger project.
- **Asynchronous Optimization:** Migrating performance-critical sections to asynchronous patterns to improve throughput during generation and scoring.
- **Robust Maintainability:** Implementing comprehensive type hinting and clear documentation to facilitate long-term development.
- **System Cleanup:** 
    - Standardizing the Hydra/YAML configuration system for a better user experience.
    - Enhancing logging and telemetry to provide clear insights into the optimization process.
