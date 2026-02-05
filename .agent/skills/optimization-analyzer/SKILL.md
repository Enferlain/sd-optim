---
name: optimization-analyzer
description: Process and analyze optimization run results from sd-optim. Use this skill when the user asks for statistics, takeaways, parameter importance, or comparisons of optimization runs. It supports analyzing both .jsonl trial logs and Optuna .db storage files.
---

# Optimization Analyzer

This skill helps you extract meaningful insights and statistical takeaways from sd-optim optimization runs.

## When to Use This Skill

- When a user wants to know the "best" trial results for a specific run.
- When analyzing which parameters had the most impact on the optimization score.
- When comparing multiple runs to see which settings or strategies performed better.
- When looking for correlations between parameter values and the target score.

## Core Workflows

### 1. Basic Run Summary (JSONL or DB)

Extracts key metrics from a single run directory or database.

**Workflow**:

- Identify the source file (`.jsonl` in `logs/` or `.db` in `optuna_db/`).
- Use `scripts/analyze_jsonl.py` or `scripts/analyze_optuna.py`.
- **Takeaway**: Summarize the best trial, average score, and time efficiency.

### 2. Parameter Correlation & Importance

Identify which parameters are actually driving the score.

**Workflow**:

- Use `scripts/calculate_correlation.py <file>`.
- **Takeaway**: Highlight parameters with high absolute correlation coefficients. These are the "levers" that matter most.
- Use `scripts/analyze_optuna.py <db>` for Optuna's native parameter importance (requires `optuna` installed).

### 3. Cross-Run Comparison

Compare the efficiency and results of two or more runs.

**Workflow**:

- Gather file paths for the runs to compare.
- Use `scripts/compare_runs.py <file1> <file2> ...`.
- **Takeaway**:
  - Compare best, average, and standard deviation of scores.
  - If comparing two runs, the script highlights parameter differences in their respective best trials.

## Data Schema Reference

For details on the log formats and Optuna attributes, see [references/run_data_specs.md](references/run_data_specs.md).

## Tools and Scripts

- **scripts/analyze_jsonl.py**: Summary statistics for JSONL logs.
- **scripts/analyze_optuna.py**: Native Optuna analysis (importance, attributes).
- **scripts/calculate_correlation.py**: Pearson correlation analysis for parameters vs score.
- **scripts/compare_runs.py**: Statistical comparison and parameter diffing across multiple runs.
