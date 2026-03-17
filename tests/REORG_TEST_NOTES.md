# Reorg Test Notes

These are follow-up tests to add after the module split is complete.

## Already Covered Now

- Bounds format contract (`list` categorical, `tuple` continuous, scalar fixed, rich dict).
- Optuna sampler wiring for TPE/CMA-ES.
- Grid sampler validation guard.
- Known issues documented as `xfail`:
  - QMC keyword typo (`warn_asyncronous_seeding`).
  - pruning flag path mismatch in `validate_optimizer_config`.

## Add After Reorganization Lands

1. Import compatibility tests
- `sd_optim.optimizers.optuna.optimizer.OptunaOptimizer` is the real runtime class path.
- New split modules (`sd_optim/optimizers/optuna/*`) expose expected public API.

2. Resume/fork integration tests (storage-backed)
- resume keeps existing sampler state.
- fork enqueues parent params correctly.
- scorer mismatch blocks resume with clear error.

3. Callback/logging contract
- trial callback JSONL schema unchanged (`trial_number`, `target`, `params`, `state`, `datetime`, `scorer_results`).
- log replay behavior for resumed studies still deterministic.

4. Objective/suggestion contract
- dependency-driven defaulting behavior unchanged.
- categorical-heavy guide path behaves identically before/after split.

5. End-to-end smoke
- mocked merge/generate/score pipeline can run 5-10 trials through `optimize()`.
- no regressions in best-trial persistence and run artifact paths.

6. Sampler policy/perf check
- short A/B run (`tpe` vs `cmaes`) with fixed seed and same guide.
- compare best score, top-10% average, and runtime/trial.
