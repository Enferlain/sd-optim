# Repo Reorg Discovery Notes (2026-03-15)

## Stable External Surfaces to Preserve

- Root entrypoints:
  - `sd_optim.py`
  - `execute_recipes.py`
  - `scripts/api.py`
- Package exports:
  - `sd_optim.OptunaOptimizer`
  - `sd_optim.BayesOptimizer`
- Config/runtime paths expected by current code:
  - `conf/`
  - `sd_optim/extensions/bundled/model_configs/`
  - `scripts/api.py` is referenced by `sd_optim/gen_adapters.py`

## Current Complexity Hotspots

- `sd_optim/merge_methods.py`: 7544 lines
- `sd_optim/utils.py`: 1769 lines
- `sd_optim/optuna_optimizer.py`: 1283 lines
- `sd_optim/scorer.py`: 1062 lines
- `sd_optim/optimizer.py`: 1032 lines
- Root scripts still carry runtime logic:
  - `sd_optim.py`: 271 lines
  - `execute_recipes.py`: 232 lines
  - `scripts/api.py`: 261 lines

## Import and Dependency Constraints

- `README.md` still documents `uv run python sd_optim.py`, so the root script must remain callable.
- Tests import the root script file directly in `tests/test_entry_optional_bayes_import.py`.
- Runtime code imports package-level lazy exports from `sd_optim/__init__.py`.
- `sd_optim.utils` is imported widely and remains a central dependency, so deeper package moves need careful sequencing.

## Obvious Reorg Candidates

- Manual or local-only artifacts at the repo root:
  - `TODO.md`
  - `UV_SETUP.md`
  - `uv_torch_guide.md`
  - `implementation_plan_dx_improvements.md`
  - `sd-optim.wiki.7z`
- Experiment-style content mixed into the package:
  - image files under `sd_optim/models/`
  - `test_*.py` helpers under `sd_optim/models/`
  - `sd_optim/random_scripts.7z`
- Analysis output already lives outside the package in `analysis_2026210/`, which is a good pattern to continue.

## Completed In This Slice

- Moved scorer scratch scripts from `sd_optim/models/test_*.py` to `tools/scorer_debug/`.
- Moved sample scorer images from `sd_optim/models/*.{png,jpg}` to `assets/test_images/scorers/`.
- Moved `sd_optim/random_scripts.7z` to `tools/archive/random_scripts.7z`.
- Moved bundled scorer modules under `sd_optim/extensions/bundled/scorers/`.
- Split bundled scorer registry/config data into `sd_optim/extensions/bundled/scorers/registry.py` and switched scorer class lookup to lazy resolution.
- Moved concrete bundled scorer implementations under `sd_optim/extensions/bundled/scorers/models/` so registry/package files stay separate from model code.
- Split scorer support logic further into `sd_optim/extensions/bundled/scorers/assets.py` and `sd_optim/extensions/bundled/scorers/loading.py`.
- Split manual scorer interaction helpers further into `sd_optim/extensions/bundled/scorers/interaction.py`.
- Moved scorer runtime/support code into `sd_optim/scoring/` so bundled scorers now only contain shipped implementation modules.
- Updated manual scoring runtime to save stable preview images under the scorer image directory and open them through the normalized platform opener path.
- Moved bundled model configs under `sd_optim/extensions/bundled/model_configs/`.
- Made `sd_optim/extensions/bundled/merge_methods/` the explicit home for bundled merge-method helper modules and shared SVD utilities.
- Split merge runtime helper concerns into `sd_optim/merge/`, with fallback handling, artifact writing, and recipe-building logic moved out of `sd_optim/merger.py` while keeping `Merger` as the stable orchestration entrypoint.
- Split merge model-selection and execution logic further into `sd_optim/merge/model_selection.py` and `sd_optim/merge/execution.py`, leaving `Merger` mainly as the orchestration shell for merge- and recipe-mode flows.
- Split model-node creation into `sd_optim/merge/model_nodes.py` and added focused characterization tests for relative-path recipe node generation.
- Updated runtime default paths and dynamic converter imports for the `extensions/bundled` layout.
- Restored the optimizer-facing `sd_optim/trial_scorer_summary.py` module so startup/import paths match the current optimizer runtime and tests.
- Tightened optimizer reuse/cache identity so runs with different merge methods or generation setups no longer share the same image hash by accident.
- Fixed merge call arity handling so tensor-valued merge parameters like `alpha` no longer get counted as extra positional model inputs.
- Restored fail-fast trial handling by default so real optimizer crashes stop the run unless `fail_on_error: false` is explicitly set.

## Future Cleanup Notes

- Archived on 2026-05-03 as historical discovery context. Active follow-up work now lives in `conductor/tracks/post_reorg_cleanup_20260503/plan.md`.
- `sd_optim/utils/` still contains a major dependency knot across `artifacts.py`, `recipes.py`, and adjacent helpers, so it remains one of the biggest long-term split candidates.
- `sd_optim/merger.py` is cleaner now, but it still contains orchestration plus some legacy helper delegation. A future pass can likely reduce it to a thin runtime shell once `_create_model_nodes`, recipe-mode orchestration, and `layer_adjust()` find better homes.
- The moved bundled merge-method modules still contain a lot of legacy shape:
  - large commented-out blocks
  - inconsistent helper naming and parameter typing
  - experimental modules with very dense logic and limited characterization coverage
- Several runtime areas still rely on broad `except Exception` handling from older code. Some of those are appropriate around external tools, but many could be narrowed now that the package boundaries are improving.
- Logging style is improved but still inconsistent across older modules:
  - some places still use f-string logging instead of parameterized logger calls
  - a few messages are very conversational or debug-heavy for normal runtime paths
- Some modules still carry old inline “version history” comments or patch-layer notes at the top of files. Those were useful during churn, but they now add noise compared with changelog/track history.
- Config access is still very dynamic in places (`DictConfig.get(...)` everywhere, implicit defaults, mixed string/int coercion). A future modernization pass could introduce clearer config normalization or typed runtime settings objects.
- There is still duplicate or near-duplicate helper logic across bundled merge-method files, especially around SVD, tensor reshaping, and fallback math. Once behavior is stable, shared helper extraction would reduce maintenance risk.
- A number of tests still reflect legacy surfaces rather than intentional long-term APIs. They’re useful as safety rails for refactoring, but some should eventually be replaced with tests that target the new package boundaries directly.

## Recommended First Moves

1. Leave `scripts/api.py` in place for now because WebUI extension installation expects that path.
2. Defer physical moves of local artifacts until a `tools/` or `assets/` destination is agreed and documented.
3. Prefer direct cutovers for real module moves rather than temporary wrapper layers unless an external integration forces them.
4. Add characterization tests around import paths and runtime contracts before splitting larger modules.

## Decision

- Keep the `sd_optim` package name stable.
- Avoid temporary compatibility layers unless a specific external integration requires one.
