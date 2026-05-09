# Track: post-reorg cleanup follow-up (2026-05-03)

## Status
- Open.
- Seeded from a 2026-05-03 audit of the current codebase after the structural repo reorg landed.
- Scope is cleanup and simplification of remaining dense or legacy surfaces, not another package-layout migration.

## Goals
- Finish the highest-value cleanup that still remains after `repo_reorg_p1_20260210`.
- Keep the completed repo reorg track as historical context while moving new work into a dedicated follow-up queue.
- Reduce maintenance risk in the largest or least-settled runtime areas without reopening stable entrypoint decisions.

## Predecessor
- Supersedes the remaining follow-up notes from `conductor/tracks/repo_reorg_p1_20260210/`.
- Use the archived repo reorg track for historical rationale and verification history.

## Current Focus
- Start with the largest remaining holdouts:
  - `sd_optim/bounds.py`
  - `sd_optim/utils/artifacts.py`
  - `sd_optim/core/optimizer_runtime.py`

## Tasks
- [x] Task: Audit the current post-reorg codebase and capture the remaining cleanup candidates worth tracking.
  - Note: Seeded from the 2026-05-03 repo review.
- [x] Task: Confirm which remaining compatibility surfaces are intentional external APIs versus removable internal leftovers.
  - Note: `sd_optim.bounds` is still an intentional active runtime surface. `ParameterHandler` / `BoundsInfo` are used directly by optimizer startup, merge recipe preparation, and bounds-focused regression tests; any split should preserve that import surface or update repo-internal imports atomically instead of adding new shims.
- [~] Task: Split or otherwise simplify `sd_optim/bounds.py` so `ParameterHandler` is easier to reason about without changing guide behavior.
  - Note: Start by carving along the existing seams: guide component/strategy expansion, custom-bounds override + summary logging, and dependency validation. Keep `ParameterHandler.validate_custom_bounds()` stable because recipe-mode fixed kwargs call it directly today.
  - Note: Added a forward-looking design note in `design_docs/bounds_graph_guide_plan.md` to anchor future guide simplification and node-UI authoring around selection, grouping, binding, and late `sd-mecha` compilation.
  - Note: Added `design_docs/bounds_graph_worked_example.md` to map the graph-first plan onto a real recipe-mode `.hydra` snapshot, emitted `.mecha` recipe, and reproducible Python artifact before changing runtime semantics.
- [x] Task: Add focused tests around bounds strategy processing, dependency mapping, and custom-bounds validation before any deeper structural split.
  - Note: Expanded the bounds test matrix to cover happy-path and defensive/error-path scenarios across strategy expansion, target-config resolution, conflict handling, dependency mapping, and custom-bounds validation. The bounds-focused suite now drives `sd_optim/bounds.py` to 100% coverage.
- [ ] Task: Revisit `sd_optim/utils/artifacts.py` and separate recipe rewrite, serialization, and artifact-export concerns where the boundaries are now clear.
  - Note: Fixed a concrete artifact naming bug in `sd_optim/merge/artifacts.py` so long truncated stems preserve the trailing `-it_<n>` marker instead of silently dropping the iteration number from saved `.mecha` and reproducible-script filenames.
- [~] Task: Reduce `sd_optim/core/optimizer_runtime.py` by extracting non-core orchestration helpers into smaller focused modules.
  - Note: Add an explicit config toggle for cross-run cached image reuse so interrupted runs can still save manifests/artifacts without automatically reusing prior scoring results on reruns.
  - Note: Added `reuse_cached_results` as a runtime/base-level gate for universal reuse scanning and per-trial cache hits, with focused tests covering the disabled path.
- [ ] Task: Re-check `sd_optim/merger.py` for any remaining helper delegation or orchestration that can move cleanly into `sd_optim/merge/*`.
- [ ] Task: Audit bundled experimental merge-method modules for dead commented blocks, duplicate helper logic, inconsistent naming, and any lingering lint or syntax problems.
- [ ] Task: Narrow broad `except Exception` handling where newer package boundaries make more specific error handling practical.
- [ ] Task: Improve consistency of logging and config access in older modules, especially where dynamic `DictConfig.get(...)` usage still obscures required settings.
- [x] Task: Replace legacy-surface regression coverage with tests that target the intended long-term package boundaries where safe.
  - Note: Added a focused recipe-facing regression test that traces guide expansion through bounds metadata, merge parameter node construction, final recipe rewrite text, and deserialized `sd-mecha` payloads for both sparse `select` and whole-component `single` key targeting.
- [~] Task: Build a clean-room graph-backed guide compiler against expected recipe artifacts instead of continuing only with incremental `ParameterHandler` reshaping.
  - Note: Use recorded `.hydra` snapshots plus emitted `.mecha` / reproducible Python artifacts as golden baselines for expected payload shape. Treat current bounds logic as a parity reference, not as the design center.
  - Note: Started a parallel typed implementation in `sd_optim/guide_graph.py` plus `tests/test_guide_graph.py`, with a reduced golden payload fixture distilled from the recorded `delta_widen` recipe artifact.
  - Note: `ParameterHandler.create_parameter_bounds_metadata()` now sources legacy-guide metadata from the graph-backed compiler while keeping the existing outer validation and summary shell, so the new path is active without immediately rewriting the rest of bounds/runtime.
  - Note: `name: null` in the current guide should be treated as an intentionally inactive placeholder when users want to keep a section around without participating in the current run, so skipping it is currently correct behavior rather than an adapter gap.
  - Note: Follow-up regressions from the first swap are now addressed: legacy `select` / `group` no-match cases warn and skip again instead of aborting guide compilation, and `prepare_param_recipe_args()` now materializes payloads from the supplied `param_info` contract instead of silently recompiling the guide from config.
- [ ] Task: Define and document the graph-native guide serialization shape, including an explicit inactive/disabled way to keep guide fragments around without participating in the current run.
- [ ] Task: Write a migration guide from the current text guide/legacy semantics to the graph-native guide model after the new serialization shape is settled.
- [ ] Task: Refresh track docs and indexes that still describe deleted modules or stale pre-package layouts.

## Exit Criteria
- [ ] The largest remaining holdout modules are smaller, better-factored, or have an explicit documented reason to remain large.
- [ ] Repo-internal callers no longer depend on obsolete legacy import surfaces.
- [ ] Logging, config access, and error-handling patterns are materially more consistent across touched modules.
- [ ] Focused tests pass for each cleanup slice and the conductor docs reflect the post-reorg steady state.
