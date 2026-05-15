## Graph Runtime Follow-Up After `bounds.py`

### Purpose

Capture the concrete runtime work still needed now that the graph-native guide semantics are settled enough to stop treating `sd_optim.bounds` as the design center.

This note assumes the intended split is:

- `sd_optim/bounds.py`: legacy guide adapter and compatibility shell
- graph-native runtime: direct consumers of compiled graph bindings

It does **not** assume we should migrate graph behavior through `BoundsInfo` or port legacy `custom_bounds` into the graph path.

### Current state

The graph-backed pieces that already exist are:

- [sd_optim/guide_compiler.py](/D:/Projects/sd-optim/sd_optim/guide_compiler.py)
  - canonical `TargetSource`, `BindingSpec`, `CompiledBinding`
  - `compile_bindings(...)`
  - `build_optimizer_bounds(...)`
  - `materialize_recipe_payloads(...)`
- [sd_optim/guide_nodes.py](/D:/Projects/sd-optim/sd_optim/guide_nodes.py)
  - graph-native authored shape -> compiled bindings
- [sd_optim/guide_legacy.py](/D:/Projects/sd-optim/sd_optim/guide_legacy.py)
  - legacy guide -> graph-backed bindings / payloads / bounds metadata

The main runtime now has a direct graph path for startup, optimizer proposals, merge payloads, recipe payloads, and Optuna dependency mapping. The legacy callers still go through `ParameterHandler` and `BoundsInfo` when no `optimization_guide.graph` is configured:

- [sd_optim/core/optimizer_base.py](/D:/Projects/sd-optim/sd_optim/core/optimizer_base.py)
  - instantiates `ParameterHandler`
  - calls `get_bounds(...)`
- [sd_optim/optimizers/optuna/study_manager.py](/D:/Projects/sd-optim/sd_optim/optimizers/optuna/study_manager.py)
  - maps graph dependencies from compiled binding scopes
  - calls `bounds_initializer.validate_dependencies(...)` only for legacy runtime data
- [sd_optim/merge/recipe_builder.py](/D:/Projects/sd-optim/sd_optim/merge/recipe_builder.py)
  - consumes `GraphRuntimeBundle` directly in graph mode
  - calls `materialize_payloads_from_legacy_bounds_info(...)` only in legacy mode
  - still calls `ParameterHandler.validate_custom_bounds(...)` only for legacy fixed kwarg behavior
- [sd_optim/merger.py](/D:/Projects/sd-optim/sd_optim/merger.py)
  - merge / recipe entrypoints still accept `BoundsInfo`
- [sd_optim/merge/recipe_optimization.py](/D:/Projects/sd-optim/sd_optim/merge/recipe_optimization.py)
  - still takes `BoundsInfo`

In other words: the graph compiler is now a runtime path, while the legacy path remains active for compatibility.

### What the graph runtime still needs

#### 1. A real graph runtime contract

The missing centerpiece is a first-class runtime object for the graph path.

Today we have:

- compiled bindings
- optimizer bounds map
- payload materialization helper

But no single runtime contract that startup, samplers, and recipe building can share.

Recommended shape:

- add a graph runtime bundle dataclass, likely in a new module such as `sd_optim/guide_runtime.py`
- bundle at least:
  - `compiled_bindings: list[CompiledBinding]`
  - `optimizer_bounds: dict[str, BoundValue]`
  - optional summarized metadata for logging/debugging

Important: this bundle should stay graph-native. It should not be another spelling of `BoundsInfo`.

#### 2. Direct startup wiring for graph-authored guides

`Optimizer.__post_init__()` still assumes the guide path is:

- config
- `ParameterHandler`
- `param_info`
- `optimizer_pbounds`

The graph path needs its own startup branch:

- load authored graph guide
- compile it via `guide_nodes`
- derive optimizer bounds via `build_optimizer_bounds(...)`
- store the resulting graph runtime bundle directly on the optimizer

That means [sd_optim/core/optimizer_base.py](/D:/Projects/sd-optim/sd_optim/core/optimizer_base.py) needs a clean branch point between:

- legacy text guide path
- graph-native guide path

The graph path should not call `ParameterHandler.get_bounds(...)`.

#### 3. Direct payload materialization in merge and recipe flows

The graph compiler already knows how to turn sampled values into method-param payload dicts:

- `materialize_recipe_payloads(sampled_values, compiled_bindings)`

But the merge stack still expects legacy metadata and reconstructs payloads from `BoundsInfo`.

Needed follow-up:

- add a graph-native recipe prep helper that consumes compiled bindings directly
- split materialized payloads by target space (`block` vs `key`) without routing through legacy metadata
- update:
  - [sd_optim/merge/recipe_builder.py](/D:/Projects/sd-optim/sd_optim/merge/recipe_builder.py)
  - [sd_optim/merger.py](/D:/Projects/sd-optim/sd_optim/merger.py)
  - [sd_optim/merge/recipe_optimization.py](/D:/Projects/sd-optim/sd_optim/merge/recipe_optimization.py)

The graph runtime should be able to feed merge and recipe optimization directly from compiled bindings.

#### 4. An explicit dependency decision

Legacy-authored dependencies are still configured through:

- `cfg.optimization_guide.dependencies`

Graph mode now maps that same dependency config through compiled graph binding scopes in `sd_optim/guide_runtime.py`, so dependency behavior can run without routing through `BoundsInfo` or `ParameterHandler.validate_dependencies(...)`.

The remaining product decision is whether the authored graph eventually gets a first-class dependency node or keeps this transitional config surface:

1. keep `optimization_guide.dependencies` as the shared transitional surface
2. add a graph-native authored dependency form
3. remove dependency support from graph mode if the concept does not fit the newer model

What should **not** happen is silently reusing `BoundsInfo` dependency mapping as the graph contract.

The runtime fallback is gone; only the authored shape is still unsettled.

#### 5. A graph loading surface in config/runtime

There is currently no real runtime loading path for an authored graph guide.

The live config still points at the legacy guide surface under `optimization_guide`.

Needed work:

- decide where graph-authored guide data lives at runtime
- decide how the run chooses between legacy and graph authoring
- keep that choice explicit in config and logs

The important rule is that graph mode should be a real caller path, not just test-only helpers.

#### 6. Logging and summaries that do not depend on legacy strategy counts

`bounds.py` still provides a lot of summary output around:

- strategy counts
- target types
- default bounds usage
- custom bounds overrides
- skipped legacy components

The graph path needs its own lighter summary:

- number of sources
- number of builds
- number of authored branches
- number of compiled optimizer params
- target-space counts
- bounds-shape counts

This should be graph-fluent logging, not legacy strategy narration recycled under new code.

### Things that should stay legacy-only

These should remain in `bounds.py` unless or until the legacy guide is removed:

- `custom_bounds`
- base-name custom-bounds override behavior
- `BoundsInfo` as a public shape for old callers
- legacy dependency mapping based on legacy-generated parameter names

Graph runtime work should not try to port those concepts forward just to preserve the old shell.

### Recommended implementation order

#### Phase 1: Introduce a graph runtime bundle

- add a graph-native runtime dataclass/module
- expose:
  - compile graph guide
  - extract optimizer bounds
  - summarize graph parameter space

Deliverable:

- tests proving graph-authored guide -> runtime bundle -> optimizer bounds

Status:

- Done. [sd_optim/guide_runtime.py](/D:/Projects/sd-optim/sd_optim/guide_runtime.py) now provides `GraphRuntimeBundle`, summary metadata, and direct graph-authored optimizer-bounds extraction with focused tests in [test_guide_runtime.py](/D:/Projects/sd-optim/tests/test_guide_runtime.py).

#### Phase 2: Teach recipe building to consume compiled bindings directly

- add graph-native payload materialization helper(s)
- update recipe builder / merger entrypoints for graph runtime data
- keep legacy path untouched in parallel

Deliverable:

- focused tests proving graph-authored guide -> merge payloads / recipe payloads

Status:

- Done for the recipe stack. [recipe_builder.py](/D:/Projects/sd-optim/sd_optim/merge/recipe_builder.py), [merger.py](/D:/Projects/sd-optim/sd_optim/merger.py), and [recipe_optimization.py](/D:/Projects/sd-optim/sd_optim/merge/recipe_optimization.py) now accept `GraphRuntimeBundle` directly and materialize payloads from compiled bindings without routing through legacy `BoundsInfo`.
- Graph-generated optimizer params now have explicit regression coverage through the merge trial loop into `Merger.merge()`, and graph-authored method parameter names are checked against the selected merge method before sd-mecha recipe construction.

#### Phase 3: Wire optimizer startup to choose graph path directly

- branch startup on guide authoring mode
- graph mode bypasses `ParameterHandler`
- optimizer stores graph runtime bundle instead of legacy `BoundsInfo`

Deliverable:

- startup/integration tests for graph-mode parameter-space setup

Status:

- Partly done. [optimizer_base.py](/D:/Projects/sd-optim/sd_optim/core/optimizer_base.py) now recognizes `optimization_guide.graph` as an explicit graph-authored runtime path, builds a graph runtime bundle directly, and passes it through trial execution. The graph load surface is still intentionally minimal and does not yet have polished user-facing config examples/templates.

#### Phase 4: Resolve dependency policy

- either:
  - explicitly disable dependencies in graph mode with a clear error/message, or
  - implement graph-native dependency authoring/compilation

Deliverable:

- no silent dependency fallback through legacy naming rules

Status:

- Runtime mapping is in place. [sd_optim/guide_runtime.py](/D:/Projects/sd-optim/sd_optim/guide_runtime.py) maps `optimization_guide.dependencies` from compiled graph binding scopes, and [study_manager.py](/D:/Projects/sd-optim/sd_optim/optimizers/optuna/study_manager.py) uses that graph-native mapper instead of `ParameterHandler.validate_dependencies(...)` when running from a `GraphRuntimeBundle`. A first-class authored dependency node is still open.

#### Phase 5: Reduce repo-internal reliance on `bounds.py`

- once graph callers are direct, leave `bounds.py` as legacy-only
- stop adding new runtime responsibilities there

Deliverable:

- new graph work no longer edits `ParameterHandler` unless legacy behavior changes

### Non-goals for this slice

- removing `bounds.py`
- porting `custom_bounds` into graph mode
- making one graph `group` node produce multiple grouped values
- forcing the legacy guide to compile through new runtime wrappers

### Practical next task

The remaining useful implementation slice is:

1. decide whether graph dependencies stay in the transitional config surface or become authored graph nodes
2. add polished graph guide config examples/templates
3. keep `custom_bounds` legacy-only and express graph domains with domain nodes

That keeps graph runtime behavior direct without dragging legacy `custom_bounds` or `BoundsInfo` deeper into the new system.
