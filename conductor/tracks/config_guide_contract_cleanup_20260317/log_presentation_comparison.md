# Guide/Bounds Log Presentation: Present vs Ideal (2026-03-18)

This note uses the latest observed run as a concrete example:

- Run dir: `logs/2026-03-17_23-18-31_weighted_sum_['manual']`
- Hydra snapshot: `logs/2026-03-17_23-18-31_weighted_sum_['manual']/.hydra/config.yaml`
- Runtime log: `logs/2026-03-17_23-18-31_weighted_sum_['manual']/sd_optim.log`

The goal is not to propose a new config artifact.
Hydra already captures the full composed config well.

The goal is to show a better way to present the already-derived guide/bounds information inside startup logging.

## What The Current Log Already Does Well

The current log already captures the key facts:

- It shows the guide processing pass.
- It shows skipped/invalid guide entries.
- It shows generated parameter count.
- It shows unmatched `custom_bounds`.
- It shows the full final parameter metadata dump.
- It shows the final flat optimizer bounds again before Optuna starts.

Relevant examples from the current run:

```text
[WARNING] Skipping component entry at index 1 due to missing 'name'.
[INFO] Generated metadata for 37 optimization parameters based on guide.
[DEBUG] Custom bound key 'k' did not match any generated optimizer parameter or base_param. It will not be optimized.
[DEBUG] Custom bound key 'min_agreement' did not match any generated optimizer parameter or base_param. It will not be optimized.
[INFO] --- Final 37 Optimization Parameter Details (Bounds Updated: 0) ---
[INFO] Prepared 37 parameters for the optimizer with specific bounds.
[DEBUG] Initial Parameter Bounds: {...37 entries...}
```

## What Feels Weak In Practice

The issue is mostly presentation, not missing information.

### 1. The guide/bounds story is split across multiple log locations

The user has to mentally stitch together:

- guide parsing
- invalid/ignored entries
- final expanded parameter metadata
- final flat optimizer bounds

These are all related, but they are not presented as one startup section.

### 2. The most useful warnings are easy to miss

Two important facts are present, but not emphasized:

- one guide component was skipped because `name` is missing
- two `custom_bounds` entries did not apply to anything

Those are good diagnostics, but they are mixed into a longer startup stream.

### 3. There is both too much detail and not enough summary

Current behavior jumps from:

- "Generated metadata for 37 optimization parameters"

to:

- full per-parameter dump for all 37 entries

without a small summary in between.

### 4. The same information is repeated in different shapes

The run logs:

- the rich metadata view in `sd_optim.bounds`
- then the flat `optimizer_pbounds` dict again in the Optuna study manager

That is useful for debugging internals, but noisy for normal run review.

## Best-Effort Ideal Presentation

Below is a proposed startup section using the same run data.
This is intentionally written as a single compact log block that a user can scan quickly.

```text
==================================================
Guide / Parameter Space Summary
==================================================
Guide file: conf/optimization_guide/guide.yaml
Mode: merge
Merge method: weighted_sum
Base config: sdxl-sgm
Custom block config: sdxl-optim_blocks_sub

Guide processing:
- Components read: 2
- Components used: 1
- Components skipped: 1
  - component[1]: missing 'name'

Strategy expansion:
- total generated parameters: 37
- target type breakdown:
  - block: 37
  - key: 0
- strategy breakdown:
  - all: 37
  - select: 0
  - group: 0
  - single: 0

Bounds:
- default bounds used: 37
- exact custom overrides applied: 0
- base-name custom overrides applied: 0
- fixed parameters: 0
- categorical parameters: 0
- continuous parameters: 37

Unused custom bounds:
- k
- min_agreement

Dependencies:
- mapped dependencies: 0

Optimizer handoff:
- parameters prepared for optimizer: 37
- sampler sees:
  - fixed: 0
  - categorical: 0
  - continuous: 37
- full parameter list: available at DEBUG
==================================================
```

## Why This Version Feels Better

### 1. It keeps the high-signal information together

The user can answer, at a glance:

- what guide was used
- what got skipped
- how many parameters were created
- whether custom bounds applied
- whether dependencies were mapped
- what kind of search space the sampler is actually receiving

### 2. It still preserves detail without flooding the log

The full per-parameter metadata dump should still exist at `DEBUG`.
The startup summary should stay compact and readable at `INFO`.

### 3. It separates "problem signals" from normal detail

These deserve dedicated visibility:

- skipped guide entries
- unused `custom_bounds`
- dependency mapping count

They are not fatal, but they are worth seeing immediately.

## Proposed Logging Levels

### `INFO`

Keep:

- one compact "Guide / Parameter Space Summary" block
- counts and top-level breakdowns
- list of skipped guide entries
- list of unused `custom_bounds`
- a note that the full itemized parameter list is available at `DEBUG`

### `DEBUG`

Keep:

- full rich parameter metadata dump
- full `optimizer_pbounds` dict
- full itemized generated parameter list
- per-strategy processing trace
- individual override application details

## Minimal Implementation Direction

If implemented later, the smallest useful change would be:

1. Gather the existing counts and unmatched entries inside `ParameterHandler.get_bounds(...)`.
2. Return or store a small summary structure alongside `params_info`.
3. Print a single startup summary block from `Optimizer.setup_parameter_space()`.
4. Leave the full current detailed logs available at `DEBUG`.

## Recommendation

Do not replace the current detailed logging.

Instead:

- keep the detailed lines for deep debugging
- add one concise, human-scannable summary block near startup

That gives a clear "present vs ideal" path without losing any current diagnostic power.
