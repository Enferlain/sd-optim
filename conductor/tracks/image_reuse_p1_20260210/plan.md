# Deterministic Image Reuse & Incremental Rescoring (Plan)

## Desired Outcome

1. Never merge+generate when doing so would produce an image that is 100% identical to one already generated previously.
2. When scorer configuration changes, never redo scoring work that is still valid:
   - If only a new scorer is added, compute only that scorer for cached identical images.
   - If an existing scorer's settings change, recompute only that scorer.
   - If only aggregation changes (weights/average type), recompute only the combined score.
3. This should work across runs (not just within a single Hydra output dir), using `logs/**` as a global reuse pool.

## Definitions (Make These Explicit in Code)

### `image_fp` (Image Fingerprint)

The key representing "this run would produce the exact same pixels".

Rule: `image_fp` must be based on the *effective* merge inputs and the *effective* generation request sent to the backend.
It must NOT include optimizer configuration (sampler type, `n_iters`, etc.) because those only affect how we *arrive* at a trial, not the trial's output.

### `scorer_fp[name]` (Per-Scorer Fingerprint)

The key representing "this scorer would produce the same numeric result for this image".

Rule: `scorer_fp` must include only output-affecting configuration for that scorer:

- thresholds/knobs (e.g. `hybridnoise_*`, `pcascorer_*`)
- model file identity (path + size + mtime, or hash if affordable)
- scorer source identity (optional but recommended: git commit SHA or module file mtime)

It should NOT include non-output-affecting settings like device selection (CPU vs CUDA) unless you observe numerical drift that matters.

### `agg_fp` (Aggregation Fingerprint)

The key representing "the combined score computation is the same".

Rule: `agg_fp` must include:

- `scorer_average_type`
- `scorer_weight` mapping (effective defaults applied)
- the ordered `scorer_method` list if ordering is significant

If `agg_fp` changes, we should recompute `combined` from cached per-scorer values without running any models.

## Data Model (Manifest Schema v2)

### Store in `run_manifest.json` per `image_fp`:

```json
{
  "<image_fp>": {
    "path": "imgs/...",
    "gen": {
      "gen_fp": "<...>",         // optional for debugging
      "merge_fp": "<...>",       // optional for debugging
      "meta": { ... }            // optional: resolved seed, model id, etc.
    },
    "scores": {
      "<scorer_name>": 7.123,
      "...": ...
    },
    "scorer_fp": {
      "<scorer_name>": "<hash>",
      "...": "..."
    },
    "agg_fp": "<hash>",
    "combined": 6.55
  }
}
```

### Backward Compatibility (Manifest v1)

Current manifests look like:

```json
{ "<sd_optim_hash>": { "path": ..., "scores": {...}, "final_score": ... } }
```

Plan:

- Continue to load v1 entries as a fallback.
- Add v2 fields going forward.
- Prefer v2 for reuse decisions.
- Allow a one-time migration utility that reads v1 manifests and rewrites them to v2 when possible.

## Where to Compute Fingerprints (Most Intuitive Design)

### Generation Fingerprints Live in Adapters

Adapters know the real request that is sent to the backend.

Add to `sd_optim/gen_adapters.py`:

- `BackendAdapter.fingerprint_generation(payload) -> (gen_fp: str, debug: dict)`
- For Comfy: fingerprint the final workflow dict *after* `_inject_parameters()` (and any other client-side mutations like websocket save-node swapping). This mutated workflow is built locally before the POST to Comfy, so no Comfy response is needed to compute the fingerprint.
- For A1111: fingerprint the exact JSON request body sent to `/sdapi/v1/txt2img` (after enforced `batch_size=1`, etc.)

### Merge Fingerprint Lives in Merger

Merger knows what affects checkpoint output.

Add to `sd_optim/merger.py` (or a helper):

- `Merger.fingerprint_merge(params, param_info) -> (merge_fp: str, debug: dict)`
- Include:
  - merge method identifier
  - base model identities (paths + stat info, or hash if feasible)
  - any merge config that changes output: dtypes, add_extra_keys, fallback/base model selection
  - optimization_mode ("merge"/"recipe"/"layer_adjust") + recipe path or recipe content hash if relevant
  - merge params (the proposed `params`, normalized)

### Image Fingerprint Composition

Compute in `sd_optim/optimizer.py`:

- `image_fp = sha256(json({merge_fp, gen_fp}))`

This replaces (or versions) `calculate_image_hash()` currently in `sd_optim/optimizer.py:295`.

## Reuse Algorithm (Default Behavior)

### Step 1: Identify Image Candidate

For each payload in a trial:

1. Compute `(merge_fp, gen_fp)` and `image_fp`
2. Look up `history_cache[image_fp]`
3. If found and image exists on disk: skip merge+gen for this payload and load the PNG for scoring

### Step 2: Incremental Scoring on Cached Images

Given cached entry with `scores`, `scorer_fp`, `agg_fp`:

1. Compute current `scorer_fp_current[name]` for each scorer in `cfg.scorer_method`
2. Determine `to_score`:
   - scorer missing from cached `scores`
   - or cached `scorer_fp[name] != scorer_fp_current[name]`
3. If `to_score` is empty and only `agg_fp` differs:
   - recompute `combined` from existing `scores`
4. If `to_score` is non-empty:
   - run only those scorers on the cached image
   - update `scores[name]` and `scorer_fp[name]`
   - recompute `combined`

### Step 3: Save Back

Always update the current run's manifest with the new fields so future runs reuse more effectively.

## Scorer Implementation Changes (Needed for "Only Affected Parts")

### Add Subset Scoring API

Modify `sd_optim/scorer.py` to support:

- `AestheticScorer.score_subset(image, prompt, scorers: list[str], name: str | None) -> dict[str, float]`

Rules:

- It must not run scorers outside `scorers`.
- It must update `last_scorer_results` for those scorers only.

Then keep `score()` as a convenience wrapper for `score_subset(..., scorers=cfg.scorer_method)` returning only `combined`.

### Combined Score Computation from Partial Results

Add a small helper:

- `AestheticScorer.combine_scores(scores_by_scorer: dict[str, float]) -> float`

This should implement the configured `scorer_average_type` + weights deterministically.

## Global Cache: Loading and Matching

### Load

Keep scanning `logs/**/run_manifest.json` but:

- Support both schemas (v1 and v2)
- Prefer v2 when present

### Match

Switch the match key from `sd_optim_hash` to `image_fp` (v2).

Keep v1 matching only as a fallback if you want a transitional period.

## Optuna / Study Interaction (Keep It Simple)

Changing scoring changes the objective function.

Instead of "resume same study with changed scorers":

- Default behavior: start a new study (or auto-fork) when `objective_fp` changes.

Define `objective_fp = hash({scorer_method, scorer_fp_all, agg_fp})`.

When user wants to continue "the same experiment with new scoring":

- Fork study (enqueue same params) and let the objective be recomputed.
- With image reuse enabled, recomputation avoids merge+gen and only rescoring parts that changed.

This preserves Optuna invariants (trials in one study share the same objective definition).

## Practical Phases (Implementation Order)

### Phase A: Make Image Identity Correct

1. Implement `gen_fp` in adapters (Comfy: hash post-injection workflow)
2. Implement `merge_fp` in merger
3. Replace `sd_optim_hash` usage with `image_fp` (versioned)
4. Write v2 manifest entries
5. Keep reading v1 for legacy reuse

### Phase B: Incremental Rescoring

1. Add scorer fingerprint functions
2. Store `scorer_fp` and `agg_fp` in manifest
3. Add `score_subset` and `combine_scores`
4. Update partial-hit path to only score missing/changed scorers

### Phase C: Optuna Behavior on Objective Change

1. Store `objective_fp` in study attrs
2. On resume, if objective differs: auto-fork or fail with a clear message

## Tests (TDD Targets)

Add tests under `tests/` that cover:

- `image_fp` stability: same inputs => same hash; different effective generation setting => different hash
- Comfy `gen_fp` changes when injected fields change (seed/steps/cfg/prompt)
- Incremental rescoring: adding a scorer only runs that scorer; changing weight only recomputes combined
- Manifest v1 read compatibility and v2 write correctness

## Notes / Non-Goals

- This plan assumes deterministic generation for a fixed effective request.
- If you later want "pixel-hash" deduplication across different recipes, that can be added as a separate layer, but it is not needed for the stated goal.
