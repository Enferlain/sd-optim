# Bounds Graph Worked Example

## Purpose

This note grounds the graph-first bounds/guide plan in a real recorded run from
this repo.

It is meant to answer:

- what the current guide actually expressed
- what the resolved Hydra config preserved
- what the rewritten recipe actually looked like
- how that maps to a future graph-oriented mental model

The goal is not to bless the current guide shape as ideal. The goal is to use a
real deterministic artifact trail as the baseline for future changes.

## Source artifacts

This worked example is based on:

- Current guide surface:
  [guide.yaml](/D:/Projects/sd-optim/conf/optimization_guide/guide.yaml:1)
- Recorded resolved recipe-mode config:
  [config.yaml](/D:/Projects/sd-optim/logs/2026-03-12_09-08-29_delta_widen_%5B'cityaes',%20'textureclean',%20'backgroundblackness',%20'pcascorer',%20'hybridnoise',%20'textureclean_fullimg',%20'hybridnoise_fullimg'%5D/.hydra/config.yaml:1)
- First emitted recipe artifact from that run:
  [it_0.mecha](</D:/Projects/sd-optim/logs/2026-03-12_09-08-29_delta_widen_['cityaes', 'textureclean', 'backgroundblackness', 'pcascorer', 'hybridnoise', 'textureclean_fullimg', 'hybridnoise_fullimg']/recipes/pop3b7vae-NoobAI-RF-v0.2-Base-pop_lora-it_142_merged-pop3b7vae-delta_widen-it_0.mecha:1>)
- Reproducible Python artifact for the same recipe:
  [it_0_run.py](</D:/Projects/sd-optim/logs/2026-03-12_09-08-29_delta_widen_['cityaes', 'textureclean', 'backgroundblackness', 'pcascorer', 'hybridnoise', 'textureclean_fullimg', 'hybridnoise_fullimg']/merge_artifacts/pop3b7vae-NoobAI-RF-v0.2-Base-pop_lora-it_142_merged-pop3b7vae-delta_widen-it_0_run.py:1>)

## What the run was doing

This run was in `recipe` optimization mode, not plain `merge` mode.

The target method was `delta_widen`, and the resolved run config targeted
these recipe kwargs:

- `critical_quantile`
- `magnitude_ratio`
- `direction_ratio`
- `temperature`

That is visible in
[recipe_optimization](/D:/Projects/sd-optim/logs/2026-03-12_09-08-29_delta_widen_%5B'cityaes',%20'textureclean',%20'backgroundblackness',%20'pcascorer',%20'hybridnoise',%20'textureclean_fullimg',%20'hybridnoise_fullimg'%5D/.hydra/config.yaml:69).

The same run also used the standard optimization guide snapshot under
[optimization_guide](/D:/Projects/sd-optim/logs/2026-03-12_09-08-29_delta_widen_%5B'cityaes',%20'textureclean',%20'backgroundblackness',%20'pcascorer',%20'hybridnoise',%20'textureclean_fullimg',%20'hybridnoise_fullimg'%5D/.hydra/config.yaml:181).

## Current guide intent

The resolved guide had two main sections:

1. A `block`-targeted `all` strategy over `unet` for:
   - `magnitude_ratio`
   - `direction_ratio`
   - `temperature`
   - `critical_quantile`

2. A `key`-targeted `group` strategy under `name: null` for:
   - `alpha`
   - `rank_ratio`

The important caveat is that the `name: null` section should not be read here
as an active global key-scope source. In current repo usage it is an
intentionally inactive placeholder for keeping a guide fragment around without
participating in the current run.

So for this run, the effective active targeting was the `unet` block section.

The broader guide format still allows mixing target spaces, but this specific
run was not actively exercising that parked `null` section.

That means the guide surface is capable of expressing at least these two kinds
of targeting:

- preset block handles from `sdxl-optim_blocks_sub`
- wildcarded state-dict key groups

That matches the broader repo reality: the guide is not only about bounds. It
is already acting as a targeting-and-binding layer on top of `sd-mecha`, even
if some sections are deliberately kept inactive.

## Important current behavior

The resolved guide for the `block` section listed only a small active subset of
sub-block handles:

- `UNET_IN05_1`
- `UNET_IN07_1`
- `UNET_IN08_1`
- `UNET_OUT01_1`
- `UNET_OUT03_1`
- `UNET_OUT04_1`
- `UNET_OUT05_1`

However, the first rewritten recipe emitted full block-space dictionaries for
each optimized kwarg, not sparse dictionaries for only those seven handles.

See the first four `dict ...` payloads in
[it_0.mecha](</D:/Projects/sd-optim/logs/2026-03-12_09-08-29_delta_widen_['cityaes', 'textureclean', 'backgroundblackness', 'pcascorer', 'hybridnoise', 'textureclean_fullimg', 'hybridnoise_fullimg']/recipes/pop3b7vae-NoobAI-RF-v0.2-Base-pop_lora-it_142_merged-pop3b7vae-delta_widen-it_0.mecha:50>).

That lines up with current `ParameterHandler` behavior:

- `_process_all_strategy()` iterates all discovered items in the component
- it does not consult `strategy_config["keys"]`

See [_process_all_strategy()](/D:/Projects/sd-optim/sd_optim/bounds.py:314).

So the current runtime behavior is:

- `all` means the full discovered component item set
- any `keys` list under `all` is currently ignored

That is a compatibility fact, even if we later decide to change it.

## What the recipe actually contained

The emitted recipe shows the runtime pipeline clearly:

1. Base and candidate models were loaded.
2. Candidate models were converted into deltas with repeated `subtract`.
3. Each optimized method kwarg got its own block-space `dict`.
4. Each block-space `dict` was wrapped as a literal in `sdxl-optim_blocks_sub`.
5. Each literal was converted into `sdxl-sgm` space with
   `convert_sdxl_optim_blocks_sub_to_sdxl_sgm`.
6. Those converted literals were plugged into `delta_widen` as named kwargs.
7. The final delta result was re-applied to the base model with
   `add_difference`.

The important point is that the guide did not directly describe a recipe.

It described enough targeting and parameter-binding intent for the runtime to
compile recipe payloads later.

## What this means conceptually

The current system is easiest to understand as:

```text
model set
  -> component target universe
  -> target handles (blocks or keys)
  -> grouping/binding rules
  -> optimizer-visible parameters
  -> concrete recipe payload dicts
  -> mecha recipe nodes
```

That reinforces the graph-first plan:

- source/discovery comes first
- grouping/binding is the real authoring step
- `sd-mecha` compilation is late

## How this maps to the future graph model

### Source

For this run, the effective source node was:

- model family using `sdxl-sgm`
- component `unet`
- alternate block handle source `sdxl-optim_blocks_sub`

### Present set

For the block-targeted part, the present set was effectively the full `unet`
block-handle universe because `all` expands the full component today.

For the parked `name: null` section, there was no active present set in this
run because that section was being retained, not executed.

### Groups

The `group` entries like `out_0`, `time_embed`, `to_out_0`, and `norm_attn`
were already explicit group nodes in spirit.

The `all` block section can be viewed as implicit one-item groups.

### Bindings

The recipe run bound each discovered block item to:

- `magnitude_ratio`
- `direction_ratio`
- `temperature`
- `critical_quantile`

Separately, the parked null/key section preserved potential grouped bindings for:

- `alpha`
- `rank_ratio`

Those grouped bindings were present in the saved guide snapshot even though this
particular recipe-mode run only targeted the `delta_widen` kwargs listed above.

### Late compilation

The emitted recipe proves the late-compile shape:

- block-space values are assembled first
- conversion into model key space happens later
- only then are kwargs attached to the target method node

This is the right boundary to preserve in a redesign.

## Design takeaways

This run suggests a few concrete rules for the future model:

1. The guide should be treated as a targeting-and-binding graph, not just as a
   bounds table.
2. Blocks are best modeled as preset group handles or alternate target spaces,
   not as a separate kind of optimizer logic.
3. `group` is already the clearest primitive.
4. `all` should be documented as "full discovered set" unless and until we
   intentionally change it.
5. The future model should have an explicit inactive/disabled representation so
   "keep this fragment around, but do not include it in the current run" does
   not have to be inferred from `name: null`.

## Candidate normalized view of this run

One reasonable graph-shaped summary of the same run would look like:

```text
ModelFamily(sdxl-sgm)
  -> Component(unet)
  -> TargetSource(block_handles from sdxl-optim_blocks_sub)
  -> ImplicitGroups(per handle)
  -> Bind(magnitude_ratio)
  -> Bind(direction_ratio)
  -> Bind(temperature)
  -> Bind(critical_quantile)
  -> Bounds(categorical [0.0, 0.5, 1.0, 1.5, 2.0])
  -> Compile(block literal)
  -> Convert(to sdxl-sgm)
  -> Attach to method node delta_widen
```

And separately, as a parked fragment rather than an active run input:

```text
ModelFamily(sdxl-sgm)
  -> InactiveFragment(key-group draft)
  -> TargetMatch(*.out.0.*, *.time_embed.*, ...)
  -> ExplicitGroups(out_0, out_2, time_embed, ...)
  -> Bind(alpha)
  -> Bind(rank_ratio)
```

That is much closer to the shape we want the runtime and node UI to share.

## Recommended implementation consequence

The next code-facing step should not be "rewrite the whole guide format."

It should be:

1. preserve the current behavior with characterization coverage
2. introduce an internal normalized model for:
   - target source
   - selected targets
   - groups
   - parameter bindings
3. compile current strategy shapes into that model
4. keep late `sd-mecha` compilation separate

That gives us a path to improve both the implementation and a future node UI
without losing sight of what the repo actually does today.
