# Guide Node Design Notes

## Purpose

This note captures the design direction agreed during the optimization guide / node sketch discussion.

It is not the final schema.
It is the current conceptual plan for:

- what the optimization guide really is
- how a future node editor should think about it
- how that relates to a human text guide
- what the runtime/compiler layer should do

## What the guide is

The guide should be treated as an **optimization targeting spec**.

It is not best understood as:

- just a bounds file
- just a merge config
- just a UI schema

Those are downstream views of the same authored intent.

The guide describes:

- what parts of the target space are in play
- how those targets are shaped or narrowed
- what value behavior they have
- which method/recipe params they drive

## Main mental model

The node sketch discussion suggested that the real system is simpler than the current YAML makes it look.

At the authored level, most of the logic is basically:

1. choose targets
2. shape them
3. optionally subtract from them
4. attach value behavior
5. bind to params
6. resolve into the compiled result

Two rough sketch patterns that came out of the discussion:

- `component/block/layer -> group/all/filter -> bound/default -> param -> resolution`
- `component/block/layer -> group/all/filter/no filter -> exclude -> resolution`

The important point is that this is a small semantic pipeline, not an inherently huge schema.

## Authored graph vs compiled graph

The distinction between these two layers is important.

### Authored graph

This is the human-facing conceptual layer.

It should stay small and semantic.

Typical authored patterns look like:

- `[component] -> [all] -> [param]`
- `[component] -> [regex] -> [exclude]`
- `[component] -> [regex] -> [group] -> [param]`

Most real authoring is expected to look like:

- broad defaults
- a few carve-outs
- a few grouped exceptions

### Compiled graph

This is the expanded internal/runtime layer.

It can fan out into:

- resolved targets
- optimizer-visible params
- payload dicts
- dependencies
- conflict handling
- final `sd-mecha`-facing values

The compiled layer can be large.
That does not mean the authored layer should be.

## Why the node sketch felt easier

The sketch made the system feel easier mostly because it removed YAML overhead:

- naming overhead
- structural overhead
- referential overhead between separate sections

The node itself can carry context directly, which reduces the need for:

- awkward section jumping
- separate `custom_bounds`-style indirection
- indirect strategy wording for simple actions

## Small visible vocabulary

The likely visible authored graph vocabulary should stay small.

Current best candidate set:

- **Source**
  - component / block / layer / other targetable source
- **Selector**
  - all / regex / exact picker / no filter
- **Modifier**
  - group / exclude
- **Param**
  - the method or recipe parameter being driven
- **Domain**
  - bounds / fixed / categorical / default
- **Resolve**
  - mostly implicit or hidden

This is intentionally smaller than the runtime/compiler object model.

## Recommended authored flow

The current preferred authored flow is:

1. choose a **Source**
2. apply a **Selector**
3. optionally apply a **Modifier**
4. choose a **Domain**
5. bind to a **Param**

Then runtime performs resolution and compilation.

Important detail:

- overlap handling
- conflict handling
- expansion into optimizer params
- payload materialization

should live mostly in compilation/runtime, not as visible authoring ceremony.

## Text guide vs node editor

We should keep both:

- a human-usable text guide
- a future node/graph editor

They should not become separate systems.

They should compile into the same internal model.

The intended split is:

- **Node editor / graph UI**
  - primary authoring experience when visual editing is helpful
- **Human text guide**
  - readable, editable, diffable persistence format
- **Canonical internal model**
  - shared runtime/compiler representation

The graph does not need to replace YAML to be useful.
It only needs to make authoring easier.

## Important implication for YAML

The current YAML probably hides the real structure too much.

That does not mean the answer is to dump raw graph/compiler objects into YAML.

Instead:

- the text guide should stay human-readable
- the graph UI can carry more context directly
- both should reflect the same authored semantics

So the future text guide should likely express the small semantic flow above, not expose internal compiler machinery directly.

## Strategy words are probably presets, not the core model

Legacy strategy words:

- `all`
- `select`
- `group`
- `single`
- `none`

should probably not remain the long-term conceptual center.

They are better treated as convenience presets over:

- selection behavior
- grouping behavior
- shared vs per-target behavior

This means old strategy-style guides should map into the shared model, rather than defining the shared model.

## `name: null`

Current understanding:

- `name: null` is best treated as an intentionally inactive placeholder
- it is for keeping a fragment around without using it in the current run
- it should not be treated as an active global targeting scope

Longer-term, the future formats should represent this more explicitly as inactive/disabled state instead of depending on a placeholder pattern.

## Graph scale concerns

The discussion conclusion was that graph ugliness is mostly a presentation problem, not proof that the model is wrong.

A graph becomes hard to use when it shows the wrong level of detail all the time.

UI strategies that should be assumed from the start:

- collapse/expand
- grouping or compound nodes
- search/filter/isolate
- per-component views or lanes
- authored-rules-first visibility
- hidden compiled/default/internal detail unless requested
- details on selection or side panel instead of always-on clutter

So the risk is not "graphs do not scale."
The risk is exposing compiled detail instead of authored intent.

## Relation to current runtime work

The graph-backed compiler work already supports this direction at the runtime level.

What we have now:

- a graph-backed internal path for guide compilation
- a legacy adapter that preserves current results
- parity checks against real run artifacts

So the remaining design work is mostly about making the authored surfaces match the underlying model more honestly.

## What still needs to be defined

These points are still open and should be handled deliberately:

1. the future human text guide format
2. the graph-native save format
3. how inactive/parked fragments are represented explicitly
4. how much of `Resolve` is visible to users, if any
5. how much naming/vocabulary should be shared between text guide and node UI

## Migration guide expectations

When the migration guide is written, it should explain:

1. what the guide is conceptually now
2. how old strategy-style guides map into the shared model
3. how that shared model appears in:
   - the human text guide
   - the node editor
4. how inactive/parked fragments should be represented
5. what stays runtime-equivalent vs what changes intentionally

## Practical takeaway

The main conclusion so far is:

- the visible authoring graph can stay small
- the current YAML likely makes the system look harder than it is
- the graph only becomes bad if it exposes compiled detail instead of authored intent
- a node UI can reduce a lot of YAML friction by carrying semantics directly
- YAML is still valuable, but probably better as persistence/export than as the only serious authoring surface
