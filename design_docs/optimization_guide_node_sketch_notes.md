# Notes from the node sketch discussion

## Main takeaway

The graph sketch made the underlying logic feel simpler than the YAML currently does.

That does **not** mean the graph should replace `guide.yaml` as the persisted authoring file. It means the current YAML is probably hiding the real structure too much, while the sketch surfaced it more directly.

## What the sketch seemed to show

A readable authoring model can be described with a small number of semantic operations:

- choose a target universe: `component / block / layer`
- optionally shape it: `all / filter / regex / group / no filter`
- optionally subtract from it: `exclude`
- attach value behavior: `bound / default / fixed / categorical`
- bind it to a recipe or method parameter: `param`
- resolve into the final compiled result

A rough mental model from the sketch:

- `component/block/layer -> group/all/whatever filter -> bound/default -> param -> resolution`
- `component/block/layer -> group/all/whatever filter/no filter -> exclude -> resolution`

That feels closer to the real semantics than the current guide wording.

## Important distinction: authored graph vs compiled graph

The discussion landed on a strong distinction between two layers:

### Authored graph
This is the human-facing conceptual layer.

It is expected to stay small and semantic.

Typical patterns are things like:

- `[component] -> [all] -> [param]`
- `[component] -> [regex] -> [exclude]`
- `[component] -> [regex] -> [group] -> [param]`

Most real authoring probably looks like broad defaults plus a few carve-outs and grouped exceptions.

### Compiled graph
This is the expanded internal/runtime layer.

It can fan out into many resolved targets, optimizer-visible parameters, payload mappings, and dependencies. That expansion should stay mostly hidden from the user by default.

So:

- the compiled graph can explode
- the authored graph usually should not

## Why the sketch felt easier than YAML

The sketch removes several kinds of overhead that YAML currently imposes:

### 1. Naming overhead
No need to remember exact field names or which concepts live in which section.

### 2. Structural overhead
No need to remember nesting, order, or where bounds/exclusions/targeting are supposed to go.

### 3. Referential overhead
No need to mentally connect separate sections like component selection, strategy choice, grouping, and custom bounds overrides.

Instead, the node itself can carry the relevant information visually.

## Why a node UI could actually be easier

A good node UI can make authoring easier by storing semantics in the interface instead of in YAML ceremony.

Examples mentioned in the discussion:

- double click to add
- a searchable drawer of presets
- search bars for components, params, and targets
- node types that already imply their scope or role
- inline bounds editing
- on-node useful info instead of scattered fields
- direct visibility of how a rule flows
- accessibility helpers that reduce friction when building graphs

This means the authoring experience would not need things like:

- a separate `custom_bounds` section
- awkward repetition of block/key/component-type metadata
- indirect strategy names just to express something simple

A lot of that can become direct UI manipulation instead.

## Why “graphs get ugly at scale” is mostly a presentation issue

The concern about graph complexity was pushed back on and refined.

The conclusion was that scale is mostly a **presentation problem**, not a proof that a graph UI is wrong.

A graph only becomes unreadable if it shows the wrong level of detail all the time.

Ways a UI could manage complexity include:

- collapse/expand
- grouping or compound nodes
- search/filter/isolate
- per-component views or lanes
- showing only authored rules by default
- hiding resolved/default/internal edges
- exposing details only on selection or in a side panel

So the real risk is not “graphs do not scale.” The real risk is choosing the wrong visible granularity.

## Likely real graph vocabulary

The discussion converged on a small likely vocabulary for the visible authored graph:

- **Source**: component / block / layer
- **Selector**: all / regex / picker / no filter
- **Modifier**: group / exclude
- **Param**: the recipe or method parameter being driven
- **Domain**: bounds / fixed / categorical / default
- **Resolve**: final resolution step, likely implicit or mostly hidden

That is a much smaller and more understandable model than exposing every internal compiler object directly.

## Why this does not mean “the graph should be the file”

A key boundary from the discussion:

- the graph is not automatically a good persisted text format
- the graph can still be a better authoring interface
- YAML can still remain the saved/reviewed/exported representation

So the likely healthy split is:

- **Node editor / graph UI** = primary authoring experience
- **Canonical internal model** = runtime/compilation representation
- **YAML** = persisted, diffable, portable representation

The main insight is that the graph does not have to replace YAML to justify itself. It only has to make authoring easier.

## Strong conclusion from the sketch discussion

The graph seems viable **if** it stays at the level of a few meaningful semantic transforms instead of exposing raw internals.

The sketch suggests the real system is not overwhelmingly complex. It may mostly be:

- choose targets
- shape them
- optionally subtract from them
- attach value behavior
- bind to params
- resolve

That is a small enough conceptual model that a purpose-built visual authoring tool could plausibly be easier than editing the guide by hand.

## Practical conclusion

The sketch did not prove that the graph should replace the guide file.

It did suggest three important things:

1. the current YAML may be making the system look harder than it really is
2. there is a readable graph-shaped mental model underneath the guide
3. a good node UI could be a better primary authoring surface, while YAML remains a good persistence format

## Short version

After the node sketch discussion, the main view was:

- the visible authoring graph can stay small
- most real authoring is broad scope plus a few exceptions or groups
- the graph only becomes bad if it exposes compiled detail instead of authored intent
- the node UI can carry context directly, which removes a lot of YAML friction
- YAML is still valuable, but it may be better as saved/exported text than as the only serious authoring experience

