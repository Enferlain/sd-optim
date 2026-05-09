# Bounds / Guide Graph Plan

## Purpose

This note sketches a future guide system for `sd-optim` that is easier to reason
about in code and also maps naturally to a node-based UI.

This is not a proposal to make the text guide more abstract or more academic.
The goal is the opposite:

- keep the user-facing concepts small
- make the internal model more explicit
- preserve the path to `sd-mecha` recipe generation
- support a graph editor where guide authoring feels like composing pieces

## Design goals

1. Keep the current feature set understandable.
2. Treat grouping as the main primitive.
3. Keep selection and exclusion simple.
4. Delay `sd-mecha`-specific compilation until late in the pipeline.
5. Make the guide easy to represent as nodes and connections.
6. Avoid introducing new concepts unless they clearly simplify authoring.

## Core view of the domain

The guide is not primarily a "bounds file."

It is a graph that describes:

1. what can be targeted
2. what is present
3. how present targets are grouped
4. which method parameter each group controls
5. how each optimizer parameter is allowed to vary
6. how the full optimization run is wired together

Conceptually:

```text
model
  -> component
  -> targetable items
  -> groups
  -> method parameters
  -> bounds / dependencies
  -> optimizer run
```

### Targetable items

The target hierarchy should be treated like this:

```text
model
  -> component
  -> existing modules / state dict layers / arbitrary groups
  -> keys (weight, bias, etc.)
```

In this framing, blocks are not a separate targeting universe.

Blocks are just preset groups or alternate config-space handles that eventually
need to compile down into values that `sd-mecha` can consume.

## The main distinction

### What our system needs to express

Our system needs to represent:

- selection
- grouping
- parameter binding
- bounds
- dependencies
- optimizer / scorer / checkpoint wiring

### What `sd-mecha` needs

`sd-mecha` needs final parameter values in the correct config space so the
recipe builder can construct literals and conversions for merge execution.

That means the guide should not be shaped around `sd-mecha`'s internal API.
Instead, the guide should describe optimization intent, and the runtime should
compile that intent into what `sd-mecha` needs at the last stage.

## Selection model

Selection should stay simple.

### Present means included

If something is present, it is in the target set.

### Exclusion is subtractive

`exclude` should work as an override on top of the present set:

```text
present set
  - excluded items
  = final selected targets
```

This avoids more complicated include/exclude algebra while still giving the UI
and the guide a practical way to refine selections.

### Why this matters for nodes

In a graph editor, this becomes natural:

- source node produces candidates
- selection node marks things present
- exclusion node removes some of them

That is much easier to explain visually than a deeply nested text schema.

## Grouping model

Grouping should be the main primitive.

The current strategy names can be treated as convenience forms of grouping:

- `all`: one implicit group per target
- `select`: present only matching targets, then bind individually
- `group`: explicit named groups
- `single`: one group containing all present targets

Under this model, `group` is the real primitive and the others are shortcuts.

This is a better fit for both the runtime and a node UI because the key idea is:

```text
Which targets share one optimizer parameter?
```

## Minimal conceptual model

The future system can be organized around five small concepts:

### 1. Source

Where targetable items come from.

Examples:

- a model
- a component
- a preset config
- an existing named group

### 2. Present set

Which discovered targets are currently included.

Examples:

- all targets in a component
- only targets matching a pattern
- only targets pulled into a group node

### 3. Exclusion

What to subtract from the present set.

Examples:

- exclude a few unstable targets
- exclude broad pattern matches
- exclude nodes dragged out of a group

### 4. Group

A named bucket of targets that share one optimizer parameter value.

### 5. Binding

A connection from one group to one method parameter plus its bounds and
dependency metadata.

## Future node system

The guide should be representable as a graph made of a small node palette.

## Suggested node types

### Target discovery

- `Model`
- `Component`
- `Preset Group`
- `Target Match`

### Selection shaping

- `Exclude`
- `Group`

### Optimization authoring

- `Bind Parameter`
- `Bounds`
- `Dependency`

### Run orchestration

- `Merge Method`
- `Optimizer`
- `Scorer`
- `Payload / Generator`
- `Checkpoint / Resume`
- `Output / Artifact`

## Example authoring flow

This is the rough UX the system should support:

1. Double click to create a `Model` node.
2. Choose a `Component`.
3. Inspect discovered targets or preset groups.
4. Pull targets into one or more `Group` nodes.
5. Connect each group to a `Bind Parameter` node.
6. Connect a `Bounds` node to the binding.
7. Optionally connect a `Dependency` node.
8. Feed all parameter bindings into an `Optimizer` node.
9. Wire `Scorer`, `Payload / Generator`, and `Checkpoint / Resume` into the run.

This makes guide authoring feel like composing blocks rather than editing a
large nested config by hand.

## Runtime normalization

Even if the text guide keeps a learning curve, the runtime should normalize the
graph into a small intermediate structure before recipe generation.

## Suggested normalized structures

### TargetSet

Represents a discovered and filtered set of targets.

Suggested fields:

```text
source_model
component
present_targets
excluded_targets
final_targets
```

### GroupSpec

Represents one bucket of targets.

Suggested fields:

```text
name
targets
origin
```

### ParameterBinding

Represents one optimizer-visible parameter.

Suggested fields:

```text
optimizer_param_name
method_param_name
group_name
bounds
dependency
target_space
```

### RunSpec

Represents the full optimization run.

Suggested fields:

```text
method
optimizer
scorers
payloads
checkpoint_policy
parameter_bindings
```

## Compilation pipeline

The runtime should behave like a compiler with distinct stages.

```text
graph / guide
  -> discover targets
  -> apply present rules
  -> apply exclusions
  -> build groups
  -> bind groups to method parameters
  -> attach bounds and dependencies
  -> build run spec
  -> compile final values for `sd-mecha`
```

This is important because it keeps the "optimization authoring" side separate
from the "recipe execution" side.

## Relationship to current runtime

Today the runtime already has the late-compile shape in spirit:

- bounds metadata is generated first
- optimizer values are sampled second
- recipe values are compiled after that
- `sd-mecha` literals and conversions are built in the merge layer

The main improvement is to make the intermediate model more explicit and more
graph-friendly, not to fundamentally change what `sd-mecha` receives.

## Mapping current strategy names

To preserve continuity, the current text guide could keep the existing terms,
but they would normalize into the same internal model.

### `all`

Compile to:

```text
one target per implicit group
```

### `select`

Compile to:

```text
filter targets by pattern
then treat each matched target as its own implicit group
```

### `group`

Compile to:

```text
explicit named groups
```

### `single`

Compile to:

```text
one explicit group containing the full present set
```

This keeps user familiarity while simplifying the internal story.

## Text guide implications

The text guide does not have to become radically simpler on day one.
It only needs to become a better serialization of the graph model.

The most useful textual improvements would be:

- explicit `exclude` support
- clearer precedence rules
- clearer naming for what is selected vs what is grouped vs what is bound

The text format can remain somewhat technical as long as the graph UI is the
easier path for authoring.

## Suggested precedence rules

The system should document and preserve clear ordering:

1. discover candidate targets
2. build present set
3. apply exclusions
4. build groups
5. bind groups to method parameters
6. apply exact parameter overrides
7. apply base-parameter overrides
8. validate dependencies

This gives both code and UI a stable model to rely on.

## Why this is better

This design improves both implementation and UX:

- the runtime becomes easier to stage and test
- the node editor gets a natural data model
- group-based thinking becomes consistent
- selection and exclusion stay boring and predictable
- `sd-mecha` integration stays a late compilation concern

Most importantly, this makes the system easier to extend later with things the
repo may want in the future:

- optimizer presets
- scorer bundles
- checkpoint policies
- run templates
- graph fragments for reusable group patterns
- richer dependency rules

## Phased adoption plan

### Phase 1: Vocabulary cleanup

- Document the current system in terms of source, present set, group, binding,
  and bounds.
- Clarify that blocks are preset groups rather than a fundamentally separate
  concept.
- Add `exclude` as a simple subtractive refinement.

### Phase 2: Internal normalization

- Introduce normalized internal structures for target sets, groups, and
  parameter bindings.
- Keep the current guide surface mostly intact.
- Compile old strategy names into the normalized model.

### Phase 3: Graph serialization

- Define a graph-shaped guide format or intermediate artifact that can be saved
  and loaded.
- Make the text guide one serialization of that graph.

### Phase 4: Node UI

- Build a graph editor using the normalized model.
- Let users discover targets, group them, bind parameters, and wire optimizers,
  scorers, and checkpoints visually.

## Questions to answer before implementation

1. Does the guide need raw state-dict key targeting beyond what current model
   configs expose?
2. Should arbitrary user-defined groups be first-class saved objects?
3. Where should exclusions live in the current text guide structure?
4. How much of optimizer/scorer/checkpoint wiring belongs in the same graph
   versus a neighboring run-config layer?
5. What is the smallest normalized intermediate model that can support both the
   existing text guide and the future node UI?

## Recommended next step

Before changing more runtime code, write one worked example that traces:

```text
model -> component -> selected targets -> grouped targets -> bound parameter
-> bounds -> optimizer param metadata -> final mecha-facing recipe values
```

That example should become the anchor for both code refactors and future UI
node behavior.
