# Graph-Native Guide Shape Draft

## Purpose

This note is the first concrete draft of the graph-native guide shape.

It is meant to answer:

- what the node editor is actually saving
- what the canonical graph concepts are
- how that differs from the human text guide
- how inactive or parked fragments are represented explicitly

It is not the final code schema yet.
It is the first shape stable enough to build around.

## Design goals

The graph-native format should:

- represent the small authored graph, not the exploded compiled graph
- be explicit about authored intent
- be easy for a node editor to save and restore
- preserve inactive fragments without weird placeholders
- compile into the same canonical runtime model as the human text guide

It should **not**:

- expose raw optimizer-visible params directly
- expose payload dicts directly
- expose internal conflict-resolution machinery directly
- become the only supported human-editing format

## Important split

There are three layers:

1. **Graph save format**
   - what the node editor stores
2. **Canonical internal model**
   - what runtime compiles from graph or text
3. **Compiled runtime result**
   - resolved targets, optimizer params, payload dicts, dependencies

This document is about layer 1, while staying consistent with layer 2.

## Small authored vocabulary

The visible authored graph should stay centered on:

- **Source**
- **Selection**
- **Type**
- **Domain**
- **Param**
- **Build**

Defaults such as default selection, default domain, and default param should usually stay implicit rather than becoming extra nodes.

That means the node editor mostly lets users express:

- where targets come from
- which explicit selection is being made, if any
- what authored operation type the branch performs
- how values behave when overridden
- which method/recipe param gets driven when overridden
- which build the branch contributes to

## Graph object shape

The graph-native file should be node-and-edge based.

At the highest level:

```yaml
version: 1

meta:
  name: weighted_sum_block_tuning
  description: broad block tuning plus a few focused exceptions

nodes:
  - id: unet_blocks
    type: source
    enabled: true
    data: {}

edges: []
```

The exact fields can still change, but these are the current likely top-level parts:

- `version`
- `meta`
- `nodes`
- `edges`

## Why explicit nodes and edges

The node editor needs to save:

- node identity
- node type
- node configuration
- whether something is enabled
- how nodes connect

That is easier and more stable if it is explicit.

The graph save format can be more structured than the human guide because it is editor-oriented rather than prose-oriented.

## Node shape

Each node should have:

```yaml
- id: node_id
  type: source | selection | type | domain | param | build
  enabled: true
  label: optional display label
  data: {}
```

Likely common fields:

- `id`
  - stable graph identity
- `type`
  - semantic node role
- `enabled`
  - whether this node participates
- `label`
  - optional human-facing name
- `data`
  - node-specific settings

## Inactive and parked fragments

Inactive state should be explicit.

Use:

```yaml
enabled: false
```

This replaces the need for placeholder patterns like `name: null`.

If a user wants to keep a branch around without using it:

- the nodes stay in the graph
- their connections stay in the graph
- the branch is disabled explicitly

This is easier to understand and easier for the editor to show visually.

## Edge shape

Edges should stay simple:

```yaml
- from: unet_blocks
  to: alpha_all
```

Optional later fields might exist, but the current assumption is that edge meaning mostly comes from the connected node types.

## Node types

### 1. Source

Represents where targetable things come from.

Examples:

- component
- block preset
- layer/key space
- future recipe-target source

Example:

```yaml
- id: unet_blocks
  type: source
  enabled: true
  label: UNet blocks
  data:
    source_kind: block
    component: unet
    preset: sdxl-optim_blocks_sub
```

Source node responsibilities:

- define the target universe
- define whether this is block-like, key-like, or another future source
- define any preset/config context needed to discover items

### 2. Selection

Represents an explicit target subset when the branch is not using the source's default selection.

Examples:

- exact block choice
- exact layer choice
- regex/wildcard selection

Example:

```yaml
- id: selected_blocks
  type: selection
  enabled: true
  data:
    mode: block
    items:
      - UNET_IN05_1
      - UNET_OUT05_1
```

Or:

```yaml
- id: out_layers
  type: selection
  enabled: true
  data:
    mode: regex
    patterns:
      - "*.out.0.*"
      - "*.out.2.*"
```

Selection node responsibilities:

- narrow the source set
- stay absent when the branch uses default selection
- avoid carrying grouping or value-behavior semantics

### 3. Type

Represents the authored operation for the branch.

Current likely type values:

- `all`
- `group`
- `exclude`

Examples:

```yaml
- id: all_targets
  type: type
  enabled: true
  data:
    mode: all
```

```yaml
- id: decoder_surface
  type: type
  enabled: true
  data:
    mode: group
    name: decoder_surface
```

```yaml
- id: drop_block_69
  type: type
  enabled: true
  data:
    mode: exclude
```

Type node responsibilities:

- tell the build how to interpret the branch
- keep `all` as a type, not a selection
- let exclusion remain stronger than inclusion inside the same build

### 4. Domain

Represents value behavior when it is explicitly overridden.

Examples:

- continuous bounds
- categorical values
- fixed value

```yaml
- id: beta_range
  type: domain
  enabled: true
  data:
    mode: range
    min: 0.0
    max: 1.0
```

```yaml
- id: ratio_choices
  type: domain
  enabled: true
  data:
    mode: categorical
    values: [0.0, 0.5, 1.0, 1.5, 2.0]
```

```yaml
- id: fixed_delta
  type: domain
  enabled: true
  data:
    mode: fixed
    value: 0.25
```

Domain node responsibilities:

- define how values may vary
- stay absent when the branch uses default value behavior

### 5. Param

Represents the method or recipe parameter being driven when it is explicitly specified.

Example:

```yaml
- id: alpha_param
  type: param
  enabled: true
  data:
    name: alpha
```

This should stay simple.
The param node answers:

- which runtime parameter gets the resulting values

### 6. Build

Represents the place where branches are gathered and resolved.

Example:

```yaml
- id: build_alpha
  type: build
  enabled: true
  data: {}
```

Build node responsibilities:

- gather all incoming branches
- apply exclusion across the whole build
- let narrower/grouped branches win before broader `all` branches
- fill in default selection, domain, and param behavior where appropriate
- only then compile the final bindings

## Recommended connection patterns

The graph should encourage a small set of patterns.

### Broad default

```text
Source -> Type(all) -> Param -> Build
```

Example:

```yaml
nodes:
  - id: unet_blocks
    type: source
    enabled: true
    data:
      source_kind: block
      component: unet
      preset: sdxl-optim_blocks_sub

  - id: all_blocks
    type: type
    enabled: true
    data:
      mode: all

  - id: alpha_param
    type: param
    enabled: true
    data:
      name: alpha

  - id: build_alpha
    type: build
    enabled: true
    data: {}

edges:
  - from: unet_blocks
    to: all_blocks
  - from: all_blocks
    to: alpha_param
  - from: alpha_param
    to: build_alpha
```

### Carve-out

```text
Source -> Selection(block/layer/regex) -> Type(exclude) -> Build
```

### Grouped exception

```text
Source -> Selection(block/layer/regex) -> Type(group) -> Domain -> Param -> Build
```

These patterns line up with the sketch discussion:

- broad defaults
- carve-outs
- grouped exceptions

## Allowed edge rules

The first practical rule set should stay small:

- `Source -> Type`
- `Source -> Selection`
- `Selection -> Type`
- `Type -> Domain`
- `Type -> Param`
- `Domain -> Param`
- `Type -> Build`
- `Param -> Build`

The editor can still allow shortcuts later, but this is the clean semantic backbone.

Important constraint:

- the graph save format should describe authored intent
- it should not require users to think about compilation stages

## What stays implicit

These parts should stay mostly implicit in the graph save format:

- overlap resolution
- conflict detection
- optimizer param name generation
- payload dict materialization
- dependency expansion across compiled params

Those belong to compilation/runtime.

The graph save format should only need enough information to author intent clearly.

## How strategy words fit

Legacy words like:

- `all`
- `select`
- `group`
- `single`

should be treated as shorthand patterns over this graph shape.

Examples:

- `all`
  - `Source -> Type(all) -> Param -> Build`
- `select`
  - `Source -> Selection(...) -> Type(all) -> Param -> Build`
- `group`
  - `Source -> Selection(...) -> Type(group) -> Domain? -> Param -> Build`
- `single`
  - `Source -> Type(group)` or `Selection(...) -> Type(group)` with no explicit selection-default override beyond the branch input

So the graph shape is the deeper model, and old strategy wording becomes a compatibility or UX convenience layer.

## Human text guide relationship

This graph-native format is **not** the same thing as the future human text guide.

The human text guide should likely:

- read more naturally
- avoid explicit edges
- reduce editor-oriented structure
- stay easy to diff and tweak

But it should still compile to the same canonical internal model.

That means:

- graph-native format is editor-first
- human guide is text-first
- canonical internal model is shared

## Open questions

These points still need follow-up:

1. whether exact item-picking and regex selection should stay under one `selection` node type or split later
2. whether `group` needs any extra saved settings beyond an optional group name or named group list
3. how much of default param/domain inheritance should be shown in editor UX versus left implicit
4. whether dependencies should become a first-class authored node later
5. whether some simple chains should be collapsed into compound nodes in saved graph form

## Recommended next steps

1. Draft the future human text guide shape separately from this file
2. Decide the canonical internal model fields that both graph and text compile into
3. Add one worked example that maps:
   - current guide
   - future graph save
   - future human text guide
   - compiled runtime result
4. Only then decide whether the graph save format needs any extra structure for editor ergonomics
