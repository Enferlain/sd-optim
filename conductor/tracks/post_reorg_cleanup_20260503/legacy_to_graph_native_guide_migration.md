# Legacy Guide to Graph-Native Guide Migration

## Purpose

This note explains how the current strategy-style optimization guide maps into the graph-native guide model.

It is meant to help with three things:

- understanding what the current guide is really expressing
- translating that intent into the graph-native authored model
- seeing how the same intent should appear in a future node editor or future human-friendly text guide

This is a migration and interpretation note.
It is not the final saved graph schema, and it is not the final future text-guide syntax.

## Current reality

Today, the active user-facing guide still lives under `conf/optimization_guide/`.

That current guide format is still the compatibility authoring surface.
Runtime preserves it through `sd_optim/guide_legacy.py`, which adapts strategy-style guide entries into the shared graph-backed compiler path.

So the current practical stack is:

1. user edits the legacy text guide
2. `sd_optim/guide_legacy.py` translates it into shared binding specs
3. the graph-backed compiler turns those bindings into optimizer-visible params and payload maps

The graph-native model is the design center for where this is going, but the legacy text guide is still the active authored format today.

## Shared mental model

The important simplification is that both the legacy guide and the graph-native guide are trying to express the same underlying authored intent:

1. choose a target source
2. optionally narrow it
3. decide whether values are per-target, grouped, or excluded
4. optionally override the value domain
5. bind the result to one method or recipe param
6. let the build resolve overlaps and exclusions

The graph-native vocabulary for that is:

- `Source`
- `Selection`
- `Type`
- `Domain`
- `Param`
- `Build`

## Legacy to graph mapping

### `target_type`

Legacy:

- `target_type: block`
- `target_type: key`

Graph-native meaning:

- choose a `Source` whose `data.source_kind` is `block` or `key`

This is how the guide chooses which target universe exists before any selection or grouping happens.

### `all`

Legacy meaning:

- the chosen target set gets one optimizer-visible value per target

Graph-native mapping:

```text
Source -> Type(all) -> Param -> Build
```

If the authored intent narrows to a subset first, then the mapping becomes:

```text
Source -> Selection(...) -> Type(all) -> Param -> Build
```

### `select`

Legacy meaning:

- choose only the matching targets
- then treat each selected target independently

Graph-native mapping:

```text
Source -> Selection(...) -> Type(all) -> Param -> Build
```

This is why `select` is better understood as a preset, not a separate long-term core concept.
It is just selection plus per-target value behavior.

### `group`

Legacy meaning:

- a selected set of targets shares one value

Graph-native mapping for one group:

```text
Source -> Selection(...) -> Type(group) -> Param -> Build
```

If there is no explicit selection node, the whole incoming set is grouped:

```text
Source -> Type(group) -> Param -> Build
```

Important locked rule:

- one `group` node receives one incoming set and produces one grouped value

So if legacy guide intent needs multiple group names, that is not one graph `group` node with internal subgroup fanout.
It becomes multiple explicit branches, each with its own `Selection -> Type(group)`.

### `single`

Legacy meaning:

- the whole source component shares one value

Graph-native mapping:

```text
Source -> Type(group) -> Param -> Build
```

Conceptually, `single` is just the broadest grouped case.
It is not a distinct long-term semantic primitive.

### `none`

Legacy meaning:

- skip this authored fragment

Graph-native mapping:

- no branch at all, or
- a disabled fragment with `enabled: false`

### `optimize_params`

Legacy meaning:

- one targeting rule may drive several method params

Graph-native meaning:

- the same targeting branch is repeated per param binding

So a legacy rule such as:

```yaml
optimize_params: [alpha, rank_ratio]
```

maps to the same authored target shape feeding two separate `Param` bindings.

### `custom_bounds`

Legacy meaning:

- override the default optimization domain for generated params

Graph-native meaning:

- a `Domain` override conceptually belongs on the branch whose values need non-default behavior

Important current limitation:

- `custom_bounds` is still a legacy/runtime override surface today
- it is not yet represented as a fully authored graph-native node workflow in the active user format

So this is one of the places where the migration is conceptual first and tooling-second.

### `name: null`

Legacy meaning today:

- keep the fragment around without participating in the current run

Graph-native meaning:

- represent the parked fragment explicitly with `enabled: false`

This is one of the clearer improvements in the graph-native model: inactive state becomes explicit instead of hiding inside placeholder values.

## Worked examples

### Example 1: broad per-target tuning

Legacy-style intent:

```yaml
components:
  - name: unet
    optimize_params: [alpha]
    strategies:
      - type: all
        target_type: block
```

Graph-native meaning:

```text
Source(block:unet) -> Type(all) -> Param(alpha) -> Build
```

Effect:

- every target in the source gets its own optimizer-visible value

### Example 2: selected grouped exception

Legacy-style intent:

```yaml
components:
  - name: unet
    optimize_params: [alpha]
    strategies:
      - type: group
        target_type: key
        groups:
          - name: out_0
            keys:
              - "*.out.0.*"
```

Graph-native meaning:

```text
Source(key:unet) -> Selection("*.out.0.*") -> Type(group name=out_0) -> Param(alpha) -> Build
```

Effect:

- every matched `out.0` key shares one `alpha` value

### Example 3: several legacy groups

Legacy-style intent:

```yaml
components:
  - name: unet
    optimize_params: [rank_ratio]
    strategies:
      - type: group
        target_type: key
        groups:
          - name: out_0
            keys: ["*.out.0.*"]
          - name: time_embed
            keys: ["*.time_embed.*"]
```

Graph-native meaning:

```text
Source(key:unet) -> Selection("*.out.0.*") -> Type(group name=out_0) -> Param(rank_ratio) -> Build
Source(key:unet) -> Selection("*.time_embed.*") -> Type(group name=time_embed) -> Param(rank_ratio) -> Build
```

Effect:

- `out_0` gets one shared value
- `time_embed` gets a different shared value

This is the key semantic cleanup from the current graph-native work:

- multiple grouped values are represented as multiple branches
- one `group` node never produces multiple groups internally

### Example 4: carve-out / exclusion

Legacy-style intent is usually expressed indirectly today through competing strategies or parked fragments.

Graph-native meaning is more direct:

```text
Source -> Selection(...) -> Type(exclude) -> Build
```

Effect:

- matching targets are removed from the build before broader `all` or `group` bindings are finalized

## How this should appear later

### Future node editor

The node editor should show explicit authored branches:

- broad defaults
- carve-outs
- grouped exceptions

For the multi-group legacy example above, the editor should show two group branches, not one magical group node that secretly emits two buckets.

### Future human-friendly text guide

The final text-guide syntax is still open, but the authored meaning should match the same branch model.

So future text should likely read more like:

- source
- optional selection
- type
- optional domain
- param
- build

and less like today's strategy-shaped YAML with mixed targeting and grouping vocabulary packed into one layer.

## What remains intentionally legacy for now

These points are still intentionally transitional:

- `conf/optimization_guide/guide.yaml` remains the active user-facing format
- `sd_optim/guide_legacy.py` still owns compatibility translation from legacy strategy-style guides
- `custom_bounds` remains a live legacy override surface
- the final future text-guide syntax is not settled yet
- the saved graph format is still a draft, even though its branch semantics are now more tightly defined

## Practical takeaway

When translating old guides into the new mental model:

- `all` means one value per target
- `group` means one selected set sharing one value
- several legacy group names become several graph branches
- `single` is just the broadest grouped case
- `name: null` should be read as inactive intent, not active targeting
- `custom_bounds` should be read as branch-domain overrides, even though the active authored surface is still legacy
