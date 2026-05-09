# Future angle: guide outputs vs merge graph inputs

## Current likely split

Right now, the merge side is probably best treated as **execution substrate**, not as a hand-authored customization surface.

During normal optimization work, the merge is not really being composed manually. It behaves more like:

- a selected merge method
- a fixed execution graph or recipe structure
- parameterized by values resolved from the guide and optimizer

So the practical flow today is closer to:

`guide + optimizer sample -> resolved param payloads -> merge graph execution`

That suggests a clean split for now:

- **guide** = authored optimization/search-space/targeting layer
- **optimizer** = search driver
- **merge graph** = execution backend

## Why this split feels right

The hand-authored part of the workflow is mostly about:

- what can move
- where it applies
- how it is grouped or excluded
- what domain it has
- how the optimizer evaluates it

The merge side is mostly just where those resolved choices get executed.

So it probably makes sense to keep merge mostly in the background unless there is a deliberate decision later to expose more of it.

## Possible future direction

A future system might stop treating guide resolution as just a subgraph or compilation step that produces final payload dicts.

Instead, individual guide-derived outputs could become pluggable inputs into a larger merge graph.

That future would look more like:

`guide rule outputs -> merge graph parameter inputs`

In that version, the merge graph could become more composable, and optimization authoring might start interacting more directly with merge-graph inputs.

## What that future could affect

If that direction becomes real, it could affect questions like:

- what counts as an input node to the merge graph
- whether guide outputs need stronger typing or clearer semantics
- whether one guide rule can feed multiple merge parameters
- whether merge graph structure itself becomes part of optimization authoring
- how previewing and debugging should work across the guide/merge boundary

## Important caution

This is a real design fork, but it is a later problem.

It does **not** need to reshape the current design work yet.

The current system can stay simpler by keeping the seam explicit:

- guide/optimizer side produces optimizer-visible variables or resolved payloads
- merge side consumes those values as execution inputs

## Practical conclusion for now

For now, the better approach is probably:

- do **not** over-unify the guide graph and merge graph yet
- treat merge as the execution backend
- treat the guide as the authored optimization/control layer
- keep the boundary between them explicit

That boundary is likely the point where resolved guide outputs become mecha-facing recipe or parameter payloads.

## Short reminder

Worth remembering later:

- today: guide resolution feeds merge execution
- possible future: guide-derived outputs become first-class pluggable inputs inside merge graphs
- not a problem to solve at 3 am

