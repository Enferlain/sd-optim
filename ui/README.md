# sd-optim UI

SvelteKit workbench for the `sd-optim` graph guide authoring surface.

## Developing

Install from the repo root with pnpm:

```sh
pnpm install
```

Run the UI locally:

```sh
pnpm --dir ui run dev
```

## Current scope

This first pass is intentionally a local graph workbench. It uses Svelte Flow to render
guide nodes, optimizer-visible method params, and dependency-shaped edges so the graph
surface can be evaluated before the backend API contract is finalized.

## Checks

```sh
pnpm --dir ui run check
pnpm --dir ui run test:unit -- --run
pnpm --dir ui run build
```
