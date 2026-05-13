# AGENTS.md - Operating Manual

This file defines how agents should work in `sd-optim`.

## 1) Mission

`sd-optim` optimizes Stable Diffusion merges and recipe-based parameterizations by:

1. proposing optimization variables
2. compiling those variables into `sd-mecha`-compatible payloads or recipes
3. generating images through WebUI APIs
4. scoring results with automatic and/or manual scorers
5. feeding results back into the optimizer

Primary goal for agents: deliver correct, reproducible changes with minimal regression risk.

## 2) Non-Negotiables

- Follow `conductor/workflow.md` where it still applies.
- Use `bd` (beads) for active task tracking.
- Use conductor tracks for longer-lived design notes, architectural handoff, and refactor history.
- Use Red-Green-Refactor for behavior changes.
- Add type hints for new or changed function signatures.
- Use structured logging (`logger = logging.getLogger(__name__)`).
- Do not introduce destructive git operations unless explicitly requested.
- Do not push or otherwise publish git changes unless explicitly requested.

## 3) Repo Orientation

- `conf/`: Hydra configuration
  - `config.yaml`: root runtime config
  - `optimization_guide/`: current text guide surface
  - `payloads/`: prompt/workflow payload definitions
- `sd_optim/`: application code
  - `core/`: optimizer startup/runtime/cache/shared orchestration
  - `merge/`: recipe building, artifact writing, execution helpers
  - `optimizers/`: concrete Optuna and Bayes implementations
  - `scoring/`: scorer runtime/support logic
  - `extensions/bundled/`: bundled model configs, merge methods, scorer assets
  - `guide_compiler.py`: canonical compiled guide/binding model
  - `guide_legacy.py`: importer for the current legacy guide format
  - `guide_nodes.py`: graph-native guide importer/compiler path
  - `bounds.py`: still-active compatibility/runtime surface around parameter generation
- `analysis_2026210/`: local analysis scripts and generated reports
- `conductor/`: workflow, tracks, and longer-form planning artifacts
- `.beads/`: local beads issue tracker state

## 4) Architecture Snapshot

Optimization loop:

1. Optimizer proposes params.
2. Guide/bounds layer maps authored intent to optimizer-visible variables.
3. Merger or recipe optimization builds and executes the `sd-mecha` graph.
4. Generator creates images via the configured WebUI backend.
5. Scorer aggregates weighted scores.
6. Optimizer records the trial and repeats.

Guide/targeting layers:

- Current user-facing guide lives under `conf/optimization_guide/`.
- `guide_legacy.py` compiles that current guide into the shared compiled model.
- `guide_nodes.py` is the newer graph-native path built around authored branches feeding a `build`.
- `guide_compiler.py` is the canonical middle layer shared by both.

## 5) Environment Assumptions

- Primary developer environment is Windows plus PowerShell, but WSL is also used in practice.
- Preferred tooling direction is Astral:
  - `uv` for environment/package/task execution
  - `ruff` for lint/format
  - `ty` for type checking
- Astral tooling is preferred when available; otherwise use existing project commands.
- Python dependencies are commonly installed in the WebUI venv:
  - `D:\stable-diffusion-webui-reForge\venv\Scripts\python.exe`
- Keep paths and commands platform-aware. Prefer repo-relative paths in docs and avoid `file:///` links.

## 6) Config and Guide Reality

- The project uses Hydra composition, but most runtime config is still `DictConfig`-based rather than fully structured dataclass-backed config.
- `conf/config.yaml` is still a large live runtime config, not yet a fully cleaned-up schema surface.
- The current optimization guide is still strategy-shaped (`all`, `select`, `group`, `single`) even though the newer internal graph model is more build-centered.
- Treat the current text guide as the compatibility authoring surface, not the final conceptual model.
- Do not force structured schemas onto the guide while the graph-native guide model is still evolving.

## 7) Optuna and Bounds Rules

- `custom_bounds` list values (e.g. `[0.0, 1.0]`) are treated as categorical choices.
- `custom_bounds` tuple values (e.g. `(0.0, 1.0)`) are treated as continuous ranges.
- Rich dict-style bounds are also supported for stepped/log domains.
- Use TPE for mostly categorical/binary or mixed search spaces.
- Use CMA-ES for mostly continuous spaces.
- When reporting importance, clearly label mode:
  - maximize intent (`--maximize` in analysis helper)
  - default/min-style view (without `--maximize`)

## 8) Required Workflow for Code Changes

1. Check `bd ready` / `bd show` and work from real tracked issues when possible.
2. Claim or update the relevant bead before substantial work.
3. If the work changes architecture, workflow, or a long-running refactor/design thread, update the relevant conductor track note as well.
4. Mark conductor plan tasks `[~]` before implementation when working from an active track.
5. Add failing test(s) first when behavior changes.
6. Implement the minimum fix.
7. Run relevant tests/checks.
8. Mark conductor plan tasks `[x]` when complete, where applicable.
9. Summarize changed files, rationale, and any remaining risks.

If tests cannot run due to environment constraints, state that explicitly in the final report.

## 9) Command Conventions

- Use `rg` for fast search.
- Use non-interactive commands.
- Keep commands copy-pastable.
- Prefer PowerShell-friendly commands in examples unless the active environment is clearly WSL/Linux.
- Prefer Astral tools when implemented:
  - `uv run ...`
  - `ruff check ...` / `ruff format ...`
  - `ty ...`
- If Astral tools are unavailable, use current Python/pytest/pip-based equivalents.
- For task state, prefer:
  - `bd ready`
  - `bd show <id>`
  - `bd update <id> --claim`
  - `bd dep tree <id>`

## 10) High-Value Areas for Caution

- `sd_optim/bounds.py`: active runtime compatibility surface; parameter-name generation and bounds interpretation are sensitive.
- `sd_optim/guide_compiler.py`: canonical compiled model for guide/binding behavior.
- `sd_optim/guide_legacy.py`: translation layer for current guide semantics.
- `sd_optim/guide_nodes.py`: graph-native guide path; semantics are still settling.
- `sd_optim/core/optimizer_base.py`: centralized startup/config loading and parameter-space setup.
- `sd_optim/core/optimizer_runtime.py`: trial orchestration and score lifecycle.
- `sd_optim/core/optimizer_runtime_cache.py`: cross-run reuse/re-score path; naming may evolve.
- `sd_optim/merge/recipe_builder.py`: guide-to-recipe payload materialization.
- `conf/config.yaml`: large live runtime surface with machine-specific values.
- `conf/optimization_guide/guide.yaml`: search-space definition still has major behavior impact.

## 11) Definition of Done

A change is complete only if:

- Behavior is validated (tests or clearly documented manual verification).
- Config/docs are updated when behavior or usage changed.
- Bead status is updated if a bead exists for the work.
- Relevant conductor notes are updated when architectural or tracked refactor work changed.
- Risks and limitations are called out explicitly.

## 12) Common Operations

### Add a scorer

1. Register metadata in the scorer registry/loading path.
2. Implement scoring path in scorer runtime/support code.
3. Add config entry in `conf/config.yaml` or related config group (method + weight/filter as needed).
4. Add at least one focused test or fixture-based check.

### Add a merge method

1. Implement method with `@merge_method` and proper type hints.
2. Ensure loader/import path registers it.
3. Add a small deterministic validation (unit/integration or analysis script).

### Touch guide behavior

1. Decide whether the change belongs to:
   - current legacy guide behavior
   - graph-native guide behavior
   - shared compiled guide model
2. Add or update focused tests around the compiled outcome, not just helper internals.
3. Preserve parity intentionally when touching the legacy path unless the change is explicitly a behavior redesign.

## 13) Troubleshooting

- API connection issues: verify `webui` + `url` in `conf/config.yaml` and endpoint availability.
- CUDA OOM in scorers: use lazy-load list and/or CPU scorer device defaults.
- Hydra confusion: run with `hydra.verbose=true` and inspect `.hydra/config.yaml` in the run dir.
- Study confusion: verify whether run was new, resumed, or forked before comparing metrics.
- Config confusion: distinguish between root runtime config, payload config, and optimization guide config before debugging behavior.
- Guide confusion: inspect whether behavior is coming from `guide_legacy.py`, `guide_nodes.py`, or the shared `guide_compiler.py`.

## 14) Beads Issue Tracker

This project uses **bd (beads)** for active issue tracking.

Quick reference:

```bash
bd ready
bd show <id>
bd update <id> --claim
bd close <id>
bd dep tree <id>
```

Rules:

- Use `bd` for active task tracking when possible.
- Use conductor tracks for longer architectural notes and refactor/design handoff.
- Do not create ad hoc markdown TODO systems when a bead or conductor note is the better home.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
