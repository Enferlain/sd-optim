# AGENTS.md - Operating Manual

This file defines how agents should work in `sd-optim`.

## 1) Mission

`sd-optim` optimizes Stable Diffusion merges by searching merge parameters (Optuna/Bayes), generating images through WebUI APIs, and scoring results with AI/manual scorers.

Primary goal for agents: deliver correct, reproducible changes with minimal regression risk.

## 2) Non-Negotiables

- Follow `conductor/workflow.md`.
- Track work in a relevant `conductor/tracks/*/plan.md`.
- Use Red-Green-Refactor for behavior changes.
- Add type hints for new/changed function signatures.
- Use structured logging (`logger = logging.getLogger(__name__)`).
- Do not introduce destructive git operations unless explicitly requested.

## 3) Repo Orientation

- `conf/`: Hydra configuration (`config.yaml`, `optimization_guide/`, payloads).
- `sd_optim/`: core runtime (optimizers, merger, scorer, generator, bounds).
- `sd_optim/model_configs/`: model block definitions and converters.
- `scripts/api.py`: WebUI bridge endpoints.
- `analysis_2026210/`: local analysis scripts and generated reports.
- `conductor/`: project workflow, tracks, and planning artifacts.

## 4) Architecture Snapshot

Optimization loop:

1. Optimizer proposes params.
2. Merger builds and executes recipe (`sd-mecha`).
3. Generator creates images via target WebUI API.
4. Scorer aggregates weighted scores.
5. Optimizer records trial and repeats.

## 5) Environment Assumptions

- Primary developer environment is Windows + PowerShell.
- Preferred tooling direction is Astral:
  - `uv` for environment/package/task execution
  - `ruff` for lint/format
  - `ty` for type checking
- Astral tooling is preferred when available; otherwise use existing project commands.
- Python dependencies are commonly installed in the WebUI venv:
  - `D:\stable-diffusion-webui-reForge\venv\Scripts\python.exe`
- Keep paths and commands platform-aware. Prefer repo-relative paths in docs and avoid `file:///` links.

## 6) Optuna-Specific Rules

- `custom_bounds` list values (e.g. `[0.0, 1.0]`) are treated as categorical choices.
- `custom_bounds` tuple values (e.g. `(0.0, 1.0)`) are treated as continuous ranges.
- Use TPE for mostly categorical/binary search spaces.
- Use CMA-ES for mostly continuous spaces.
- When reporting importance, clearly label mode:
  - maximize intent (`--maximize` in analysis helper)
  - default/min-style view (without `--maximize`)

## 7) Required Workflow for Code Changes

1. Pick or create a track plan in `conductor/tracks/.../plan.md`.
2. Mark task `[~]` before implementation.
3. Add failing test(s) first when behavior changes.
4. Implement minimum fix.
5. Run relevant tests/checks.
6. Update plan item to `[x]` when complete.
7. Summarize changed files and rationale.

If tests cannot run due to environment constraints, state that explicitly in the final report.

## 8) Command Conventions

- Use `rg` for fast search.
  - `https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md`
- Use non-interactive commands.
- Keep commands copy-pastable.
- Prefer PowerShell-friendly commands in examples.
- Prefer Astral tools when implemented:
  - `uv run ...`
  - `ruff check ...` / `ruff format ...`
  - `ty ...`
- If Astral tools are unavailable, use current Python/pytest/pip-based equivalents.

## 9) High-Value Areas for Caution

- `sd_optim/optuna_optimizer.py`: study resume/fork semantics, sampler config, callback logging.
- `sd_optim/bounds.py`: parameter-name generation and bounds interpretation.
- `sd_optim/scorer.py`: weighted averaging and scorer filters/lazy-load behavior.
- `conf/optimization_guide/guide.yaml`: search-space definition has major behavior impact.

## 10) Definition of Done

A change is complete only if:

- Behavior is validated (tests or clearly documented manual verification).
- Config/docs are updated when behavior or usage changed.
- Plan tracking is updated.
- Risks/limitations are called out explicitly.

## 11) Common Operations

### Add a scorer

1. Register metadata in scorer registry (`MODEL_DATA` path used by current scorer manager).
2. Implement scoring path in scorer runtime.
3. Add config entry in `conf/config.yaml` (method + weight/filter as needed).
4. Add at least one focused test or fixture-based check.

### Add a merge method

1. Implement method with `@merge_method` and proper type hints.
2. Ensure loader/import path registers it.
3. Add a small deterministic validation (unit/integration or analysis script).

## 12) Troubleshooting

- API connection issues: verify `webui` + `url` in `conf/config.yaml` and endpoint availability.
- CUDA OOM in scorers: use lazy-load list and/or CPU scorer device defaults.
- Hydra confusion: run with `hydra.verbose=true` and inspect `.hydra/config.yaml` in run dir.
- Study confusion: verify whether run was new, resumed, or forked before comparing metrics.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
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

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
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
