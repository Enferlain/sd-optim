# Scorer Startup Performance (2026-02-13)

- [x] Task: Quantify startup regressions from timestamped logs and isolate the primary hangup segment(s).
- [x] Task: Restore straightforward eager scorer/rembg imports in `sd_optim/scorer.py` to remove runtime class-import stalls introduced by dynamic scorer class resolution.
- [x] Task: Align scorer dependency tests with the eager-loading behavior and drop deferred-loading-only test coverage.
- [x] Task: Run targeted scorer tests with `uv run` when possible, and record environment constraints if blocked.
- [x] Task: Simplify scorer class import helper/readability without changing behavior.
