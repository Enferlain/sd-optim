# Track: sd-mecha converter modernization (2026-03-15)

- [x] Task: Add regression tests for `sd-mecha` 1.1.3 converter import and key-map compatibility.
- [x] Task: Refactor custom SDXL block converters to class-based `sd-mecha` conversion methods with explicit `map_keys(...)`.
- [x] Task: Use `sd_mecha.skip_key(...)` for missing optimized blocks instead of older fallback-only patterns.
- [x] Task: Run focused converter tests and record outcomes.
  - Command: `CI=true .venv-wsl/bin/pytest -q tests/test_granular_conversion.py`
  - Result: `4 passed in 25.93s`
  - Command: `CI=true .venv-wsl/bin/ruff check tests/test_granular_conversion.py sd_optim/model_configs/convert_sdxl_optim_blocks.py sd_optim/model_configs/convert_sdxl_optim_blocks_sub.py`
  - Result: `All checks passed!`
