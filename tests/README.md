# Tests (Planned Coverage)

This repository currently has limited automated tests. The list below captures the highest-value areas to cover as tests are added.

## Coverage Targets

- [ ] Reuse/cache correctness: full hit vs partial hit vs miss; never merge+gen on hit (`sd_optim/optimizer.py`)
- [ ] Image fingerprint determinism/sensitivity: same inputs => same hash; any output-affecting change => different hash (`sd_optim/optimizer.py`, `sd_optim/gen_adapters.py`, `sd_optim/merger.py`)
- [ ] Comfy workflow injection: seed/steps/cfg/sampler/scheduler/prompt/neg/width/height/model injected as expected (`sd_optim/gen_adapters.py`)
- [ ] Manifest schema: read/write, backward compatibility, migration, corruption handling (`run_manifest.json`)
- [ ] Score aggregation: arithmetic/geometric/quadratic + weights + edge cases (`sd_optim/scorer.py`)
- [ ] Scorer orchestration: filters, lazy-load behavior, partial rescoring (subset scoring) (`sd_optim/scorer.py`)
- [ ] Scorer unit tests on fixtures: HybridNoiseScorer boundary/mask-size cases; TextureScorer/PCAScorer stability (`sd_optim/models/*.py`)
- [ ] Config validation: `validate_run_config` and recipe validation paths (`sd_optim/utils.py`)
- [ ] Optuna study rules: objective-definition consistency on resume vs fork (`sd_optim/optuna_optimizer.py`)

## Suggested Test Types

- Unit tests: fingerprints, manifest parsing, score aggregation, config validation.
- Integration tests: mock generator backend returning a fixed image; verify manifest reuse and partial rescoring.
- Fixture (golden) tests: small set of PNGs in `tests/fixtures/images/`, asserting scorer outputs within tolerance.
