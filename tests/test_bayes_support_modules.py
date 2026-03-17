from __future__ import annotations

import json

from omegaconf import OmegaConf

from sd_optim.optimizers.bayes.history import load_previous_iterations


def test_load_previous_iterations_copies_prior_log_when_not_resetting(tmp_path) -> None:
    previous_log = tmp_path / "previous.jsonl"
    new_log = tmp_path / "current.jsonl"
    entry = {"target": 1.25, "params": {"alpha": 0.5}}
    previous_log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "optimizer": {
                "bayes_config": {
                    "load_log_file": str(previous_log),
                    "reset_log_file": False,
                }
            }
        }
    )

    loaded = load_previous_iterations(cfg, target_log_path=new_log)

    assert loaded == [entry]
    assert new_log.read_text(encoding="utf-8").strip() == json.dumps(entry)


def test_load_previous_iterations_does_not_copy_prior_log_when_resetting(tmp_path) -> None:
    previous_log = tmp_path / "previous.jsonl"
    new_log = tmp_path / "current.jsonl"
    entry = {"target": 0.75, "params": {"beta": 0.2}}
    previous_log.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "optimizer": {
                "bayes_config": {
                    "load_log_file": str(previous_log),
                    "reset_log_file": True,
                }
            }
        }
    )

    loaded = load_previous_iterations(cfg, target_log_path=new_log)

    assert loaded == [entry]
    assert not new_log.exists()
