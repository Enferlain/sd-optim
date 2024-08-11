from pathlib import Path

import hydra
from omegaconf import DictConfig

from sd_webui_bayesian_merger import ATPEOptimiser, BayesOptimiser, TPEOptimiser

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Remove the draw_unet_weights and draw_unet_base_alpha block, handled by artist

    if cfg["optimiser"] == "bayes":
        cls = BayesOptimiser
    elif cfg["optimiser"] == "tpe":
        cls = TPEOptimiser
    elif cfg["optimiser"] == "atpe":
        cls = ATPEOptimiser
    else:
        exit(f"Invalid optimiser:{cfg['optimiser']}")

    bo = cls(cfg)
    bo.optimise()
    bo.postprocess()

    # Trigger visualizations
    bo.artist.visualize_optimization()  # Call the visualization method


if __name__ == "__main__":
    main()