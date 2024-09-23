import hydra
from omegaconf import DictConfig

from sd_interim_bayesian_merger import ATPEOptimizer, BayesOptimizer, TPEOptimizer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Remove the draw_unet_weights and draw_unet_base_alpha block, handled by artist

    if cfg["optimizer"] == "bayes":
        cls = BayesOptimizer
    elif cfg["optimizer"] == "tpe":
        cls = TPEOptimizer
    elif cfg["optimizer"] == "atpe":
        cls = ATPEOptimizer
    else:
        exit(f"Invalid optimizer:{cfg['optimizer']}")

    bo = cls(cfg)
    bo.optimize()
    bo.postprocess()

if __name__ == "__main__":
    main()