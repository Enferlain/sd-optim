import json
import os
import shutil

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import requests

from bayes_opt.logger import JSONLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from sd_webui_bayesian_merger.bounds import Bounds
from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.merger import Merger
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.scorer import AestheticScorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

PathT = os.PathLike


@dataclass
class Optimiser:
    cfg: DictConfig
    best_rolling_score: float = 0.0

    def __post_init__(self) -> None:
        self.bounds_initialiser = Bounds()
        self.generator = Generator(self.cfg.url, self.cfg.batch_size)
        self.merger = Merger(self.cfg)
        self.start_logging()
        self.scorer = AestheticScorer(self.cfg, {}, {}, {})
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self.best_model_path = None

        # import artist inside
        from sd_webui_bayesian_merger.artist import Artist
        self.artist = Artist(self)

    def start_logging(self) -> None:
        run_name = "-".join(self.merger.output_file.stem.split("-")[:-1])
        self.log_name = f"{run_name}-{self.cfg.optimiser}"
        self.logger = JSONLogger(
            path=str(
                Path(
                    HydraConfig.get().runtime.output_dir,
                    f"{self.log_name}.json",
                )
            )
        )

    def init_params(self) -> Dict:
        for guide in ["frozen_params", "custom_ranges", "groups"]:
            if guide not in self.cfg.optimisation_guide.keys():
                with open_dict(self.cfg):
                    self.cfg["optimisation_guide"][guide] = None
        return self.bounds_initialiser.get_bounds(
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.custom_ranges
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else None,
            self.cfg
        )

    def sd_target_function(self, **params) -> float:
        logger.info(f"Parameters for Optimization Iteration: {params}")

        self.iteration += 1
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.init_points else "optimisation"
        )

        if self.iteration in {1, self.cfg.init_points + 1}:
            logger.info("\n" + "-" * 10 + f" {iteration_type} " + "-" * 10 + ">")

        logger.info(f"\n{iteration_type} - Iteration: {self.iteration}")

        # Assemble parameters using bounds_initialiser
        weights_list, base_values = self.bounds_initialiser.assemble_params(
            params, self.cfg.optimisation_guide.frozen_params, self.cfg.optimisation_guide.groups, self.cfg
        )

        # Update the output file name with the current iteration
        self.merger.create_model_out_name(self.iteration)

        # Merge the model to disk
        model_path = self.merger.merge(weights_list, base_values, cfg=self.cfg)

        # Send a request to the API to load the merged model
        r = requests.post(url=f"{self.cfg.url}/bbwm/load-model", json={"model_path": str(model_path), "model_arch": self.cfg.model_arch})
        r.raise_for_status()

        # Generate images and score
        images, gen_paths, payloads = self.generate_images()
        scores, norm = self.score_images(images, gen_paths, payloads)
        avg_score = self.scorer.average_calc(scores, norm, self.cfg.img_average_type)

        # Update best score and handle best model saving
        self.update_best_score(base_values, weights_list, avg_score)

        # Collect data for visualization
        self.artist.collect_data(avg_score, params)

        # Send a request to the API to unload the merged model
        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model", json={})
        r.raise_for_status()

        logger.info(f"Average Score for Iteration: {avg_score}")
        return avg_score

    def generate_images(self) -> Tuple[List, List, List]:
        images = []
        gen_paths = []
        payloads, paths = self.prompter.render_payloads(self.cfg.batch_size)
        for i, payload in tqdm(enumerate(list(payloads)), desc="Batches generation"):
            generated_images = self.generator.generate(payload)
            images.extend(generated_images)
            gen_paths.extend([paths[i]] * len(generated_images))
            payloads[i: i + 1] = [payloads[i]] * len(generated_images)
        return images, gen_paths, payloads

    def score_images(self, images, gen_paths, payloads) -> List[float]:
        logger.info("\nScoring")
        return self.scorer.batch_score(images, gen_paths, payloads, self.iteration)

    def update_best_score(self, base_values, weights_list, avg_score):
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")

        for base_param_name in base_values:  # Iterate over base parameter names (e.g., "base_alpha")
            logger.info(f"\nrun {base_param_name}: {base_values[base_param_name]}")
            # Extract the corresponding parameter name without the "base_" prefix
            param_name = base_param_name[5:]  # Remove "base_"
            if param_name in weights_list:
                logger.info(f"run weights_{param_name}: {weights_list[param_name]}")

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # Update the best model filename
            self.merger.create_best_model_out_name(self.iteration)

            # Move the current model to the new best model filename (overwriting if necessary)
            shutil.move(self.merger.output_file, self.merger.best_output_file)
            logger.info(f"Saved new best model as: {self.merger.best_output_file}")

            Optimiser.save_best_log(base_values, weights_list, self.iteration)

    @abstractmethod
    def optimise(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def save_best_log(base_values: Dict, weights_list: Dict, iteration: int) -> None:  # Accept iteration as argument
        logger.info("Saving best.log")
        with open(
                Path(HydraConfig.get().runtime.output_dir, "best.log"),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(f"Best Iteration: {iteration}\n\n")

            for param_name in base_values:
                f.write(f"Parameter: {param_name}\n")
                f.write(f"Base Value: {base_values[param_name]}\n")
                f.write(f"Weights: {weights_list.get(param_name, [])}\n\n")  # Use .get() with default value

    @staticmethod
    def load_log(log: PathT) -> List[Dict]:
        iterations = []
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break
                iterations.append(json.loads(iteration))
        return iterations
