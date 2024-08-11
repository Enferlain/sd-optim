import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from bayes_opt.logger import JSONLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from sd_webui_bayesian_merger.artist import Artist  # Import the Artist class
from sd_webui_bayesian_merger.bounds import Bounds
from sd_webui_bayesian_merger.generator import Generator
from sd_webui_bayesian_merger.merger import Merger
from sd_webui_bayesian_merger.prompter import Prompter
from sd_webui_bayesian_merger.scorer import AestheticScorer
from mecha_recipe_generator import translate_optimiser_parameters

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
        # Remove this line:
        # self.sdxl = self.cfg.sdxl
        # Create an instance of the Artist class
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
            self.cfg.get("greek_letters", ["alpha"]),
            self.cfg.optimisation_guide.frozen_params
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.custom_ranges
            if self.cfg.guided_optimisation
            else None,
            self.cfg.optimisation_guide.groups
            if self.cfg.guided_optimisation
            else None,
            cfg=self.cfg,  # Pass the configuration object (self.cfg)
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

        # Directly use the params dictionary
        weights = params
        bases = params

        # Merge the model in memory
        self.merger.merge(weights, bases)

        images, gen_paths, payloads = self.generate_images()
        scores, norm = self.score_images(images, gen_paths, payloads)
        avg_score = self.scorer.average_calc(scores, norm, self.cfg.img_average_type)

        # Update and save only if it's the best score so far
        self.update_best_score(bases, weights, avg_score)

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

    def update_best_score(self, bases, weights, avg_score):
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")

        # Translate parameters to the new format
        base_values, weights_list = translate_optimiser_parameters(bases, weights)

        for greek_letter in base_values:
            logger.info(f"\nrun base_{greek_letter}: {base_values[greek_letter]}")
            logger.info(f"run weights_{greek_letter}: {weights_list[greek_letter]}")

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # Save the best model
            self.merger.merge(weights, bases, save_best=True)

            Optimiser.save_best_log(base_values, weights_list)  # Pass the translated parameters

    @abstractmethod
    def optimise(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def save_best_log(bases: Dict, weights: Dict) -> None:
        logger.info("Saving best.log")
        with open(
            Path(HydraConfig.get().runtime.output_dir, "best.log"),
            "w",
            encoding="utf-8",
        ) as f:
            for greek_letter in bases:
                f.write(f"{bases[greek_letter]}\n\n{weights[greek_letter]}\n\n")

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