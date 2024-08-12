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

#from sd_webui_bayesian_merger.artist import Artist  # Import the Artist class
from .bounds import Bounds
from .generator import Generator
from .merger import Merger
from .prompter import Prompter
from .scorer import AestheticScorer
from .mecha_recipe_generator import translate_optimiser_parameters

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
        # Create an instance of the Artist class
#        self.artist = Artist(self)

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

        # Merge the model in memory
        self.merger.merge(weights_list, base_values, cfg=self.cfg)  # Pass the assembled parameters

        images, gen_paths, payloads = self.generate_images()
        scores, norm = self.score_images(images, gen_paths, payloads)
        avg_score = self.scorer.average_calc(scores, norm, self.cfg.img_average_type)

        # Update and save only if it's the best score so far
        self.update_best_score(base_values, weights_list, avg_score)

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

        # Use the received base_values and weights_list directly
        for greek_letter in base_values:
            logger.info(f"\nrun base_{greek_letter}: {base_values[greek_letter]}")
            if greek_letter in weights:  # Check if the key exists in weights
                logger.info(f"run weights_{greek_letter}: {weights[greek_letter]}")

        for param_name in base_values:
            logger.info(f"\nrun base_{param_name}: {base_values[param_name]}")
            logger.info(f"run weights_{param_name}: {weights_list[param_name]}")

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # Save the best model
            self.merger.merge(weights_list, base_values, save_best=True)

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
            for param_name in bases:
                f.write(f"{bases[param_name]}\n\n{weights[param_name]}\n\n")

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