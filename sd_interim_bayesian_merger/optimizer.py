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

from sd_interim_bayesian_merger.bounds import Bounds
from sd_interim_bayesian_merger.generator import Generator
from sd_interim_bayesian_merger.merger import Merger
from sd_interim_bayesian_merger.prompter import Prompter
from sd_interim_bayesian_merger.scorer import AestheticScorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

PathT = os.PathLike


@dataclass
class Optimizer:
    cfg: DictConfig
    best_rolling_score: float = 0.0

    def __post_init__(self) -> None:
        self.bounds_initializer = Bounds()
        self.generator = Generator(self.cfg.url, self.cfg.batch_size)
        self.merger = Merger(self.cfg)
        self.start_logging()
        self.scorer = AestheticScorer(self.cfg, {}, {}, {})
        self.prompter = Prompter(self.cfg)
        self.iteration = 0
        self.best_model_path = None
        self.cache = {}

        # import artist inside
        from sd_interim_bayesian_merger.artist import Artist
        self.artist = Artist(self)

    def start_logging(self) -> None:
        run_name = "-".join(self.merger.output_file.stem.split("-")[:-1])
        self.log_name = f"{run_name}-{self.cfg.optimizer}"
        self.logger = JSONLogger(
            path=str(
                Path(
                    HydraConfig.get().runtime.output_dir,
                    f"{self.log_name}.json",
                )
            )
        )

    def init_params(self) -> Dict:
        if self.cfg.optimization_guide is None:  # Handle missing optimization_guide
            self.cfg.optimization_guide = {}

        for guide in ["frozen_params", "custom_ranges", "groups"]:
            if guide not in self.cfg.optimization_guide:
                with open_dict(self.cfg):
                    self.cfg.optimization_guide[guide] = {}  # Use empty dictionaries for missing keys

        return self.bounds_initializer.get_bounds(
            self.cfg.optimization_guide.get("frozen_params", {}),
            self.cfg.optimization_guide.get("custom_ranges", {}),
            self.cfg.optimization_guide.get("groups", []),
            self.cfg
        )

    def sd_target_function(self, **params) -> float:
        logger.info(f"Parameters for Optimization Iteration: {params}")

        self.iteration += 1
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.init_points else "optimization"
        )

        if self.iteration in {1, self.cfg.init_points + 1}:
            logger.info("\n" + "-" * 10 + f" {iteration_type} " + "-" * 10 + ">")

        logger.info(f"\n{iteration_type} - Iteration: {self.iteration}")

        # Assemble parameters using bounds_initializer
        weights_list, base_values = self.bounds_initializer.assemble_params(
            params, self.cfg.optimization_guide.frozen_params, self.cfg.optimization_guide.groups, self.cfg
        )

        # Update the output file name with the current iteration
        self.merger.create_model_out_name(self.iteration)

        # Unload the currently loaded model
        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model?webui={self.cfg.webui}")  # Use query parameter
        r.raise_for_status()

        # Pass the models directory to the merge function
        model_path = self.merger.merge(weights_list, base_values, cfg=self.cfg, device=self.cfg.device, cache=self.cache,
                                       models_dir=Path(self.cfg.model_paths[0]).parent)

        # Send a request to the API to load the merged model
        r = requests.post(url=f"{self.cfg.url}/bbwm/load-model",
                          json={"model_path": str(model_path), "webui": self.cfg.webui})
        r.raise_for_status()

        # Generate images and score
        images, gen_paths, payloads = self.generate_images()
        scores, norm = self.score_images(images, gen_paths, payloads)
        avg_score = self.scorer.average_calc(scores, norm, self.cfg.img_average_type)

        # Update best score and handle best model saving
        self.update_best_score(base_values, weights_list, avg_score)

        # Collect data for visualization, including weights_list and base_values
        self.artist.collect_data(avg_score, params, weights_list, base_values)  # Pass the dictionaries here

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

        for param_name in weights_list:
            logger.info(f"\nrun base_{param_name}: {base_values.get(f'base_{param_name}')}")
            logger.info(f"run weights_{param_name}: {weights_list[param_name]}")

        if avg_score > self.best_rolling_score:
            logger.info("\n NEW BEST!")
            self.best_rolling_score = avg_score

            # Check if a previous best model exists and delete it BEFORE updating the filename
            if os.path.exists(self.merger.best_output_file):
                os.remove(self.merger.best_output_file)
                logger.info(f"Deleted previous best model: {self.merger.best_output_file}")

            # Update the best model filename
            self.merger.create_best_model_out_name(self.iteration)

            # Move the current model to the new best model filename
            shutil.move(self.merger.output_file, self.merger.best_output_file)
            logger.info(f"Saved new best model as: {self.merger.best_output_file}")

            Optimizer.save_best_log(base_values, weights_list, self.iteration)
        else:
            # Delete the current iteration's model file if it's not the best
            if os.path.exists(self.merger.output_file):
                os.remove(self.merger.output_file)
                logger.info(f"Deleted non-best model: {self.merger.output_file}")

    @abstractmethod
    def optimize(self) -> None:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def postprocess(self) -> None:
        raise NotImplementedError("Not implemented")

    @staticmethod
    def save_best_log(base_values: Dict, weights_list: Dict, iteration: int) -> None:
        logger.info("Saving best.log")
        with open(
                Path(HydraConfig.get().runtime.output_dir, "best.log"),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(f"Best Iteration: {iteration}.\n\n")

            for param_name in base_values:
                # Remove "base_" prefix from parameter name in log output
                clean_param_name = param_name.replace("base_", "")
                f.write(f"Parameter: {clean_param_name}\n")
                f.write(f"Base Value: {base_values[param_name]}\n")
                f.write(f"Weights: {weights_list.get(clean_param_name, [])}\n\n")

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
