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
from sd_interim_bayesian_merger.utils import load_and_prepare_recipe

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
        self.log_name = run_name
        log_file_path = Path(HydraConfig.get().runtime.output_dir, f"{self.log_name}.json")

        # Load data from previous log file (if specified)
        previous_iterations = []
        if self.cfg.optimizer.get("load_log_file", None) and os.path.isfile(self.cfg.optimizer.load_log_file):
            try:
                with open(self.cfg.optimizer.load_log_file, "r") as f:
                    previous_iterations = [json.loads(line) for line in f]
                logger.info(f"Loaded {len(previous_iterations)} iterations from {self.cfg.optimizer.load_log_file}")
            except Exception as e:
                logger.warning(f"Failed to load previous optimization data: {e}")

        # Create the log file and append previous data *before* initializing JSONLogger
        try:
            with open(log_file_path, "w") as f:  # Open in write mode to create the file
                if previous_iterations:  # Append previous data if available
                    for iteration_data in previous_iterations:
                        f.write(json.dumps(iteration_data) + "\n")
                logger.info(f"Previous optimization data appended to {log_file_path}")
        except Exception as e:
            logger.warning(f"Failed to create or append to log file: {e}")

        # Initialize JSONLogger (in append mode to continue logging)
        self.logger = JSONLogger(path=str(log_file_path), reset=self.cfg.optimizer.reset_log_file)

    def init_params(self) -> Dict:
        with open_dict(self.cfg):
            self.cfg.optimization_guide.setdefault("custom_ranges", {})
            self.cfg.optimization_guide.setdefault("custom_bounds", {})
            self.cfg.optimization_guide.setdefault("components", [])

        # Check if recipe optimization is enabled
        if self.cfg.recipe_optimization.enabled:
            return load_and_prepare_recipe(self.cfg)  # Call get_recipe_bounds from utils
        else:
            # Check if components are specified
            if not self.cfg.optimization_guide.components:
                raise ValueError("No components specified for optimization in the configuration.")

            return self.bounds_initializer.get_bounds(
                self.cfg.optimization_guide.custom_ranges,
                self.cfg.optimization_guide.custom_bounds,
                self.cfg
            )

    def sd_target_function(self, **params) -> float:
        self.iteration += 1
        iteration_type = (
            "warmup" if self.iteration <= self.cfg.optimizer.init_points else "optimization"
        )

        if self.iteration in {1, self.cfg.optimizer.init_points + 1}:
            logger.info("\n" + "-" * 10 + f" {iteration_type} " + "-" * 10 + ">")

        logger.info(f"\n{iteration_type} - Iteration: {self.iteration}")

        # Assemble parameters using bounds_initializer
        assembled_params = self.bounds_initializer.assemble_params(params, self.cfg)
        logger.info(f"Assembled Hyperparameters for Iteration {self.iteration}: {assembled_params}")

        # Update the output file name with the current iteration
        self.merger.create_model_out_name(self.iteration)

        # Unload the currently loaded model
        r = requests.post(url=f"{self.cfg.url}/bbwm/unload-model?webui={self.cfg.webui}")  # Use query parameter
        r.raise_for_status()

        # Pass the models directory to the merge function
        model_path = self.merger.merge(assembled_params, cfg=self.cfg, device=self.cfg.device, cache=self.cache,
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
        self.update_best_score(assembled_params, avg_score)

        # Collect data for visualization
        self.artist.collect_data(avg_score, params, assembled_params)  # Pass the dictionaries here

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

    def update_best_score(self, assembled_params, avg_score: float):
        logger.info(f"{'-' * 10}\nRun score: {avg_score}")

        for param_name, param_value in assembled_params.items():
            logger.info(f"{param_name}: {param_value}")

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

            Optimizer.save_best_log(assembled_params, self.iteration)
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
    def save_best_log(assembled_params: Dict[str, Dict[str, float]], iteration: int) -> None:
        """Saves the best hyperparameters and iteration number to a log file."""
        logger.info("Saving best.log")
        with open(
                Path(HydraConfig.get().runtime.output_dir, "best.log"),
                "w",
                encoding="utf-8",
        ) as f:
            f.write(f"Best Iteration: {iteration}.\n\n")

            for param_name, param_values in assembled_params.items():
                f.write(f"Parameter: {param_name}\n")
                for key, value in param_values.items():
                    f.write(f"\t{key}: {value}\n")  # Write the nested key-value pairs
                f.write("\n")
