import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
from ruamel.yaml import YAML

PathT = os.PathLike


class CardDealer:
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = Path(wildcards_dir)
        self.wildcards = {}
        self.load_wildcards()

    def load_wildcards(self):
        wildcard_files = list(self.wildcards_dir.rglob("*.txt"))
        for file in wildcard_files:
            # Use relative path from wildcards directory
            relative_path = file.relative_to(self.wildcards_dir)
            # Replace slashes with underscores
            wildcard_name = str(relative_path).replace("/", "_").replace(".txt", "")
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.wildcards[wildcard_name] = [line.strip() for line in lines]

    def sample_wildcard(self, wildcard_name: str) -> str:
        if wildcard_name in self.wildcards:
            content = self.wildcards[wildcard_name]
            if content:
                return random.choice(content)
            else:
                raise ValueError(f"Wildcard file for '{wildcard_name}' is empty.")  # Raise an exception
        else:
            raise FileNotFoundError(f"Wildcard file for '{wildcard_name}' not found.")  # Raise an exception

    def replace_wildcards(self, prompt: str) -> str:
        chunks = re.split("(__\w+__)", prompt)
        replacements = []
        for wildcard_pattern in chunks:
            if wildcard_pattern.startswith("__") and wildcard_pattern.endswith("__"):
                replacement = self.sample_wildcard(wildcard_pattern[2:-2])
                replacements.append(replacement)
            else:
                replacements.append(wildcard_pattern)
        return "".join(replacements)


@dataclass
class Prompter:
    cfg: DictConfig

    def __post_init__(self):
        """Initializes paths based on the main configuration."""
        self.dealer = CardDealer(self.cfg.get("wildcards_dir"))
        project_root = Path(os.getcwd())
        self.conf_dir = project_root / "conf"
        self.payloads_dir = self.conf_dir / "payloads"
        self.presets_dir = self.conf_dir / "webui_presets"
        logger.info(f"Prompter Initialized. Reading Presets from: '{self.presets_dir}' and Payloads from: '{self.payloads_dir}'")

    def render_payloads(self) -> Tuple[List[Dict], List[str]]:
        """
        The new, **strict** payload rendering pipeline.
        This function will now raise errors on any configuration mistake.
        """
        final_payloads = []
        payload_names_for_run = []
        yaml = YAML(typ='safe')

        # --- Strict Check 1: Ensure payloads_to_run exists and is not empty ---
        payloads_to_run = self.cfg.get("payloads_to_run", [])
        if not payloads_to_run:
            raise ValueError(
                "Configuration Error: 'payloads_to_run' is missing or empty in your config.yaml.\n"
                "Please add the names of the payloads you want to run, for example:\n"
                "payloads_to_run:\n"
                "  - my_first_payload\n"
                "  - my_second_payload"
            )
            
        logger.info(f"Found {len(payloads_to_run)} payloads to process: {payloads_to_run}")

        # --- Loop through each payload name ---
        for payload_name in payloads_to_run:
            
            # --- Strict Check 2: The payload file must exist ---
            payload_path = self.payloads_dir / f"{payload_name}.yaml"
            if not payload_path.is_file():
                raise FileNotFoundError(
                    f"Configuration Error: Payload file '{payload_name}.yaml' not found!\n"
                    f"Please check the spelling in your config.yaml or make sure the file exists in '{self.payloads_dir}'."
                )
            
            try:
                with open(payload_path, 'r', encoding='utf-8') as f:
                    payload_specifics = yaml.load(f)
            except Exception as e:
                # --- Strict Check 3: The payload file must be valid YAML ---
                raise ValueError(
                    f"Configuration Error: Could not parse the payload file '{payload_path}'. It might be malformed.\n"
                    f"YAML Parser Error: {e}"
                ) from e

            # --- Strict Check 4: The payload must specify which preset it uses ---
            preset_filename = payload_specifics.get("webui_preset")
            if not preset_filename:
                raise KeyError(
                    f"Configuration Error: Payload '{payload_name}.yaml' is missing the required 'webui_preset' key.\n"
                    f"Please open the file and add a line like: 'webui_preset: forge.yaml'"
                )

            # --- Strict Check 5: The specified preset file must exist ---
            preset_path = self.presets_dir / preset_filename
            if not preset_path.is_file():
                raise FileNotFoundError(
                    f"Configuration Error: The payload '{payload_name}.yaml' specifies a preset '{preset_filename}' which was not found.\n"
                    f"Please make sure the file exists in '{self.presets_dir}'."
                )
            
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    preset_defaults = yaml.load(f)
            except Exception as e:
                raise ValueError(
                    f"Configuration Error: Could not parse the preset file '{preset_path}'. It might be malformed.\n"
                    f"YAML Parser Error: {e}"
                ) from e
            
            # --- Assemble the payloads for the batch ---
            for i in range(self.cfg.batch_size):
                final_payload = {**preset_defaults, **payload_specifics}
                
                if "__" in final_payload.get("prompt", ""):
                    final_payload["prompt"] = self.dealer.replace_wildcards(final_payload["prompt"])
                
                final_payloads.append(final_payload)
                payload_name_with_batch_index = f"{payload_name}_{i+1}"
                payload_names_for_run.append(payload_name_with_batch_index)

        logger.info(f"Successfully rendered {len(final_payloads)} total payloads for generation.")
        return final_payloads, payload_names_for_run