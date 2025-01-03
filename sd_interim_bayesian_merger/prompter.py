import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf

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


def assemble_payload(defaults: Dict, payload: Dict, webui_type: str) -> Dict:
    """Assembles payloads differently based on the webui_type."""
    if webui_type in ["a1111", "forge", "reforge"]: # Handle the structure for a1111-like UIs
        for k, v in defaults.items():
            if k not in payload.keys():
                payload[k] = v
        return payload
    elif webui_type == "comfy":
       for k, v in defaults.items():
            if k not in payload.keys():
               payload[k] = { "value": v}
       return payload
    elif webui_type == "swarm":
        for k, v in defaults.items():
            if k not in payload.keys():
                payload[k] = { "value": v}
            else:
                if "id" not in payload[k]:
                     payload[k] = { "value": payload[k], "id": k }

        return payload
    else:
        raise ValueError(f"Unsupported webui_type: {webui_type}")


def unpack_cargo(cargo: DictConfig, webui_type: str) -> Tuple[Dict, Dict]:
    defaults = {}
    payloads = {}
    for k, v in cargo.items():
        if k == "cargo":
           for p_name, p in v.items():
                payloads[p_name] = OmegaConf.to_container(p)
        elif isinstance(v, (DictConfig, ListConfig)):
            defaults[k] = OmegaConf.to_container(v)
        else:
            defaults[k] = v
    return defaults, payloads

@dataclass
class Prompter:
    cfg: DictConfig

    def __post_init__(self):
        self.load_payloads()
        self.dealer = CardDealer(self.cfg.wildcards_dir)

    def load_payloads(self) -> None:
        self.raw_payloads = {}
        cargo_file = f"cargo_{self.cfg.webui}.yaml"
        defaults, payloads = unpack_cargo(self.cfg.payloads.get(cargo_file, self.cfg.payloads), self.cfg.webui)
        for payload_name, payload in payloads.items():
            self.raw_payloads[payload_name] = assemble_payload(defaults, payload, self.cfg.webui)

    def render_payloads(self, batch_size: int = 0) -> Tuple[List[Dict], List[str]]:
        payloads = []
        paths = []
        for p_name, p in self.raw_payloads.items():
            for _ in range(batch_size):
                rendered_payload = p.copy()  # Always create a copy
                if "__" in p["prompt"]:
                    processed_prompt = self.dealer.replace_wildcards(p["prompt"])
                    rendered_payload["prompt"] = processed_prompt

                # Add alwayson_scripts if vpred_enabled is true
                if rendered_payload.get("vpred_enabled", False):
                    extension_name = rendered_payload.get("extension_name", "Forge2 extras")
                    rendered_payload["alwayson_scripts"] = {
                        extension_name: {
                            "args": [True, 0, 0, 0, 0, 'v_prediction']
                        }
                    }

                paths.append(p_name)
                payloads.append(rendered_payload)
        return payloads, paths
