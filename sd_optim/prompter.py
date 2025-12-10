# sd_optim/prompter.py
import os
import random
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)

class CardDealer:
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = Path(wildcards_dir)
        self.wildcards = {}
        # Only load if directory exists to prevent crash on missing dir
        if self.wildcards_dir.exists():
            self.load_wildcards()
        else:
            logger.warning(f"Wildcards directory not found: {self.wildcards_dir}")

    def load_wildcards(self):
        wildcard_files = list(self.wildcards_dir.rglob("*.txt"))
        for file in wildcard_files:
            try:
                # Relative path for ID: "character/protagonist" from "character/protagonist.txt"
                relative_path = file.relative_to(self.wildcards_dir)
                wildcard_name = str(relative_path.with_suffix("")).replace(os.sep, "_")
                
                with open(file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if lines:
                        self.wildcards[wildcard_name] = lines
            except Exception as e:
                logger.warning(f"Failed to load wildcard {file}: {e}")

    def sample_wildcard(self, wildcard_name: str) -> str:
        if wildcard_name in self.wildcards:
            return random.choice(self.wildcards[wildcard_name])
        # Return the string intact if wildcard not found, so user knows it failed
        return f"__{wildcard_name}__"

    def replace_wildcards(self, prompt: str) -> str:
        if not prompt: return ""
        
        # Regex to find __wildcard_name__
        def replace_match(match):
            key = match.group(1) # content inside underscores
            return self.sample_wildcard(key)

        # Iteratively replace until no wildcards remain (handles nested wildcards if needed)
        # Using a loop limit to prevent infinite recursion
        for _ in range(3): 
            if "__" not in prompt: break
            prompt = re.sub(r"__([a-zA-Z0-9_/\-\\]+)__", replace_match, prompt)
            
        return prompt

# --- CHANGED: Simplified Assembly Logic ---
def assemble_payload(defaults: Dict, payload: Dict) -> Dict:
    """
    Merges defaults into the payload.
    Now WebUI-agnostic: returns a flat dict. Adapters handle API formatting.
    """
    final_payload = defaults.copy()
    
    # Payload overrides defaults
    for k, v in payload.items():
        # Hydra/OmegaConf compatibility: unwrap if needed
        if isinstance(v, (DictConfig, ListConfig)):
            final_payload[k] = OmegaConf.to_container(v, resolve=True)
        else:
            final_payload[k] = v
            
    return final_payload

def unpack_cargo(cargo: DictConfig) -> Tuple[Dict, Dict]:
    defaults = {}
    payloads = {}
    
    # Convert entire config to container once to avoid repeated conversions
    cargo_container = OmegaConf.to_container(cargo, resolve=True)
    
    for k, v in cargo_container.items():
        if k == "cargo":
            # These are the specific test cases
            if isinstance(v, list):
                # Handle list format if cargo is a list of dicts (rare but possible)
                for item in v:
                    payloads.update(item)
            elif isinstance(v, dict):
                # Standard dict format
                payloads = v
        else:
            # These are global defaults (steps, cfg, workflow_json, etc.)
            defaults[k] = v
            
    return defaults, payloads

@dataclass
class Prompter:
    cfg: DictConfig

    def __post_init__(self):
        self.dealer = CardDealer(self.cfg.get("wildcards_dir", "wildcards"))
        self.load_payloads()

    def load_payloads(self) -> None:
        self.raw_payloads = {}
        # Determine cargo file based on webui type
        cargo_file = f"cargo_{self.cfg.webui}.yaml"
        
        # Fetch cargo config, fallback to empty if missing
        cargo_data = self.cfg.payloads.get(cargo_file)
        if not cargo_data:
            # Fallback to generic cargo if specific one missing
            cargo_data = self.cfg.payloads.get("cargo", {})

        defaults, payloads = unpack_cargo(cargo_data)
        
        for payload_name, payload in payloads.items():
            self.raw_payloads[payload_name] = assemble_payload(defaults, payload)

    def render_payloads(self, batch_size: int = 0) -> Tuple[List[Dict], List[str]]:
        payloads = []
        paths = []
        
        for p_name, p in self.raw_payloads.items():
            # In optimization, batch_size is strictly controlled by the Optimizer loop,
            # but we allow generating multiple variations of the *same* payload if needed.
            # Usually batch_size=1 from config.
            
            iterations = max(1, batch_size)
            
            for _ in range(iterations):
                rendered = p.copy()
                
                # Process Wildcards in Prompts
                if "prompt" in rendered and isinstance(rendered["prompt"], str):
                    rendered["prompt"] = self.dealer.replace_wildcards(rendered["prompt"])
                
                if "negative_prompt" in rendered and isinstance(rendered["negative_prompt"], str):
                    rendered["negative_prompt"] = self.dealer.replace_wildcards(rendered["negative_prompt"])

                # Handle specific logic for Forge Extensions (vpred)
                # This could arguably move to the Adapter, but it's pure data manipulation
                if rendered.get("vpred_enabled", False):
                    ext_name = rendered.get("extension_name", "Forge2 extras")
                    # Only add if not already present
                    if "alwayson_scripts" not in rendered:
                        rendered["alwayson_scripts"] = {}
                    
                    rendered["alwayson_scripts"][ext_name] = {
                        "args": [True, False, 0, 0, 0, 0, 'default', 'v_prediction']
                    }

                paths.append(p_name)
                payloads.append(rendered)
                
        return payloads, paths
