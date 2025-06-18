from fastapi import APIRouter, HTTPException
from ruamel.yaml import YAML, YAMLError
from pathlib import Path
from typing import List
from sd_optim.scorer import MODEL_DATA

from backend.schemas import ConfigUpdate

# --- APIRouter for Main Configuration ---
router = APIRouter(
    prefix="/config",
    tags=["Configuration"],
)

# --- Define Directory ---
base_dir = Path(__file__).resolve().parent.parent.parent
config_path = (base_dir / "conf" / "config.yaml").resolve()

@router.get("/")
async def get_main_config() -> dict:
    """Reads the main configuration and returns its content as JSON."""
    yaml = YAML(typ='rt')
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config_content = yaml.load(config_file)
        return config_content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Config file not found at {config_path}.")
    except YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing config file: {e}")

@router.put("/")
async def update_config(config_data: ConfigUpdate):
    """Updates the main configuration file."""
    yaml = YAML(typ='rt')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            existing_config = yaml.load(f)
        
        update_data = config_data.model_dump(exclude_unset=True)
        existing_config.update(update_data)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing_config, f)
        return {"message": "Config updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config file: {e}")

@router.get("/scorers")
async def list_available_scorers() -> List[str]:
    """Dynamically lists all available scorer identifiers."""
    try:
        available_scorers = list(MODEL_DATA.keys())
        available_scorers.extend(['manual', 'background_blackness'])
        return sorted(list(set(available_scorers)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scorers: {e}")