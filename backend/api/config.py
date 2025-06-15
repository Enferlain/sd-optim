from fastapi import APIRouter, HTTPException, Path
from ruamel.yaml import YAML, YAMLError
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from sd_optim.scorer import MODEL_DATA

from backend.schemas import (
    ConfigUpdate,
    OptimizationGuideUpdate,
    PayloadData,
    CargoData,
    # Import all other necessary Pydantic models from schemas.py
    HydraRunConfig, HydraConfig, WebUIUrlsConfig, RecipeOptimizationConfig,
    BayesAcquisitionFunctionConfig, BayesBoundsTransformerConfig, BayesConfig,
    OptunaSamplerConfig, OptunaConfig, OptimizerConfig, ImageGenerationConfig,
    GeneratorSettingsConfig, ScoringConfig, VisualizationsConfig,
    GroupStrategy, Strategy, GroupStrategyModel, Component
)


# --- APIRouter for Configuration Endpoints ---
router = APIRouter(
    prefix="/config",
    tags=["Configuration"],
)

# Define payloads_dir here as it's used by multiple endpoints
base_dir = Path(__file__).resolve().parent.parent.parent
payloads_dir = (base_dir / "conf" / "payloads").resolve()
config_tmpl_path = (base_dir / "conf" / "config.yaml").resolve()
guide_tmpl_path = (base_dir / "conf" / "optimization_guide" / "guide.yaml").resolve()


@router.get("/")
async def get_main_config() -> dict:
    """Reads the main configuration template and returns its content as JSON."""
    yaml = YAML(typ='rt')
    try:
        with open(config_tmpl_path, 'r', encoding='utf-8') as config_file:
            config_content = yaml.load(config_file)
        return config_content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Config file not found at {config_tmpl_path}.")
    except YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing config file: {e}")

@router.put("/")
async def update_config(config_data: ConfigUpdate):
    """Updates the main configuration template file."""
    yaml = YAML(typ='rt')
    try:
        with open(config_tmpl_path, 'r', encoding='utf-8') as f:
            existing_config = yaml.load(f)
        existing_config.update(config_data.model_dump(exclude_unset=True))
        with open(config_tmpl_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing_config, f)
        return {"message": "Config updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config file: {e}")

@router.get("/optimization-guide")
async def read_optimization_guide() -> dict:
    """Reads the optimization guide template and returns its content as JSON."""
    yaml = YAML(typ='rt')
    try:
        with open(guide_tmpl_path, 'r', encoding='utf-8') as guide_file:
            guide_config = yaml.load(guide_file)
        return guide_config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Optimization guide file not found at {guide_tmpl_path}.")
    except YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing optimization guide file: {e}")

@router.put("/optimization-guide")
async def update_optimization_guide(guide_data: OptimizationGuideUpdate):
    """Updates the optimization guide template file."""
    yaml = YAML(typ='rt')
    try:
        with open(guide_tmpl_path, 'r', encoding='utf-8') as f:
            existing_guide = yaml.load(f)
        existing_guide.update(guide_data.model_dump(exclude_unset=True))
        with open(guide_tmpl_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing_guide, f)
        return {"message": "Optimization guide updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating optimization guide file: {e}")

@router.get("/cargo")
async def list_cargo_files() -> list[str]:
    """Lists all available cargo template files."""
    cargo_files = [f.name for f in payloads_dir.glob("*.yaml") if f.is_file()]
    return cargo_files

@router.get("/cargo/{filename}")
async def read_cargo_file(filename: str):
    """Reads the content of a specific cargo template file."""
    yaml = YAML(typ='safe')
    cargo_path = payloads_dir / filename
    try:
        with open(cargo_path, 'r', encoding='utf-8') as cargo_file:
            cargo_content = yaml.load(cargo_file)
        return cargo_content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Cargo file '{filename}' not found.")
    except YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing cargo file '{filename}': {e}")

@router.put("/cargo/{filename}")
async def update_cargo_file(filename: str, cargo_data: CargoData):
    """Updates the content of a specific cargo template file."""
    file_path = payloads_dir / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Cargo file not found: {filename}")
    try:
        yaml = YAML(typ='rt')
        with open(file_path, "r", encoding="utf-8") as f:
            existing_cargo = yaml.load(f)
        existing_cargo.update(cargo_data.model_dump(exclude_unset=True))
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(existing_cargo, f)
        return {"message": f"Cargo file '{filename}' updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating cargo file: {e}")

@router.get("/cargo/{cargo_filename}/payloads")
async def list_payload_files(cargo_filename: str) -> list[str]:
    """Lists all available payload template files for a given cargo."""
    cargo_payloads_dir = payloads_dir / cargo_filename

    if not cargo_payloads_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Payloads directory for cargo '{cargo_filename}' not found.")

    payload_files = [f.name for f in cargo_payloads_dir.glob("*.yaml") if f.is_file()]
    return payload_files

@router.get("/cargo/{cargo_filename}/payloads/{payload_filename}")
async def read_payload_file(cargo_filename: str, payload_filename: str) -> dict:
    """Reads the content of a specific payload template file for a given cargo."""
    yaml = YAML(typ='safe')
    payload_path = payloads_dir / cargo_filename / payload_filename

    try:
        with open(payload_path, 'r', encoding='utf-8') as payload_file:
            payload_content = yaml.load(payload_file)
        return payload_content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Payload file '{payload_filename}' for cargo '{cargo_filename}' not found.")
    except YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing payload file '{payload_filename}' for cargo '{cargo_filename}': {e}")

@router.put("/cargo/{cargo_filename}/payloads/{payload_filename}")
async def update_payload_file(cargo_filename: str, payload_filename: str, payload_data: PayloadData):
    """Updates the content of a specific payload template file for a given cargo."""
    payload_path = payloads_dir / cargo_filename / payload_filename

    if not payload_path.is_file():
        raise HTTPException(status_code=404, detail=f"Payload file '{payload_filename}' for cargo '{cargo_filename}' not found.")

    try:
        yaml = YAML(typ='rt')
        with open(payload_path, "r", encoding="utf-8") as f:
            existing_payload = yaml.load(f)
        existing_payload.update(payload_data.model_dump(exclude_unset=True))
        with open(payload_path, "w", encoding="utf-8") as f:
            yaml.dump(existing_payload, f)
        return {"message": f"Payload file '{payload_filename}' for cargo '{cargo_filename}' updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating payload file: {e}")

# --- ADD THIS NEW ENDPOINT ---
@router.get("/scorers")
async def list_available_scorers() -> List[str]:
    """
    Dynamically lists all available scorer identifiers from the project configuration.
    """
    try:
        # Get all keys from our MODEL_DATA dictionary in scorer.py
        available_scorers = list(MODEL_DATA.keys())
        
        # Add the special, non-model-based scorers
        available_scorers.extend(['manual', 'background_blackness'])
        
        # Return a sorted, unique list for a clean UI
        return sorted(list(set(available_scorers)))
    except Exception as e:
        # This shouldn't fail, but it's good practice to have error handling
        raise HTTPException(status_code=500, detail=f"Failed to list scorers: {e}")

# --- ADD THIS NEW ENDPOINT to the file ---
@router.get("/cargo/{filename}/defined-payloads")
async def get_defined_payloads_from_cargo(filename: str) -> List[str]:
    """
    Parses a specific cargo YAML file to extract the list of defined payloads.
    """
    yaml = YAML(typ='safe')  # Use safe loader
    cargo_path = payloads_dir / filename
    if not cargo_path.is_file():
        raise HTTPException(status_code=404, detail=f"Cargo file '{filename}' not found.")

    try:
        with open(cargo_path, 'r', encoding='utf-8') as f:
            cargo_content = yaml.load(f)
        
        # Navigate the YAML structure to find the list of payloads
        # It's inside 'defaults', which is a list, and the item we want has a 'cargo' key.
        payload_list = []
        if 'defaults' in cargo_content and isinstance(cargo_content['defaults'], list):
            for item in cargo_content['defaults']:
                if isinstance(item, dict) and 'cargo' in item:
                    # Found it!
                    payload_list = item['cargo']
                    break
        
        if not isinstance(payload_list, list):
            # Handle cases where the structure is wrong or 'cargo' isn't a list
            return []

        return payload_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing cargo file '{filename}': {e}")


# --- CARGO TEMPLATES ---
@router.get("/cargo_templates")
async def list_cargo_templates():
    """Lists all available cargo template files."""
    # Returns ["cargo_forge.yaml", "cargo_comfy.yaml"]

@router.get("/cargo_templates/{template_name}")
async def get_cargo_template(template_name: str):
    """Gets the full contents of a specific cargo template."""
    # Returns the full YAML content as JSON

# --- PAYLOADS (Full CRUD!) ---
@router.get("/payloads")
async def list_payloads():
    """Lists all available payload files."""
    # Returns ["my_character.yaml", "style_test.yaml"]

@router.get("/payloads/{payload_name}")
async def get_payload(payload_name: str):
    """Gets the contents of a specific payload file."""
    # Returns the payload's YAML content as JSON

@router.post("/payloads")
async def create_payload(payload_data: dict):
    """Creates a new payload .yaml file."""
    # Expects a JSON body with a 'filename' and 'content'
    # Saves the content to a new file in conf/payloads/

@router.put("/payloads/{payload_name}")
async def update_payload(payload_name: str, payload_data: dict):
    """Updates an existing payload .yaml file."""
    # Overwrites the specified file with new content

@router.delete("/payloads/{payload_name}")
async def delete_payload(payload_name: str):
    """Deletes a payload .yaml file."""
    # Deletes the specified file from conf/payloads/