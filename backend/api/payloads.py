from fastapi import APIRouter, HTTPException, Body
from ruamel.yaml import YAML
from pathlib import Path
from typing import Dict, List, Any

from backend.schemas import PayloadUpdate, PayloadCreate

# --- APIRouter for Payloads and Presets ---
router = APIRouter(
    prefix="/payloads",  # The main prefix for this whole section
    tags=["Payload Workshop"],
)

# --- Define Directories ---
base_dir = Path(__file__).resolve().parent.parent.parent
payloads_dir = (base_dir / "conf" / "payloads").resolve()
presets_dir = (base_dir / "conf" / "webui_presets").resolve()

# --- WebUI Presets ---
@router.get("/presets")
async def list_webui_presets() -> List[str]:
    """Lists all available WebUI preset files."""
    if not presets_dir.is_dir():
        raise HTTPException(status_code=404, detail="WebUI presets directory not found.")
    preset_files = [f.name for f in presets_dir.glob("*.yaml") if f.is_file()]
    return preset_files

@router.get("/presets/{preset_name}")
async def get_webui_preset(preset_name: str) -> Dict:
    """Gets the contents of a specific WebUI preset file."""
    if not preset_name.endswith('.yaml'):
        preset_name += '.yaml'
    preset_path = presets_dir / preset_name
    if not preset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Preset file not found: {preset_name}")
    
    yaml = YAML(typ='safe')
    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            return yaml.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading preset file '{preset_name}': {e}")

# --- Payload Management (Full CRUD) ---
@router.get("/")
async def list_payloads() -> List[str]:
    """Lists all available payload files by name (without extension)."""
    if not payloads_dir.is_dir():
        raise HTTPException(status_code=404, detail="Payloads directory not found.")
    payload_files = [f.stem for f in payloads_dir.glob("*.yaml") if f.is_file()]
    return payload_files

@router.get("/{payload_name}")
async def get_payload(payload_name: str) -> Dict:
    """Gets the contents of a specific payload file."""
    payload_path = payloads_dir / f"{payload_name}.yaml"
    if not payload_path.is_file():
        raise HTTPException(status_code=404, detail=f"Payload file not found: {payload_name}")

    yaml = YAML(typ='safe')
    try:
        with open(payload_path, 'r', encoding='utf-8') as f:
            return yaml.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading payload file '{payload_name}': {e}")

@router.post("/", status_code=201)
async def create_payload(payload_data: PayloadCreate):
    """Creates a new payload .yaml file."""
    file_path = payloads_dir / f"{payload_data.filename}.yaml"
    if file_path.exists():
        raise HTTPException(status_code=409, detail=f"Payload file '{payload_data.filename}' already exists.")

    yaml = YAML(typ='rt')
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(payload_data.content, f)
        return {"message": f"Payload '{payload_data.filename}' created successfully.", "filename": payload_data.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating payload file: {e}")

@router.put("/{payload_name}")
async def update_payload(payload_name: str, payload_data: PayloadUpdate):
    """Updates an existing payload .yaml file."""
    file_path = payloads_dir / f"{payload_name}.yaml"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Payload file not found: {payload_name}")
    
    yaml = YAML(typ='rt')
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(payload_data.content, f)
        return {"message": f"Payload '{payload_name}' updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating payload file: {e}")

@router.delete("/{payload_name}", status_code=204)
async def delete_payload(payload_name: str):
    """Deletes a payload .yaml file."""
    file_path = payloads_dir / f"{payload_name}.yaml"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Payload file not found: {payload_name}")
    
    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting payload file: {e}")