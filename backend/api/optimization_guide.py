from fastapi import APIRouter, HTTPException
from ruamel.yaml import YAML, YAMLError
from pathlib import Path

from backend.schemas import OptimizationGuideUpdate

# --- APIRouter for Optimization Guide ---
router = APIRouter(
    prefix="/optimization-guide",
    tags=["Optimization Guide"],
)

# --- Define Directory ---
base_dir = Path(__file__).resolve().parent.parent.parent
guide_path = (base_dir / "conf" / "optimization_guide" / "guide.yaml").resolve()

@router.get("/")
async def read_optimization_guide() -> dict:
    """Reads the optimization guide and returns its content as JSON."""
    yaml = YAML(typ='rt')
    try:
        with open(guide_path, 'r', encoding='utf-8') as guide_file:
            guide_config = yaml.load(guide_file)
        return guide_config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Optimization guide file not found at {guide_path}.")
    except YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing optimization guide file: {e}")

@router.put("/")
async def update_optimization_guide(guide_data: OptimizationGuideUpdate):
    """Updates the optimization guide file."""
    yaml = YAML(typ='rt')
    try:
        with open(guide_path, 'r', encoding='utf-8') as f:
            existing_guide = yaml.load(f)
        
        update_data = guide_data.model_dump(exclude_unset=True)
        existing_guide.update(update_data)

        with open(guide_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing_guide, f)
        return {"message": "Optimization guide updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating optimization guide file: {e}")