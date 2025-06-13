from fastapi import APIRouter, HTTPException, Body, WebSocket, WebSocketDisconnect
from backend.services.optimization_manager import optimization_manager
from backend.schemas import ConfigUpdate # Import the ConfigUpdate Pydantic model
from typing import Dict, Any

router = APIRouter(
    prefix="/optimization",
    tags=["Optimization"],
)

@router.post("/start")
async def start_optimization_endpoint(config: ConfigUpdate = Body(...)) -> Dict[str, Any]:
    """Starts a new optimization process."""
    try:
        response = await optimization_manager.start_optimization(config)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {e}")

@router.post("/pause")
async def pause_optimization_endpoint() -> Dict[str, Any]:
    """Pauses the ongoing optimization process."""
    response = await optimization_manager.pause_optimization()
    return response

@router.post("/resume")
async def resume_optimization_endpoint() -> Dict[str, Any]:
    """Resumes a paused optimization process."""
    response = await optimization_manager.resume_optimization()
    return response

@router.post("/cancel")
async def cancel_optimization_endpoint() -> Dict[str, Any]:
    """Cancels the ongoing optimization process."""
    response = await optimization_manager.cancel_optimization()
    return response

@router.get("/status")
async def get_optimization_status() -> Dict[str, Any]:
    """Gets the current status of the optimization process."""
    return optimization_manager.get_status()

@router.websocket("/ws")
async def websocket_status_endpoint(websocket: WebSocket):
 """WebSocket endpoint for real-time optimization status updates."""
    await websocket.accept()
    optimization_manager.add_websocket(websocket)
    try:
 # Send initial status on connection
        initial_status = optimization_manager.get_status()
        await websocket.send_json(initial_status)
        # Keep the connection open
 while True:
 await websocket.receive_text() # Just receive to keep connection open, manager pushes updates
    except WebSocketDisconnect:
        optimization_manager.remove_websocket(websocket)
# Future: Endpoints for getting detailed history, best parameters etc.
# These would likely call methods on the optimization_manager that retrieve data
# from the optimizer_instance or its study (for Optuna).
# @router.get("/history")
# async def get_optimization_history():
#     if optimization_manager.optimizer_instance:
#         return optimization_manager.optimizer_instance.get_optimization_history()
#     return []

# @router.get("/best-parameters")
# async def get_best_parameters():
#     if optimization_manager.optimizer_instance:
#         return optimization_manager.optimizer_instance.get_best_parameters()
#     return {}
