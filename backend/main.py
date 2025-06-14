from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import config, optimization

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")  # Change this to avoid conflict
async def read_root():
    return {"message": "SD-Optim Standalone Backend"}

# Include API routers FIRST
app.include_router(config.router)
app.include_router(optimization.router)

# Mount static files LAST (catches remaining requests)
app.mount("/", StaticFiles(directory="/home/user/sdoptimui/sd-optim/frontend/build", html=True), name="static")