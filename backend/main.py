from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Now we import all our clean, single-responsibility routers!
from backend.api import config, optimization, payloads, optimization_guide

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
async def read_root():
    return {"message": "SD-Optim Standalone Backend"}

# Include all our API routers
app.include_router(config.router)
app.include_router(optimization.router)
app.include_router(payloads.router)
app.include_router(optimization_guide.router) # <-- Add the new one!

# Mount static files LAST
app.mount("/", StaticFiles(directory="/home/user/sdoptimui/sd-optim/frontend/build", html=True), name="static")