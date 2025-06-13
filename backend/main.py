from fastapi import FastAPI

from backend.api import config, optimization

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "SD-Optim Standalone Backend"}

# Include the configuration router
app.include_router(config.router)

# Include the optimization router
app.include_router(optimization.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
