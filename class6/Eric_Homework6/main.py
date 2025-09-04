# main.py
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.chat import router as chat_router

app = FastAPI(title="Week6 Voice Agent (Function Calling)")

# Mount the ./static directory at /static
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# include routes
app.include_router(chat_router)

# health check
@app.get("/")
def root():
    return {"status": "ok", "service": "Week6 Voice Agent"}
