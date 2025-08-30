# # main.py
# from fastapi import FastAPI
# from routes.chat import router as chat_router

# app = FastAPI(title="Week3 Voice Agent")

# # include routes
# app.include_router(chat_router)

# # health check
# @app.get("/")
# def root():
#     return {"status": "ok", "service": "Week3 Voice Agent"}
from fastapi import FastAPI
from routes.chat import router as chat_router

# NEW: add this import
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Week3 Voice Agent")

# Mount the ./static directory at /static
# Uses a path relative to where you launch uvicorn (your project root)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# include routes
app.include_router(chat_router)

@app.get("/")
def root():
    return {"status": "ok", "service": "Week3 Voice Agent"}
