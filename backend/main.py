import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from routers import chat
from database import init_db

load_dotenv()
init_db()

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(
    title="Legal LLM Backend",
    version="1.0",
    description="Vietnamese legal consulting system with RAG and LLM integration"
)

# 1. Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Serve the chat interface
@app.get("/chat")
def serve_chat_ui():
    return FileResponse("static/index.html")

app.include_router(chat.router)

@app.get("/")
def health_check():
    return {
        "status": "Backend server is running live",
        "chat_ui": "http://localhost:8000/chat"
    }

@app.get("/health")
def quick_health():
    return {"status": "ok"}
