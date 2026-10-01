import os
import uuid
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from routers import chat
from database import init_db
from schemas import LoginRequest
from rate_limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

load_dotenv()

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(
    title="Legal LLM Backend",
    version="1.0",
    description="Vietnamese legal consulting system with RAG and LLM integration"
)

# 0. Initialize the async database when the server starts
@app.on_event("startup")
async def startup_event():
    await init_db()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 1. Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Serve the chat interface
@app.get("/chat")
def serve_chat_ui():
    return FileResponse("static/index.html")

app.include_router(chat.router)

# LOG-IN (IMPORTANT)
@app.post("/mock-login")
def mock_login(request: LoginRequest):
    # Generates a deterministic UUID based on the username string
    user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, request.username)
    return {"user_id": str(user_uuid)}

@app.get("/")
def health_check():
    return {
        "status": "Backend server is running live",
        "chat_ui": "http://localhost:8000/chat"
    }

@app.get("/health")
def quick_health():
    return {"status": "ok"}
