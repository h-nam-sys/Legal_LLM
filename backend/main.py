from fastapi import FastAPI
from dotenv import load_dotenv
import os
from routers import chat
from database import init_db

# Load environment variables from .env file
load_dotenv()

# Initialize database tables
init_db()

# Read service URLs from environment
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(
    title="Legal LLM Backend",
    version="1.0",
    description="Vietnamese legal consulting system with RAG and LLM integration"
)

# Include routers
app.include_router(chat.router)

@app.get("/")
def health_check():
    return {
        "status": "Backend server is running live",
        "version": "1.0",
        "llm_service_url": LLM_SERVICE_URL,
        "rag_service_url": RAG_SERVICE_URL,
        "log_level": LOG_LEVEL,
        "database": "SQLite initialized"
    }

@app.get("/health")
def quick_health():
    return {"status": "ok"}
