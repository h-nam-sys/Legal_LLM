import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# Encoding
# ============================================================

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# Global RAG components
# ============================================================

model = None
client = None


# ============================================================
# Request / Response
# ============================================================

class RetrieveRequest(BaseModel):
    query: str


class RetrieveResponse(BaseModel):
    documents: list[str]
    scores: list[float]


# ============================================================
# Startup / Shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, client

    print("[RAG] Loading model...", file=sys.stderr, flush=True)

    from retrieval import load_model, connect_qdrant

    model = load_model()

    print("[RAG] Connecting to Qdrant...", file=sys.stderr, flush=True)

    client = connect_qdrant()

    print("[RAG] RAG service ready", file=sys.stderr, flush=True)

    yield

    if client is not None:
        client.close()
        print("[RAG] Qdrant client closed", file=sys.stderr, flush=True)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Vietnamese Legal RAG Service",
    description="RAG retrieval service using BKAI Embedding and Qdrant",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "vietnamese-legal-rag"
    }


# ============================================================
# Retrieve
# ============================================================

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_documents(request: RetrieveRequest):

    global model, client

    if model is None or client is None:
        raise RuntimeError("RAG components are not initialized")

    from retrieval import retrieve

    print(
        f"[RAG] Query: {request.query}",
        file=sys.stderr,
        flush=True
    )

    results = retrieve(
        request.query,
        model,
        client
    )

    documents = []
    scores = []

    for item in results:

        result = item["result"]

        payload = result.payload or {}

        text = payload.get("text", "")

        score = float(item["final_score"])

        documents.append(text)
        scores.append(score)

    print(
        f"[RAG] Returned {len(documents)} documents",
        file=sys.stderr,
        flush=True
    )

    return {
        "documents": documents,
        "scores": scores
    }


# ============================================================
# Local Run
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "rag:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )