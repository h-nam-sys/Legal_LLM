import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

model = None
client = None
# The 'procedures' global variable is removed since retrieval.py handles it internally.

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrieveResponse(BaseModel):
    documents: list[str]
    scores: list[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, client
    print("[RAG] Loading model...", file=sys.stderr, flush=True)

    # Import from your new retrieval script
    from retrieval import load_model, connect_qdrant

    model = load_model()
    print("[RAG] Connecting to Qdrant...", file=sys.stderr, flush=True)
    client = connect_qdrant()

    # The load_procedure_names call has been completely removed to fix the startup crash.

    print("[RAG] RAG service ready", file=sys.stderr, flush=True)
    yield

    if client is not None:
        client.close()
        print("[RAG] Qdrant client closed", file=sys.stderr, flush=True)

app = FastAPI(
    title="Vietnamese Legal RAG Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "vietnamese-legal-rag"}

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_documents(request: RetrieveRequest):
    global model, client
    if model is None or client is None:
        raise RuntimeError("RAG components are not initialized")

    from retrieval import retrieve

    # Removed the 'procedures=procedures' argument because the updated
    # retrieve() function in retrieval.py no longer accepts it.
    results = retrieve(
        query=request.query,
        model=model,
        client=client,
        top_k=20,
        final_top_k=request.top_k
    )

    MIN_THRESHOLD = 0.4

    # Access object attributes (.final_score and .text) instead of dict keys
    valid_results = [res for res in results if res.final_score >= MIN_THRESHOLD]
    limited_results = valid_results[:1]

    documents = [res.text for res in limited_results]
    scores = [res.final_score for res in limited_results]

    return {"documents": documents, "scores": scores}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag:app", host="0.0.0.0", port=8001, reload=False)
