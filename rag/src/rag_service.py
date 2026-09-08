from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_documents_bkai"
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


# ============================================================
# INITIALIZE
# ============================================================

app = FastAPI(
    title="Legal RAG Retrieval Service",
    description="Vietnamese Legal Document Retrieval Service",
    version="1.0.0",
)

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL
)

print("Connected to Qdrant!")

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded!")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrieveResponse(BaseModel):
    documents: list[str]
    scores: list[float]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "rag",
        "collection": COLLECTION_NAME
    }


# ============================================================
# RETRIEVE
# ============================================================

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):

    try:

        # 1. Encode query
        query_vector = model.encode(
            request.query,
            normalize_embeddings=True
        )

        # 2. Search Qdrant
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=request.top_k,
        ).points

        # 3. Prepare response
        documents = []
        scores = []

        for result in results:

            documents.append(
                result.payload.get("text", "")
            )

            scores.append(
                float(result.score)
            )

        # 4. Return UTF-8 JSON
        return JSONResponse(
            content={
                "documents": documents,
                "scores": scores
            },
            media_type="application/json"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"RAG retrieval error: {str(e)}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002
    )