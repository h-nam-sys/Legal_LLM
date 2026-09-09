from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import retrieve, load_model, connect_qdrant

app = FastAPI()

# Load model and DB globally on startup
model = load_model()
client = connect_qdrant()

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/retrieve")
async def retrieve_docs(request: RAGRequest):
    results = retrieve(request.query, model, client, retrieval_k=20, final_k=request.top_k)

    # Format the results to match what backend expects
    documents = [res["result"].payload.get("text", "") for res in results]
    scores = [res["final_score"] for res in results]

    return {"documents": documents, "scores": scores}
