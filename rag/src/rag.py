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
    # Fetch initial results
    results = retrieve(request.query, model, client, retrieval_k=20, final_k=request.top_k)

    # 1. Define the minimum acceptable similarity score
    MIN_THRESHOLD = 0.45

    # 2. Filter out low-confidence documents
    valid_results = [res for res in results if res["final_score"] >= MIN_THRESHOLD]

    # 3. Restrict to maximum of 2 documents
    limited_results = valid_results[:2]

    # Format the results for the backend
    documents = [res["result"].payload.get("text", "") for res in limited_results]
    scores = [res["final_score"] for res in limited_results]

    return {"documents": documents, "scores": scores}
