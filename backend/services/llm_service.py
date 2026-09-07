import httpx
import os
from schemas import LLMServiceRequest, LLMServiceResponse, RAGServiceRequest, RAGServiceResponse

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")

async def fetch_rag_context(query: str) -> tuple[list[str], list[float]]:
    """Fetch legal document context from RAG service"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            rag_request = RAGServiceRequest(query=query, top_k=5)
            response = await client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json=rag_request.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            rag_result = RAGServiceResponse(**response.json())
            return rag_result.documents, rag_result.scores
        except httpx.ConnectError:
            print(f"[RAG] Service not available at {RAG_SERVICE_URL}")
            return [], []
        except httpx.HTTPStatusError as e:
            print(f"[RAG] HTTP Error {e.response.status_code}: {e}")
            return [], []
        except Exception as e:
            print(f"[RAG] Unexpected error: {e}")
            return [], []

async def get_legal_response(prompt: str) -> tuple[str, list[str]]:
    """Main orchestration: RAG → LLM pipeline"""
    # Step 1: Fetch context from RAG
    context_docs, scores = await fetch_rag_context(prompt)

    # Step 2: Send prompt + context to LLM
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            llm_request = LLMServiceRequest(prompt=prompt, context=context_docs)
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json=llm_request.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            llm_result = LLMServiceResponse(**response.json())
            return llm_result.answer, llm_result.sources
        except httpx.ConnectError:
            fallback_msg = f"LLM service unavailable at {LLM_SERVICE_URL}. Please try again shortly."
            return fallback_msg, []
        except httpx.HTTPStatusError as e:
            fallback_msg = f"LLM service error: {e.response.status_code}. Please try again."
            return fallback_msg, []
        except Exception as e:
            return f"Error processing request: {str(e)}", []
