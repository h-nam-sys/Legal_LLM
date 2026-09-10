import httpx
import os
import random
from schemas import LLMServiceRequest, LLMServiceResponse, RAGServiceRequest, RAGServiceResponse

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")

NUDGE_MESSAGES = [
    "Xin chào! Tôi là trợ lý tư vấn pháp lý. Vui lòng nhập câu hỏi pháp luật bạn muốn tìm hiểu (ví dụ: luật hôn nhân, hình sự, doanh nghiệp...).",
    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thắc mắc về pháp luật Việt Nam. Bạn đang cần tìm hiểu về vấn đề pháp lý nào?",
    "Dạ chào bạn. Bạn cần tôi tư vấn về quy định pháp luật hay thủ tục hành chính nào hôm nay?",
    "Xin chào! Là một trợ lý AI chuyên về luật pháp, tôi có thể giúp gì cho bạn trong các vấn đề như dân sự, đất đai, hay lao động?",
    "Chào bạn, rất vui được hỗ trợ! Hãy đặt câu hỏi liên quan đến pháp luật Việt Nam để tôi có thể tư vấn chính xác nhất nhé."
]

async def fetch_rag_context(query: str) -> tuple[list[str], list[float]]:
    print(f"\n[DEBUG] Calling RAG Service at {RAG_SERVICE_URL}/retrieve with query: '{query}'")
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
            print(f"[DEBUG] RAG Response received. Documents count: {len(rag_result.documents)}")
            return rag_result.documents, rag_result.scores
        except Exception as e:
            print(f"[ERROR] RAG Service unexpected error: {e}")
            return [], []

async def get_legal_response(prompt: str, search_query: str) -> tuple[str, list[str]]:
    print(f"\n==========================================")
    print(f"[DEBUG] Original Prompt: '{prompt}'")
    print(f"[DEBUG] Search Query for RAG: '{search_query}'")
    print(f"==========================================")

    # 1. Trigger RAG using the bundled query
    context_docs, scores = await fetch_rag_context(search_query)

    max_score = max(scores) if scores else 0.0
    SCORE_THRESHOLD = 0.45
    print(f"[DEBUG] Max Similarity Score: {max_score:.4f} (Threshold: {SCORE_THRESHOLD})")

    if max_score < SCORE_THRESHOLD or not context_docs:
        print(f"[DEBUG] Score is BELOW threshold. Triggering casual nudge message.")
        return random.choice(NUDGE_MESSAGES), []

    print(f"[DEBUG] Score is ABOVE threshold. Forwarding short prompt to LLM Service.")

    # 2. Send only the short prompt to the LLM
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
        except Exception as e:
            return f"Error processing request: {str(e)}", []
