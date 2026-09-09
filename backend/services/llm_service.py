import httpx
import os
import random
from schemas import LLMServiceRequest, LLMServiceResponse, RAGServiceRequest, RAGServiceResponse

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")

# Array of varied nudge responses to feign natural AI conversation
NUDGE_MESSAGES = [
    "Xin chào! Tôi là trợ lý tư vấn pháp lý. Vui lòng nhập câu hỏi pháp luật bạn muốn tìm hiểu (ví dụ: luật hôn nhân, hình sự, doanh nghiệp...).",
    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thắc mắc về pháp luật Việt Nam. Bạn đang cần tìm hiểu về vấn đề pháp lý nào?",
    "Dạ chào bạn. Bạn cần tôi tư vấn về quy định pháp luật hay thủ tục hành chính nào hôm nay?",
    "Xin chào! Là một trợ lý AI chuyên về luật pháp, tôi có thể giúp gì cho bạn trong các vấn đề như dân sự, đất đai, hay lao động?",
    "Chào bạn, rất vui được hỗ trợ! Hãy đặt câu hỏi liên quan đến pháp luật Việt Nam để tôi có thể tư vấn chính xác nhất nhé."
]

async def fetch_rag_context(query: str) -> tuple[list[str], list[float]]:
    """Fetch legal document context and similarity scores from RAG service with logging"""
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
            print(f"[DEBUG] RAG Response received. Documents count: {len(rag_result.documents)}, Scores: {rag_result.scores}")
            return rag_result.documents, rag_result.scores
        except httpx.ConnectError as e:
            print(f"[ERROR] RAG Service connection failed at {RAG_SERVICE_URL}: {e}")
            return [], []
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] RAG Service HTTP Error {e.response.status_code}: {e}")
            return [], []
        except Exception as e:
            print(f"[ERROR] RAG Service unexpected error: {e}")
            return [], []

async def get_legal_response(prompt: str) -> tuple[str, list[str]]:
    """Main orchestration with full diagnostic printing"""
    print(f"\n==========================================")
    print(f"[DEBUG] Incoming User Prompt: '{prompt}'")
    print(f"==========================================")

    # 1. Trigger RAG first
    context_docs, scores = await fetch_rag_context(prompt)

    # 2. Evaluate scores
    max_score = max(scores) if scores else 0.0
    SCORE_THRESHOLD = 0.35  # Adjust threshold here if needed
    print(f"[DEBUG] Max Similarity Score: {max_score:.4f} (Threshold: {SCORE_THRESHOLD})")

    # 3. Branch based on score
    if max_score < SCORE_THRESHOLD or not context_docs:
        print(f"[DEBUG] Score is BELOW threshold. Triggering casual nudge message.")
        nudge_message = random.choice(NUDGE_MESSAGES)
        return nudge_message, []

    print(f"[DEBUG] Score is ABOVE threshold. Forwarding to LLM Service at {LLM_SERVICE_URL}/generate")

    # 4. Send to LLM Service
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
            print(f"[DEBUG] LLM Response successfully generated.")
            return llm_result.answer, llm_result.sources
        except httpx.ConnectError as e:
            print(f"[ERROR] LLM Service connection failed at {LLM_SERVICE_URL}: {e}")
            return f"LLM service unavailable at {LLM_SERVICE_URL}. Please try again shortly.", []
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] LLM Service HTTP Error {e.response.status_code}: {e}")
            return f"LLM service error: {e.response.status_code}. Please try again.", []
        except Exception as e:
            print(f"[ERROR] LLM Service unexpected error: {e}")
            return f"Error processing request: {str(e)}", []
