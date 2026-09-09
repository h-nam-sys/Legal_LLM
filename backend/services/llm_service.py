import httpx
import os
import re
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

def is_chit_chat(prompt: str) -> bool:
    clean_prompt = prompt.strip().lower()
    word_count = len(clean_prompt.split())

    # 1. Catch exact standalone conversational phrases
    exact_patterns = r"^(hi|hello|hey|alo|chào|xin chào|cảm ơn|bạn là ai|help|bye|tạm biệt|ok|vâng|dạ)( there| bạn| bot)?[\.!\?]*$"
    if re.match(exact_patterns, clean_prompt):
        return True

    # 2. 4-word rule: Short prompts containing chatty or insult words bypass RAG
    chatty_words = ["hello", "hi", "hey", "chào", "stupid", "ngu", "bot", "cảm ơn", "haha", "ok", "dở hơi"]
    if word_count <= 4 and any(word in clean_prompt for word in chatty_words):
        return True

    return False

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
    """Main orchestration: Always RAG pipeline with smart intercept"""

    # 1. Fast intercept for greetings and short garbage queries
    if is_chit_chat(prompt):
        # Pick a random response from the array
        nudge_message = random.choice(NUDGE_MESSAGES)
        return nudge_message, []

    # 2. Always RAG for everything else
    context_docs, scores = await fetch_rag_context(prompt)

    # 3. Send to LLM
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
            return f"LLM service unavailable at {LLM_SERVICE_URL}. Please try again shortly.", []
        except httpx.HTTPStatusError as e:
            return f"LLM service error: {e.response.status_code}. Please try again.", []
        except Exception as e:
            return f"Error processing request: {str(e)}", []
