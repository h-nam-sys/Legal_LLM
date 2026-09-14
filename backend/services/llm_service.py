import httpx
import os
import random
from rapidfuzz import fuzz
from schemas import LLMServiceRequest, LLMServiceResponse, RAGServiceRequest, RAGServiceResponse

# IMPORT YOUR NEW DISCRETE MCP MODULE
from mcp_service import should_trigger_mcp, execute_mcp_search

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")

NUDGE_MESSAGES = [
    "Xin chào! Tôi là trợ lý hướng dẫn thủ tục hành chính. Vui lòng cho tôi biết bạn cần làm thủ tục gì nhé (ví dụ: đăng ký khai sinh, kết hôn, xác nhận tình trạng hôn nhân...).",
    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thủ tục hành chính công. Bạn đang cần thực hiện thủ tục nào?",
    "Dạ chào bạn. Bạn cần tôi hướng dẫn giấy tờ, địa điểm nộp hay lệ phí cho thủ tục hành chính nào hôm nay?",
]

DOCUMENT_ALIASES = {
    "Chứng minh nhân dân": ["cccd", "cmnd", "căn cước", "chứng minh", "hộ chiếu"],
    "Giấy chứng sinh": ["chứng sinh", "giấy viện cấp", "giấy sinh", "giấy chứng sinh"],
    "Giấy chứng nhận kết hôn": ["kết hôn", "đăng ký kết hôn", "giấy kết hôn"],
    "Quyết định ly hôn": ["giấy ly hôn", "quyết định ly hôn", "giấy toà", "ly hôn"],
    "Giấy xác nhận thông tin cư trú": ["cư trú", "ct07", "hộ khẩu", "giấy cư trú", "tạm trú", "giấy xác nhận thông tin cư trú"]
}

def calculate_missing_docs(user_history: str, rag_context_text: str, threshold: int = 85) -> list[str]:
    history_lower = user_history.lower()
    rag_lower = rag_context_text.lower()
    missing_docs = []

    for official_doc, aliases in DOCUMENT_ALIASES.items():
        if official_doc.lower() in rag_lower:
            has_doc = False
            for alias in aliases:
                similarity_score = fuzz.partial_ratio(alias, history_lower)
                if similarity_score >= threshold:
                    has_doc = True
                    break
            if not has_doc:
                missing_docs.append(official_doc)

    return missing_docs

async def fetch_rag_context(query: str) -> tuple[list[str], list[float]]:
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
        except Exception:
            return [], []

async def get_legal_response(prompt: str, search_query: str) -> tuple[str, list[str], float]:
    context_docs, scores = await fetch_rag_context(search_query)

    max_score = max(scores) if scores else 0.0
    SCORE_THRESHOLD = 0.45

    # Extract just the latest message to determine routing intent
    latest_user_msg = prompt.split(" . ")[-1].lower().strip()

    # 1. RAG Failed: Route to MCP or Reject
    if max_score < SCORE_THRESHOLD or not context_docs:
        if should_trigger_mcp(latest_user_msg):
            print(f"[ROUTER] Valid admin query not in local DB. Triggering MCP for: {latest_user_msg}")
            answer, sources = await execute_mcp_search(latest_user_msg)
            return answer, sources, max_score
        else:
            print("[ROUTER] Unrelated chat. Nudging user.")
            return random.choice(NUDGE_MESSAGES), [], max_score

    # 2. RAG Succeeded: Normal Local Flow
    rag_context_str = " ".join(context_docs)

    # Rerouting trigger words for local DB results
    is_fee = any(word in latest_user_msg for word in ["tiền", "phí", "lệ phí", "nhiêu"])
    is_location = any(word in latest_user_msg for word in ["ở đâu", "nơi nào", "địa điểm", "nộp"])
    is_time = any(word in latest_user_msg for word in ["bao lâu", "thời gian", "ngày"])

    if is_fee or is_location or is_time:
        # Route 1: Direct Q&A (Bypass missing documents logic entirely)
        formatted_prompt = f"""Dựa vào tài liệu pháp lý dưới đây, hãy trả lời ngắn gọn, trực tiếp câu hỏi của người dân: "{latest_user_msg}"

TÀI LIỆU:
{rag_context_str}

TUYỆT ĐỐI không liệt kê giấy tờ thủ tục nếu người dân không hỏi. Chỉ trả lời đúng trọng tâm câu hỏi."""
    else:
        # Route 2: Default Missing Documents Flow
        missing_items = calculate_missing_docs(prompt, rag_context_str)
        missing_str = ", ".join(missing_items) if missing_items else "Không thiếu giấy tờ cơ bản."

        formatted_prompt = f"""Dựa vào TÀI LIỆU RAG, hãy đóng vai cán bộ thân thiện trả lời người dân.

THÔNG TIN BẮT BUỘC PHẢI THÔNG BÁO CHO NGƯỜI DÂN:
- Giấy tờ họ đang thiếu: {missing_str}

TÀI LIỆU RAG (Dùng để xem lệ phí và địa điểm):
{rag_context_str}

YÊU CẦU: KHÔNG lặp lại hướng dẫn. Viết một câu chào hỏi và nhắc họ chuẩn bị giấy tờ thiếu.

CÁN BỘ TRẢ LỜI:"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            llm_request = LLMServiceRequest(prompt=formatted_prompt, context=context_docs)
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json=llm_request.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            llm_result = LLMServiceResponse(**response.json())
            return llm_result.answer, llm_result.sources, max_score
        except Exception as e:
            return f"Lỗi xử lý hệ thống: {str(e)}", [], 0.0
