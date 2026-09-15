import httpx
import os
import json
import random
from schemas import LLMServiceRequest, LLMServiceResponse, RAGServiceRequest, RAGServiceResponse

from mcp_service import should_trigger_mcp, execute_mcp_search

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8002")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8001")
CONFIG_PATH = os.getenv("PROMPTS_CONFIG_PATH", "prompts.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    PROMPT_CONFIG = json.load(f)

NUDGE_MESSAGES = [
    "Xin chào! Tôi là trợ lý hướng dẫn thủ tục hành chính. Vui lòng cho tôi biết bạn cần làm thủ tục gì nhé (ví dụ: đăng ký khai sinh, kết hôn...).",
    "Chào bạn! Tôi ở đây để hỗ trợ giải đáp các thủ tục hành chính công. Bạn đang cần thực hiện thủ tục nào?",
]

async def fetch_rag_context(query: str) -> tuple[list[str], list[float]]:
    print(f"\n[DIAGNOSTIC] Sending RAG request to: {RAG_SERVICE_URL}/retrieve")
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
        except Exception as e:
            print(f"[DIAGNOSTIC] FATAL EXCEPTION: {str(e)}")
            return [], []

async def get_legal_response(prompt: str, search_query: str) -> tuple[str, list[str], float]:
    try:
        context_docs, scores = await fetch_rag_context(search_query)

        max_score = max(scores) if scores else 0.0
        SCORE_THRESHOLD = 0.45

        latest_user_msg = prompt.split(" . ")[-1].lower().strip()

        # 1. RAG Failed: Route to MCP or Nudge
        if max_score < SCORE_THRESHOLD or not context_docs:
            if should_trigger_mcp(latest_user_msg):
                try:
                    answer, sources = await execute_mcp_search(latest_user_msg)
                    return answer, sources, max_score
                except Exception as mcp_e:
                    return f"Lỗi hệ thống phụ trợ: {str(mcp_e)}", [], max_score
            else:
                return random.choice(NUDGE_MESSAGES), [], max_score

        # 2. RAG Succeeded: Select config route
        rag_context_str = "\n".join(context_docs)

        is_fee = any(word in latest_user_msg for word in ["tiền", "phí", "lệ phí", "nhiêu"])
        is_location = any(word in latest_user_msg for word in ["ở đâu", "nơi nào", "địa điểm", "nộp"])
        is_time = any(word in latest_user_msg for word in ["bao lâu", "thời gian", "ngày"])

        if is_fee or is_location or is_time:
            cfg = PROMPT_CONFIG["route_direct_qa"]
        else:
            cfg = PROMPT_CONFIG["route_missing_docs"]

        # Populate the template with the RAG context and the raw user query
        user_prompt = cfg["user_template"].format(context=rag_context_str, query=search_query)

        llm_request = LLMServiceRequest(
            system_prompt=cfg["system_prompt"],
            user_prompt=user_prompt,
            prompt=None,
            context=context_docs,
            max_tokens=cfg.get("max_tokens", 300),
            temperature=cfg.get("temperature", 0.0)
        )

        # 3. Send to LLM Engine (Port 8002)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json=llm_request.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            llm_result = LLMServiceResponse(**response.json())
            return llm_result.answer, llm_result.sources, max_score

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Lỗi xử lý hệ thống: {str(e)}", [], 0.0
