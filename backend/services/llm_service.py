import os
import sys
import json
import httpx
from cachetools import TTLCache

# Fix imports so backend can see the new rag folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the newly isolated RAG logic from your teammate's folder
from rag.query_pipeline import build_evidence_bundle

PROMPTS_PATH = os.path.join(ROOT_DIR, "backend", "prompts.json")
LLM_URL = os.getenv("LLM_SERVICE_URLS", "http://127.0.0.1:8002").split(",")[0].strip()

qa_cache = TTLCache(maxsize=100, ttl=3600)

try:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        QA_PROMPT_CONFIG = json.load(f).get("baseline_qa", {})
except FileNotFoundError:
    print("[LLM Service] ERROR: prompts.json not found! Using fallback.")
    QA_PROMPT_CONFIG = {}

async def get_legal_response_stream(prompt: str, search_query: str):
    """Pure API Layer: Triggers RAG -> Injects Prompts -> Streams LLM."""
    if prompt in qa_cache:
        cached = qa_cache[prompt]
        async def fake_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': cached['sources']})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': cached['text']})}\n\n"
            yield "data: [DONE]\n\n"
        return fake_stream(), cached["score"]

    # 1. Call Teammate's RAG Pipeline (Black Box)
    context_str, score, sources = await build_evidence_bundle(prompt, search_query)

    # 2. Handle low confidence / fallback logic
    if not context_str or score < 0.01:
        async def low_conf_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return low_conf_stream(), 0.0

    # 3. Dynamic Prompt Injection (Your Job)
    system_prompt = QA_PROMPT_CONFIG.get(
        "system_prompt",
        "Bạn là trợ lý tra cứu thông tin pháp lý. Chỉ cung cấp thông tin dựa trực tiếp vào tài liệu cung cấp."
    )
    user_template = QA_PROMPT_CONFIG.get(
        "user_template",
        "TÀI LIỆU CUNG CẤP:\n{context}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\nTRẢ LỜI:"
    )
    user_prompt = user_template.replace("{context}", context_str).replace("{query}", prompt)

    openai_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": True,
        "max_tokens": QA_PROMPT_CONFIG.get("max_tokens", 800),
        "temperature": QA_PROMPT_CONFIG.get("temperature", 0.1)
    }

    # 4. Stream response to UI
    async def stream_generator():
        yield f"data: {json.dumps({'type': 'metadata', 'sources': sources})}\n\n"
        full_text = ""

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream("POST", f"{LLM_URL}/v1/chat/completions", json=openai_payload, timeout=300.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line.replace("data: ", "").strip()
                            if data_str == "[DONE]":
                                yield "data: [DONE]\n\n"
                            elif data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    content = parsed["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        full_text += content
                                        yield f"data: {json.dumps({'type': 'chunk', 'text': content})}\n\n"
                                except Exception:
                                    pass
            except Exception as e:
                yield f"data: {json.dumps({'type': 'chunk', 'text': f'Lỗi hệ thống: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"

        if full_text:
            qa_cache[prompt] = {"sources": sources, "text": full_text, "score": score}

    return stream_generator(), score
