import asyncio
import httpx
import os
import json
import re
import time
from cachetools import TTLCache
from schemas import LLMServiceRequest, LLMServiceResponse, RAGServiceRequest, RAGServiceResponse

LLM_URLS_ENV = os.getenv("LLM_SERVICE_URLS", "http://127.0.0.1:8002")
AVAILABLE_LLMS = [url.strip() for url in LLM_URLS_ENV.split(",")]

llm_queue = asyncio.Queue()
for url in AVAILABLE_LLMS:
    llm_queue.put_nowait(url)

qa_cache = TTLCache(maxsize=100, ttl=3600)

def clean_query(text: str) -> str:
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()

def jaccard_similarity(query1: str, query2: str) -> float:
    set1 = set(query1.split())
    set2 = set(query2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def check_smart_cache(user_query: str) -> dict | None:
    cleaned_user_q = clean_query(user_query)

    if cleaned_user_q in qa_cache:
        print("[CACHE HIT] Exact match!")
        return qa_cache[cleaned_user_q]

    for cached_q in qa_cache.keys():
        if jaccard_similarity(cleaned_user_q, cached_q) >= 1.0:
            print(f"[CACHE HIT] 100% Jaccard word-order match with: '{cached_q}'")
            return qa_cache[cached_q]

    return None

def extract_procedure_name(doc_text: str) -> str:
    if not doc_text:
        return ""
    match = re.search(r"Tên thủ tục hành chính:\s*(.*?)(?:\n|$)", doc_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
    return lines[0] if lines else ""

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8001")
CONFIG_PATH = os.getenv("PROMPTS_CONFIG_PATH", "prompts.json")
VOCAB_PATH = os.getenv("INTENT_VOCAB_PATH", "data/intent_vocab.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    PROMPT_CONFIG = json.load(f)

try:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        INTENT_VOCAB = json.load(f)
except FileNotFoundError:
    INTENT_VOCAB = {
        "Hình thức nộp": ["hình thức", "nộp thế nào", "online hay offline", "trực tuyến"],
        "Thành phần hồ sơ": ["hồ sơ", "giấy tờ", "cần những gì", "chuẩn bị gì"],
        "Thời gian giải quyết": ["thời gian", "bao lâu", "mấy ngày", "khi nào xong"],
        "Lệ phí": ["lệ phí", "bao nhiêu tiền", "chi phí", "mất tiền không"],
        "Địa điểm tiếp nhận hồ sơ trực tiếp": ["địa điểm", "ở đâu", "nơi nộp", "cơ quan nào"]
    }
    print(f"[WARNING] Intent vocab file not found at {VOCAB_PATH}. Using fallback defaults.")

def detect_requested_fields(query: str) -> list[str]:
    query_lower = query.lower()
    requested_fields = []

    for field_name, keywords in INTENT_VOCAB.items():
        if any(keyword in query_lower for keyword in keywords):
            requested_fields.append(field_name)

    return requested_fields

def extract_fields_from_doc(doc_text: str, fields: list[str]) -> str:
    extracted_text = []

    proc_name = extract_procedure_name(doc_text)
    if proc_name:
        extracted_text.append(f"Tên thủ tục hành chính: {proc_name}")

    for field in fields:
        pattern = rf"{re.escape(field)}:\s*(.*?)(?=\n[A-ZĐ][a-zà-ỹ\s]+:|$)"
        match = re.search(pattern, doc_text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            extracted_text.append(f"{field}: {content}")

    return "\n\n".join(extracted_text)

async def fetch_rag_context(query: str, top_k: int = 3) -> tuple[list[str], list[float]]:
    print(f"\n[DIAGNOSTIC] Sending RAG request to: {RAG_SERVICE_URL}/retrieve")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            rag_request = RAGServiceRequest(query=query, top_k=top_k)
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

async def get_legal_response_stream(prompt: str, search_query: str):
    try:
        t_start_cache = time.perf_counter()
        cached_data = check_smart_cache(prompt)
        print(f"[PROFILER] Cache Lookup Time: {(time.perf_counter() - t_start_cache):.4f}s")

        if cached_data:
            async def fake_cached_stream():
                yield f"data: {json.dumps({'type': 'metadata', 'sources': cached_data['sources']})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': cached_data['answer']})}\n\n"
                yield "data: [DONE]\n\n"
            return fake_cached_stream(), cached_data['score']

        t_start_rag = time.perf_counter()
        context_docs, scores = await fetch_rag_context(search_query, top_k=1)
        print(f"[PROFILER] RAG Retrieval Time: {(time.perf_counter() - t_start_rag):.4f}s")

        max_score = max(scores) if scores else 0.0

        if not context_docs:
            async def empty_stream():
                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield "data: [DONE]\n\n"
            return empty_stream(), 0.0

        primary_doc = context_docs[0]
        detected_proc_title = extract_procedure_name(primary_doc)
        requested_fields = detect_requested_fields(prompt)

        if not requested_fields:
            rag_context_str = primary_doc
        else:
            rag_context_str = extract_fields_from_doc(primary_doc, requested_fields)
            if len(rag_context_str.splitlines()) <= 1:
                rag_context_str = primary_doc

        rag_context_str = re.sub(
            r"(Hình thức nộp\s*:\s*)Cả hai",
            r"\1Trực tiếp hoặc trực tuyến",
            rag_context_str,
            flags=re.IGNORECASE
        )

        rag_sources = [detected_proc_title] if detected_proc_title else []
        rag_sources.append(f"RAG_CONTENT:{rag_context_str}")

        cfg = PROMPT_CONFIG.get("baseline_qa")
        if not cfg:
            async def error_stream():
                yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': 'Lỗi: Không tìm thấy cấu hình baseline_qa.'})}\n\n"
                yield "data: [DONE]\n\n"
            return error_stream(), max_score

        modified_query = f"/no_think {prompt}"
        user_prompt = cfg["user_template"].format(context=rag_context_str, query=modified_query)

        # === RESTORED CHAT PAYLOAD ===
        openai_payload = {
            "messages": [
                {"role": "system", "content": cfg["system_prompt"]},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True,
            "max_tokens": cfg.get("max_tokens", 800),
            "temperature": cfg.get("temperature", 0.0)
        }

        async def stream_from_engine():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': rag_sources})}\n\n"

            full_answer = ""
            t_start_queue = time.perf_counter()
            target_llm_url = await llm_queue.get()
            print(f"[PROFILER] LLM Queue Wait Time: {(time.perf_counter() - t_start_queue):.4f}s")

            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    t_start_llm = time.perf_counter()
                    first_token_received = False

                    # RESTORED ENDPOINT: /v1/chat/completions
                    async with client.stream(
                        "POST",
                        f"{target_llm_url}/v1/chat/completions",
                        json=openai_payload,
                        timeout=300.0
                    ) as response:
                        response.raise_for_status()

                        async for line in response.aiter_lines():
                            if not first_token_received:
                                print(f"[PROFILER] LLM Time to First Token (TTFT): {(time.perf_counter() - t_start_llm):.4f}s")
                                first_token_received = True

                            if line.startswith("data: "):
                                data_str = line.replace("data: ", "").strip()

                                if data_str == "[DONE]":
                                    yield "data: [DONE]\n\n"
                                elif data_str:
                                    try:
                                        parsed = json.loads(data_str)
                                        choices = parsed.get("choices", [])
                                        if choices:
                                            # RESTORED DELTA PARSING
                                            delta = choices[0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                full_answer += content
                                                yield f"data: {json.dumps({'type': 'chunk', 'text': content})}\n\n"
                                    except Exception:
                                        pass

                    print(f"[PROFILER] LLM Total Stream Time: {(time.perf_counter() - t_start_llm):.4f}s")

                except Exception as e:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': f'Lỗi kết nối LLM ({target_llm_url}): {str(e)}'})}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    llm_queue.put_nowait(target_llm_url)

                    t_start_save = time.perf_counter()
                    if full_answer:
                        cleaned_q = clean_query(prompt)
                        qa_cache[cleaned_q] = {
                            "sources": rag_sources,
                            "answer": full_answer,
                            "score": max_score
                        }
                    print(f"[PROFILER] Cache Save Time: {(time.perf_counter() - t_start_save):.4f}s")

        return stream_from_engine(), max_score

    except Exception as e:
        import traceback
        traceback.print_exc()
        async def fatal_error_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': f'Lỗi xử lý hệ thống: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"
        return fatal_error_stream(), 0.0
