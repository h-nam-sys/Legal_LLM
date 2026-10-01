import os
import json
import httpx
import time
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from cachetools import TTLCache

# LlamaIndex Imports
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Cấu hình đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
SQLITE_DB_URL = f"sqlite+aiosqlite:///{os.path.join(DATA_DIR, 'legal_llm.db')}"
VECTOR_STORAGE_DIR = os.path.join(DATA_DIR, "vector_storage")
SCHEMA_DICT_PATH = os.path.join(DATA_DIR, "schema_dictionary.json")

LLM_URL = os.getenv("LLM_SERVICE_URLS", "http://127.0.0.1:8002").split(",")[0].strip()

# Bộ nhớ đệm câu trả lời
qa_cache = TTLCache(maxsize=100, ttl=3600)

# Khởi tạo mô hình Embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="keepitreal/vietnamese-sbert",
    device="cpu"
)

# Tải Vector Store từ đĩa vào bộ nhớ RAM
print("[Runtime] Đang tải Vector Index vào RAM...")
storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORAGE_DIR)
vector_index = load_index_from_storage(storage_context)
retriever = vector_index.as_retriever(similarity_top_k=1)

# Tải Schema Dictionary tiếng Việt do Scout Agent tạo
with open(SCHEMA_DICT_PATH, "r", encoding="utf-8") as f:
    SCHEMA_DICT = json.load(f)

async def call_llm_json(prompt: str) -> str:
    """Gọi model 1.7B để phân loại ý định ở chế độ JSON."""
    payload = {
        "messages": [
            {"role": "system", "content": "Bạn là bộ phân loại ý định. Chỉ trả về kết quả định dạng JSON array hợp lệ, không giải thích gì thêm."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 128
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLM Intent Error] {e}")
            return "[]"

async def classify_requested_columns(user_query: str) -> list[str]:
    """Sử dụng từ điển tiếng Việt để model 1.7B nhận diện đúng cột cần lấy."""
    field_descriptions = "\n".join([f"- {k}: {v}" for k, v in SCHEMA_DICT.items() if k not in ["timestamp", "email", "id"]])
    prompt = f"""Dưới đây là danh sách các trường dữ liệu hành chính:
{field_descriptions}

Câu hỏi của người dùng: "{user_query}"

Hãy chọn ra các khóa (keys) tiếng Anh tương ứng với thông tin người dùng đang hỏi.
Ví dụ: nếu hỏi về lệ phí trả về ["fee"], nếu hỏi về thời gian giải quyết trả về ["processing_time"].
Nếu hỏi chung chung hoặc muốn biết thủ tục gồm những gì, trả về tất cả các trường chính: ["submission_method", "required_documents", "processing_time", "fee"].

Chỉ trả về JSON array danh sách khóa tiếng Anh:"""

    raw_json = await call_llm_json(prompt)
    try:
        clean_json = raw_json.replace("```json", "").replace("```", "").strip()
        cols = json.loads(clean_json)
        return cols if isinstance(cols, list) and cols else list(SCHEMA_DICT.keys())
    except Exception:
        return list(SCHEMA_DICT.keys())

async def fetch_record_from_sqlite(procedure_id: int) -> dict:
    """Truy vấn dữ liệu chính xác từ SQLite theo id thủ tục."""
    engine = create_async_engine(SQLITE_DB_URL)
    async with engine.begin() as conn:
        res = await conn.execute(
            text("SELECT * FROM procedures WHERE id = :id"),
            {"id": procedure_id}
        )
        row = res.mappings().first()
    await engine.dispose()
    return dict(row) if row else {}

async def get_legal_response_stream(prompt: str, search_query: str):
    """Router chính: Semantic search -> Trích xuất cột -> Truy vấn SQLite -> Streaming câu trả lời."""
    # 1. Kiểm tra cache
    if prompt in qa_cache:
        cached = qa_cache[prompt]
        async def fake_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': cached['sources']})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': cached['text']})}\n\n"
            yield "data: [DONE]\n\n"
        return fake_stream(), cached["score"]

    # 2. Tìm kiếm thủ tục qua LlamaIndex Vector Retriever
    nodes = retriever.retrieve(search_query)
    if not nodes:
        async def empty_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return empty_stream(), 0.0

    matched_node = nodes[0]
    similarity_score = float(matched_node.score) if matched_node.score is not None else 1.0
    procedure_id = matched_node.node.metadata.get("id")
    procedure_name = matched_node.node.metadata.get("procedure_name", "")

    # Ngưỡng kích hoạt công cụ bên ngoài nếu không khớp dữ liệu
    if similarity_score < 0.45 or procedure_id is None:
        async def low_conf_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"
        return low_conf_stream(), 0.0

    # 3. Lấy dữ liệu thực tế từ SQLite
    full_row = await fetch_record_from_sqlite(int(procedure_id))

    # 4. Model 1.7B phân loại các cột cần lấy
    requested_cols = await classify_requested_columns(prompt)

    # Đảm bảo luôn giữ lại tên thủ tục
    filtered_data = {"procedure_name": full_row.get("procedure_name", procedure_name)}
    for col in requested_cols:
        if col in full_row and full_row[col] and full_row[col] != "nan":
            vietnamese_label = SCHEMA_DICT.get(col, col)
            filtered_data[vietnamese_label] = full_row[col]

    # Chuẩn bị ngữ cảnh chính xác truyền cho model 1.7B tổng hợp
    context_str = json.dumps(filtered_data, ensure_ascii=False, indent=2)
    sources = [procedure_name, f"RAG_CONTENT:{context_str}"]

    system_prompt = (
        "Bạn là trợ lý pháp lý hỗ trợ thủ tục hành chính công. "
        "Dựa vào thông tin chính xác được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách rõ ràng, mạch lạc bằng tiếng Việt. "
        "Tuyệt đối không suy diễn thêm thông tin ngoài ngữ cảnh được cung cấp."
    )
    user_prompt = f"Thông tin căn cứ:\n{context_str}\n\nCâu hỏi: {prompt}"

    openai_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": True,
        "max_tokens": 600,
        "temperature": 0.1
    }

    # 5. Thực hiện stream kết quả
    async def stream_generator():
        yield f"data: {json.dumps({'type': 'metadata', 'sources': sources})}\n\n"
        full_text = ""

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{LLM_URL}/v1/chat/completions",
                    json=openai_payload,
                    timeout=300.0
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line.replace("data: ", "").strip()
                            if data_str == "[DONE]":
                                yield "data: [DONE]\n\n"
                            elif data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    delta = parsed["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_text += content
                                        yield f"data: {json.dumps({'type': 'chunk', 'text': content})}\n\n"
                                except Exception:
                                    pass
            except Exception as e:
                yield f"data: {json.dumps({'type': 'chunk', 'text': f'Lỗi hệ thống: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"

        if full_text:
            qa_cache[prompt] = {
                "sources": sources,
                "text": full_text,
                "score": similarity_score
            }

    return stream_generator(), similarity_score
