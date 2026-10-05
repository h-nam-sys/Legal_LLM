import os
import json
import httpx
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Import the vector search from the same folder
from .retrieval_service import retrieve_and_rerank

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SQLITE_DB_URL = f"sqlite+aiosqlite:///{os.path.join(DATA_DIR, 'legal_llm.db')}"
DOMAIN_PROFILE_PATH = os.path.join(DATA_DIR, "domain_profile.json")
LLM_URL = os.getenv("LLM_SERVICE_URLS", "http://127.0.0.1:8002").split(",")[0].strip()

try:
    with open(DOMAIN_PROFILE_PATH, "r", encoding="utf-8") as f:
        DOMAIN_PROFILE = json.load(f)
        SCHEMA_MAPPING = DOMAIN_PROFILE.get("schema_mapping", {})
except FileNotFoundError:
    DOMAIN_PROFILE = {}
    SCHEMA_MAPPING = {}

async def call_llm_json(prompt: str) -> str:
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
        except Exception:
            return "[]"

async def classify_requested_columns(user_query: str) -> list[str]:
    primary_entity = DOMAIN_PROFILE.get("primary_entity", "procedure_name")
    field_descriptions = "\n".join([f"- {k}: {v}" for k, v in SCHEMA_MAPPING.items() if k not in ["timestamp", "email", "id", primary_entity]])

    prompt = f"Dưới đây là danh sách các trường dữ liệu hành chính:\n{field_descriptions}\n\nCâu hỏi của người dùng: \"{user_query}\"\n\nHãy chọn ra các khóa (keys) tiếng Anh tương ứng. Chỉ trả về JSON array:"
    raw_json = await call_llm_json(prompt)
    try:
        clean_json = raw_json.replace("```json", "").replace("```", "").strip()
        cols = json.loads(clean_json)
        return cols if isinstance(cols, list) and cols else list(SCHEMA_MAPPING.keys())
    except Exception:
        return list(SCHEMA_MAPPING.keys())

async def fetch_record_from_sqlite(procedure_id: int) -> dict:
    engine = create_async_engine(SQLITE_DB_URL)
    async with engine.begin() as conn:
        res = await conn.execute(text("SELECT * FROM procedures WHERE id = :id"), {"id": procedure_id})
        row = res.mappings().first()
    await engine.dispose()
    return dict(row) if row else {}

async def build_evidence_bundle(prompt: str, search_query: str) -> tuple[str, float, list]:
    """The only function the backend needs to call. Returns Context String, Score, and Sources."""
    procedure_id, score, meta = retrieve_and_rerank(search_query)
    print(f"\n[RERANKER DEBUG] Top Match ID: {procedure_id} | Confidence: {score:.4f} | Procedure: {meta.get('procedure_name', '')}")

    if procedure_id is None or score < 0.01:
        return "", 0.0, []

    full_row = await fetch_record_from_sqlite(int(procedure_id))
    requested_cols = await classify_requested_columns(prompt)

    primary_entity = DOMAIN_PROFILE.get("primary_entity", "procedure_name")
    procedure_name = full_row.get(primary_entity, meta.get(primary_entity, ""))

    filtered_data = {SCHEMA_MAPPING.get(primary_entity, primary_entity): procedure_name}
    for col in requested_cols:
        if col in full_row and full_row[col] and str(full_row[col]).lower() not in ["nan", "không có thông tin"]:
            filtered_data[SCHEMA_MAPPING.get(col, col)] = full_row[col]

    context_str = json.dumps(filtered_data, ensure_ascii=False, indent=2)
    sources = [procedure_name, f"RAG_CONTENT:{context_str}"]

    return context_str, score, sources
