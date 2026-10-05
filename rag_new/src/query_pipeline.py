import os
import json
import re
from typing import Tuple, Dict, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

try:
    from .retrieval_service import retrieve_and_rerank
except ImportError:
    from retrieval_service import retrieve_and_rerank

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOMAINS_DIR = os.path.join(DATA_DIR, "domains")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "knowledge_base.db")
SQLITE_DB_URL = f"sqlite+aiosqlite:///{SQLITE_DB_PATH}"

# ============================================================
# TASK 2: CHẶN DỮ LIỆU RỖNG TRƯỚC KHIN TRUYỀN CHO LLM
# ============================================================
DEFAULT_MISSING_TEXT = "Hệ thống hiện chưa có dữ liệu chi tiết cho mục này."

_BLANK_LIKE = {
    "", "nan", "none", "nat", "null", "n/a", "na", "undefined", "-",
    "không có thông tin", "khong co thong tin"
}

def sanitize_identifier(name: str) -> str:
    name = name.strip().lower()
    vn_map = str.maketrans(
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ",
        "a" * 17 + "e" * 11 + "i" * 5 + "o" * 17 + "u" * 11 + "y" * 5 + "d",
    )
    name = name.translate(vn_map)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name if name else "default_domain"

def sanitize_value(value: Any) -> str:
    if value is None:
        return DEFAULT_MISSING_TEXT
    text_val = str(value).strip()
    if text_val.lower() in _BLANK_LIKE:
        return DEFAULT_MISSING_TEXT
    return text_val

def load_domain_profile(domain: str) -> dict:
    profile_path = os.path.join(DOMAINS_DIR, sanitize_identifier(domain), "domain_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

async def fetch_record_from_sqlite(record_id: str, domain: str) -> dict:
    table_name = sanitize_identifier(domain)
    engine = create_async_engine(SQLITE_DB_URL)
    profile = load_domain_profile(domain)
    id_col = profile.get("id_column") or "row_id"
    
    # Xử lý chuỗi ID: "thu_tuc_hanh_chinh_26" -> lấy "26"
    clean_id = str(record_id)
    if "_" in clean_id:
        clean_id = clean_id.rsplit("_", 1)[-1]

    # Ép kiểu số nguyên nếu truy vấn theo row_id tự tăng
    try:
        query_id = int(clean_id)
    except ValueError:
        query_id = clean_id

    query_str = f'SELECT * FROM "{table_name}" WHERE "{id_col}" = :id'

    async with engine.begin() as conn:
        try:
            res = await conn.execute(text(query_str), {"id": query_id})
            row = res.mappings().first()
        except Exception:
            row = None
            
    await engine.dispose()
    return dict(row) if row else {}

async def build_evidence_bundle(prompt: str, search_query: str, domain: str = "thu_tuc_hanh_chinh") -> Tuple[str, float, list]:
    record_id, score, meta = retrieve_and_rerank(search_query, domain=domain)

    if record_id is None or score < 0.01:
        return "", 0.0, []

    full_row = await fetch_record_from_sqlite(str(record_id), domain=domain)
    if not full_row:
        return "", score, []

    filtered_data = {}
    for col, val in full_row.items():
        if col == "row_id":
            continue
        filtered_data[col] = sanitize_value(val)

    context_str = json.dumps(filtered_data, ensure_ascii=False, indent=2)
    sources = [meta.get("title", f"ID_{record_id}"), f"RAG_CONTENT:{context_str}"]

    return context_str, score, sources

if __name__ == "__main__":
    import asyncio

    async def main():
        prompt = "Thành phần hồ sơ và thời hạn giải quyết thủ tục là gì?"
        search_query = "thành phần hồ sơ thời hạn giải quyết thủ tục hành chính"
        domain = "thu_tuc_hanh_chinh"

        print(f"\n=================== KẾT QUẢ TEST QUERY PIPELINE ===================")
        print(f"Domain: {domain}")
        print(f"Search Query: {search_query}\n")

        context_str, score, sources = await build_evidence_bundle(
            prompt=prompt,
            search_query=search_query,
            domain=domain
        )

        print(f"[1] Rerank Confidence Score: {score:.4f}")
        print(f"[2] Sources: {sources[0] if sources else 'None'}\n")
        print("[3] Context JSON (Truyền cho LLM - Đã xử lý Task 2):")
        print(context_str)
        print(f"===================================================================\n")

    asyncio.run(main())