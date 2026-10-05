"""
============================================================
INGEST PIPELINE - Đa lĩnh vực (Multi-domain)
============================================================

Cú pháp chạy:
    python ingest_pipeline.py --file path/to/data.xlsx --domain it_support
    python ingest_pipeline.py --file path/to/data.csv  --domain hr
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOMAINS_DIR = os.path.join(DATA_DIR, "domains")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "knowledge_base.db")
SQLITE_DB_URL = f"sqlite+aiosqlite:///{SQLITE_DB_PATH}"

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cpu")
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)


# 1. ĐỌC FILE ĐA ĐỊNH DẠNG
def _load_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0, dtype=str)

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def _load_json(path: str) -> pd.DataFrame:
    return pd.read_json(path, dtype=str)

def _load_pdf(path: str) -> pd.DataFrame:
    raise NotImplementedError("Cần viết thêm hàm trích xuất PDF (pdfplumber/camelot) phù hợp với cấu trúc file.")

def _load_docx(path: str) -> pd.DataFrame:
    raise NotImplementedError("Cần viết thêm hàm đọc file Word (.docx) bóc tách theo heading hoặc bảng.")

def _load_database(path: str) -> pd.DataFrame:
    raise NotImplementedError("Cần cung cấp Connection String và câu lệnh SQL cụ thể.")

def _load_api(path: str) -> pd.DataFrame:
    raise NotImplementedError("Cần endpoint API và cấu trúc JSON response tương ứng.")

LOADERS: dict[str, Callable[[str], pd.DataFrame]] = {
    ".xlsx": _load_excel,
    ".xls": _load_excel,
    ".csv": _load_csv,
    ".json": _load_json,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,
}

_BLANK_LIKE = {"", "nan", "none", "nat", "null", "n/a", "na", "undefined", "-"}

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v).strip()
        )
        df[col] = df[col].apply(lambda v: "" if v.lower() in _BLANK_LIKE else v)
    return df

def load_dataframe(file_path: str) -> pd.DataFrame:
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return _load_api(file_path)
    if file_path.startswith("postgresql://") or file_path.startswith("mysql://"):
        return _load_database(file_path)

    ext = Path(file_path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Định dạng '{ext}' chưa hỗ trợ.")
    return normalize_dataframe(loader(file_path))


# 2. CHUẨN HOÁ TÊN DOMAIN & ĐƯỜNG DẪN
def sanitize_identifier(name: str) -> str:
    name = name.strip().lower()
    vn_map = str.maketrans(
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ",
        "a" * 17 + "e" * 11 + "i" * 5 + "o" * 17 + "u" * 11 + "y" * 5 + "d",
    )
    name = name.translate(vn_map)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if not name:
        raise ValueError("Tên domain không hợp lệ.")
    if name[0].isdigit():
        name = f"d_{name}"
    return name

def get_domain_paths(domain: str) -> dict[str, str]:
    domain_dir = os.path.join(DOMAINS_DIR, domain)
    return {
        "domain_dir": domain_dir,
        "profile_path": os.path.join(domain_dir, "domain_profile.json"),
        "vector_dir": os.path.join(domain_dir, "vector_storage"),
    }


# 3. TẠO TÀI LIỆU EMBEDDING
def detect_id_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.strip().lower() == "id":
            return col
    for col in df.columns:
        if col.strip().lower().endswith("_id"):
            return col
    return None

def detect_text_column(df: pd.DataFrame, override: str | None) -> str | None:
    if override and override in df.columns:
        return override
    for col in df.columns:
        if col.strip().lower() == "rag_text":
            return col
    return None

def build_row_text(row: pd.Series, text_column: str | None, id_column: str | None) -> str:
    if text_column:
        return str(row[text_column]).strip()
    lines = []
    for col, val in row.items():
        if col == id_column:
            continue
        val = str(val).strip()
        if val:
            lines.append(f"{col}: {val}")
    return "\n".join(lines)

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# 4. TÁCH BẢNG SQLITE THEO DOMAIN
async def save_to_sqlite(df: pd.DataFrame, domain: str, id_column: str | None) -> str:
    table_name = sanitize_identifier(domain)
    engine = create_async_engine(SQLITE_DB_URL)

    columns_sql = []
    for col in df.columns:
        col_sql = f"{quote_ident(col)} TEXT"
        if id_column and col == id_column:
            col_sql += " PRIMARY KEY"
        columns_sql.append(col_sql)

    if not id_column:
        columns_sql.insert(0, "row_id INTEGER PRIMARY KEY AUTOINCREMENT")

    create_sql = f"CREATE TABLE IF NOT EXISTS {quote_ident(table_name)} ({', '.join(columns_sql)})"
    insert_cols = list(df.columns)
    insert_cols_sql = ", ".join(quote_ident(c) for c in insert_cols)
    insert_placeholders = ", ".join(f":{i}" for i in range(len(insert_cols)))
    insert_sql = f"INSERT INTO {quote_ident(table_name)} ({insert_cols_sql}) VALUES ({insert_placeholders})"

    rows = [{str(i): r[c] for i, c in enumerate(insert_cols)} for _, r in df.iterrows()]

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {quote_ident(table_name)}"))
        await conn.execute(text(create_sql))
        if rows:
            await conn.execute(text(insert_sql), rows)

    await engine.dispose()
    print(f"[SQLite] Domain '{domain}' -> Bảng \"{table_name}\" ({len(rows)} dòng)")
    return table_name


# 5. LƯU VECTOR THEO DOMAIN (TĂNG DẦN)
def ingest_vectors_incremental(documents: list[Document], vector_dir: str) -> None:
    os.makedirs(vector_dir, exist_ok=True)
    if any(Path(vector_dir).iterdir()):
        print(f"[LlamaIndex] Cập nhật index tăng dần tại {vector_dir}...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=vector_dir)
            index = load_index_from_storage(storage_context)
            refreshed = index.refresh_ref_docs(documents)
            n_changed = sum(1 for r in refreshed if r)
            print(f"[LlamaIndex] Đã cập nhật {n_changed}/{len(documents)} tài liệu.")
        except Exception:
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
    else:
        index = VectorStoreIndex.from_documents(documents, show_progress=True)

    index.storage_context.persist(persist_dir=vector_dir)

async def run(file_path: str, domain_raw: str, text_column_override: str | None) -> None:
    domain = sanitize_identifier(domain_raw)
    paths = get_domain_paths(domain)
    os.makedirs(paths["domain_dir"], exist_ok=True)

    df = load_dataframe(file_path)
    id_column = detect_id_column(df)
    text_column = detect_text_column(df, text_column_override)

    table_name = await save_to_sqlite(df, domain, id_column)

    documents = []
    for i, row in df.iterrows():
        row_text = build_row_text(row, text_column, id_column)
        if not row_text:
            continue
        doc_id = str(row[id_column]).strip() if id_column and row[id_column] else f"{domain}_{i}"
        documents.append(Document(text=row_text, doc_id=doc_id, metadata={"domain": domain, "id": doc_id}))

    ingest_vectors_incremental(documents, paths["vector_dir"])

    profile = {
        "domain": domain,
        "domain_raw": domain_raw,
        "source_file": os.path.abspath(file_path),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "columns": list(df.columns),
        "id_column": id_column,
        "text_column": text_column,
        "sqlite_table": table_name,
        "sqlite_db": SQLITE_DB_PATH,
        "vector_dir": paths["vector_dir"],
    }
    with open(paths["profile_path"], "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Ingest dữ liệu đa domain vào SQLite & LlamaIndex")
    parser.add_argument("--file", required=True, help="Đường dẫn file dữ liệu")
    parser.add_argument("--domain", required=True, help="Tên lĩnh vực (vd: it_support, hr)")
    parser.add_argument("--text-column", default=None, help="Cột dùng làm embedding text")
    args = parser.parse_args()
    asyncio.run(run(args.file, args.domain, args.text_column))

if __name__ == "__main__":
    main()