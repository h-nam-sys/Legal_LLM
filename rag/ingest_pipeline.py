import os
import asyncio
import pandas as pd
import json
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# LlamaIndex Imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Tổng hợp các thủ tục hành chính (22.12) (1).xlsx")
SQLITE_DB_URL = f"sqlite+aiosqlite:///{os.path.join(DATA_DIR, 'legal_llm.db')}"
VECTOR_STORAGE_DIR = os.path.join(DATA_DIR, "vector_storage")
DOMAIN_PROFILE_PATH = os.path.join(DATA_DIR, "domain_profile.json")

# 1. Configure Universal Agent Settings
# Heavyweight Embedding for superior multilingual contextual depth
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cpu"
)
# Highly capable Scout model for schema understanding
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")

async def build_domain_profile_with_gemini(columns: list) -> dict:
    """Uses the Scout Agent (Gemini) to generate a comprehensive domain profile."""
    prompt = f"""
    You are a Data Profiler AI for a Universal Domain Architecture.
    I am providing you a list of columns from a Vietnamese administrative document dataset:
    {columns}

    Analyze these columns and output a valid JSON representing the 'Domain Profile'.
    It MUST contain exactly these keys:
    {{
        "domain_name": "A short English name for this domain (e.g., 'administrative_procedures')",
        "primary_entity": "The English key representing the main entity (e.g., 'procedure_name')",
        "schema_mapping": {{
            "procedure_name": "Tên thủ tục hành chính",
            "fee": "Lệ phí",
            "processing_time": "Thời gian giải quyết"
            // Map the rest of the relevant columns to short English keys.
        }},
        "retrieval_policy": "hybrid"
    }}

    Output ONLY valid JSON. No markdown tags, no explanations.
    """
    response = await Settings.llm.acomplete(prompt)
    try:
        clean_json = str(response).replace("```json", "").replace("```", "").strip()
        profile = json.loads(clean_json)
        return profile
    except Exception as e:
        print(f"Error parsing Gemini domain profile: {e}")
        # Fallback profile
        return {
            "domain_name": "administrative_procedures",
            "primary_entity": "procedure_name",
            "schema_mapping": {"procedure_name": columns[0], "metadata": columns[1]},
            "retrieval_policy": "hybrid"
        }

async def init_sqlite_and_store(df: pd.DataFrame, schema_mapping: dict):
    """Creates the SQLite table mapping raw data to the clean schema keys."""
    engine = create_async_engine(SQLITE_DB_URL)

    columns = ["id INTEGER PRIMARY KEY"]
    for eng_col in schema_mapping.keys():
        columns.append(f"{eng_col} TEXT")
    create_table_sql = f"CREATE TABLE IF NOT EXISTS procedures ({', '.join(columns)})"

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS procedures"))
        await conn.execute(text(create_table_sql))

        for idx, row in df.iterrows():
            insert_cols = ["id"] + list(schema_mapping.keys())
            placeholders = [":id"] + [f":{col}" for col in schema_mapping.keys()]

            insert_sql = f"INSERT INTO procedures ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})"

            values = {"id": idx + 1} # 1-based indexing for cleaner IDs
            for eng_col, vie_desc in schema_mapping.items():
                matched_val = ""
                for orig_col in df.columns:
                    if str(orig_col).lower() in vie_desc.lower() or vie_desc.lower() in str(orig_col).lower():
                        matched_val = str(row[orig_col])
                        break
                values[eng_col] = matched_val

            await conn.execute(text(insert_sql), values)

    await engine.dispose()
    print(f"[SQLite] Stored {len(df)} structured records.")

def ingest_semantic_vectors(df: pd.DataFrame, schema_mapping: dict, primary_entity_key: str):
    """Embeds the primary entities using BGE-M3 for broad retrieval."""
    print("[LlamaIndex] Building Documents for BGE-M3 Embedding...")
    documents = []

    primary_col_vie = schema_mapping.get(primary_entity_key, df.columns[0])

    for idx, row in df.iterrows():
        proc_name = str(row.get(primary_col_vie, '')).strip()
        if not proc_name or proc_name.lower() == "nan":
            continue

        # Embed just the title for precise semantic matching before reranking
        embed_text = f"Thủ tục: {proc_name}"

        doc = Document(
            text=embed_text,
            metadata={"id": int(idx + 1), primary_entity_key: proc_name},
            excluded_llm_metadata_keys=["id", primary_entity_key],
            doc_id=str(idx + 1)
        )
        documents.append(doc)

    print("[LlamaIndex] Indexing documents with BGE-M3 (This may take a moment on CPU)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=VECTOR_STORAGE_DIR)
    print(f"[LlamaIndex] Successfully indexed {len(documents)} vectors.")

async def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Could not find raw dataset at {RAW_DATA_PATH}")
        return

    print(f"Reading Dataset: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH, sheet_name=0)
    df = df.fillna("Không có thông tin")

    print("[Scout Agent] Building Domain Profile...")
    domain_profile = await build_domain_profile_with_gemini(df.columns.tolist())

    # Save the formal domain profile
    with open(DOMAIN_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(domain_profile, f, ensure_ascii=False, indent=4)
    print(f"[Scout Agent] Saved domain_profile.json with schema mapping.")

    print("[SQLite] Normalizing and storing factual data...")
    await init_sqlite_and_store(df, domain_profile["schema_mapping"])

    print("[VectorStore] Creating semantic index with BGE-M3...")
    ingest_semantic_vectors(df, domain_profile["schema_mapping"], domain_profile["primary_entity"])

    print("\n[Phase 1 Complete] Domain Onboarding finished. Ready for Phase 2 Runtime!")

if __name__ == "__main__":
    asyncio.run(main())
