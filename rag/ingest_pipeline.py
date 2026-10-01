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

# Load environment variables from the root .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Tong_hop_thu_tuc.csv")
SQLITE_DB_URL = f"sqlite+aiosqlite:///{os.path.join(DATA_DIR, 'legal_llm.db')}"
VECTOR_STORAGE_DIR = os.path.join(DATA_DIR, "vector_storage")
SCHEMA_DICT_PATH = os.path.join(DATA_DIR, "schema_dictionary.json")

# 1. Configure LlamaIndex
Settings.llm = GoogleGenAI(model="gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="keepitreal/vietnamese-sbert",
    device="cpu"
)

async def init_sqlite_and_store(df: pd.DataFrame, schema_dict: dict, primary_col: str):
    """Creates the SQLite table and stores the factual data."""
    engine = create_async_engine(SQLITE_DB_URL)

    columns = ["id INTEGER PRIMARY KEY"]
    for eng_col in schema_dict.keys():
        columns.append(f"{eng_col} TEXT")
    create_table_sql = f"CREATE TABLE IF NOT EXISTS procedures ({', '.join(columns)})"

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS procedures"))
        await conn.execute(text(create_table_sql))

        for idx, row in df.iterrows():
            insert_cols = ["id"] + list(schema_dict.keys())
            placeholders = [":id"] + [f":{col}" for col in schema_dict.keys()]
            insert_sql = f"INSERT INTO procedures ({', '.join(insert_cols)}) VALUES ({', '.join(placeholders)})"

            values = {"id": idx}
            for eng_col, vie_desc in schema_dict.items():
                matched_val = ""
                if eng_col in df.columns:
                    matched_val = str(row[eng_col])
                else:
                    for orig_col in df.columns:
                        if str(orig_col).lower() in eng_col.lower() or str(orig_col).lower() in vie_desc.lower():
                            matched_val = str(row[orig_col])
                            break
                values[eng_col] = matched_val
            await conn.execute(text(insert_sql), values)

    await engine.dispose()
    print(f"[SQLite] Stored {len(df)} records.")

async def extract_schema_with_gemini(columns: list) -> dict:
    """Uses Gemini to translate columns into a clean schema."""
    prompt = f"""
    Here is a list of columns from a spreadsheet:
    {columns}

    Your task is to normalize these columns into a structured schema dictionary.
    Output ONLY a valid JSON dictionary where:
    - The keys are short, lowercase English variable names (use exact original if already in English).
    - The values are a clear Vietnamese description of what that column means.
    """
    response = await Settings.llm.acomplete(prompt)
    try:
        clean_json = str(response).replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Error parsing Gemini schema: {e}")
        return {"procedure_name": columns[0]}

def ingest_semantic_vectors(df: pd.DataFrame, schema_dict: dict):
    """Embeds a clean, universal string for semantic matching regardless of dataset type."""
    print("[LlamaIndex] Building Documents...")
    documents = []

    # Identify the main name/title column
    primary_col_df = None
    for eng in schema_dict.keys():
        if "title" in eng.lower() or "name" in eng.lower():
            if eng in df.columns:
                primary_col_df = eng
                break
    if not primary_col_df:
        primary_col_df = df.columns[0]

    for idx, row in df.iterrows():
        proc_name = str(row.get(primary_col_df, '')).strip()
        if not proc_name or proc_name.lower() == "nan":
            continue

        # UNIVERSAL FIX: Dynamically grab short data points (like category, city, price)
        # to build context without hardcoding any specific column names.
        context_parts = [proc_name]
        for col in df.columns:
            if col == primary_col_df:
                continue
            val = str(row.get(col, '')).strip()
            # Only append short strings (avoids dumping long paragraphs/reviews into vectors)
            if val and val.lower() not in ["nan", "không có thông tin"] and len(val) < 60:
                context_parts.append(val)

        # Example output: "trưởng phòng kinh doanh - nhân viên chính thức - hồ chí minh - 15.0"
        # Example output: "Wayona USB Cable - Computers & Accessories - 4.2"
        embed_text = " - ".join(context_parts)

        doc = Document(
            text=embed_text,
            metadata={"id": int(idx), "title": proc_name},
            excluded_llm_metadata_keys=["id", "title"],
            doc_id=str(idx)
        )
        documents.append(doc)

    print("[LlamaIndex] Indexing documents in memory...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=VECTOR_STORAGE_DIR)
    print(f"[LlamaIndex] Successfully indexed {len(documents)} records.")

async def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Could not find raw file at {RAW_DATA_PATH}")
        return

    print(f"Reading file: {RAW_DATA_PATH}")
    # Universal fallback for csv vs excel can be added here later if needed
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8')
    df = df.fillna("Không có thông tin")

    print("[Scout Agent] Extracting schema...")
    schema_dict = await extract_schema_with_gemini(df.columns.tolist())

    with open(SCHEMA_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, ensure_ascii=False, indent=2)

    await init_sqlite_and_store(df, schema_dict, df.columns[0])
    ingest_semantic_vectors(df, schema_dict)
    print("\nPhase 1 Complete!")

if __name__ == "__main__":
    asyncio.run(main())
