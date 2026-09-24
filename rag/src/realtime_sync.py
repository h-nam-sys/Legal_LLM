# src/realtime_sync.py

import os
import sys
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_FILE = BASE_DIR / "data" / "raw" / "administrative_procedures.xlsx"

# Nếu file Excel hiện tại nằm ở tên khác thì sửa dòng trên.
# Ví dụ:
# SOURCE_FILE = BASE_DIR / "data" / "raw" / "administrative_procedures(2).xlsx"

STATE_FILE = BASE_DIR / "data" / "processed" / "realtime_sync_state.json"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vietnamese_administrative_procedures"

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

BATCH_SIZE = 8


# ============================================================
# REQUIRED COLUMNS
# ============================================================

PROCEDURE_COL = "Tên thủ tục hành chính"

IMPORTANT_COLUMNS = [
    "Tên thủ tục hành chính",
    "Lĩnh vực",
    "Hình thức nộp",
    "Thành phần hồ sơ",
    "Thời gian giải quyết",
    "Lệ phí",
    "Địa điểm tiếp nhận hồ sơ trực tiếp (nếu có)",
    "Ghi chú",
    "Trong trường hợp hồ sơ bao gồm các biểu mẫu vui lòng dính kèm file mẫu",
    "trạng thái",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value) -> str:
    """
    Chuẩn hóa text để:
    - tránh NaN
    - loại bỏ khoảng trắng thừa
    - giữ newline có ý nghĩa
    """
    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Chuẩn hóa khoảng trắng quanh từng dòng
    lines = []

    for line in text.split("\n"):
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def canonicalize_row(row: pd.Series) -> Dict[str, str]:
    """
    Chuyển một dòng Excel thành dictionary chuẩn hóa.
    """

    result = {}

    for column in IMPORTANT_COLUMNS:
        if column in row.index:
            result[column] = normalize_text(row[column])
        else:
            result[column] = ""

    return result


# ============================================================
# HASH
# ============================================================

def calculate_procedure_hash(data: Dict[str, str]) -> str:
    """
    Hash toàn bộ nội dung có ý nghĩa của một procedure.

    Nếu bất kỳ trường quan trọng nào thay đổi
    => hash thay đổi
    => procedure được xem là CHANGED.
    """

    canonical_string = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_string.encode("utf-8")
    ).hexdigest()


# ============================================================
# STATE
# ============================================================

def load_state() -> Dict[str, str]:

    if not STATE_FILE.exists():
        return {}

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception as e:
        print(
            f"[WARNING] Cannot load sync state: {e}",
            file=sys.stderr,
        )
        return {}


def save_state(state: Dict[str, str]):

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = STATE_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp_file.replace(STATE_FILE)


# ============================================================
# LOAD EXCEL
# ============================================================

def load_source() -> pd.DataFrame:

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy source Excel:\n{SOURCE_FILE}"
        )

    df = pd.read_excel(SOURCE_FILE)

    if PROCEDURE_COL not in df.columns:
        raise ValueError(
            f"Không tìm thấy cột '{PROCEDURE_COL}'.\n"
            f"Các cột hiện có:\n{df.columns.tolist()}"
        )

    return df


# ============================================================
# DETECT CHANGES
# ============================================================

def detect_changes(
    df: pd.DataFrame,
    old_state: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    new_items = []
    changed_items = []
    unchanged_items = []

    # tránh duplicate procedure name
    current_procedures = {}

    for _, row in df.iterrows():

        data = canonicalize_row(row)

        procedure_name = data[PROCEDURE_COL].strip()

        if not procedure_name:
            continue

        current_procedures[procedure_name] = data

    for procedure_name, data in current_procedures.items():

        current_hash = calculate_procedure_hash(data)

        old_hash = old_state.get(procedure_name)

        item = {
            "procedure_name": procedure_name,
            "hash": current_hash,
            "data": data,
        }

        if old_hash is None:

            new_items.append(item)

        elif old_hash != current_hash:

            changed_items.append(item)

        else:

            unchanged_items.append(item)

    return (
        new_items,
        changed_items,
        unchanged_items,
    )


# ============================================================
# CHUNKING
# ============================================================

def build_chunks_for_procedure(item: Dict) -> List[Dict]:
    """
    Reuse the project's canonical chunking.py logic.

    This keeps realtime sync consistent with the offline
    preprocessing/chunking pipeline.
    """

    from chunking import create_chunks_for_document

    data = item["data"]

    procedure_name = data["Tên thủ tục hành chính"]

    row = {
        "document_id": procedure_name,
        "procedure_name": procedure_name,
        "field": data.get(
            "Lĩnh vực",
            ""
        ),
        "submission_method": data.get(
            "Hình thức nộp",
            ""
        ),
        "required_documents": data.get(
            "Thành phần hồ sơ",
            ""
        ),
        "processing_time": data.get(
            "Thời gian giải quyết",
            ""
        ),
        "fee": data.get(
            "Lệ phí",
            ""
        ),
        "location": data.get(
            "Địa điểm tiếp nhận hồ sơ trực tiếp (nếu có)",
            ""
        ),
        "notes": data.get(
            "Ghi chú",
            ""
        ),
        "form_link": data.get(
            "Trong trường hợp hồ sơ bao gồm các biểu mẫu vui lòng dính kèm file mẫu",
            ""
        ),
    }

    # Dùng logic chunking chính thức của project
    chunks = create_chunks_for_document(row)

    # create_chunks_for_document() chưa tạo chunk_id.
    # create_all_chunks() mới tạo chunk_id.
    # Realtime phải tạo tương tự.
    for index, chunk in enumerate(
        chunks,
        start=1
    ):
        chunk["chunk_id"] = (
            f"doc_{procedure_name}_chunk_{index}"
        )

    return chunks


# ============================================================
# STABLE QDRANT ID
# ============================================================

def make_point_id(
    procedure_name: str,
    chunk_type: str,
    text: str,
) -> str:

    stable_key = (
        f"{procedure_name}|"
        f"{chunk_type}|"
        f"{text}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            stable_key,
        )
    )


# ============================================================
# DELETE OLD PROCEDURE
# ============================================================

def delete_procedure(
    client: QdrantClient,
    procedure_name: str,
):

    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
    )

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="procedure_name",
                    match=MatchValue(
                        value=procedure_name
                    ),
                )
            ]
        ),
    )


# ============================================================
# EMBEDDING
# ============================================================

def load_embedding_model():

    print(
        f"[INFO] Loading BKAI model: {MODEL_NAME}",
        file=sys.stderr,
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "[OK] BKAI model loaded",
        file=sys.stderr,
    )

    return model


def generate_embeddings(model, texts):

    from pyvi import ViTokenizer

    segmented_texts = [
        ViTokenizer.tokenize(text)
        for text in texts
    ]

    embeddings = model.encode(
        segmented_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings


# ============================================================
# UPSERT
# ============================================================

def upsert_procedure(
    client: QdrantClient,
    model,
    item: Dict,
):

    procedure_name = item["procedure_name"]

    chunks = build_chunks_for_procedure(
        item
    )

    if not chunks:
        print(
            f"[WARNING] No chunks generated: "
            f"{procedure_name}",
            file=sys.stderr,
        )
        return 0

    print(
        f"[INFO] Procedure: {procedure_name}",
        file=sys.stderr,
    )

    print(
        f"[INFO] Chunks: {len(chunks)}",
        file=sys.stderr,
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        model,
        texts,
    )

    points = []

    for chunk, vector in zip(
        chunks,
        embeddings,
    ):

        point_id = make_point_id(
            procedure_name,
            chunk["chunk_type"],
            chunk["text"],
        )

        payload = {
            "chunk_id": chunk["chunk_id"],
            "document_id": procedure_name,
            "procedure_name": procedure_name,
            "field": chunk["field"],
            "submission_method": chunk[
                "submission_method"
            ],
            "chunk_type": chunk[
                "chunk_type"
            ],
            "text": chunk["text"],
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"[OK] Upserted {len(points)} vectors",
        file=sys.stderr,
    )

    return len(points)


# ============================================================
# MAIN SYNC
# ============================================================

def sync():

    print(
        "=" * 70,
        file=sys.stderr,
    )

    print(
        "REALTIME LEGAL RAG SYNC",
        file=sys.stderr,
    )

    print(
        "=" * 70,
        file=sys.stderr,
    )

    # --------------------------------------------------------
    # 1. Load Excel
    # --------------------------------------------------------

    df = load_source()

    print(
        f"[INFO] Excel rows: {len(df)}",
        file=sys.stderr,
    )

    # --------------------------------------------------------
    # 2. Load state
    # --------------------------------------------------------

    old_state = load_state()

    print(
        f"[INFO] Previous procedures: "
        f"{len(old_state)}",
        file=sys.stderr,
    )

    # --------------------------------------------------------
    # 3. Detect changes
    # --------------------------------------------------------

    (
        new_items,
        changed_items,
        unchanged_items,
    ) = detect_changes(
        df,
        old_state,
    )

    print(
        f"[NEW]       {len(new_items)}",
        file=sys.stderr,
    )

    print(
        f"[CHANGED]   {len(changed_items)}",
        file=sys.stderr,
    )

    print(
        f"[UNCHANGED] {len(unchanged_items)}",
        file=sys.stderr,
    )

    # --------------------------------------------------------
    # Không có thay đổi
    # --------------------------------------------------------

    if not new_items and not changed_items:

        print(
            "[OK] No changes detected.",
            file=sys.stderr,
        )

        return {
            "status": "unchanged",
            "new": 0,
            "changed": 0,
            "unchanged": len(unchanged_items),
            "vectors_upserted": 0,
        }

    # --------------------------------------------------------
    # 4. Connect Qdrant
    # --------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    model = None

    vectors_upserted = 0

    try:

        # ----------------------------------------------------
        # 5. Load BKAI ONLY when there is data to update
        # ----------------------------------------------------

        model = load_embedding_model()

        # ----------------------------------------------------
        # 6. Process NEW
        # ----------------------------------------------------

        for item in new_items:

            print(
                f"\n[NEW] {item['procedure_name']}",
                file=sys.stderr,
            )

            # Remove any existing points for this procedure first.
            # This also makes the first realtime sync safe when Qdrant
            # already contains vectors from the previous batch ingestion.
            delete_procedure(
                client,
                item["procedure_name"],
            )

            print(
                "[OK] Existing vectors checked/deleted",
                file=sys.stderr,
            )

            vectors_upserted += (
                upsert_procedure(
                    client,
                    model,
                    item,
                )
            )

        # ----------------------------------------------------
        # 7. Process CHANGED
        # ----------------------------------------------------

        for item in changed_items:

            print(
                f"\n[CHANGED] "
                f"{item['procedure_name']}",
                file=sys.stderr,
            )

            # Xóa vector cũ của procedure
            delete_procedure(
                client,
                item["procedure_name"],
            )

            print(
                "[OK] Old vectors deleted",
                file=sys.stderr,
            )

            # Embed + upsert version mới
            vectors_upserted += (
                upsert_procedure(
                    client,
                    model,
                    item,
                )
            )

        # ----------------------------------------------------
        # 8. Update state
        # ----------------------------------------------------

        new_state = {}

        for item in (
            new_items
            + changed_items
            + unchanged_items
        ):

            new_state[
                item["procedure_name"]
            ] = item["hash"]

        save_state(new_state)

        print(
            "\n[OK] Sync state updated",
            file=sys.stderr,
        )

    finally:

        # ----------------------------------------------------
        # 9. Close Qdrant
        # ----------------------------------------------------

        client.close()

        print(
            "[OK] Qdrant client closed",
            file=sys.stderr,
        )

    return {
        "status": "synced",
        "new": len(new_items),
        "changed": len(changed_items),
        "unchanged": len(unchanged_items),
        "vectors_upserted": vectors_upserted,
    }


if __name__ == "__main__":

    result = sync()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )