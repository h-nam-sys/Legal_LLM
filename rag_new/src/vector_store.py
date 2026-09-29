from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_embeddings.parquet"
)

# Qdrant Embedded Local Mode - không cần Docker
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_storage")

COLLECTION_NAME = "vietnamese_administrative_procedures"

VECTOR_SIZE = 768

TOP_K_TEST = 5

BATCH_SIZE = 64

# Các cột chunk đưa vào payload của Qdrant (schema mới từ chunking.py).
# legal_codes = "số hiệu văn bản": BẮT BUỘC có trong payload để MỌI kết quả
# truy xuất đều trả về cột này.
PAYLOAD_COLUMNS = [
    "chunk_id",
    "document_id",
    "procedure_name",
    "executing_agency",
    "legal_codes",
    "chunk_type",
    "text",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_embeddings():

    print("=" * 80)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("QDRANT VECTOR STORE")
    print("=" * 80)

    print("\nLoading embeddings...")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    for column in PAYLOAD_COLUMNS + ["embedding"]:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}. "
                "Hãy chạy lại chunking.py rồi embedding.py với "
                "schema mới."
            )

    empty_codes = (
        df["legal_codes"].fillna("").astype(str).str.strip().eq("").sum()
    )

    if empty_codes:
        raise ValueError(
            f"{empty_codes} chunk có legal_codes (số hiệu văn bản) trống."
        )

    # Đảm bảo index liên tục để embeddings[index] luôn khớp hàng
    return df.reset_index(drop=True)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(df):

    print("\n" + "=" * 80)
    print("VECTOR VALIDATION")
    print("=" * 80)

    embeddings = np.array(
        df["embedding"].tolist(),
        dtype=np.float32
    )

    print(f"\nNumber of vectors: {len(embeddings):,}")
    print(f"Vector dimension : {embeddings.shape[1]}")

    if embeddings.shape[1] != VECTOR_SIZE:

        raise ValueError(
            f"Expected {VECTOR_SIZE}-dimensional vectors, "
            f"got {embeddings.shape[1]}"
        )

    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN.")

    if np.isinf(embeddings).any():
        raise ValueError("Embeddings contain Inf.")

    print("\n[OK] Correct vector dimension")
    print("[OK] No NaN")
    print("[OK] No Inf")

    return embeddings


# ============================================================
# CREATE QDRANT CLIENT
# ============================================================

def create_client():

    print("\n" + "=" * 80)
    print("CONNECTING TO QDRANT - LOCAL EMBEDDED MODE")
    print("=" * 80)

    print(f"\nQdrant storage: {QDRANT_PATH}")

    # Script này luôn DỰNG LẠI toàn bộ từ file embeddings, nên xoá sạch thư
    # mục lưu trữ trước khi mở. Lý do: ở chế độ local, delete_collection +
    # create_collection từng để sót điểm của lần build cũ (17.302 điểm thay
    # vì 9.154), khiến truy xuất trả về chunk cũ không có số hiệu văn bản.
    if Path(QDRANT_PATH).exists():
        try:
            shutil.rmtree(QDRANT_PATH)
        except PermissionError as error:
            raise RuntimeError(
                f"Không xoá được {QDRANT_PATH}: {error}\n"
                "Có tiến trình khác đang mở Qdrant (retrieval/app/notebook)? "
                "Hãy tắt rồi chạy lại."
            ) from error
        print("[OK] Đã xoá dữ liệu Qdrant cũ")

    Path(QDRANT_PATH).mkdir(
        parents=True,
        exist_ok=True,
    )

    client = QdrantClient(path=QDRANT_PATH)

    collections = client.get_collections()

    print("\n[OK] Qdrant local storage opened")

    print(
        f"Existing collections: "
        f"{len(collections.collections)}"
    )

    return client


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(client):

    print("\n" + "=" * 80)
    print("CREATING COLLECTION")
    print("=" * 80)

    existing = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing:

        print(f"\nExisting collection found: {COLLECTION_NAME}")
        print("Deleting old collection...")

        client.delete_collection(collection_name=COLLECTION_NAME)

        print("[OK] Old collection deleted")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    stale = client.count(collection_name=COLLECTION_NAME, exact=True).count

    if stale != 0:
        raise RuntimeError(
            f"Collection mới tạo nhưng đã có {stale} điểm - dữ liệu cũ "
            "chưa được xoá sạch."
        )

    print(f"\n[OK] Collection created (rỗng): {COLLECTION_NAME}")
    print(f"Vector size: {VECTOR_SIZE}")
    print("Distance   : COSINE")


# ============================================================
# CREATE STABLE POINT ID
# ============================================================

def create_point_id(chunk_id):
    """
    Convert chunk_id into a deterministic UUID (cùng chunk_id -> cùng UUID
    mỗi lần chạy), cần cho đồng bộ dữ liệu về sau.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


def build_payload(row):
    """
    Payload của 1 chunk. Tất cả là chuỗi; document_id (proc_id, vd
    "1.000028") KHÔNG được ép sang số vì sẽ mất số 0 / lỗi.
    """

    return {column: str(row[column]) for column in PAYLOAD_COLUMNS}


# ============================================================
# INSERT VECTORS
# ============================================================

def insert_vectors(client, df, embeddings):

    print("\n" + "=" * 80)
    print("INSERTING VECTORS")
    print("=" * 80)

    points = []

    for index, row in df.iterrows():

        points.append(
            PointStruct(
                id=create_point_id(row["chunk_id"]),
                vector=embeddings[index].tolist(),
                payload=build_payload(row),
            )
        )

    print(f"\nTotal points to insert: {len(points):,}")

    for start in range(0, len(points), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(points))

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:end],
        )

        print(f"Inserted: {end:,}/{len(points):,}")

    print(f"\n[OK] Inserted {len(points):,} vectors")


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection(client, expected_count):

    print("\n" + "=" * 80)
    print("QDRANT COLLECTION VALIDATION")
    print("=" * 80)

    stored = client.count(collection_name=COLLECTION_NAME, exact=True).count

    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Vectors: {stored}")
    print(f"Expected: {expected_count}")
    print(f"Vector size: {VECTOR_SIZE}")

    if stored == expected_count:

        print(f"\n[OK] All {expected_count:,} vectors stored")

    else:

        raise ValueError(
            "Vector count mismatch: "
            f"expected {expected_count}, "
            f"got {stored}"
        )


# ============================================================
# TEST VECTOR SEARCH
# ============================================================

def test_search(client, embeddings):

    print("\n" + "=" * 80)
    print("TEST VECTOR SEARCH")
    print("=" * 80)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embeddings[0].tolist(),
        limit=TOP_K_TEST,
        with_payload=True,
    ).points

    print(f"\nTop {TOP_K_TEST} results:")

    missing_codes = 0

    for rank, result in enumerate(results, start=1):

        payload = result.payload or {}

        if not str(payload.get("legal_codes", "")).strip():
            missing_codes += 1

        print("\n" + "-" * 80)
        print(f"Rank        : {rank}")
        print(f"Score       : {result.score:.4f}")
        print(f"Chunk ID    : {payload.get('chunk_id')}")
        print(f"Document ID : {payload.get('document_id')}")
        print(f"Procedure   : {payload.get('procedure_name')}")
        print(f"Cơ quan     : {payload.get('executing_agency')}")
        print(f"Số hiệu VB  : {payload.get('legal_codes')}")
        print(f"Chunk type  : {payload.get('chunk_type')}")
        print("\nText:")
        print(str(payload.get("text", ""))[:500])

    print()

    if missing_codes == 0:
        print("[OK] Mọi kết quả truy xuất đều có số hiệu văn bản")
    else:
        print(
            f"[ERROR] {missing_codes} kết quả thiếu số hiệu văn bản "
            "trong payload"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    client = None

    try:

        df = load_embeddings()

        embeddings = validate_embeddings(df)

        client = create_client()

        create_collection(client)

        insert_vectors(client, df, embeddings)

        verify_collection(client, expected_count=len(df))

        test_search(client, embeddings)

        print("\n" + "=" * 80)
        print("QDRANT VECTOR STORE COMPLETED SUCCESSFULLY")
        print("=" * 80)

    finally:

        if client is not None:

            client.close()

            print("\n[OK] Qdrant client closed")


if __name__ == "__main__":

    main()