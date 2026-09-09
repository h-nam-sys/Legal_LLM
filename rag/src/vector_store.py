from pathlib import Path

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

COLLECTION_NAME = "vietnamese_administrative_procedures"

VECTOR_SIZE = 768

TOP_K_TEST = 5


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

    required_columns = [
        "chunk_id",
        "document_id",
        "procedure_name",
        "field",
        "submission_method",
        "chunk_type",
        "text",
        "embedding",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    return df


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
    print("CONNECTING TO QDRANT")
    print("=" * 80)

    # Local in-memory Qdrant.
    #
    # Later we can switch to Docker Qdrant.
    client = QdrantClient(path=str(PROJECT_ROOT / "qdrant_data"))

    print("\n[OK] Qdrant client created")

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
        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print(
            f"Existing collection '{COLLECTION_NAME}' "
            "deleted."
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"\n[OK] Collection created: "
        f"{COLLECTION_NAME}"
    )

    print("Vector size: 768")
    print("Distance   : COSINE")


# ============================================================
# INSERT VECTORS
# ============================================================

def insert_vectors(client, df, embeddings):

    print("\n" + "=" * 80)
    print("INSERTING VECTORS")
    print("=" * 80)

    points = []

    for index, row in df.iterrows():

        payload = {
            "chunk_id": str(row["chunk_id"]),
            "document_id": int(row["document_id"]),
            "procedure_name": str(
                row["procedure_name"]
            ),
            "field": str(row["field"]),
            "submission_method": str(
                row["submission_method"]
            ),
            "chunk_type": str(
                row["chunk_type"]
            ),
            "chunk_index": int(
                row["chunk_index"]
            ),
            "text": str(row["text"]),
            "processing_time": str(
                row["processing_time"]
            ),
            "fee": str(row["fee"]),
            "location": str(row["location"]),
            "notes": str(row["notes"]),
            "form_link": str(
                row["form_link"]
            ),
            "status": str(row["status"]),
        }

        point = PointStruct(
            id=index,
            vector=embeddings[index].tolist(),
            payload=payload,
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"\n[OK] Inserted {len(points):,} vectors"
    )


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection(client):

    print("\n" + "=" * 80)
    print("QDRANT COLLECTION VALIDATION")
    print("=" * 80)

    info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print(
        f"\nCollection: {COLLECTION_NAME}"
    )

    print(
        f"Vectors: {info.points_count}"
    )

    print(
        f"Vector size: {VECTOR_SIZE}"
    )

    if info.points_count == 124:
        print(
            "[OK] All 124 vectors stored"
        )
    else:
        print(
            "[WARNING] Expected 124 vectors, "
            f"got {info.points_count}"
        )


# ============================================================
# TEST VECTOR SEARCH
# ============================================================

def test_search(client, embeddings):

    print("\n" + "=" * 80)
    print("TEST VECTOR SEARCH")
    print("=" * 80)

    # Use the first vector as a test query.
    query_vector = embeddings[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_TEST,
        with_payload=True,
    ).points

    print(
        f"\nTop {TOP_K_TEST} results:"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print("\n" + "-" * 80)

        print(f"Rank       : {rank}")
        print(f"Score      : {result.score:.4f}")
        print(
            f"Chunk ID   : "
            f"{payload['chunk_id']}"
        )
        print(
            f"Document ID: "
            f"{payload['document_id']}"
        )
        print(
            f"Procedure  : "
            f"{payload['procedure_name']}"
        )
        print(
            f"Chunk type : "
            f"{payload['chunk_type']}"
        )

        print("\nText:")
        print(payload["text"][:500])


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load embeddings
    df = load_embeddings()

    # 2. Validate
    embeddings = validate_embeddings(df)

    # 3. Create Qdrant client
    client = create_client()

    # 4. Create collection
    create_collection(client)

    # 5. Insert vectors
    insert_vectors(
        client,
        df,
        embeddings
    )

    # 6. Verify
    verify_collection(client)

    # 7. Test similarity search
    test_search(
        client,
        embeddings
    )

    print("\n" + "=" * 80)
    print("QDRANT VECTOR STORE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
