from pathlib import Path
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

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "vietnamese_administrative_procedures"

VECTOR_SIZE = 768

TOP_K_TEST = 5

BATCH_SIZE = 64


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
        raise ValueError(
            "Embeddings contain NaN."
        )

    if np.isinf(embeddings).any():
        raise ValueError(
            "Embeddings contain Inf."
        )

    print("\n[OK] Correct vector dimension")
    print("[OK] No NaN")
    print("[OK] No Inf")

    return embeddings


# ============================================================
# CREATE QDRANT CLIENT
# ============================================================

def create_client():

    print("\n" + "=" * 80)
    print("CONNECTING TO QDRANT SERVER")
    print("=" * 80)

    print(f"\nQdrant URL: {QDRANT_URL}")

    client = QdrantClient(
        url=QDRANT_URL
    )

    # Check connection
    collections = client.get_collections()

    print("\n[OK] Connected to Qdrant Server")

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

        print(
            f"\nExisting collection found: "
            f"{COLLECTION_NAME}"
        )

        print("Deleting old collection...")

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print("[OK] Old collection deleted")

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

    print(f"Vector size: {VECTOR_SIZE}")
    print("Distance   : COSINE")


# ============================================================
# CREATE STABLE POINT ID
# ============================================================

def create_point_id(chunk_id):

    """
    Convert chunk_id into a deterministic UUID.

    Example:
        doc_2_chunk_2
        -> same UUID every time

    This is important for future real-time synchronization.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            str(chunk_id)
        )
    )


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
            "chunk_id": str(
                row["chunk_id"]
            ),

            "document_id": int(
                row["document_id"]
            ),

            "procedure_name": str(
                row["procedure_name"]
            ),

            "field": str(
                row["field"]
            ),

            "submission_method": str(
                row["submission_method"]
            ),

            "chunk_type": str(
                row["chunk_type"]
            ),

            "text": str(
                row["text"]
            ),
        }

        point = PointStruct(

            id=create_point_id(
                row["chunk_id"]
            ),

            vector=embeddings[index].tolist(),

            payload=payload,
        )

        points.append(point)

    print(
        f"\nTotal points to insert: "
        f"{len(points):,}"
    )

    # --------------------------------------------------------
    # Batch upsert
    # --------------------------------------------------------

    for start in range(
        0,
        len(points),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(points)
        )

        batch = points[start:end]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        print(
            f"Inserted: {end:,}/{len(points):,}"
        )

    print(
        f"\n[OK] Inserted "
        f"{len(points):,} vectors"
    )


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection(client, expected_count):

    print("\n" + "=" * 80)
    print("QDRANT COLLECTION VALIDATION")
    print("=" * 80)

    info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print(
        f"\nCollection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vectors: "
        f"{info.points_count}"
    )

    print(
        f"Expected: "
        f"{expected_count}"
    )

    print(
        f"Vector size: "
        f"{VECTOR_SIZE}"
    )

    if info.points_count == expected_count:

        print(
            f"\n[OK] All "
            f"{expected_count:,} vectors stored"
        )

    else:

        raise ValueError(
            "Vector count mismatch: "
            f"expected {expected_count}, "
            f"got {info.points_count}"
        )


# ============================================================
# TEST VECTOR SEARCH
# ============================================================

def test_search(client, embeddings):

    print("\n" + "=" * 80)
    print("TEST VECTOR SEARCH")
    print("=" * 80)

    # Use the first vector as test query

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

        payload = result.payload or {}

        print("\n" + "-" * 80)

        print(
            f"Rank       : {rank}"
        )

        print(
            f"Score      : "
            f"{result.score:.4f}"
        )

        print(
            f"Chunk ID   : "
            f"{payload.get('chunk_id')}"
        )

        print(
            f"Document ID: "
            f"{payload.get('document_id')}"
        )

        print(
            f"Procedure  : "
            f"{payload.get('procedure_name')}"
        )

        print(
            f"Chunk type : "
            f"{payload.get('chunk_type')}"
        )

        print("\nText:")

        print(
            str(
                payload.get("text", "")
            )[:500]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    client = None

    try:

        # 1. Load embeddings

        df = load_embeddings()

        # 2. Validate embeddings

        embeddings = validate_embeddings(df)

        # 3. Connect Qdrant Server

        client = create_client()

        # 4. Create/recreate collection

        create_collection(client)

        # 5. Insert vectors

        insert_vectors(
            client,
            df,
            embeddings
        )

        # 6. Verify collection

        verify_collection(
            client,
            expected_count=len(df)
        )

        # 7. Test similarity search

        test_search(
            client,
            embeddings
        )

        print("\n" + "=" * 80)
        print("QDRANT VECTOR STORE COMPLETED SUCCESSFULLY")
        print("=" * 80)

    finally:

        if client is not None:

            client.close()

            print(
                "\n[OK] Qdrant client closed"
            )


if __name__ == "__main__":

    main()