from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "embeddings.npy"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "embeddings.jsonl"
)

QDRANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "qdrant_storage"
)

COLLECTION_NAME = (
    "vietnamese_administrative_procedures"
)

VECTOR_DIMENSION = 768

DISTANCE = Distance.COSINE

BATCH_SIZE = 64


# =============================================================================
# LOAD EMBEDDINGS
# =============================================================================

def load_embeddings() -> np.ndarray:

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"Embeddings not found:\n"
            f"{EMBEDDINGS_FILE}"
        )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        f"Loaded embeddings: "
        f"{embeddings.shape}"
    )

    if embeddings.ndim != 2:

        raise ValueError(
            f"Expected 2D embeddings, "
            f"got {embeddings.shape}"
        )

    if embeddings.shape[1] != VECTOR_DIMENSION:

        raise ValueError(
            f"Expected dimension "
            f"{VECTOR_DIMENSION}, "
            f"got {embeddings.shape[1]}"
        )

    return embeddings


# =============================================================================
# LOAD PAYLOAD METADATA
# =============================================================================

def load_metadata() -> list[dict]:

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n"
            f"{METADATA_FILE}"
        )

    records = []

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    print(
        f"Loaded metadata: "
        f"{len(records)} records"
    )

    return records


# =============================================================================
# VALIDATE
# =============================================================================

def validate_data(
    embeddings: np.ndarray,
    metadata: list[dict],
) -> None:

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Embedding / metadata count mismatch: "
            f"{len(embeddings)} vs "
            f"{len(metadata)}"
        )

    required_payload_fields = {
        "chunk_id",
        "procedure_id",
        "procedure_name",
        "domain",
        "intent",
        "chunk_type",
        "text",
    }

    for index, record in enumerate(metadata):

        missing = (
            required_payload_fields
            - set(record.keys())
        )

        if missing:

            raise ValueError(
                f"Metadata record {index} "
                f"is missing: {missing}"
            )

    chunk_ids = [
        record["chunk_id"]
        for record in metadata
    ]

    if len(chunk_ids) != len(set(chunk_ids)):

        raise ValueError(
            "Duplicate chunk_id found."
        )

    print("Data validation: PASS")


# =============================================================================
# CONNECT QDRANT
# =============================================================================

def create_client() -> QdrantClient:

    QDRANT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("QDRANT")
    print("=" * 80)

    print(
        f"Storage: {QDRANT_PATH}"
    )

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    return client


# =============================================================================
# CREATE COLLECTION
# =============================================================================

def recreate_collection(
    client: QdrantClient,
) -> None:

    collections = client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    if COLLECTION_NAME in existing_names:

        print(
            f"Collection '{COLLECTION_NAME}' "
            f"already exists."
        )

        print(
            "Deleting old collection..."
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    print(
        f"Creating collection "
        f"'{COLLECTION_NAME}'..."
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,
            distance=DISTANCE,
        ),
    )

    print("Collection created.")


# =============================================================================
# BUILD POINTS
# =============================================================================

def build_points(
    embeddings: np.ndarray,
    metadata: list[dict],
) -> list[PointStruct]:

    points = []

    for index, record in enumerate(metadata):

        payload = {
            "chunk_id": record["chunk_id"],
            "procedure_id": record["procedure_id"],
            "procedure_name": record["procedure_name"],
            "domain": record["domain"],
            "intent": record["intent"],
            "chunk_type": record["chunk_type"],
            "text": record["text"],
        }

        point = PointStruct(
            id=index,
            vector=embeddings[index].tolist(),
            payload=payload,
        )

        points.append(point)

    return points


# =============================================================================
# INSERT
# =============================================================================

def insert_points(
    client: QdrantClient,
    points: list[PointStruct],
) -> None:

    total = len(points)

    print()
    print("=" * 80)
    print("INSERTING VECTORS")
    print("=" * 80)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = points[start:end]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

        print(
            f"Inserted "
            f"{end}/{total}"
        )


# =============================================================================
# VALIDATE QDRANT
# =============================================================================

def validate_collection(
    client: QdrantClient,
) -> None:

    collection = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print()
    print("=" * 80)
    print("QDRANT VALIDATION")
    print("=" * 80)

    print(
        f"Collection : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vectors    : "
        f"{collection.points_count}"
    )

    print(
        f"Dimension  : "
        f"{collection.config.params.vectors.size}"
    )

    print(
        f"Distance   : "
        f"{collection.config.params.vectors.distance}"
    )

    if collection.points_count != 242:

        raise ValueError(
            f"Expected 242 vectors, "
            f"got {collection.points_count}"
        )

    if (
        collection.config.params.vectors.size
        != VECTOR_DIMENSION
    ):

        raise ValueError(
            "Vector dimension mismatch."
        )

    print()
    print("Qdrant validation: PASS")


# =============================================================================
# TEST SEARCH
# =============================================================================

def test_search(
    client: QdrantClient,
    embeddings: np.ndarray,
) -> None:

    print()
    print("=" * 80)
    print("VECTOR SEARCH TEST")
    print("=" * 80)

    query_vector = embeddings[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
        with_payload=True,
    ).points

    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = result.payload

        print()
        print(
            f"Rank {rank}"
        )

        print(
            f"Score     : "
            f"{result.score:.4f}"
        )

        print(
            f"Chunk ID   : "
            f"{payload['chunk_id']}"
        )

        print(
            f"Procedure : "
            f"{payload['procedure_name']}"
        )

        print(
            f"Intent    : "
            f"{payload['intent']}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - QDRANT INDEXING / PHASE 4")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load
    # -------------------------------------------------------------------------

    embeddings = load_embeddings()

    metadata = load_metadata()

    # -------------------------------------------------------------------------
    # 2. Validate
    # -------------------------------------------------------------------------

    validate_data(
        embeddings,
        metadata,
    )

    # -------------------------------------------------------------------------
    # 3. Qdrant
    # -------------------------------------------------------------------------

    client = create_client()

    # -------------------------------------------------------------------------
    # 4. Recreate collection
    # -------------------------------------------------------------------------

    recreate_collection(client)

    # -------------------------------------------------------------------------
    # 5. Build points
    # -------------------------------------------------------------------------

    points = build_points(
        embeddings,
        metadata,
    )

    print(
        f"Points prepared: "
        f"{len(points)}"
    )

    # -------------------------------------------------------------------------
    # 6. Insert
    # -------------------------------------------------------------------------

    insert_points(
        client,
        points,
    )

    # -------------------------------------------------------------------------
    # 7. Validate
    # -------------------------------------------------------------------------

    validate_collection(client)

    # -------------------------------------------------------------------------
    # 8. Search test
    # -------------------------------------------------------------------------

    test_search(
        client,
        embeddings,
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 4 COMPLETED")
    print("=" * 80)

    print(
        "242 vectors successfully indexed "
        "into Qdrant."
    )


if __name__ == "__main__":
    main()