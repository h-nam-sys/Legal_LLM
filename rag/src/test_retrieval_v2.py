from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_documents_bkai"

TOP_K = 10

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")

# ============================================================
# CONNECT QDRANT
# ============================================================

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL
)

print("Connected to Qdrant!")

# ============================================================
# FUNCTION: RETRIEVE
# ============================================================

def retrieve(query, top_k=10):

    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    # --------------------------------------------------------
    # Embedding query
    # --------------------------------------------------------

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"TOP {top_k} RETRIEVAL RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, 1):

        payload = result.payload or {}

        text = payload.get("text", "")

        print("\n" + "-" * 70)
        print(f"RESULT {i}")
        print("-" * 70)

        print(f"Score: {result.score:.4f}")
        print(f"ID: {payload.get('id')}")

        print("\nTEXT:")
        print(text[:1500])

    return results


# ============================================================
# TEST
# ============================================================

queries = [
    "Quyền sở hữu là gì?",
    "Quyền sở hữu bao gồm những quyền nào?",
    "Người có quyền sở hữu tài sản có những quyền gì?"
]

for query in queries:
    retrieve(query, TOP_K)