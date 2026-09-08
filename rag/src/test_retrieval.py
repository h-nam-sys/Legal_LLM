`from sentence_transformers import SentenceTransformer
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
client = QdrantClient(url=QDRANT_URL)
print("Connected to Qdrant!")

# ============================================================
# QUERY
# ============================================================
query = "Quyền sở hữu là gì?"
print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)
print(query)

# ============================================================
# EMBEDDING QUERY
# ============================================================

query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

# ============================================================
# RETRIEVE
# ============================================================

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=TOP_K,
    with_payload=True,
).points

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)

for i, result in enumerate(results, 1):

    payload = result.payload

    print("\n" + "-" * 70)
    print(f"RESULT {i}")
    print("-" * 70)

    print(f"Score: {result.score}")
    print(f"ID: {payload.get('id')}")
    print(f"Law name: {payload.get('law_name')}")
    print(f"Article: {payload.get('article')}")
    print(f"Title: {payload.get('title')}")

    print("\nTEXT:")
    print(payload.get("text", ""))

print("\n" + "=" * 70)`