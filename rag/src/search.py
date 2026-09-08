from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_documents"

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"


# ============================================================
# 1. CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL
)

print("Connected to Qdrant!")


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded!")


# ============================================================
# 3. USER QUERY
# ============================================================

query = "Tôi muốn nghỉ việc thì cần báo trước như thế nào?"

print("\nQuery:")
print(query)


# ============================================================
# 4. CREATE QUERY EMBEDDING
# ============================================================

query_vector = model.encode(
    query,
    normalize_embeddings=True
)

print("\nQuery vector dimension:")
print(len(query_vector))


# ============================================================
# 5. SEARCH QDRANT
# ============================================================

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    limit=3,
).points


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SEARCH RESULTS")
print("=" * 70)

for i, result in enumerate(results, start=1):

    payload = result.payload

    print(f"\n--- Result {i} ---")

    print("Score:", result.score)
    print("Law:", payload.get("law_name"))
    print("Article:", payload.get("article"))
    print("Title:", payload.get("title"))

    print("Text:")
    print(payload.get("text"))