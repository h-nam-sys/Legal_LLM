from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_documents_bkai"

client = QdrantClient(url=QDRANT_URL)

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_payload=True,
    with_vectors=False
)

for i, point in enumerate(points, 1):
    print("=" * 70)
    print(f"POINT {i}")
    print("=" * 70)
    print("ID:", point.id)
    print("PAYLOAD:")
    print(point.payload)