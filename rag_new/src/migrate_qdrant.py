from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


OLD_QDRANT_PATH = r"D:\legal-rag\qdrant_data"
NEW_QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "vietnamese_administrative_procedures"


# ============================================================
# 1. Kết nối Qdrant cũ
# ============================================================

old_client = QdrantClient(path=OLD_QDRANT_PATH)

old_info = old_client.get_collection(COLLECTION_NAME)

print("=" * 80)
print("OLD QDRANT")
print("=" * 80)
print("Collection   :", COLLECTION_NAME)
print("Vectors count:", old_info.points_count)
print("Vector size  :", old_info.config.params.vectors.size)


# ============================================================
# 2. Đọc toàn bộ points từ Qdrant cũ
# ============================================================

print()
print("Reading vectors from old Qdrant...")

all_points = []

offset = None

while True:
    points, offset = old_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        offset=offset,
        with_vectors=True,
        with_payload=True
    )

    all_points.extend(points)

    if offset is None:
        break


print("Read:", len(all_points), "vectors")


# ============================================================
# 3. Kết nối Qdrant Server
# ============================================================

new_client = QdrantClient(url=NEW_QDRANT_URL)

print()
print("=" * 80)
print("NEW QDRANT SERVER")
print("=" * 80)

print("Connected successfully!")


# ============================================================
# 4. Tạo collection trên Qdrant Server
# ============================================================

collections = new_client.get_collections()

existing_names = [
    c.name for c in collections.collections
]


if COLLECTION_NAME not in existing_names:

    print()
    print("Creating collection...")

    new_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

    print("Collection created!")

else:

    print()
    print("Collection already exists.")


# ============================================================
# 5. Upload vectors + payload
# ============================================================

print()
print("Uploading vectors...")

points_to_upload = []

for point in all_points:

    points_to_upload.append(
        PointStruct(
            id=point.id,
            vector=point.vector,
            payload=point.payload
        )
    )


new_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points_to_upload
)

print("Upload completed!")
print("Uploaded:", len(points_to_upload), "vectors")


# ============================================================
# 6. Kiểm tra lại
# ============================================================

new_info = new_client.get_collection(COLLECTION_NAME)

print()
print("=" * 80)
print("MIGRATION RESULT")
print("=" * 80)

print("Collection   :", COLLECTION_NAME)
print("Vectors count:", new_info.points_count)
print("Vector size  :", new_info.config.params.vectors.size)

print()
print("Migration finished successfully!")