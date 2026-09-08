from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "vohuutridung/vietnamese-legal-documents"

# Dataset có 2 configs: metadata và content
DATASET_CONFIG = "content"

COLLECTION_NAME = "legal_documents_bkai"

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

QDRANT_URL = "http://localhost:6333"

# Chỉ xử lý 1000 document để test trước
MAX_DOCUMENTS = 100

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("VIETNAMESE LEGAL DOCUMENTS - BKAI INGESTION")
print("=" * 70)

print("\nLoading dataset in streaming mode...")

ds = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split="data",
    streaming=True,
)

print("Dataset loaded!")

# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded!")

# ============================================================
# 3. CONNECT QDRANT
# ============================================================

print("\nConnecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL
)

print("Connected to Qdrant!")

# ============================================================
# 4. CREATE / RESET COLLECTION
# ============================================================

print("\nCreating Qdrant collection...")

if client.collection_exists(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' already exists.")
    print("Deleting old collection...")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
)

print(f"Collection '{COLLECTION_NAME}' created.")

# ============================================================
# 5. CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

# ============================================================
# 6. PROCESS DATASET
# ============================================================

print("\nStarting ingestion...")
print("=" * 70)

all_chunks = []

documents_processed = 0

for item in ds:

    # --------------------------------------------------------
    # Dataset fields:
    # id
    # content
    # --------------------------------------------------------

    document = Document(
        page_content=item["content"],
        metadata={
            "id": str(item["id"]),
        },
    )

    chunks = splitter.split_documents([document])

    all_chunks.extend(chunks)

    documents_processed += 1

    # In progress
    if documents_processed % 10 == 0:
        print(
            f"Documents processed: {documents_processed} | "
            f"Chunks created: {len(all_chunks)}"
        )

    # Stop after MAX_DOCUMENTS
    if documents_processed >= MAX_DOCUMENTS:
        break

# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHUNKING COMPLETED")
print("=" * 70)

print(f"Documents processed: {documents_processed}")
print(f"Total chunks: {len(all_chunks)}")

# ============================================================
# 8. CREATE EMBEDDINGS
# ============================================================

print("\nCreating BKAI embeddings...")

texts = [
    chunk.page_content
    for chunk in all_chunks
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32,
)

print("Embeddings created!")

print("Embedding shape:", embeddings.shape)
print("Vector dimension:", embeddings.shape[1])

# ============================================================
# 9. CREATE QDRANT POINTS
# ============================================================

print("\nCreating Qdrant points...")

points = []

for chunk, vector in zip(all_chunks, embeddings):

    point = PointStruct(
        id=str(uuid.uuid4()),

        vector=vector.tolist(),

        payload={
            "text": chunk.page_content,
            **chunk.metadata,
        },
    )

    points.append(point)

print(f"Created {len(points)} Qdrant points.")

# ============================================================
# 10. INSERT INTO QDRANT
# ============================================================

print("\nUploading vectors to Qdrant...")

# Chia thành batch để tránh gửi một request quá lớn
BATCH_SIZE = 256

for i in range(0, len(points), BATCH_SIZE):

    batch = points[i:i + BATCH_SIZE]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
    )

    print(
        f"Uploaded {min(i + BATCH_SIZE, len(points))}/{len(points)}"
    )

print("\nUpload completed!")

# ============================================================
# 11. VERIFY QDRANT
# ============================================================

collection_info = client.get_collection(
    COLLECTION_NAME
)

print("\n" + "=" * 70)
print("QDRANT VERIFICATION")
print("=" * 70)

print("Collection:", COLLECTION_NAME)
print("Vectors:", collection_info.points_count)

print("\n" + "=" * 70)
print("INGESTION COMPLETED SUCCESSFULLY!")
print("=" * 70)