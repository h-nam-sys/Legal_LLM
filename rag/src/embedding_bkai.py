from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


print("=" * 70)
print("BKAI VIETNAMESE BI-ENCODER")
print("=" * 70)

print("\nLoading model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")

# Các câu test giống nhau để so sánh với keepitreal
sentences = [
    "Tôi muốn nghỉ việc thì cần báo trước như thế nào?",
    "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động.",
    "Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng."
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

print("\nEmbedding shape:", embeddings.shape)
print("Vector dimension:", embeddings.shape[1])

print("\nFirst vector:")
print(embeddings[0][:10])

print("\nSimilarity matrix:")
similarity = np.dot(embeddings, embeddings.T)

print(np.round(similarity, 4))