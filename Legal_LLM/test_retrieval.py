import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Same setup as the main app
Settings.embed_model = HuggingFaceEmbedding(
    model_name="keepitreal/vietnamese-sbert",
    device="cpu"
)

# 2. Point to the data folder
VECTOR_STORAGE_DIR = os.path.join("data", "vector_storage")

print("Loading index...")
storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORAGE_DIR)
vector_index = load_index_from_storage(storage_context)

# 3. Retrieve top 3 matches
retriever = vector_index.as_retriever(similarity_top_k=3)
query = "Vị trí trưởng phòng kinh doanh tại Hồ Chí Minh yêu cầu kinh nghiệm bao lâu và mức lương là bao nhiêu?"

print(f"\nSearching for: '{query}'\n")
nodes = retriever.retrieve(query)

for i, node in enumerate(nodes):
    print(f"--- Match {i+1} ---")
    print(f"Score: {node.score}")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}\n")
