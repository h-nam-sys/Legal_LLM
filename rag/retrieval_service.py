import os

os.environ["HF_HOME"] = r"D:\Program Files\.cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import math
from typing import Optional, Tuple, Dict, Any
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from FlagEmbedding import FlagReranker

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORAGE_DIR = os.path.join(ROOT_DIR, "data", "vector_storage")

# 1. Embedding Setup (BGE-M3)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cpu"
)

# 2. In-memory Index Loader
print("[RetrievalService] Loading BGE-M3 Vector Index into RAM...")
storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORAGE_DIR)
vector_index = load_index_from_storage(storage_context)
base_retriever = vector_index.as_retriever(similarity_top_k=5)

# 3. Cross-Encoder Reranker Setup
print("[RetrievalService] Initializing BGE-Reranker-v2-m3...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))

def retrieve_and_rerank(query: str, top_k: int = 5) -> Tuple[Optional[int], float, Dict[str, Any]]:
    """
    1. Fetches top_k candidates via dense bi-encoder (BGE-M3).
    2. Reranks candidates via cross-encoder (BGE-Reranker-v2-m3).
    Returns (procedure_id, normalized_score, metadata).
    """
    candidates = base_retriever.retrieve(query)
    if not candidates:
        return None, 0.0, {}

    # Format pairs for cross-encoder scoring: [query, doc_text]
    pairs = [[query, node.text] for node in candidates]
    raw_scores = reranker.compute_score(pairs)

    if isinstance(raw_scores, float):
        raw_scores = [raw_scores]

    # Rank candidate nodes by reranker score
    scored_candidates = []
    for node, raw_score in zip(candidates, raw_scores):
        normalized_score = _sigmoid(raw_score)
        scored_candidates.append((node, normalized_score))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    best_node, best_score = scored_candidates[0]

    procedure_id = best_node.metadata.get("id")
    return procedure_id, best_score, best_node.metadata
