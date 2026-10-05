import os
import math
import re
from typing import Optional, Tuple, Dict, Any
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from FlagEmbedding import FlagReranker

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOMAINS_DIR = os.path.join(DATA_DIR, "domains")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cpu")

_INDEX_CACHE = {}
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def sanitize_identifier(name: str) -> str:
    name = name.strip().lower()
    vn_map = str.maketrans(
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ",
        "a" * 17 + "e" * 11 + "i" * 5 + "o" * 17 + "u" * 11 + "y" * 5 + "d",
    )
    name = name.translate(vn_map)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name if name else "default_domain"

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))

def get_vector_retriever(domain: str):
    domain_clean = sanitize_identifier(domain)
    if domain_clean in _INDEX_CACHE:
        return _INDEX_CACHE[domain_clean]
    
    vector_dir = os.path.join(DOMAINS_DIR, domain_clean, "vector_storage")
    if not os.path.exists(vector_dir):
        vector_dir = os.path.join(DATA_DIR, "vector_storage")

    if os.path.exists(vector_dir):
        storage_context = StorageContext.from_defaults(persist_dir=vector_dir)
        vector_index = load_index_from_storage(storage_context)
        retriever = vector_index.as_retriever(similarity_top_k=5)
        _INDEX_CACHE[domain_clean] = retriever
        return retriever
    return None

def retrieve_and_rerank(query: str, domain: str = "it_support", top_k: int = 5) -> Tuple[Optional[str], float, Dict[str, Any]]:
    retriever = get_vector_retriever(domain)
    if not retriever:
        return None, 0.0, {}

    candidates = retriever.retrieve(query)
    if not candidates:
        return None, 0.0, {}

    pairs = [[query, node.text] for node in candidates]
    raw_scores = reranker.compute_score(pairs)
    if isinstance(raw_scores, float):
        raw_scores = [raw_scores]

    scored_candidates = []
    for node, raw_score in zip(candidates, raw_scores):
        scored_candidates.append((node, _sigmoid(raw_score)))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    best_node, best_score = scored_candidates[0]

    record_id = best_node.metadata.get("id") or best_node.node_id
    return record_id, best_score, best_node.metadata
# ============================================================
# KHỐI TEST CHẠY TRỰC TIẾP RETRIEVAL SERVICE
# ============================================================
if __name__ == "__main__":
    test_query = "thành phần hồ sơ thời hạn giải quyết thủ tục hành chính"
    test_domain = "thu_tuc_hanh_chinh"

    print(f"\n=================== KẾT QUẢ TEST RETRIEVAL SERVICE ===================")
    print(f"Domain: {test_domain}")
    print(f"Query: {test_query}\n")

    record_id, score, metadata = retrieve_and_rerank(test_query, domain=test_domain)

    print(f"[1] Top Match Record ID : {record_id}")
    print(f"[2] Reranker Score     : {score:.4f}")
    print(f"[3] Metadata đi kèm    : {metadata}")
    print(f"======================================================================\n")