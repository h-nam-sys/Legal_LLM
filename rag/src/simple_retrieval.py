# -*- coding: utf-8 -*-
import os
from pathlib import Path
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", str(PROJECT_ROOT / "qdrant_storage"))
COLLECTION_NAME = "vietnamese_administrative_procedures"

QDRANT_TOP_K = 20
FINAL_TOP_K = 3
MIN_VECTOR_SCORE = 0.20

@dataclass
class RetrievalResult:
    rank: int
    final_score: float
    vector_score: float
    procedure_score: float
    document_id: str
    procedure: str
    text: str
    payload: Dict[str, Any]

def normalize_without_accents(text: Any) -> str:
    if text is None: return ""
    text = str(text).lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    normalized = unicodedata.normalize("NFC", normalized)
    return re.sub(r"[^\w\s]", " ", normalized).strip()

GENERIC_ADMIN_WORDS = {
    "thu", "tuc", "dang", "ky", "giay", "to", "ho", "so", "xac", "nhan", "cap", "lam",
    "gioi", "thieu", "quy", "dinh", "doi", "tuong", "nguoi", "viec", "the", "tinh", "trang",
    "chung", "nhat", "van", "ban", "nay", "duoc", "va", "hoac", "cua", "tai", "theo",
    "co", "khong", "trong", "gom", "nhung", "gi", "can", "nao", "muon", "hoi", "la",
    "thi", "nho", "xin", "hay", "ve", "bao", "nhieu", "mot", "toi",
}

def _meaningful_tokens(text: str) -> set[str]:
    return {t for t in normalize_without_accents(text).split() if t and t not in GENERIC_ADMIN_WORDS}

# RESTORED: Robust payload extraction so it doesn't fail on alternate key names
def payload_get(payload: Dict[str, Any], keys: List[str], default: str = "") -> str:
    if not payload: return default
    for key in keys:
        if key in payload and payload[key] is not None:
            return str(payload[key])
    return default

def extract_payload_data(point: Any) -> Dict[str, Any]:
    payload = getattr(point, "payload", {}) if hasattr(point, "payload") else (point.get("payload", {}) if isinstance(point, dict) else {})
    if not isinstance(payload, dict): payload = {}

    return {
        "chunk_id": payload_get(payload, ["chunk_id", "id", "chunkId"]),
        "document_id": payload_get(payload, ["document_id", "documentId", "doc_id", "id"]),
        "procedure": payload_get(payload, ["procedure", "procedure_name", "procedureName", "ten_thu_tuc", "Tên thủ tục hành chính"]),
        "chunk_type": payload_get(payload, ["chunk_type", "type", "section_type", "loai_chunk"]),
        "text": payload_get(payload, ["text", "content", "page_content", "chunk"]),
        "payload": payload
    }

def load_model() -> SentenceTransformer:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    model.eval()
    return model

def connect_qdrant() -> QdrantClient:
    return QdrantClient(path=QDRANT_STORAGE_PATH)

def retrieve(query: str, model: SentenceTransformer, client: QdrantClient, top_k: int = QDRANT_TOP_K, final_top_k: int = FINAL_TOP_K) -> List[RetrievalResult]:
    if not query or not query.strip(): return []

    with torch.inference_mode():
        query_vector = model.encode(query, batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).tolist()

    try:
        points = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k, with_payload=True).points
    except Exception:
        points = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k, with_payload=True)

    if not points: return []

    q_tokens = _meaningful_tokens(query)
    q_norm = normalize_without_accents(query)

    candidates = []
    for point in points:
        data = extract_payload_data(point)
        vec_score = float(getattr(point, "score", 0.0))

        # Dynamic Procedure Lexical Scoring
        proc_score = 0.0
        proc_name = data["procedure"]

        if proc_name and q_tokens:
            p_tokens = _meaningful_tokens(proc_name)
            overlap = p_tokens & q_tokens
            if overlap:
                proc_score = len(overlap) / len(p_tokens)

            # Massive boost if exact substring match
            if q_norm in normalize_without_accents(proc_name):
                proc_score = 1.0

        # Blend scores
        final_score = (0.7 * vec_score) + (0.3 * proc_score)

        if proc_score >= 0.5:
            final_score += 0.1

        candidates.append({
            "data": data, "vector_score": vec_score, "procedure_score": proc_score, "final_score": min(final_score, 1.0)
        })

    candidates = [c for c in candidates if c["vector_score"] >= MIN_VECTOR_SCORE]
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Fetch parent document pieces
    document_ids = {str(c["data"]["document_id"]).strip() for c in candidates if str(c["data"]["document_id"]).strip()}

    all_points = []
    if document_ids:
        offset = None
        while True:
            pts, next_offset = client.scroll(collection_name=COLLECTION_NAME, limit=150, offset=offset, with_payload=True, with_vectors=False)
            all_points.extend(pts)
            if next_offset is None: break
            offset = next_offset

    grouped = {}
    for pt in all_points:
        d = extract_payload_data(pt)
        doc_id = str(d["document_id"]).strip()
        if doc_id in document_ids:
            grouped.setdefault(doc_id, []).append(d)

    final_results = []
    seen_docs = set()
    rank = 1

    for c in candidates:
        doc_id = str(c["data"]["document_id"]).strip()

        if doc_id in seen_docs:
            continue

        chunks = grouped.get(doc_id, [])
        if not chunks:
            continue

        # Build full document text, ignoring generic summaries
        context_parts = []
        seen_texts = set()
        for chunk in chunks:
            text = chunk["text"].strip()
            if not text or normalize_without_accents(chunk["chunk_type"]) == "general_information":
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)

            # Pure raw text, no [tags]
            context_parts.append(text)

        full_text = "\n\n".join(context_parts)

        final_results.append(RetrievalResult(
            rank=rank,
            final_score=c["final_score"],
            vector_score=c["vector_score"],
            procedure_score=c["procedure_score"],
            document_id=doc_id,
            procedure=c["data"]["procedure"],
            text=full_text,
            payload={"parent_context": full_text}
        ))

        seen_docs.add(doc_id)
        rank += 1

        if len(final_results) >= final_top_k:
            break

    return final_results
