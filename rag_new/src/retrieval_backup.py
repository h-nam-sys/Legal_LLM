# -*- coding: utf-8 -*-

"""
VIETNAMESE ADMINISTRATIVE PROCEDURES
LEGAL RAG RETRIEVER

Pipeline:

User Query
    |
    v
BKAI Vietnamese Bi-Encoder
    |
    v
Qdrant Vector Search
    |
    v
Candidate Retrieval
    |
    v
Procedure / Intent Detection
    |
    v
Candidate Scoring
    |
    v
Reranking
    |
    v
Final Results

Requirements:
    sentence-transformers
    qdrant-client
    torch

Expected Qdrant:
    URL: http://localhost:6333
    Collection:
        vietnamese_administrative_procedures

Expected vector:
    dimension = 768
    distance = COSINE
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

QDRANT_STORAGE_PATH = r"D:/legal-rag/qdrant_storage"
COLLECTION_NAME = "vietnamese_administrative_procedures"


# Number of vectors requested from Qdrant
QDRANT_TOP_K = 20

# Number of final documents returned
FINAL_TOP_K = 3

# Minimum vector similarity
MIN_VECTOR_SCORE = 0.20

# Minimum final score for weak/general queries.
# We don't use this as a hard filter for normal queries because
# legal retrieval should prefer returning the best available evidence.
MIN_FINAL_SCORE = 0.25

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For RTX 4060 8GB, this is safe for query embedding.
EMBED_BATCH_SIZE = 32


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RetrievalResult:
    rank: int
    final_score: float
    vector_score: float
    procedure_score: float
    chunk_score: float
    keyword_score: float
    text_intent_score: float

    chunk_id: str
    document_id: str
    procedure: str
    field: str
    submission: str
    chunk_type: str
    text: str

    payload: Dict[str, Any]


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_separator(char: str = "=", width: int = 80) -> None:
    print(char * width)


def print_header(title: str) -> None:
    print()
    print_separator("=")
    print(title)
    print_separator("=")


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

VIETNAMESE_MAP = {
    "à": "a",
    "á": "a",
    "ạ": "a",
    "ả": "a",
    "ã": "a",
    "â": "a",
    "ầ": "a",
    "ấ": "a",
    "ậ": "a",
    "ẩ": "a",
    "ẫ": "a",
    "ă": "a",
    "ằ": "a",
    "ắ": "a",
    "ặ": "a",
    "ẳ": "a",
    "ẵ": "a",

    "è": "e",
    "é": "e",
    "ẹ": "e",
    "ẻ": "e",
    "ẽ": "e",
    "ê": "e",
    "ề": "e",
    "ế": "e",
    "ệ": "e",
    "ể": "e",
    "ễ": "e",

    "ì": "i",
    "í": "i",
    "ị": "i",
    "ỉ": "i",
    "ĩ": "i",

    "ò": "o",
    "ó": "o",
    "ọ": "o",
    "ỏ": "o",
    "õ": "o",
    "ô": "o",
    "ồ": "o",
    "ố": "o",
    "ộ": "o",
    "ổ": "o",
    "ỗ": "o",
    "ơ": "o",
    "ờ": "o",
    "ớ": "o",
    "ợ": "o",
    "ở": "o",
    "ỡ": "o",

    "ù": "u",
    "ú": "u",
    "ụ": "u",
    "ủ": "u",
    "ũ": "u",
    "ư": "u",
    "ừ": "u",
    "ứ": "u",
    "ự": "u",
    "ử": "u",
    "ữ": "u",

    "ỳ": "y",
    "ý": "y",
    "ỵ": "y",
    "ỷ": "y",
    "ỹ": "y",

    "đ": "d",
}


def normalize_text(text: Any) -> str:
    """
    Normalize text for matching.

    Keeps Vietnamese Unicode in the original text,
    but creates a lowercase normalized representation.
    """

    if text is None:
        return ""

    text = str(text)

    # lowercase
    text = text.lower()

    # Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_without_accents(text: Any) -> str:
    """
    Convert Vietnamese text to accent-free form.

    Example:
        "Đăng ký khai sinh"
        ->
        "dang ky khai sinh"
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Special Vietnamese đ
    text = text.replace("đ", "d")

    normalized = unicodedata.normalize("NFD", text)

    normalized = "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )

    normalized = unicodedata.normalize("NFC", normalized)

    normalized = re.sub(r"[^\w\s]", " ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def tokenize(text: Any) -> List[str]:
    text = normalize_without_accents(text)

    if not text:
        return []

    return text.split()


# =============================================================================
# PROCEDURE DETECTION
# =============================================================================

PROCEDURES = [
    "Thủ tục đăng ký khai sinh",
    "Thủ tục đăng ký kết hôn",
    "Thủ tục đăng ký khai tử",
    "Thủ tục xác nhận tình trạng hôn nhân",
]


PROCEDURE_ALIASES = {
    "Thủ tục đăng ký khai sinh": [
        "dang ky khai sinh",
        "khai sinh",
        "lam khai sinh",
        "dang ky giay khai sinh",
        "giay khai sinh",
    ],

    "Thủ tục đăng ký kết hôn": [
        "dang ky ket hon",
        "ket hon",
        "lam dang ky ket hon",
        "dang ky hon nhan",
    ],

    "Thủ tục đăng ký khai tử": [
        "dang ky khai tu",
        "khai tu",
        "lam khai tu",
        "giay khai tu",
    ],

    "Thủ tục xác nhận tình trạng hôn nhân": [
        "xac nhan tinh trang hon nhan",
        "tinh trang hon nhan",
        "xac nhan hon nhan",
        "giay xac nhan tinh trang hon nhan",
    ],
}


def detect_procedure(query: str) -> Tuple[Optional[str], float]:
    """
    Detect administrative procedure from query.

    We deliberately use deterministic lexical matching instead of
    depending only on vector similarity.
    """

    q = normalize_without_accents(query)

    best_procedure = None
    best_score = 0.0

    for procedure, aliases in PROCEDURE_ALIASES.items():

        local_score = 0.0

        for alias in aliases:
            alias = normalize_without_accents(alias)

            # Exact phrase is strongest
            if alias in q:
                score = 1.0
            else:
                alias_tokens = set(alias.split())
                query_tokens = set(q.split())

                if not alias_tokens:
                    score = 0.0
                else:
                    overlap = len(alias_tokens & query_tokens)
                    score = overlap / len(alias_tokens)

            local_score = max(local_score, score)

        if local_score > best_score:
            best_score = local_score
            best_procedure = procedure

    return best_procedure, best_score


# =============================================================================
# INTENT DETECTION
# =============================================================================

INTENT_KEYWORDS = {
    "required_documents": [
        "giay to gi",
        "giay to",
        "ho so",
        "can nhung gi",
        "can gi",
        "chuan bi gi",
        "thanh phan ho so",
        "thanh phan",
        "can chuan bi",
        "nop nhung gi",
    ],

    "processing_time": [
        "mat bao lau",
        "bao lau",
        "thoi gian",
        "thoi han",
        "trong bao lau",
        "giai quyet bao lau",
        "khi nao co ket qua",
        "bao gio co ket qua",
    ],

    "fee": [
        "le phi",
        "phi",
        "mat phi",
        "co mat phi",
        "bao nhieu tien",
        "chi phi",
        "thu phi",
    ],

    "location": [
        "o dau",
        "nop o dau",
        "nop ho so o dau",
        "dia diem",
        "noi nop",
        "co quan nao",
        "trung tam nao",
        "tiep nhan o dau",
    ],

    "general_information": [
        "la gi",
        "nhu the nao",
        "the nao",
        "thong tin",
        "thu tuc",
        "dang ky",
        "quy trinh",
        "huong dan",
        "khong",
    ],
}


def detect_intent(query: str) -> Tuple[str, float]:
    """
    Detect query intent.

    Returns:
        intent
        confidence
    """

    q = normalize_without_accents(query)

    scores: Dict[str, float] = {}

    query_tokens = set(q.split())

    for intent, keywords in INTENT_KEYWORDS.items():

        best = 0.0

        for keyword in keywords:

            keyword = normalize_without_accents(keyword)

            if keyword in q:
                # Exact phrase
                best = max(best, 1.0)
                continue

            keyword_tokens = set(keyword.split())

            if not keyword_tokens:
                continue

            overlap = len(keyword_tokens & query_tokens)

            score = overlap / len(keyword_tokens)

            best = max(best, score)

        scores[intent] = best

    # Special rules for common Vietnamese questions
    if any(x in q for x in [
        "giay to",
        "ho so",
        "thanh phan",
        "chuan bi",
    ]):
        scores["required_documents"] = max(
            scores["required_documents"],
            1.0,
        )

    if any(x in q for x in [
        "bao lau",
        "thoi gian",
        "thoi han",
    ]):
        scores["processing_time"] = max(
            scores["processing_time"],
            1.0,
        )

    if any(x in q for x in [
        "le phi",
        "mat phi",
        "bao nhieu tien",
    ]):
        scores["fee"] = max(
            scores["fee"],
            1.0,
        )

    if any(x in q for x in [
        "o dau",
        "nop o dau",
        "dia diem",
        "noi nop",
    ]):
        scores["location"] = max(
            scores["location"],
            1.0,
        )

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # If no clear intent
    if best_score <= 0:
        return "general_information", 0.0

    return best_intent, min(best_score, 1.0)


# =============================================================================
# KEYWORD MATCHING
# =============================================================================

STOPWORDS = {
    "toi",
    "muon",
    "cho",
    "hoi",
    "la",
    "gi",
    "nhung",
    "mot",
    "cua",
    "co",
    "can",
    "the",
    "nao",
    "thi",
    "nho",
    "xin",
    "hay",
    "ve",
    "duoc",
    "khong",
    "bao",
    "nhieu",
}


def keyword_score(query: str, text: str) -> float:
    """
    Lexical overlap score between query and candidate text.
    """

    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))

    q_tokens -= STOPWORDS

    if not q_tokens:
        return 0.0

    overlap = q_tokens & t_tokens

    return len(overlap) / len(q_tokens)


# =============================================================================
# INTENT <-> CHUNK TYPE
# =============================================================================

INTENT_TO_CHUNK = {
    "required_documents": {
        "required_documents": 1.0,
        "general_information": 0.20,
    },

    "processing_time": {
        "processing_time": 1.0,
        "general_information": 0.20,
    },

    "fee": {
        "fee": 1.0,
        "general_information": 0.20,
    },

    "location": {
        "location": 1.0,
        "general_information": 0.20,
    },

    "general_information": {
        "general_information": 1.0,
        "required_documents": 0.15,
        "processing_time": 0.15,
        "fee": 0.15,
        "location": 0.15,
    },
}


def get_chunk_type_score(
    detected_intent: str,
    chunk_type: str,
) -> float:

    chunk_type = normalize_without_accents(chunk_type)

    mapping = INTENT_TO_CHUNK.get(
        detected_intent,
        {},
    )

    for key, score in mapping.items():
        if normalize_without_accents(key) == chunk_type:
            return score

    return 0.0


# =============================================================================
# QDRANT PAYLOAD HELPERS
# =============================================================================

def payload_get(
    payload: Dict[str, Any],
    keys: List[str],
    default: str = "",
) -> str:
    """
    Safely retrieve a value from Qdrant payload.

    Supports common key variants.
    """

    if not payload:
        return default

    for key in keys:
        if key in payload and payload[key] is not None:
            value = payload[key]

            if isinstance(value, (dict, list)):
                return str(value)

            return str(value)

    return default


def extract_payload_data(
    point: Any,
) -> Dict[str, Any]:

    payload = getattr(point, "payload", None)

    if payload is None and isinstance(point, dict):
        payload = point.get("payload", {})

    if not isinstance(payload, dict):
        payload = {}

    # Try several possible field names.
    chunk_id = payload_get(
        payload,
        [
            "chunk_id",
            "id",
            "chunkId",
        ],
        default="",
    )

    document_id = payload_get(
        payload,
        [
            "document_id",
            "documentId",
            "doc_id",
            "id",
        ],
        default="",
    )

    procedure = payload_get(
        payload,
        [
            "procedure",
            "procedure_name",
            "procedureName",
            "ten_thu_tuc",
            "Tên thủ tục hành chính",
        ],
        default="",
    )

    field = payload_get(
        payload,
        [
            "field",
            "linh_vuc",
            "category",
            "Lĩnh vực",
        ],
        default="",
    )

    submission = payload_get(
        payload,
        [
            "submission",
            "submission_type",
            "form",
            "Hình thức nộp",
        ],
        default="",
    )

    chunk_type = payload_get(
        payload,
        [
            "chunk_type",
            "type",
            "section_type",
            "loai_chunk",
        ],
        default="",
    )

    text = payload_get(
        payload,
        [
            "text",
            "content",
            "page_content",
            "chunk",
        ],
        default="",
    )

    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "procedure": procedure,
        "field": field,
        "submission": submission,
        "chunk_type": chunk_type,
        "text": text,
        "payload": payload,
    }


# =============================================================================
# MODEL
# =============================================================================

def check_gpu() -> None:
    print_header("PYTORCH / GPU CHECK")

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"CUDA version    : {torch.version.cuda}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

        total_memory = torch.cuda.get_device_properties(
            0
        ).total_memory / (1024 ** 3)

        print(f"GPU             : {gpu_name}")
        print(f"GPU memory      : {total_memory:.2f} GB")
    else:
        print("GPU             : NONE")

    print_separator("=")


def load_model() -> SentenceTransformer:

    print_header("LOADING BKAI EMBEDDING MODEL")

    print()
    print(f"Model: {MODEL_NAME}")
    print()

    if torch.cuda.is_available():
        print("[OK] CUDA is available")
        print("[INFO] Using GPU")

        device = "cuda:0"

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print("[WARNING] CUDA is not available")
        print("[INFO] Using CPU")

        device = "cpu"

    print()

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        trust_remote_code=True,
    )

    model.eval()

    print()
    print("[OK] BKAI model loaded")
    print(f"Device: {device}")

    try:
        dimension = model.get_embedding_dimension()
    except AttributeError:
        dimension = model.get_sentence_embedding_dimension()

    print(f"Embedding dimension: {dimension}")

    return model


# =============================================================================
# EMBEDDING
# =============================================================================

def encode_query(
    model: SentenceTransformer,
    query: str,
) -> List[float]:

    with torch.inference_mode():

        embedding = model.encode(
            query,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return embedding.tolist()


# =============================================================================
# QDRANT CONNECTION
# =============================================================================

def connect_qdrant() -> QdrantClient:
    print_header("CONNECTING TO QDRANT - LOCAL EMBEDDED MODE")
    print(f"\nQdrant storage: {QDRANT_STORAGE_PATH}")

    client = QdrantClient(path=QDRANT_STORAGE_PATH)

    # Check connection
    try:
        client.get_collections()
    except Exception as exc:
        print()
        print("[ERROR] Cannot connect to Qdrant")
        print(f"Reason: {exc}")

        raise

    print("[OK] Qdrant connected")

    # Check collection
    try:
        collection_info = client.get_collection(
            COLLECTION_NAME
        )

        vectors_count = getattr(
            collection_info,
            "vectors_count",
            None,
        )

        if vectors_count is None:
            vectors_count = getattr(
                collection_info,
                "points_count",
                None,
            )

        print(f"Collection: {COLLECTION_NAME}")
        print(f"Vectors: {vectors_count}")

    except Exception as exc:
        print()
        print(
            f"[ERROR] Collection does not exist or "
            f"cannot be accessed: {COLLECTION_NAME}"
        )
        print(f"Reason: {exc}")

        raise

    return client


# =============================================================================
# QDRANT SEARCH
# =============================================================================

def qdrant_search(
    client: QdrantClient,
    query_vector: List[float],
    limit: int = QDRANT_TOP_K,
) -> List[Any]:
    """
    Search Qdrant.

    We intentionally do NOT apply procedure/intent filters
    at the Qdrant level.

    Reason:
        The existing collection payload may use slightly different
        field names/types. Filtering in Python after vector search
        is safer and prevents the "0 candidates" problem.
    """

    # New Qdrant API
    try:

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        points = response.points

        return points

    except AttributeError:
        pass

    except TypeError:
        pass

    except Exception as exc:

        print()
        print(
            "[WARNING] query_points() failed."
        )
        print(f"Reason: {exc}")

    # Older Qdrant API
    try:

        points = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return points

    except Exception as exc:

        print()
        print("[ERROR] Qdrant vector search failed")
        print(f"Reason: {exc}")

        raise


# =============================================================================
# SCORE HELPERS
# =============================================================================

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def calculate_procedure_score(
    detected_procedure: Optional[str],
    candidate_procedure: str,
) -> float:

    if not detected_procedure:
        return 0.0

    if not candidate_procedure:
        return 0.0

    detected = normalize_without_accents(
        detected_procedure
    )

    candidate = normalize_without_accents(
        candidate_procedure
    )

    # Exact match
    if detected == candidate:
        return 1.0

    # One contained in the other
    if detected in candidate or candidate in detected:
        return 0.90

    detected_tokens = set(detected.split())
    candidate_tokens = set(candidate.split())

    if not detected_tokens:
        return 0.0

    overlap = len(
        detected_tokens & candidate_tokens
    )

    return clamp(
        overlap / len(detected_tokens)
    )


def calculate_text_intent_score(
    detected_intent: str,
    chunk_type: str,
    text: str,
) -> float:

    # Strong signal from metadata
    metadata_score = get_chunk_type_score(
        detected_intent,
        chunk_type,
    )

    if metadata_score > 0:
        return metadata_score

    # Backup signal from text
    normalized_text = normalize_without_accents(text)

    keywords = INTENT_KEYWORDS.get(
        detected_intent,
        [],
    )

    best = 0.0

    for keyword in keywords:
        keyword = normalize_without_accents(keyword)

        if keyword in normalized_text:
            best = max(best, 1.0)

    return best


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def select_candidates(
    points: List[Any],
    query: str,
    detected_intent: str,
    detected_procedure: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Convert Qdrant points into candidates.

    IMPORTANT:
        We do not discard candidates simply because the metadata
        does not exactly match.

    This solves the previous:
        Procedure candidates: 0
        Intent candidates: 0
    problem.
    """

    print_header("CANDIDATE SELECTION")

    print(
        f"Detected intent    : {detected_intent}"
    )

    print(
        f"Detected procedure : "
        f"{detected_procedure}"
    )

    candidates: List[Dict[str, Any]] = []

    procedure_candidates = 0
    intent_candidates = 0

    for point in points:

        data = extract_payload_data(point)

        procedure_score = calculate_procedure_score(
            detected_procedure,
            data["procedure"],
        )

        intent_score = calculate_text_intent_score(
            detected_intent,
            data["chunk_type"],
            data["text"],
        )

        vector_score = float(
            getattr(point, "score", 0.0)
        )

        # Keyword matching
        combined_text = " ".join(
            [
                data["procedure"],
                data["chunk_type"],
                data["text"],
            ]
        )

        kw_score = keyword_score(
            query,
            combined_text,
        )

        if procedure_score >= 0.80:
            procedure_candidates += 1

        if intent_score >= 0.80:
            intent_candidates += 1

        candidates.append(
            {
                "point": point,
                "data": data,
                "vector_score": vector_score,
                "procedure_score": procedure_score,
                "intent_score": intent_score,
                "keyword_score": kw_score,
            }
        )

    print(
        f"Procedure candidates: "
        f"{procedure_candidates}"
    )

    print(
        f"Intent candidates    : "
        f"{intent_candidates}"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT
    # -------------------------------------------------------------------------
    #
    # Previous implementation apparently removed everything when the
    # candidate counts were 0.
    #
    # We DON'T do that anymore.
    #
    # Qdrant already returned relevant vector candidates, so we keep them
    # and let reranking decide.
    # -------------------------------------------------------------------------

    selected = candidates

    # Sort temporarily by a combination of strong metadata signals.
    selected.sort(
        key=lambda x: (
            x["procedure_score"],
            x["intent_score"],
            x["vector_score"],
            x["keyword_score"],
        ),
        reverse=True,
    )

    print(
        f"Selected candidates: "
        f"{len(selected)}"
    )

    return selected


# =============================================================================
# RERANKING
# =============================================================================

def rerank_candidates(
    candidates: List[Dict[str, Any]],
    query: str,
    detected_intent: str,
    detected_procedure: Optional[str],
) -> List[RetrievalResult]:

    print_header("RERANKING INFORMATION")

    print(
        f"Detected intent     : "
        f"{detected_intent}"
    )

    print(
        f"Detected procedure  : "
        f"{detected_procedure}"
    )

    # Estimate confidence
    procedure_confidence = 0.0
    if detected_procedure:
        procedure_confidence = 0.90

    intent_confidence = 0.0

    if detected_intent != "general_information":
        intent_confidence = 0.90
    else:
        intent_confidence = 0.70

    print(
        f"Intent confidence   : "
        f"{intent_confidence:.4f}"
    )

    print(
        f"Procedure confidence: "
        f"{procedure_confidence:.4f}"
    )

    reranked: List[RetrievalResult] = []

    for candidate in candidates:

        data = candidate["data"]

        vector_score = clamp(
            candidate["vector_score"]
        )

        procedure_score = clamp(
            candidate["procedure_score"]
        )

        chunk_score = get_chunk_type_score(
            detected_intent,
            data["chunk_type"],
        )

        # If metadata doesn't give a chunk score,
        # use text intent score.
        if chunk_score <= 0:
            chunk_score = clamp(
                candidate["intent_score"]
            )

        keyword = clamp(
            candidate["keyword_score"]
        )

        text_intent = clamp(
            candidate["intent_score"]
        )

        # ---------------------------------------------------------------------
        # WEIGHTING
        # ---------------------------------------------------------------------
        #
        # Procedure relevance is critical in legal RAG.
        # Intent/chunk type prevents retrieving e.g. "fee" when user asks
        # for required documents.
        #
        # Vector score remains important because it represents semantic
        # similarity from BKAI.
        # ---------------------------------------------------------------------

        if detected_procedure:

            final_score = (
                0.35 * vector_score
                + 0.25 * procedure_score
                + 0.15 * chunk_score
                + 0.15 * text_intent
                + 0.10 * keyword
            )

        else:

            final_score = (
                0.45 * vector_score
                + 0.20 * chunk_score
                + 0.20 * text_intent
                + 0.15 * keyword
            )

        # ---------------------------------------------------------------------
        # Small bonus for exact procedure match
        # ---------------------------------------------------------------------

        if procedure_score >= 1.0:
            final_score += 0.05

        # Bonus for exact intent chunk
        if chunk_score >= 1.0:
            final_score += 0.05

        final_score = clamp(
            final_score
        )

        result = RetrievalResult(
            rank=0,
            final_score=final_score,
            vector_score=vector_score,
            procedure_score=procedure_score,
            chunk_score=chunk_score,
            keyword_score=keyword,
            text_intent_score=text_intent,
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            procedure=data["procedure"],
            field=data["field"],
            submission=data["submission"],
            chunk_type=data["chunk_type"],
            text=data["text"],
            payload=data["payload"],
        )

        reranked.append(result)

    # Sort by final score
    reranked.sort(
        key=lambda x: x.final_score,
        reverse=True,
    )

    # Assign rank
    for index, result in enumerate(
        reranked,
        start=1,
    ):
        result.rank = index

    return reranked


# =============================================================================
# RESULT FILTERING
# =============================================================================

def filter_final_results(
    results: List[RetrievalResult],
    detected_procedure: Optional[str],
    detected_intent: str,
) -> List[RetrievalResult]:

    if not results:
        return []

    # -------------------------------------------------------------------------
    # For a clearly identified procedure + intent:
    #
    # We prefer candidates from the same procedure.
    #
    # But we NEVER return zero just because metadata is imperfect.
    # -------------------------------------------------------------------------

    if detected_procedure:

        same_procedure = [
            r
            for r in results
            if calculate_procedure_score(
                detected_procedure,
                r.procedure,
            ) >= 0.80
        ]

        if same_procedure:
            results = same_procedure

    # Remove extremely weak vector results only when there are enough
    # alternatives.
    strong = [
        r
        for r in results
        if r.vector_score >= MIN_VECTOR_SCORE
    ]

    if strong:
        results = strong

    # Final top-k
    results = results[:FINAL_TOP_K]

    return results


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(
    query: str,
    results: List[RetrievalResult],
) -> None:

    print_header("FINAL RETRIEVAL RESULTS")

    print()
    print("Query:")
    print(query)

    print()

    if not results:
        print("No retrieval results.")
        return

    print(
        f"Top {len(results)} results:"
    )

    for result in results:

        print()
        print_separator("-")

        print(
            f"Rank             : "
            f"{result.rank}"
        )

        print(
            f"Final score      : "
            f"{result.final_score:.4f}"
        )

        print(
            f"Vector score     : "
            f"{result.vector_score:.4f}"
        )

        print(
            f"Procedure score  : "
            f"{result.procedure_score:.4f}"
        )

        print(
            f"Chunk score      : "
            f"{result.chunk_score:.4f}"
        )

        print(
            f"Keyword score    : "
            f"{result.keyword_score:.4f}"
        )

        print(
            f"Text intent      : "
            f"{result.text_intent_score:.4f}"
        )

        print(
            f"Chunk ID         : "
            f"{result.chunk_id}"
        )

        print(
            f"Document ID      : "
            f"{result.document_id}"
        )

        print(
            f"Procedure        : "
            f"{result.procedure}"
        )

        print(
            f"Field            : "
            f"{result.field}"
        )

        print(
            f"Submission       : "
            f"{result.submission}"
        )

        print(
            f"Chunk type       : "
            f"{result.chunk_type}"
        )

        print()
        print("TEXT:")
        print(result.text)


# =============================================================================
# RETRIEVAL PIPELINE
# =============================================================================

def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    top_k: int = QDRANT_TOP_K,
    final_top_k: int = FINAL_TOP_K,
) -> List[RetrievalResult]:

    if not query or not query.strip():
        return []

    query = query.strip()

    # -------------------------------------------------------------------------
    # STEP 1: DETECT INTENT
    # -------------------------------------------------------------------------

    detected_intent, intent_confidence = detect_intent(
        query
    )

    # -------------------------------------------------------------------------
    # STEP 2: DETECT PROCEDURE
    # -------------------------------------------------------------------------

    detected_procedure, procedure_confidence = detect_procedure(
        query
    )

    # -------------------------------------------------------------------------
    # STEP 3: EMBEDDING
    # -------------------------------------------------------------------------

    query_vector = encode_query(
        model,
        query,
    )

    # -------------------------------------------------------------------------
    # STEP 4: QDRANT
    # -------------------------------------------------------------------------

    points = qdrant_search(
        client,
        query_vector,
        limit=top_k,
    )

    print()
    print(
        f"Qdrant candidates: "
        f"{len(points)}"
    )

    if not points:
        return []

    # -------------------------------------------------------------------------
    # STEP 5: CANDIDATE SELECTION
    # -------------------------------------------------------------------------

    candidates = select_candidates(
        points,
        query,
        detected_intent,
        detected_procedure,
    )

    # -------------------------------------------------------------------------
    # STEP 6: RERANK
    # -------------------------------------------------------------------------

    reranked = rerank_candidates(
        candidates,
        query,
        detected_intent,
        detected_procedure,
    )

    # -------------------------------------------------------------------------
    # STEP 7: FINAL FILTER
    # -------------------------------------------------------------------------

    final_results = filter_final_results(
        reranked,
        detected_procedure,
        detected_intent,
    )

    # Ensure requested top_k
    final_results = final_results[:final_top_k]

    return final_results


# =============================================================================
# SINGLE QUERY TEST
# =============================================================================

def run_query(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
) -> List[RetrievalResult]:

    results = retrieve(
        query=query,
        model=model,
        client=client,
    )

    print_results(
        query,
        results,
    )

    return results


# =============================================================================
# TEST SUITE
# =============================================================================

TEST_QUERIES = [
    "Đăng ký khai sinh cần những giấy tờ gì?",

    "Đk kết hôn cần chuẩn bị hồ sơ gì?",


    "Đăng ký kết hôn không?",
]


def run_test_suite(
    model: SentenceTransformer,
    client: QdrantClient,
) -> None:

    print_header(
        "RUNNING RETRIEVAL TEST SUITE"
    )

    total = len(TEST_QUERIES)

    success = 0

    start_time = time.time()

    for index, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        print()
        print_separator("=")

        print(
            f"[TEST {index}/{total}]"
        )

        print_separator("=")

        results = retrieve(
            query=query,
            model=model,
            client=client,
        )

        print_results(
            query,
            results,
        )

        if results:
            success += 1

    elapsed = time.time() - start_time

    print()
    print_separator("=")
    print("TEST SUMMARY")
    print_separator("=")

    print(
        f"Tests       : {total}"
    )

    print(
        f"Successful  : {success}"
    )

    print(
        f"Failed      : {total - success}"
    )

    print(
        f"Success rate: "
        f"{success / total * 100:.2f}%"
    )

    print(
        f"Elapsed time: "
        f"{elapsed:.2f}s"
    )


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode(
    model: SentenceTransformer,
    client: QdrantClient,
) -> None:

    print_header(
        "INTERACTIVE RETRIEVAL MODE"
    )

    print()
    print(
        "Nhập câu hỏi pháp luật hành chính."
    )

    print(
        "Gõ 'exit' hoặc 'quit' để thoát."
    )

    print()

    while True:

        try:
            query = input(
                "Question > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print()
            print("Exiting...")
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Exiting...")
            break

        try:
            run_query(
                query,
                model,
                client,
            )

        except Exception as exc:

            print()
            print_separator("-")

            print(
                "[ERROR] Retrieval failed"
            )

            print(
                f"Reason: {exc}"
            )

            print_separator("-")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print_separator("=")
    print(
        "VIETNAMESE ADMINISTRATIVE PROCEDURES"
    )
    print(
        "LEGAL RAG RETRIEVER"
    )
    print_separator("=")

    # -------------------------------------------------------------------------
    # GPU
    # -------------------------------------------------------------------------

    check_gpu()

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = load_model()

    # -------------------------------------------------------------------------
    # Connect Qdrant
    # -------------------------------------------------------------------------

    client = connect_qdrant()

    try:

        # ---------------------------------------------------------------------
        # TEST SUITE
        # ---------------------------------------------------------------------

        run_test_suite(
            model=model,
            client=client,
        )

        # ---------------------------------------------------------------------
        # Optional interactive mode
        # ---------------------------------------------------------------------

        print()
        print_separator("=")
        print(
            "RETRIEVAL TEST COMPLETED"
        )
        print_separator("=")

        print()
        print(
            "Interactive mode? "
            "Type 'yes' to continue."
        )

        try:
            answer = input(
                "> "
            ).strip().lower()

        except (
            KeyboardInterrupt,
            EOFError,
        ):
            answer = "no"

        if answer in {
            "yes",
            "y",
        }:

            interactive_mode(
                model=model,
                client=client,
            )

    finally:

        try:
            client.close()
        except Exception:
            pass

        print()
        print(
            "[OK] Qdrant client closed"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print()
        print(
            "[INFO] Program interrupted by user."
        )

        sys.exit(0)

    except Exception as exc:

        print()
        print_separator("=")
        print("[FATAL ERROR]")
        print_separator("=")
        print(exc)
        print_separator("=")

        sys.exit(1)