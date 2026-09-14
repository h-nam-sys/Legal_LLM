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

Expected:

    Qdrant URL: http://localhost:6333

    Collection: vietnamese_administrative_procedures

    Vector dimension: 768

    Distance: COSINE

"""

from __future__ import annotations

import re

import sys

import time

import unicodedata

from typing import Any, Dict, List, Optional, Tuple

import torch

from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer

# =============================================================================

# CONFIGURATION

# =============================================================================

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "vietnamese_administrative_procedures"

# Retrieve more candidates than the final number so reranking has room to work.

QDRANT_TOP_K = 30

FINAL_TOP_K = 5

# Only used for reporting / optional fallback decisions.

MIN_VECTOR_SCORE = 0.20

# Device

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def normalize_text(text: Any) -> str:

    """

    Lowercase + remove Vietnamese accents + normalize punctuation.

    Used only for lexical matching.

    Original Vietnamese text is NOT modified in Qdrant or output.

    """

    if text is None:

        return ""

    text = str(text).strip().lower()

    text = text.replace("đ", "d")

    text = unicodedata.normalize("NFD", text)

    text = "".join(

        char

        for char in text

        if unicodedata.category(char) != "Mn"

    )

    text = re.sub(r"[^a-z0-9*\s*]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize(text: Any) -> List[str]:

    normalized = normalize_text(text)

    return normalized.split() if normalized else []

# =============================================================================

# PROCEDURE / INTENT DETECTION

# =============================================================================

# These are only aliases for common user wording.

# The actual procedure names are loaded from Qdrant payloads.

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

INTENT_KEYWORDS = {

    "location": [

        "nop ho so o dau",

        "nop o dau",

        "o dau",

        "dia diem",

        "dia chi",

        "noi nop",

        "noi tiep nhan",

        "co quan nao",

        "co quan tiep nhan",

        "tiep nhan ho so",

    ],

    "processing_time": [

        "mat bao lau",

        "bao lau",

        "thoi gian",

        "thoi han",

        "bao nhieu ngay",

        "trong bao nhieu ngay",

        "giai quyet bao lau",

        "khi nao co ket qua",

        "bao gio co ket qua",

    ],

    "fee": [

        "le phi",

        "phi",

        "mat phi",

        "co mat phi",

        "co mat tien",

        "bao nhieu tien",

        "chi phi",

        "thu phi",

        "khong thu phi",

    ],

    "required_documents": [

        "giay to gi",

        "giay to",

        "ho so",

        "ho so gom",

        "ho so can",

        "thanh phan ho so",

        "thanh phan",

        "can nhung gi",

        "can gi",

        "chuan bi gi",

        "can chuan bi",

        "nop nhung gi",

    ],

    "procedure": [

        "trinh tu",

        "cac buoc",

        "thuc hien nhu the nao",

        "quy trinh",

        "cach thuc thuc hien",

        "cach thuc",

    ],

}

def detect_intent(query: str) -> Tuple[str, float]:

    """

    Detect the most likely user intent.

    Priority is intentional:

        location > processing_time > fee > required_documents > procedure

    This prevents a query such as

    "đăng ký khai sinh nộp hồ sơ ở đâu?"

    from being classified as required_documents just because

    it contains "hồ sơ".

    """

    q = normalize_text(query)

    if not q:

        return "general", 0.0

    # Strong phrase checks first.

    priority = [

        "location",

        "processing_time",

        "fee",

        "required_documents",

        "procedure",

    ]

    scores: Dict[str, float] = {}

    for intent in priority:

        best = 0.0

        for keyword in INTENT_KEYWORDS[intent]:

            kw = normalize_text(keyword)

            if kw in q:

                best = max(best, 1.0)

                continue

            kw_tokens = set(kw.split())

            q_tokens = set(q.split())

            if kw_tokens:

                overlap = len(kw_tokens & q_tokens)

                best = max(best, overlap / len(kw_tokens))

        scores[intent] = best

    best_intent = max(

        priority,

        key=lambda intent: (

            scores[intent],

            -priority.index(intent),

        ),

    )

    best_score = scores[best_intent]

    if best_score <= 0:

        return "general", 0.0

    return best_intent, min(best_score, 1.0)

def _alias_procedure(query: str) -> Optional[str]:

    """

    Fast alias-based detection for the four common civil-status procedures.

    """

    q = normalize_text(query)

    best_name = None

    best_len = 0

    for procedure, aliases in PROCEDURE_ALIASES.items():

        for alias in aliases:

            alias_norm = normalize_text(alias)

            if alias_norm and alias_norm in q:

                if len(alias_norm) > best_len:

                    best_name = procedure

                    best_len = len(alias_norm)

    return best_name

def _procedure_token_score(query: str, procedure_name: str) -> float:

    q_tokens = set(tokenize(query))

    p_tokens = set(tokenize(procedure_name))

    # Words too generic to identify a procedure.

    stopwords = {

        "thu", "tuc", "can", "gi", "nhung", "nhu", "the",

        "nao", "toi", "muon", "cho", "hoi", "ho", "so",

        "giay", "to", "lam", "thuc", "hien", "dang",

        "ky", "cap", "xin", "co", "yeu", "cau",

    }

    q_tokens -= stopwords

    p_tokens -= stopwords

    if not p_tokens:

        return 0.0

    overlap = q_tokens & p_tokens

    coverage = len(overlap) / len(p_tokens)

    union = q_tokens | p_tokens

    jaccard = len(overlap) / len(union) if union else 0.0

    return 0.7 * coverage + 0.3 * jaccard

def detect_procedure(

    query: str,

    procedure_names: Optional[List[str]] = None,

) -> Tuple[Optional[str], float]:

    """

    Detect procedure from:

        1. Known aliases

        2. Exact/substring match against actual Qdrant procedure names

        3. Token similarity against actual procedure names

    The returned procedure name is the REAL payload value whenever

    procedure_names are available.

    """

    alias_match = _alias_procedure(query)

    if procedure_names:

        q = normalize_text(query)

        # First prefer an exact procedure phrase in the actual dataset.

        normalized = [

            (name, normalize_text(name))

            for name in procedure_names

            if normalize_text(name)

        ]

        normalized.sort(key=lambda x: len(x[1]), reverse=True)

        for original, name_norm in normalized:

            if name_norm in q:

                return original, 1.0

        # If an alias was detected, find the actual dataset procedure

        # that corresponds to it.

        if alias_match:

            alias_norm = normalize_text(alias_match)

            best_actual = None

            best_score = 0.0

            for original, name_norm in normalized:

                score = _procedure_token_score(

                    alias_norm,

                    name_norm,

                )

                if score > best_score:

                    best_score = score

                    best_actual = original

            if best_actual and best_score >= 0.40:

                return best_actual, min(0.95, best_score + 0.40)

        # General token matching against actual dataset procedures.

        scored = []

        for original in procedure_names:

            score = _procedure_token_score(

                query,

                original,

            )

            if score > 0:

                scored.append((score, original))

        if scored:

            scored.sort(

                key=lambda item: (

                    item[0],

                    len(normalize_text(item[1])),

                ),

                reverse=True,

            )

            best_score, best_name = scored[0]

            if best_score >= 0.45:

                return best_name, best_score

    if alias_match:

        return alias_match, 0.90

    return None, 0.0

# =============================================================================

# PAYLOAD HELPERS

# =============================================================================

def payload_get(

    payload: Dict[str, Any],

    keys: List[str],

    default: str = "",

) -> str:

    if not isinstance(payload, dict):

        return default

    for key in keys:

        value = payload.get(key)

        if value is not None:

            if isinstance(value, (dict, list)):

                return str(value)

            return str(value)

    return default

def extract_payload(point: Any) -> Dict[str, Any]:

    payload = getattr(point, "payload", None)

    if payload is None and isinstance(point, dict):

        payload = point.get("payload", {})

    if not isinstance(payload, dict):

        payload = {}

    return {

        "chunk_id": payload_get(

            payload,

            ["chunk_id", "chunkId", "id"],

        ),

        "document_id": payload_get(

            payload,

            ["document_id", "documentId", "doc_id"],

        ),

        "procedure": payload_get(

            payload,

            [

                "procedure_name",

                "procedure",

                "procedureName",

                "ten_thu_tuc",

                "Tên thủ tục hành chính",

            ],

        ),

        "field": payload_get(

            payload,

            [

                "field",

                "linh_vuc",

                "category",

                "Lĩnh vực",

            ],

        ),

        "submission": payload_get(

            payload,

            [

                "submission_method",

                "submission",

                "submission_type",

                "form",

                "Hình thức nộp",

            ],

        ),

        "chunk_type": payload_get(

            payload,

            [

                "chunk_type",

                "type",

                "section_type",

                "loai_chunk",

            ],

        ),

        "text": payload_get(

            payload,

            [

                "text",

                "content",

                "page_content",

                "chunk",

            ],

        ),

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

        print(f"GPU             : {torch.cuda.get_device_name(0)}")

        total_memory = (

            torch.cuda.get_device_properties(0).total_memory

            / (1024 ** 3)

        )

        print(f"GPU memory      : {total_memory:.2f} GB")

    else:

        print("GPU             : NONE")

    print_separator("=")

def load_model() -> SentenceTransformer:

    print_header("LOADING BKAI EMBEDDING MODEL")

    print(f"\nModel: {MODEL_NAME}")

    print(f"Device: {DEVICE}")

    model = SentenceTransformer(

        MODEL_NAME,

        device=DEVICE,

        trust_remote_code=True,

    )

    model.eval()

    dimension = model.get_embedding_dimension()

    print("\n[OK] BKAI model loaded")

    print(f"Embedding dimension: {dimension}")

    if dimension != 768:

        print(

            "[WARNING] Model dimension is not 768. "

            "Your Qdrant collection must use the same dimension."

        )

    return model

# =============================================================================

# EMBEDDING

# =============================================================================

def encode_query(

    model: SentenceTransformer,

    query: str,

) -> List[float]:

    query = str(query).strip()

    if not query:

        raise ValueError("Query cannot be empty.")

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

# QDRANT

# =============================================================================

def connect_qdrant() -> QdrantClient:

    print_header("CONNECTING TO QDRANT SERVER")

    print(f"\nQdrant URL: {QDRANT_URL}")

    client = QdrantClient(url=QDRANT_URL)

    try:

        client.get_collections()

    except Exception as exc:

        print("\n[ERROR] Cannot connect to Qdrant")

        print(f"Reason: {exc}")

        raise

    print("[OK] Qdrant connected")

    try:

        info = client.get_collection(

            collection_name=COLLECTION_NAME,

        )

    except Exception as exc:

        print(

            f"\n[ERROR] Collection not found: "

            f"{COLLECTION_NAME}"

        )

        print(f"Reason: {exc}")

        raise

    print(f"Collection: {COLLECTION_NAME}")

    print(f"Points: {info.points_count}")

    # Check vector size when available.

    try:

        vectors = info.config.params.vectors

        if hasattr(vectors, "size"):

            vector_size = vectors.size

            print(f"Vector dimension: {vector_size}")

            if vector_size != 768:

                raise ValueError(

                    f"Qdrant vector dimension is {vector_size}, "

                    f"but BKAI model outputs 768."

                )

    except AttributeError:

        # Some qdrant-client versions expose this differently.

        pass

    return client

def search_qdrant(

    client: QdrantClient,

    query_vector: List[float],

    top_k: int = QDRANT_TOP_K,

) -> List[Any]:

    """

    Works with modern qdrant-client and falls back to the old search API.

    """

    try:

        response = client.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            limit=top_k,

            with_payload=True,

            with_vectors=False,

        )

        return response.points

    except (AttributeError, TypeError):

        pass

    # Older qdrant-client

    return client.search(

        collection_name=COLLECTION_NAME,

        query_vector=query_vector,

        limit=top_k,

        with_payload=True,

        with_vectors=False,

    )

def load_procedure_names(

    client: QdrantClient,

    limit: int = 1000,

) -> List[str]:

    """

    Read actual procedure_name values from Qdrant.

    This removes the old hard-coded assumption that the collection

    contains only four procedures.

    """

    names = set()

    offset = None

    while True:

        points, next_offset = client.scroll(

            collection_name=COLLECTION_NAME,

            limit=limit,

            offset=offset,

            with_payload=True,

            with_vectors=False,

        )

        for point in points:

            payload = point.payload or {}

            name = payload_get(

                payload,

                [

                    "procedure_name",

                    "procedure",

                    "procedureName",

                    "ten_thu_tuc",

                    "Tên thủ tục hành chính",

                ],

            ).strip()

            if name:

                names.add(name)

        if next_offset is None:

            break

        offset = next_offset

    result = sorted(

        names,

        key=lambda x: len(normalize_text(x)),

        reverse=True,

    )

    print(f"Procedure index: {len(result)} unique names")

    return result

# =============================================================================

# SCORING

# =============================================================================

def clamp(

    value: float,

    low: float = 0.0,

    high: float = 1.0,

) -> float:

    return max(low, min(high, float(value)))

def calculate_procedure_score(

    target_procedure: Optional[str],

    candidate_procedure: str,

) -> float:

    if not target_procedure or not candidate_procedure:

        return 0.0

    target = normalize_text(target_procedure)

    current = normalize_text(candidate_procedure)

    if not target or not current:

        return 0.0

    if target == current:

        return 1.0

    if target in current:

        return 0.90

    if current in target:

        return 0.85

    target_tokens = set(target.split())

    current_tokens = set(current.split())

    overlap = target_tokens & current_tokens

    if not target_tokens:

        return 0.0

    return clamp(

        len(overlap) / len(target_tokens)

    )

def calculate_chunk_score(

    intent: str,

    chunk_type: str,

) -> float:

    """

    Score how well the candidate chunk type matches the query intent.

    Do NOT hard-filter here. Scoring is safer because real-world payloads

    may have missing or slightly different metadata.

    """

    intent = normalize_text(intent)

    chunk_type = normalize_text(chunk_type)

    mapping = {

        "required_documents": {

            "required_documents": 1.00,

            "general_information": 0.20,

        },

        "processing_time": {

            "processing_time": 1.00,

            "general_information": 0.20,

        },

        "fee": {

            "fee": 1.00,

            "general_information": 0.20,

        },

        "location": {

            "location": 1.00,

            "general_information": 0.20,

        },

        "procedure": {

            "procedure": 1.00,

            "general_information": 0.20,

        },

        "general": {

            "general_information": 1.00,

        },

    }

    for key, score in mapping.get(intent, {}).items():

        if normalize_text(key) == chunk_type:

            return score

    return 0.0

INTENT_TEXT_CUES = {

    "location": [

        ("địa điểm tiếp nhận hồ sơ", 1.00),

        ("địa điểm tiếp nhận", 0.95),

        ("nơi tiếp nhận hồ sơ", 0.90),

        ("cơ quan tiếp nhận", 0.90),

        ("tiếp nhận hồ sơ", 0.85),

    ],

    "processing_time": [

        ("thời gian giải quyết", 1.00),

        ("thời hạn giải quyết", 0.95),

        ("ngày làm việc", 0.70),

    ],

    "fee": [

        ("lệ phí", 1.00),

        ("không thu phí", 0.90),

        ("thu phí", 0.80),

        ("đồng/bản", 0.75),

    ],

    "required_documents": [

        ("thành phần hồ sơ", 1.00),

        ("hồ sơ gồm", 0.90),

    ],

    "procedure": [

        ("trình tự thực hiện", 1.00),

        ("các bước thực hiện", 0.90),

        ("quy trình thực hiện", 0.90),

        ("cách thức thực hiện", 0.85),

    ],

}

def calculate_text_intent_score(

    text: str,

    intent: str,

) -> float:

    if not text:

        return 0.0

    cues = INTENT_TEXT_CUES.get(intent, [])

    if not cues:

        return 0.0

    normalized = normalize_text(text)

    original_lower = text.lower()

    best = 0.0

    for cue, weight in cues:

        # Check both original Vietnamese and normalized form.

        if cue in original_lower or normalize_text(cue) in normalized:

            best = max(best, weight)

    return best

STOPWORDS = {

    "toi", "muon", "cho", "hoi", "la", "gi", "nhung",

    "mot", "cua", "co", "can", "the", "nao", "thi",

    "nho", "xin", "hay", "ve", "duoc", "khong", "bao",

    "nhieu", "ho", "so", "giay", "to",

}

def calculate_keyword_score(

    query: str,

    text: str,

) -> float:

    q_tokens = set(tokenize(query))

    t_tokens = set(tokenize(text))

    q_tokens -= STOPWORDS

    if not q_tokens or not t_tokens:

        return 0.0

    overlap = q_tokens & t_tokens

    return len(overlap) / len(q_tokens)

# =============================================================================

# CANDIDATE PREPARATION

# =============================================================================

def prepare_candidates(

    points: List[Any],

    query: str,

    target_procedure: Optional[str],

    intent: str,

) -> List[Dict[str, Any]]:

    candidates = []

    for point in points:

        data = extract_payload(point)

        vector_score = clamp(

            float(getattr(point, "score", 0.0))

        )

        procedure_score = calculate_procedure_score(

            target_procedure,

            data["procedure"],

        )

        chunk_score = calculate_chunk_score(

            intent,

            data["chunk_type"],

        )

        keyword_score = calculate_keyword_score(

            query,

            " ".join(

                [

                    data["procedure"],

                    data["field"],

                    data["chunk_type"],

                    data["text"],

                ]

            ),

        )

        text_intent_score = calculate_text_intent_score(

            data["text"],

            intent,

        )

        candidates.append(

            {

                "point": point,

                "data": data,

                "vector_score": vector_score,

                "procedure_score": procedure_score,

                "chunk_score": chunk_score,

                "keyword_score": keyword_score,

                "text_intent_score": text_intent_score,

            }

        )

    return candidates

# =============================================================================

# RERANKING

# =============================================================================

def rerank_candidates(

    candidates: List[Dict[str, Any]],

    final_k: int = FINAL_TOP_K,

) -> List[Dict[str, Any]]:

    """

    Score ALL retrieved candidates first, then take final_k.

    This is important: do not select only 3-5 candidates before reranking,

    otherwise the reranker has no opportunity to recover a better result.

    """

    scored = []

    for candidate in candidates:

        vector = candidate["vector_score"]

        procedure = candidate["procedure_score"]

        chunk = candidate["chunk_score"]

        keyword = candidate["keyword_score"]

        text_intent = candidate["text_intent_score"]

        final_score = (

            0.50 * vector

            + 0.25 * procedure

            + 0.15 * chunk

            + 0.05 * keyword

            + 0.05 * text_intent

        )

        # Small bonus for exact procedure / exact intent chunk.

        if procedure >= 1.0:

            final_score += 0.05

        if chunk >= 1.0:

            final_score += 0.05

        candidate = dict(candidate)

        candidate["final_score"] = clamp(final_score)

        scored.append(candidate)

    scored.sort(

        key=lambda item: item["final_score"],

        reverse=True,

    )

    return scored[:final_k]

# =============================================================================

# RETRIEVE

# =============================================================================

def retrieve(

    query: str,

    model: SentenceTransformer,

    client: QdrantClient,

    procedure_names: Optional[List[str]] = None,

    top_k: int = QDRANT_TOP_K,

    final_k: int = FINAL_TOP_K,

) -> List[Dict[str, Any]]:

    query = str(query).strip()

    if not query:

        return []

    # 1. Intent

    intent, intent_confidence = detect_intent(query)

    # 2. Procedure

    target_procedure, procedure_confidence = detect_procedure(

        query,

        procedure_names,

    )

    print()

    print("=" * 80)

    print("QUERY ANALYSIS")

    print("=" * 80)

    print(f"Query               : {query}")

    print(f"Detected intent     : {intent}")

    print(f"Intent confidence   : {intent_confidence:.4f}")

    print(f"Detected procedure  : {target_procedure}")

    print(f"Procedure confidence: {procedure_confidence:.4f}")

    # 3. Embed

    query_vector = encode_query(

        model,

        query,

    )

    # 4. Vector retrieval

    points = search_qdrant(

        client,

        query_vector,

        top_k,

    )

    print(f"\nQdrant candidates: {len(points)}")

    if not points:

        return []

    # 5. Prepare ALL candidates

    candidates = prepare_candidates(

        points,

        query,

        target_procedure,

        intent,

    )

    procedure_matches = sum(

        1

        for item in candidates

        if item["procedure_score"] >= 0.80

    )

    intent_matches = sum(

        1

        for item in candidates

        if item["chunk_score"] >= 0.80

        or item["text_intent_score"] >= 0.80

    )

    print()

    print("=" * 80)

    print("CANDIDATE SELECTION")

    print("=" * 80)

    print(f"Procedure candidates: {procedure_matches}")

    print(f"Intent candidates   : {intent_matches}")

    print(f"Candidates retained : {len(candidates)}")

    # 6. Rerank ALL candidates.

    final_results = rerank_candidates(

        candidates,

        final_k,

    )

    return final_results

# =============================================================================

# DISPLAY

# =============================================================================

def display_results(

    query: str,

    results: List[Dict[str, Any]],

) -> None:

    print_header("FINAL RETRIEVAL RESULTS")

    print("\nQuery:")

    print(query)

    if not results:

        print("\nNo retrieval results.")

        return

    print(f"\nTop {len(results)} results:")

    for rank, item in enumerate(results, start=1):

        data = item["data"]

        print()

        print_separator("-")

        print(f"Rank             : {rank}")

        print(f"Final score      : {item['final_score']:.4f}")

        print(f"Vector score     : {item['vector_score']:.4f}")

        print(f"Procedure score  : {item['procedure_score']:.4f}")

        print(f"Chunk score      : {item['chunk_score']:.4f}")

        print(f"Keyword score    : {item['keyword_score']:.4f}")

        print(f"Text intent      : {item['text_intent_score']:.4f}")

        print(f"Chunk ID          : {data['chunk_id']}")

        print(f"Document ID       : {data['document_id']}")

        print(f"Procedure         : {data['procedure']}")

        print(f"Field             : {data['field']}")

        print(f"Submission        : {data['submission']}")

        print(f"Chunk type        : {data['chunk_type']}")

        print("\nTEXT:")

        print(data["text"])

# =============================================================================

# TEST SUITE

# =============================================================================

TEST_QUERIES = [

    "Đăng ký khai sinh cần những giấy tờ gì?",

    "Đăng ký kết hôn cần chuẩn bị hồ sơ gì?",

    "Thủ tục xác nhận tình trạng hôn nhân mất bao lâu?",

    "Đăng ký khai tử có mất lệ phí không?",

    "Tôi muốn đăng ký khai sinh thì nộp hồ sơ ở đâu?",

    "Đăng ký khai tử nộp hồ sơ ở đâu?",

    "Đăng ký kết hôn mất bao lâu?",

    "Đăng ký khai sinh có mất lệ phí không?",

    "Xác nhận tình trạng hôn nhân cần giấy tờ gì?",

    "Đăng ký kết hôn không?",

]

def run_test_suite(

    model: SentenceTransformer,

    client: QdrantClient,

    procedure_names: List[str],

) -> None:

    print_header("RUNNING RETRIEVAL TEST SUITE")

    total = len(TEST_QUERIES)

    success = 0

    start_time = time.time()

    for index, query in enumerate(TEST_QUERIES, start=1):

        print()

        print_separator("=")

        print(f"[TEST {index}/{total}]")

        print_separator("=")

        try:

            results = retrieve(

                query=query,

                model=model,

                client=client,

                procedure_names=procedure_names,

                top_k=QDRANT_TOP_K,

                final_k=FINAL_TOP_K,

            )

            display_results(

                query,

                results,

            )

            if results:

                success += 1

        except Exception as exc:

            print("\n[ERROR] Test failed")

            print(f"Reason: {exc}")

    elapsed = time.time() - start_time

    print()

    print_separator("=")

    print("TEST SUMMARY")

    print_separator("=")

    print(f"Tests        : {total}")

    print(f"Successful   : {success}")

    print(f"Failed       : {total - success}")

    print(f"Success rate : {success / total * 100:.2f}%")

    print(f"Elapsed time : {elapsed:.2f}s")

# =============================================================================

# INTERACTIVE MODE

# =============================================================================

def interactive_mode(

    model: SentenceTransformer,

    client: QdrantClient,

    procedure_names: List[str],

) -> None:

    print_header("INTERACTIVE RETRIEVAL MODE")

    print("\nNhập câu hỏi pháp luật hành chính.")

    print("Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:

        try:

            query = input("Question > ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\nExiting...")

            break

        if not query:

            continue

        if query.lower() in {"exit", "quit", "q"}:

            print("Exiting...")

            break

        try:

            results = retrieve(

                query=query,

                model=model,

                client=client,

                procedure_names=procedure_names,

                top_k=QDRANT_TOP_K,

                final_k=FINAL_TOP_K,

            )

            display_results(

                query,

                results,

            )

        except Exception as exc:

            print()

            print_separator("-")

            print("[ERROR] Retrieval failed")

            print(f"Reason: {exc}")

            print_separator("-")

# =============================================================================

# MAIN

# =============================================================================

def main() -> None:

    print_separator("=")

    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")

    print("LEGAL RAG RETRIEVER")

    print_separator("=")

    model = None

    client = None

    try:

        # 1. GPU

        check_gpu()

        # 2. Model

        model = load_model()

        # 3. Qdrant

        client = connect_qdrant()

        # 4. Load procedure names ONCE.

        #    Do not scan the whole Qdrant collection for every query.

        procedure_names = load_procedure_names(client)

        # 5. Test suite

        run_test_suite(

            model,

            client,

            procedure_names,

        )

        # 6. Interactive mode

        print()

        print_separator("=")

        print("RETRIEVAL TEST COMPLETED")

        print_separator("=")

        print("\nInteractive mode? Type 'yes' to continue.")

        try:

            answer = input("> ").strip().lower()

        except (KeyboardInterrupt, EOFError):

            answer = "no"

        if answer in {"yes", "y"}:

            interactive_mode(

                model,

                client,

                procedure_names,

            )

    finally:

        if client is not None:

            try:

                client.close()

            except Exception:

                pass

            print("\n[OK] Qdrant client closed")

# =============================================================================

# ENTRY POINT

# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n[INFO] Program interrupted by user.")

        sys.exit(0)

    except Exception as exc:

        print()

        print_separator("=")

        print("[FATAL ERROR]")

        print_separator("=")

        print(exc)

        print_separator("=")

        sys.exit(1)