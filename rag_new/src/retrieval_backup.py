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
    Mode: Embedded / local storage
    Storage: D:/legal-rag/qdrant_storage
    Collection:
        vietnamese_administrative_procedures

Expected vector:
    dimension = 768
    distance = COSINE
"""

from __future__ import annotations

import math
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

QDRANT_STORAGE_PATH = r"D:\legal-rag\qdrant_storage"
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

# NOTE (fix):
# The manual PROCEDURES / PROCEDURE_ALIASES lists were removed.
#
# They only covered 4 of the ~40 procedures that actually exist in the
# Qdrant collection, and everything they matched is already covered (more
# robustly) by the name_coverage scoring in Pass 2 below, which runs
# against the full, dynamically-loaded catalog of procedure names pulled
# straight from Qdrant payloads (see load_known_procedures()).
#
# If you ever need to force a specific colloquial phrase to a specific
# procedure regardless of scoring, reintroduce a small dict here and add
# it back as "Pass 1" inside detect_procedure().


# -----------------------------------------------------------------------------
# FIX (v3): the hand-written GENERIC_ADMIN_WORDS list above caused two new
# bugs found via log review:
#
#   Bug A - a procedure name that, after stripping every hand-listed
#   "generic" word, collapses to a single leftover token (e.g. "Hồ sơ cấp
#   số nhà" -> only "nhà" survives, since hồ/sơ/cấp/số were all hand-listed
#   as generic). That single, accidentally-shared word ("nhà" also shows up
#   in "người NHÀ tôi" = "my family", unrelated meaning) then matched with
#   confidence 1.0.
#
#   Bug B - the list conflated "generic across the catalog" with "generic
#   in isolation". "ký" was hand-listed as generic because it usually rides
#   along with "đăng ký", so a query about "chữ ký" (signature) lost its
#   most specific word. Worse, "chứng" (from "chứng thực" = notarize)
#   becomes "chung" once accents are stripped, which collided with the
#   hand-listed generic word "chung" ("chung chung" = generic/vague) and
#   was wrongly filtered too. Result: "chứng thực chữ ký" matched nothing.
#
# Root cause: a manually curated list can't know how rare/common a word
# actually is in the real procedure catalog, and it can't tell two
# same-looking-after-accent-stripping words apart.
#
# Fix: only remove a small CLOSED CLASS of Vietnamese function words
# (pronouns, particles, conjunctions, question words) unconditionally -
# words that structurally never carry procedure-identifying meaning by
# themselves. Everything else (administrative jargon like "hồ sơ", "cấp",
# "giấy", "đăng ký", as well as genuinely distinctive words like "khuyết
# tật", "chữ ký", "xây dựng") is kept as a token and weighted automatically
# by how RARE it is across the real, dynamically-loaded procedure catalog
# (IDF: inverse document frequency). A word used in almost every procedure
# name naturally ends up with a low weight; a word used in only one or two
# names naturally ends up with a high weight. No hand-listing required, and
# it updates itself whenever the catalog changes.
# -----------------------------------------------------------------------------
FUNCTION_STOPWORDS = {
    # closed-class grammatical words only: pronouns, particles, question
    # words, conjunctions. Deliberately does NOT include anything that
    # could plausibly be the accent-stripped form of a meaningful content
    # word in a procedure name (that ambiguity is exactly what caused Bug
    # B above) - when in doubt, a word is left OUT of this list and IDF
    # weighting is left to handle it instead of a hand-picked exclusion.
    "gi", "can", "nao", "muon", "hoi", "la", "thi", "nho",
    "xin", "hay", "ve", "duoc", "khong", "bao", "nhieu",
    "mot", "toi", "cua", "va", "hoac", "theo", "trong",
    "gom", "nhung", "nay", "co",
}


# Primary threshold: weighted share of the PROCEDURE NAME's tokens that the
# query explains. Kept at 0.60, same as before, but now computed over
# IDF-weighted tokens instead of raw counts, so a match built entirely out
# of common admin words (low weight) no longer coincidentally clears it.
MIN_PROCEDURE_MATCH_SCORE = 0.60

# Second signal (per the log review): for long/compound procedure names
# ("chứng thực chữ ký - chứng thực Hợp đồng giao dịch/Di chúc"), sharing
# every distinctive word in the QUERY with the name is strong evidence of a
# match even when name_coverage looks diluted by the name's extra
# qualifying words. Kept deliberately strict (0.85) and requires at least
# two overlapping tokens, so a single coincidental shared word can never
# qualify through this path alone (that would just reintroduce Bug A).
MIN_QUERY_COVERAGE_STRICT = 0.85
MIN_OVERLAP_TOKENS_FOR_QUERY_PATH = 2


def _content_tokens(text: str) -> List[str]:
    """
    Tokenize text, dropping only closed-class function words.

    Unlike the old _meaningful_tokens(), this keeps administrative jargon
    ("hồ", "sơ", "cấp", "ký"...) as real tokens - their influence on
    matching is controlled by IDF weight (see compute_idf_weights()), not
    by removing them outright.
    """
    normalized = normalize_without_accents(text)
    return [
        token
        for token in normalized.split()
        if token and token not in FUNCTION_STOPWORDS
    ]


# Cache of IDF weights, keyed by the catalog that produced them so it's
# automatically invalidated if load_known_procedures() ever returns a
# different catalog (e.g. after a re-ingest) within the same process.
_IDF_CACHE: Dict[str, float] = {}
_IDF_CACHE_KEY: Optional[Tuple[str, ...]] = None


def compute_idf_weights(catalog: List[str]) -> Dict[str, float]:
    """
    Data-driven replacement for the old hand-written GENERIC_ADMIN_WORDS
    list.

    weight(token) = ln((N + 1) / (df(token) + 1)) + 1

    where N is the number of procedure names and df(token) is how many of
    those names contain the token at least once. A token in almost every
    name (e.g. "cap", "ho", "so") ends up close to the floor weight of
    ~1.0; a token in only one or two names (e.g. "khuyet", "tat", "chu",
    "nha") ends up with a much higher weight - automatically, from the
    real catalog, instead of anyone's guess about what "sounds generic".
    """

    global _IDF_CACHE, _IDF_CACHE_KEY

    cache_key = tuple(catalog)

    if _IDF_CACHE_KEY == cache_key:
        return _IDF_CACHE

    document_frequency: Dict[str, int] = {}
    doc_count = 0

    for name in catalog:
        tokens = set(_content_tokens(name))

        if not tokens:
            continue

        doc_count += 1

        for token in tokens:
            document_frequency[token] = (
                document_frequency.get(token, 0) + 1
            )

    doc_count = max(doc_count, 1)

    weights = {
        token: math.log((doc_count + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }

    _IDF_CACHE = weights
    _IDF_CACHE_KEY = cache_key

    return weights


def _token_weight(token: str, idf: Dict[str, float]) -> float:
    # A token never seen anywhere in the catalog (typo, or a word that
    # genuinely never occurs in any procedure name) is treated as maximally
    # specific - same ceiling as the rarest catalog word - rather than 0,
    # so it can still contribute to query_coverage.
    if not idf:
        return 1.0

    return idf.get(token, max(idf.values()))


# Cache of procedure names loaded from Qdrant, populated by
# load_known_procedures() the first time it's called for a given client.
_KNOWN_PROCEDURES_CACHE: List[str] = []


def load_known_procedures(client: "QdrantClient") -> List[str]:
    """
    Pull the full, real list of distinct procedure names straight from the
    Qdrant collection's payloads, instead of relying on a hand-maintained
    (and easily out-of-date) list in this file.

    Cached in-process after the first call.
    """

    global _KNOWN_PROCEDURES_CACHE

    if _KNOWN_PROCEDURES_CACHE:
        return _KNOWN_PROCEDURES_CACHE

    names = set()

    try:
        next_offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = getattr(record, "payload", None) or {}
                data = extract_payload_data(record)
                name = data.get("procedure", "")

                if name:
                    names.add(name.strip())

            if next_offset is None:
                break

    except Exception as exc:
        print()
        print(
            f"[WARN] Could not load procedure catalog from Qdrant: {exc}"
        )
        print(
            "[WARN] Falling back to empty procedure catalog "
            "(detect_procedure will return None until Qdrant is reachable)."
        )

    _KNOWN_PROCEDURES_CACHE = sorted(names)

    return _KNOWN_PROCEDURES_CACHE


def detect_procedure(
    query: str,
    known_procedures: Optional[List[str]] = None,
) -> Tuple[Optional[str], float]:
    """
    Detect administrative procedure from query.

    We deliberately use deterministic lexical matching instead of
    depending only on vector similarity.

    known_procedures: the full catalog of real procedure names, normally
    from load_known_procedures(client). If empty/not provided, no
    procedure can be detected and (None, 0.0) is returned - that's a
    deliberately safe fallback (pure vector + intent ranking) rather than
    guessing at one of a handful of hardcoded names.
    """

    q_norm = normalize_without_accents(query)

    catalog = known_procedures or []

    # FIX: collect every full-phrase match first instead of returning on
    # the first hit. `catalog` is alphabetically sorted, so a short name
    # that happens to be a PREFIX of a more specific one (e.g. "Đăng ký
    # kết hôn" vs "Đăng ký kết hôn có yếu tố nước ngoài") used to win just
    # because it sorted first - even though the query text also contained
    # the extra words that make it the more specific procedure. Among all
    # full-phrase matches we now keep the LONGEST (most specific) name.
    exact_matches = [
        name
        for name in catalog
        if normalize_without_accents(name)
        and normalize_without_accents(name) in q_norm
    ]

    if exact_matches:
        return max(exact_matches, key=len), 1.0

    if not catalog:
        return None, 0.0

    q_tokens = set(_content_tokens(query))

    if not q_tokens:
        return None, 0.0

    idf = compute_idf_weights(catalog)
    query_weight_total = sum(_token_weight(t, idf) for t in q_tokens)

    best_procedure = None
    best_name_coverage = 0.0
    best_query_coverage = 0.0

    for name in catalog:

        name_tokens = set(_content_tokens(name))

        if not name_tokens:
            continue

        overlap = name_tokens & q_tokens

        if not overlap:
            continue

        overlap_weight = sum(_token_weight(t, idf) for t in overlap)
        name_weight_total = sum(
            _token_weight(t, idf) for t in name_tokens
        )

        # Score by how much of the PROCEDURE NAME's distinguishing weight
        # was found in the query (name_coverage), IDF-weighted so common
        # admin words ("hồ sơ", "cấp", "giấy") barely move the score and
        # rare/specific words ("khuyết tật", "xây dựng") dominate it.
        name_coverage = overlap_weight / name_weight_total

        query_coverage = (
            overlap_weight / query_weight_total
            if query_weight_total
            else 0.0
        )

        # FIX (Bug A guard): if the procedure name has only one
        # distinguishing token left (or the overlap is just one token),
        # a single shared word - however weighted - is never enough on
        # its own to claim the primary confidence threshold. This is what
        # used to let "Hồ sơ cấp số nhà" absorb "người NHÀ tôi" at 1.0.
        if len(name_tokens) <= 1 and len(overlap) <= 1:
            name_coverage = min(
                name_coverage, MIN_PROCEDURE_MATCH_SCORE - 0.01
            )

        # FIX (Bug B / diluted compound names): a long or hyphenated name
        # ("chứng thực chữ ký - chứng thực Hợp đồng giao dịch/Di chúc")
        # can have low name_coverage purely because of its own extra
        # qualifying words, even when the query matches it almost
        # perfectly. query_coverage is the second signal for exactly this
        # case: if the query's distinguishing words are essentially all
        # explained by this one name (>= 0.85) AND at least two tokens
        # overlap (never just one coincidental word), treat it as a match
        # via this path even though name_coverage alone falls short.
        qualifies_by_name = name_coverage >= MIN_PROCEDURE_MATCH_SCORE
        qualifies_by_query = (
            query_coverage >= MIN_QUERY_COVERAGE_STRICT
            and len(overlap) >= MIN_OVERLAP_TOKENS_FOR_QUERY_PATH
        )

        if not (qualifies_by_name or qualifies_by_query):
            continue

        if (name_coverage, query_coverage) > (
            best_name_coverage,
            best_query_coverage,
        ):
            best_name_coverage = name_coverage
            best_query_coverage = query_coverage
            best_procedure = name

    if best_procedure is None:
        return None, 0.0

    confidence = min(1.0, max(best_name_coverage, best_query_coverage))

    return best_procedure, confidence


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
# QDRANT EMBEDDED CONNECTION
# =============================================================================

def connect_qdrant() -> QdrantClient:

    print_header("CONNECTING TO QDRANT EMBEDDED")

    print()
    print(f"Qdrant storage path: {QDRANT_STORAGE_PATH}")

    client = QdrantClient(
        path=QDRANT_STORAGE_PATH,
    )

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


# FIX (v3): reuse the same IDF-weighted, function-word-only filtering as
# detect_procedure() instead of the old hand-written GENERIC_ADMIN_WORDS
# list. Previously this function shared that list's two bugs too: it could
# collapse a candidate procedure name down to a single accidental token, and
# it could wrongly strip real content words like "ký" or "chứng" (via its
# accent-stripped collision with "chung"). Scoring is now IDF-weighted so
# common admin words ("hồ sơ", "cấp", "giấy") barely move the score and
# rare/specific words carry it, exactly like detect_procedure().
def _procedure_core_tokens(text: str) -> set[str]:
    return set(_content_tokens(text))


def calculate_procedure_score(
    detected_procedure: Optional[str],
    candidate_procedure: str,
    idf: Optional[Dict[str, float]] = None,
) -> float:
    """
    Score procedure similarity without rewarding partial-name collisions.

    Example:
        Thủ tục đăng ký khai sinh
        vs
        Thủ tục đăng ký khai tử

    must NOT receive a high score just because both contain
    ``thủ tục đăng ký khai``.

    idf: token -> weight, normally from compute_idf_weights(catalog). If
    not provided, falls back to the last catalog seen by detect_procedure()
    in this process (via the module-level cache), and to unweighted
    (all-tokens-equal) scoring only if no catalog has been loaded at all.
    """

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

    # Exact full normalized name.
    if detected == candidate:
        return 1.0

    detected_core = _procedure_core_tokens(
        detected_procedure
    )
    candidate_core = _procedure_core_tokens(
        candidate_procedure
    )

    if not detected_core or not candidate_core:
        return 0.0

    # Ignore function words and compare the actual procedure name.
    if detected_core == candidate_core:
        return 1.0

    # A candidate with extra meaningful words is a different procedure.
    # Example: ``đăng ký kết hôn có yếu tố nước ngoài`` vs
    # ``đăng ký kết hôn``. Keep a low score so it can only survive
    # when there is genuinely no exact procedure evidence.
    overlap = detected_core & candidate_core

    if not overlap:
        return 0.0

    if idf is None:
        idf = _IDF_CACHE

    # FIX (Bug A guard, same as detect_procedure): a name that reduces to
    # a single core token can't reach a high score off one coincidentally
    # shared word (e.g. detected="Xác nhận tình trạng nhà ở" candidate=
    # "Hồ sơ cấp số nhà" sharing only "nhà"). Capped low, not zeroed, so it
    # can still rank below genuine matches instead of vanishing outright.
    if (len(detected_core) <= 1 or len(candidate_core) <= 1) and len(overlap) <= 1:
        return 0.3

    overlap_weight = sum(_token_weight(t, idf) for t in overlap)
    detected_weight_total = sum(
        _token_weight(t, idf) for t in detected_core
    )
    candidate_weight_total = sum(
        _token_weight(t, idf) for t in candidate_core
    )

    detected_coverage = overlap_weight / detected_weight_total
    candidate_coverage = overlap_weight / candidate_weight_total

    balanced = min(
        detected_coverage,
        candidate_coverage,
    )

    # Never allow partial procedure-name overlap to reach the
    # same-procedure threshold used by final filtering.
    return min(balanced, 0.69)


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
    known_procedures: Optional[List[str]] = None,
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

    # Compute IDF weights once for the whole batch instead of once per
    # candidate - same catalog is reused across every point.
    idf = compute_idf_weights(known_procedures) if known_procedures else _IDF_CACHE

    for point in points:

        data = extract_payload_data(point)

        procedure_score = calculate_procedure_score(
            detected_procedure,
            data["procedure"],
            idf=idf,
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
    procedure_confidence: float = 0.0,
    intent_confidence: float = 0.0,
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

    # FIX: these used to be hardcoded to 0.90 / 0.70 regardless of how
    # weak the actual match was, which made every detection look equally
    # "confident" in the logs even when it was a coincidental word overlap.
    # We now print the real scores computed by detect_intent()/
    # detect_procedure() and passed in by the caller.

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

    # ============================================================
    # STEP 1: FILTER BY PROCEDURE
    # ============================================================
    if detected_procedure:
        exact_procedure = [
            r
            for r in results
            if r.procedure_score >= 0.95
        ]

        if exact_procedure:
            results = exact_procedure

    # ============================================================
    # STEP 2: FILTER BY EXACT CHUNK TYPE
    # ============================================================
    if detected_intent != "general_information":

        exact_intent = [
            r
            for r in results
            if r.chunk_score >= 1.0
        ]

        if exact_intent:
            results = exact_intent

    # ============================================================
    # STEP 3: REMOVE VERY WEAK VECTOR RESULTS
    # ============================================================
    strong = [
        r
        for r in results
        if r.vector_score >= MIN_VECTOR_SCORE
    ]

    if strong:
        results = strong

    # ============================================================
    # STEP 4: SORT
    # ============================================================
    results.sort(
        key=lambda r: r.final_score,
        reverse=True,
    )

    return results[:FINAL_TOP_K]
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

        parent_context = result.payload.get(
            "parent_context",
            "",
        )

        if parent_context:
            print()
            print("PARENT CONTEXT:")
            print(parent_context)


# =============================================================================
# PARENT DOCUMENT EXPANSION
# =============================================================================

def expand_to_parent_context(
    results: List[RetrievalResult],
    client: QdrantClient,
) -> List[RetrievalResult]:
    """
    Expand retrieved child chunks back to their parent document.

    Retrieval still happens on small child chunks for precision. After
    reranking/filtering, all chunks belonging to the selected document_id
    are loaded from Qdrant and attached to the result as parent_context.

    FIX: this used to scroll the ENTIRE collection (every point, every
    query) and filter in Python. That is O(collection size) work on every
    single retrieval call and gets slower as the collection grows. We now
    ask Qdrant to filter by document_id server-side (a payload index is
    not required for this to work, it just makes it faster), so only the
    chunks belonging to the selected documents are pulled over the wire.
    Falls back to the old "scroll everything + filter locally" behavior
    only if the filtered scroll call fails for some reason (e.g. very old
    qdrant-client that doesn't support this filter shape).
    """

    if not results:
        return []

    document_ids = {
        str(result.document_id).strip()
        for result in results
        if str(result.document_id).strip()
    }

    if not document_ids:
        return results

    all_points: List[Any] = []

    try:
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchAny,
        )

        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=sorted(document_ids)),
                )
            ]
        )

        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=doc_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            all_points.extend(points)

            if next_offset is None:
                break

            offset = next_offset

    except Exception as exc:

        print()
        print(
            f"[WARN] Filtered scroll for parent expansion failed "
            f"({exc}); falling back to full-collection scroll."
        )

        all_points = []
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            all_points.extend(points)

            if next_offset is None:
                break

            offset = next_offset

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for point in all_points:
        data = extract_payload_data(point)
        document_id = str(data.get("document_id", "")).strip()

        if document_id in document_ids:
            grouped.setdefault(document_id, []).append(data)

    for document_id, chunks in grouped.items():
        chunks.sort(
            key=lambda item: str(item.get("chunk_id", ""))
        )

    for result in results:
        document_id = str(result.document_id).strip()
        chunks = grouped.get(document_id, [])

        if not chunks:
            continue

        parent_chunks = []
        context_parts = []

        seen_texts = set()

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            chunk_type = str(chunk.get("chunk_type", "")).strip()
            text = str(chunk.get("text", "")).strip()

            if not text:
                continue

            # ``general_information`` is usually a duplicate aggregate of
            # the structured child chunks. Keep the structured sections and
            # skip this duplicate block to avoid bloating the LLM context.
            if normalize_without_accents(chunk_type) == "general_information":
                continue

            normalized_text = normalize_without_accents(text)
            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            parent_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_type": chunk_type,
                    "text": text,
                }
            )

            context_parts.append(
                f"[{chunk_type or 'section'}]\n{text}"
            )

        result.payload["parent_context"] = "\n\n".join(
            context_parts
        )
        result.payload["parent_chunks"] = parent_chunks
        result.payload["parent_document_id"] = document_id

    return results


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

    known_procedures = load_known_procedures(client)

    detected_procedure, procedure_confidence = detect_procedure(
        query,
        known_procedures,
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
        known_procedures=known_procedures,
    )

    # -------------------------------------------------------------------------
    # STEP 6: RERANK
    # -------------------------------------------------------------------------

    reranked = rerank_candidates(
        candidates,
        query,
        detected_intent,
        detected_procedure,
        procedure_confidence=procedure_confidence,
        intent_confidence=intent_confidence,
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

    # -------------------------------------------------------------------------
    # STEP 8: PARENT DOCUMENT EXPANSION
    # -------------------------------------------------------------------------
    # Keep child chunks for retrieval precision, but attach the full parent
    # document context for downstream LLM/API use.
    final_results = expand_to_parent_context(
        results=final_results,
        client=client,
    )

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
        # INTERACTIVE MODE ONLY
        # ---------------------------------------------------------------------
        # Benchmark/test suite is intentionally not run automatically.
        # The current phase is manual retrieval testing.
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