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
    Mode: local embedded
    Storage: D:/legal-rag/qdrant_storage
    Collection:
        vietnamese_administrative_procedures

Expected vector:
    dimension = 768
    distance = COSINE
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from rapidfuzz import fuzz
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_STORAGE_PATH = str(PROJECT_ROOT / "qdrant_data")

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


def _procedure_variants(
    procedure: str,
) -> List[str]:
    """
    Build normalized matching variants for one procedure.

    Example:
        "Thủ tục đăng ký khai sinh"

    variants:
        - "thu tuc dang ky khai sinh"
        - "dang ky khai sinh"
        - "khai sinh"

    The last variant is important for natural-language queries such as:
        "làm giấy khai sinh cho con cần những gì"
    """

    normalized = normalize_without_accents(
        procedure
    )

    if not normalized:
        return []

    tokens = normalized.split()

    variants = {
        normalized
    }

    # Administrative prefix words.
    generic_words = {
        "thu",
        "tuc",
        "dang",
        "ky",
        "giai",
        "quyet",
        "thuc",
        "hien",
    }

    # -------------------------------------------------------------
    # Remove generic words from the beginning.
    #
    # "thu tuc dang ky khai sinh"
    #              ↓
    # "khai sinh"
    # -------------------------------------------------------------
    meaningful_tokens = list(tokens)

    while (
        meaningful_tokens
        and meaningful_tokens[0] in generic_words
    ):
        meaningful_tokens.pop(0)

    if meaningful_tokens:
        variants.add(
            " ".join(meaningful_tokens)
        )

    # Keep "dang ky ..." variant explicitly.
    if normalized.startswith("thu tuc "):
        shortened = normalized[
            len("thu tuc "):
        ].strip()

        if shortened:
            variants.add(shortened)

    return sorted(
        variants,
        key=lambda x: (
            len(x.split()),
            len(x),
        ),
        reverse=True,
    )

def build_procedure_statistics(
    procedures: List[str],
) -> Dict[str, Any]:
    """
    Build automatic statistics from the current procedure index.

    Important:
        No manually maintained list of query words is required.

    The system derives token importance directly from the procedure names.
    """

    PROCEDURE_GENERIC_WORDS = {
    "thu",
    "tuc",
    "giai",
    "quyet",
    "thuc",
    "hien",
    "cap",
}
    # -------------------------------------------------------------
    # Group procedure names by normalized procedure key.
    #
    # Example:
    #
    # "Giấy ..." and "giấy ..."
    # become the same logical procedure key.
    # -------------------------------------------------------------
    grouped: Dict[str, Dict[str, Any]] = {}

    for procedure in procedures:

        normalized = normalize_without_accents(
            procedure
        )

        if not normalized:
            continue

        if normalized not in grouped:
            grouped[normalized] = {
                "display_name": procedure,
                "variants": set(),
            }

        for variant in _procedure_variants(
            procedure
        ):
            grouped[normalized]["variants"].add(
                variant
            )

    # -------------------------------------------------------------
    # Calculate document frequency for tokens.
    #
    # A token appearing in many procedures is less useful.
    # A token appearing in only one/few procedures is more useful.
    # -------------------------------------------------------------
    document_frequency: Dict[str, int] = {}

    for entry in grouped.values():

        token_set = set()

        for variant in entry["variants"]:

            tokens = set(variant.split())

            tokens -= PROCEDURE_GENERIC_WORDS

            token_set.update(tokens)

        entry["tokens"] = token_set

        for token in token_set:
            document_frequency[token] = (
                document_frequency.get(token, 0)
                + 1
            )

    procedure_count = len(grouped)

    # -------------------------------------------------------------
    # Automatic IDF-like weighting.
    #
    # Rare token:
    #   high weight
    #
    # Common token:
    #   low weight
    # -------------------------------------------------------------
    token_weights: Dict[str, float] = {}

    for token, df in document_frequency.items():

        token_weights[token] = (
            1.0
            + __import__("math").log(
                (procedure_count + 1)
                / (df + 1)
            )
        )

    # -------------------------------------------------------------
    # Finalize entries
    # -------------------------------------------------------------
    for entry in grouped.values():

        entry["token_weights"] = {
            token: token_weights[token]
            for token in entry["tokens"]
        }

        entry["variants"] = list(
            entry["variants"]
        )

    return {
        "procedures": list(
            grouped.values()
        ),
        "document_frequency": document_frequency,
        "token_weights": token_weights,
        "procedure_count": procedure_count,
        "generic_words": PROCEDURE_GENERIC_WORDS,
    }


def get_intent_query_tokens(
    detected_intent: str,
) -> set[str]:
    """
    Return tokens that belong to the question/intent part.

    The tokens are derived automatically from INTENT_PHRASES.
    """

    intent_tokens: set[str] = set()

    phrases = INTENT_PHRASES.get(
        detected_intent,
        [],
    )

    for phrase, _weight in phrases:

        normalized_phrase = (
            normalize_without_accents(
                phrase
            )
        )

        if not normalized_phrase:
            continue

        # "dang ky" is important for procedure detection.
        # Do not remove it from the query.
        if normalized_phrase == "dang ky":
            continue

        intent_tokens.update(
            normalized_phrase.split()
        )

    return intent_tokens

def detect_procedure(
    query: str,
    procedures: List[str],
    detected_intent: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """
    Automatically detect the most likely administrative procedure.

    The detector is data-driven:

    1. Normalize the query.
    2. Normalize common abbreviations.
    3. Remove intent/question words using INTENT_PHRASES.
    4. Ignore generic administrative words.
    5. Keep only query tokens that actually occur in the
       procedure index.
    6. Weight rare procedure tokens more strongly.
    7. Compare weighted query coverage and candidate coverage.
    8. Detect ambiguity between similar procedures.
    9. Return None when confidence is insufficient.

    This avoids manually adding words such as:
        "nhieu", "tien", "ton", "het", "nuoc ngoai", ...
    """

    q = normalize_without_accents(
        query
    )

    if not q or not procedures:
        return None, 0.0

    # -------------------------------------------------------------
    # Normalize common user abbreviations / spelling variants
    # -------------------------------------------------------------

    # "đk", "dk" -> "dang ky"
    q = re.sub(
        r"\bdk\b",
        "dang ky",
        q,
    )

    # After accent normalization:
    # "kí" -> "ki"
    # Normalize it to "ky"
    q = re.sub(
        r"\bki\b",
        "ky",
        q,
    )

    query_tokens = list(
        dict.fromkeys(
            q.split()
        )
    )

    # -------------------------------------------------------------
    # Build automatic procedure statistics
    # -------------------------------------------------------------

    stats = build_procedure_statistics(
        procedures
    )

    procedure_entries = stats["procedures"]
    generic_words = stats["generic_words"]
    token_weights = stats["token_weights"]

    # -------------------------------------------------------------
    # Words belonging to the detected intent/question
    # -------------------------------------------------------------

    intent_tokens = (
        get_intent_query_tokens(
            detected_intent
        )
        if detected_intent
        else set()
)

    # -------------------------------------------------------------
    # Remove generic question/admin words.
    #
    # We do NOT maintain a giant manually written
    # QUERY_GENERIC_WORDS list.
    # -------------------------------------------------------------

    remaining_tokens = [
        token
        for token in query_tokens
        if token not in intent_tokens
        and token not in generic_words
    ]

    if not remaining_tokens:
        return None, 0.0

    # -------------------------------------------------------------
    # Only query tokens that occur somewhere in the procedure index
    # are useful for procedure identification.
    #
    # This automatically ignores words such as:
    #
    #   nhieu
    #   tien
    #   het
    #   ton
    #
    # without hard-coding them.
    # -------------------------------------------------------------

    procedure_vocabulary = set(
        token_weights.keys()
    )

    meaningful_tokens = [
        token
        for token in remaining_tokens
        if token in procedure_vocabulary
    ]

    if not meaningful_tokens:
        return None, 0.0

    query_token_set = set(
        meaningful_tokens
    )

    query_total_weight = sum(
        token_weights.get(
            token,
            0.0,
        )
        for token in query_token_set
    )

    if query_total_weight <= 0:
        return None, 0.0

    # -------------------------------------------------------------
    # Compare against every procedure
    # -------------------------------------------------------------

    scored_candidates = []

    for entry in procedure_entries:

        candidate_tokens = set(
            entry["tokens"]
        )

        if not candidate_tokens:
            continue

        overlap = (
            query_token_set
            & candidate_tokens
        )

        if not overlap:
            continue

        # ---------------------------------------------------------
        # Weighted query coverage
        #
        # How much of the meaningful query is explained?
        # ---------------------------------------------------------

        overlap_weight = sum(
            token_weights.get(
                token,
                0.0,
            )
            for token in overlap
        )

        query_coverage = (
            overlap_weight
            / query_total_weight
        )

        # ---------------------------------------------------------
        # Weighted candidate coverage
        #
        # How much of this procedure is supported by the query?
        # ---------------------------------------------------------

        candidate_total_weight = sum(
            token_weights.get(
                token,
                0.0,
            )
            for token in candidate_tokens
        )

        candidate_coverage = (
            overlap_weight
            / candidate_total_weight
            if candidate_total_weight > 0
            else 0.0
        )

        # ---------------------------------------------------------
        # Weighted overlap score
        #
        # Query coverage is slightly more important because
        # additional query-specific words should help select
        # a more specific procedure.
        # ---------------------------------------------------------

        weighted_score = (
            0.40 * query_coverage
            + 0.60 * candidate_coverage
            )

        # ---------------------------------------------------------
        # Fuzzy support
        # ---------------------------------------------------------

        query_text = " ".join(
            meaningful_tokens
        )

        candidate_text = " ".join(
            sorted(candidate_tokens)
        )

        fuzzy_score = (
            fuzz.token_set_ratio(
                query_text,
                candidate_text,
            )
            / 100.0
        )

        # Fuzzy is supporting evidence only.
        final_score = (
            0.85 * weighted_score
            + 0.15 * fuzzy_score
        )

        # ---------------------------------------------------------
        # Exact variant support
        #
        # Only add a small bonus.
        # It does NOT override additional specific query terms.
        # ---------------------------------------------------------

        exact_phrase_match = False

        for variant in entry["variants"]:
            if variant in q:
                exact_phrase_match = True
                break

        if exact_phrase_match:
            final_score = max(
                final_score,
                0.93,
    )

        final_score = clamp(
            final_score
        )

        scored_candidates.append(
            {
                "procedure": entry["display_name"],
                "score": final_score,
                "query_coverage": query_coverage,
                "candidate_coverage": candidate_coverage,
                "fuzzy_score": fuzzy_score,
                "overlap": overlap,
                "candidate_tokens": candidate_tokens,
            }
        )

    if not scored_candidates:
        return None, 0.0

    # -------------------------------------------------------------
    # Sort candidates
    # -------------------------------------------------------------

    scored_candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = scored_candidates[0]

    best_score = best["score"]

    # -------------------------------------------------------------
    # Confidence threshold
    # -------------------------------------------------------------

    if best_score < 0.80:
        return None, 0.0

    # -------------------------------------------------------------
    # Ambiguity detection
    #
    # If another procedure is almost as good as the best one,
    # do not guess.
    # -------------------------------------------------------------

    if len(scored_candidates) >= 2:

        second = scored_candidates[1]

        margin = (
            best["score"]
            - second["score"]
        )

        AMBIGUITY_MARGIN = 0.08

        if (
            second["score"] >= 0.75
            and margin < AMBIGUITY_MARGIN
        ):
            print()
            print(
                "[WARNING] Ambiguous procedure detected."
            )
            print(
                f"Candidate 1: "
                f"{best['procedure']} "
                f"({best['score']:.4f})"
            )
            print(
                f"Candidate 2: "
                f"{second['procedure']} "
                f"({second['score']:.4f})"
            )

            return None, 0.0

    # -------------------------------------------------------------
    # Return best procedure
    # -------------------------------------------------------------

    return (
        best["procedure"],
        clamp(best_score),
    )
# =============================================================================
# INTENT DETECTION
# =============================================================================

# Weighted phrases used by detect_intent()
INTENT_PHRASES = {
    "required_documents": [
        ("giay to gi", 1.00),
        ("giay to", 0.95),
        ("ho so gom gi", 1.00),
        ("ho so can gi", 1.00),
        ("ho so gom nhung gi", 1.00),
        ("can nhung gi", 0.95),
        ("can chuan bi gi", 1.00),
        ("chuan bi gi", 0.95),
        ("nop nhung gi", 0.95),
        ("thanh phan ho so", 1.00),
        ("thanh phan", 0.90),
    ],

    "processing_time": [
        ("mat bao lau", 1.00),
        ("bao lau", 0.95),
        ("thoi gian", 0.90),
        ("thoi gian giai quyet", 1.00),
        ("thoi han", 0.95),
        ("khi nao co ket qua", 1.00),
        ("bao gio co ket qua", 1.00),
        ("may ngay", 0.95),
    ],

    "fee": [
        ("le phi", 1.00),
        ("phi bao nhieu", 1.00),
        ("bao nhieu tien", 1.00),
        ("co mat phi khong", 1.00),
        ("co ton phi khong", 1.00),
        ("mat phi khong", 1.00),
        ("chi phi", 0.90),
        ("thu phi", 0.90),
    ],

    "location": [
        ("o dau", 1.00),
        ("nop o dau", 1.00),
        ("nop ho so o dau", 1.00),
        ("dia diem", 1.00),
        ("noi nop", 0.95),
        ("co quan nao", 0.95),
        ("tiep nhan o dau", 1.00),
        ("nop tai dau", 1.00),
    ],

    "general_information": [
        ("la gi", 0.80),
        ("nhu the nao", 0.80),
        ("quy trinh", 0.90),
        ("huong dan", 0.90),
        ("thu tuc", 0.60),
        ("dang ky", 0.50),
    ],
}


# Simple keywords used by calculate_text_intent_score()
INTENT_KEYWORDS = {
    "required_documents": [
        "giay to gi",
        "giay to",
        "ho so",
        "thanh phan ho so",
        "thanh phan",
        "chuan bi",
        "can nhung gi",
        "can chuan bi",
        "nop nhung gi",
    ],

    "processing_time": [
        "mat bao lau",
        "bao lau",
        "thoi gian",
        "thoi han",
        "khi nao co ket qua",
        "bao gio co ket qua",
        "may ngay",
    ],

    "fee": [
        "le phi",
        "phi bao nhieu",
        "bao nhieu tien",
        "co mat phi khong",
        "co ton phi khong",
        "mat phi khong",
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
        "tiep nhan o dau",
        "nop tai dau",
    ],

    "general_information": [
        "la gi",
        "nhu the nao",
        "quy trinh",
        "huong dan",
        "thu tuc",
        "dang ky",
    ],
}

def detect_intent(query: str) -> Tuple[str, float]:
    """
    Detect query intent using exact phrase, token overlap,
    fuzzy matching, and strong special rules.
    """

    q = normalize_without_accents(query)

    if not q:
        return "general_information", 0.0

    query_tokens = set(q.split())
    scores: Dict[str, float] = {}

    from rapidfuzz import fuzz

    # =========================================================
    # 1. Phrase matching
    # =========================================================
    for intent, phrases in INTENT_PHRASES.items():

        best_score = 0.0

        for phrase, weight in phrases:
            phrase = normalize_without_accents(phrase)

            if not phrase:
                continue

            # Exact phrase
            if phrase in q:
                best_score = max(best_score, weight)
                continue

            # Token overlap
            phrase_tokens = set(phrase.split())

            if phrase_tokens:
                overlap = len(
                    phrase_tokens & query_tokens
                )

                overlap_score = (
                    overlap / len(phrase_tokens)
                ) * weight

                best_score = max(
                    best_score,
                    overlap_score
                )

            # Fuzzy phrase
            fuzzy_score = (
                fuzz.partial_ratio(
                    phrase,
                    q,
                ) / 100.0
            )

            if fuzzy_score >= 0.80:
                fuzzy_weighted = (
                    0.85 * fuzzy_score * weight
                )

                best_score = max(
                    best_score,
                    fuzzy_weighted
                )

        scores[intent] = clamp(best_score)

    # =========================================================
    # 2. Strong rules
    # =========================================================
    if any(
        x in q
        for x in [
            "giay to",
            "ho so",
            "thanh phan ho so",
            "chuan bi",
            "can nhung gi",
        ]
    ):
        scores["required_documents"] = max(
            scores.get("required_documents", 0.0),
            1.0,
        )

    if any(
        x in q
        for x in [
            "bao lau",
            "thoi gian",
            "thoi han",
            "khi nao co ket qua",
            "bao gio co ket qua",
        ]
    ):
        scores["processing_time"] = max(
            scores.get("processing_time", 0.0),
            1.0,
        )

    if any(
        x in q
        for x in [
            "le phi",
            "bao nhieu tien",
            "phi bao nhieu",
            "mat phi",
        ]
    ):
        scores["fee"] = max(
            scores.get("fee", 0.0),
            1.0,
        )

    if any(
        x in q
        for x in [
            "o dau",
            "nop o dau",
            "dia diem",
            "noi nop",
            "tiep nhan o dau",
        ]
    ):
        scores["location"] = max(
            scores.get("location", 0.0),
            1.0,
        )

    # =========================================================
    # 3. Best intent
    # =========================================================
    best_intent = max(
        scores,
        key=scores.get,
    )

    best_score = scores[best_intent]

    if best_score <= 0:
        return "general_information", 0.0

    return best_intent, clamp(best_score)


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
    Keyword similarity using exact + fuzzy token matching.

    Exact matches receive 1.0. Fuzzy matches are accepted only when the
    similarity is high enough. Very short tokens are excluded from fuzzy
    matching to reduce false positives.
    """

    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))

    q_tokens -= STOPWORDS
    t_tokens -= STOPWORDS

    if not q_tokens or not t_tokens:
        return 0.0

    matched_scores: List[float] = []

    for q_token in q_tokens:
        best_score = 0.0

        for t_token in t_tokens:

            # Exact token match
            if q_token == t_token:
                best_score = 1.0
                break

            # Do not fuzzy-match very short tokens.
            if len(q_token) < 4 or len(t_token) < 4:
                continue

            fuzzy_score = fuzz.ratio(q_token, t_token) / 100.0

            if fuzzy_score >= 0.80:
                best_score = max(best_score, fuzzy_score)

        matched_scores.append(best_score)

    return sum(matched_scores) / len(matched_scores)


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
# PROCEDURE INDEX FROM QDRANT
# =============================================================================


def load_procedure_names(
    client: QdrantClient,
    batch_size: int = 256,
) -> List[str]:
    """
    Load unique procedure names directly from Qdrant payloads.

    This keeps procedure detection data-driven: the retriever automatically
    uses every procedure present in the collection instead of hard-coding a
    small list in the source code.
    """

    procedures = set()
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            data = extract_payload_data(point)
            procedure = data.get("procedure", "").strip()

            if procedure:
                procedures.add(procedure)

        if next_offset is None:
            break

        offset = next_offset

    result = sorted(procedures)

    return result


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


def lexical_search(
    client: QdrantClient,
    query: str,
    limit: int = QDRANT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Lexical retrieval ưu tiên:

    1. Tên thủ tục
    2. Đúng loại thông tin theo intent
    3. Nội dung text của chunk

    Không dùng keyword_score() trên toàn bộ
    procedure + chunk_type + text như bản cũ.
    """

    scored: List[Dict[str, Any]] = []

    # =========================================================
    # 1. DETECT INTENT
    # =========================================================

    detected_intent, intent_confidence = detect_intent(
        query
    )

    # =========================================================
    # 2. LOAD PROCEDURES
    # =========================================================

    procedure_names = load_procedure_names(
        client
    )

    # =========================================================
    # 3. DETECT PROCEDURE
    # =========================================================

    detected_procedure, procedure_confidence = (
        detect_procedure(
            query,
            procedure_names,
            detected_intent,
        )
    )

    # =========================================================
    # 4. BUILD PROCEDURE QUERY
    # =========================================================

    normalized_query = normalize_without_accents(
        query
    )

    query_tokens = (
        normalized_query.split()
    )

    intent_tokens = (
        get_intent_query_tokens(
            detected_intent
        )
    )

    procedure_query_tokens = [
        token
        for token in query_tokens
        if token not in intent_tokens
    ]

    procedure_query = " ".join(
        procedure_query_tokens
    )

    # =========================================================
    # 5. LOAD ALL QDRANT POINTS
    # =========================================================

    points, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=256,
        offset=None,
        with_payload=True,
        with_vectors=False,
    )

    all_points = list(points)

    while next_offset is not None:

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        all_points.extend(points)

    # =========================================================
    # 6. SCORE EVERY CHUNK
    # =========================================================

    for point in all_points:

        data = extract_payload_data(
            point
        )

        procedure = data["procedure"]
        chunk_type = data["chunk_type"]
        text = data["text"]

        # -----------------------------------------------------
        # A. Lexical score on procedure name
        # -----------------------------------------------------

        procedure_lexical_score = 0.0

        if procedure_query:

            procedure_lexical_score = (
                keyword_score(
                    procedure_query,
                    procedure,
                )
            )

        # -----------------------------------------------------
        # B. Procedure semantic score
        # -----------------------------------------------------

        procedure_score = calculate_procedure_score(
            detected_procedure,
            procedure,
        )

        # -----------------------------------------------------
        # C. Intent / chunk type score
        # -----------------------------------------------------

        chunk_score = get_chunk_type_score(
            detected_intent,
            chunk_type,
        )

        # -----------------------------------------------------
        # D. Keyword score on TEXT ONLY
        # -----------------------------------------------------

        text_keyword_score = keyword_score(
            query,
            text,
        )

        # -----------------------------------------------------
        # E. Exact procedure bonus
        # -----------------------------------------------------

        exact_procedure_bonus = 0.0

        if (
            detected_procedure
            and normalize_without_accents(
                detected_procedure
            )
            == normalize_without_accents(
                procedure
            )
        ):
            exact_procedure_bonus = 0.10

        # -----------------------------------------------------
        # F. Final lexical score
        # -----------------------------------------------------

        lexical_score = (
            0.50 * procedure_lexical_score
            + 0.25 * procedure_score
            + 0.15 * chunk_score
            + 0.10 * text_keyword_score
            + exact_procedure_bonus
        )

        lexical_score = clamp(
            lexical_score
        )

        # -----------------------------------------------------
        # Ignore completely irrelevant chunks
        # -----------------------------------------------------

        if lexical_score <= 0:
            continue

        scored.append(
            {
                "point": point,
                "data": data,
                "lexical_score": lexical_score,
                "procedure_lexical_score":
                    procedure_lexical_score,
                "procedure_score":
                    procedure_score,
                "chunk_score":
                    chunk_score,
                "text_keyword_score":
                    text_keyword_score,
            }
        )

    # =========================================================
    # 7. SORT
    # =========================================================

    scored.sort(
        key=lambda item: (
            item["lexical_score"],
            item["procedure_score"],
            item["chunk_score"],
            item["procedure_lexical_score"],
        ),
        reverse=True,
    )

    return scored[:limit]

def rrf_fusion(
    vector_results: List[Any],
    lexical_results: List[Dict[str, Any]],
    detected_intent: str,
    detected_procedure: Optional[str],
    k: int = 60,
    limit: int = QDRANT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Intent-aware Reciprocal Rank Fusion (RRF).

    Kết hợp:
        - Vector retrieval
        - Lexical retrieval
        - Intent / chunk-type relevance
        - Procedure relevance

    Base:
        RRF(d) = 1 / (k + rank_vector)
               + 1 / (k + rank_lexical)

    Sau đó cộng thêm:
        - bonus cho đúng chunk type
        - bonus cho đúng procedure

    Rank bắt đầu từ 1.
    """

    fused: Dict[str, Dict[str, Any]] = {}

    # =========================================================
    # 1. VECTOR RESULTS
    # =========================================================

    for rank, point in enumerate(
        vector_results,
        start=1,
    ):

        data = extract_payload_data(
            point
        )

        chunk_id = data["chunk_id"]

        if not chunk_id:
            chunk_id = str(
                getattr(
                    point,
                    "id",
                    "",
                )
            )

        rrf_score = 1.0 / (
            k + rank
        )

        if chunk_id not in fused:

            fused[chunk_id] = {
                "point": point,
                "data": data,
                "vector_rank": rank,
                "lexical_rank": None,
                "base_rrf_score": 0.0,
            }

        fused[chunk_id][
            "base_rrf_score"
        ] += rrf_score

    # =========================================================
    # 2. LEXICAL RESULTS
    # =========================================================

    for rank, item in enumerate(
        lexical_results,
        start=1,
    ):

        data = item["data"]

        chunk_id = data["chunk_id"]

        if not chunk_id:
            chunk_id = str(
                getattr(
                    item["point"],
                    "id",
                    "",
                )
            )

        rrf_score = 1.0 / (
            k + rank
        )

        if chunk_id not in fused:

            fused[chunk_id] = {
                "point": item["point"],
                "data": data,
                "vector_rank": None,
                "lexical_rank": rank,
                "base_rrf_score": rrf_score,
            }

        else:

            fused[chunk_id][
                "lexical_rank"
            ] = rank

            fused[chunk_id][
                "base_rrf_score"
            ] += rrf_score

    # =========================================================
    # 3. INTENT / PROCEDURE BONUS
    # =========================================================

    results = []

    for item in fused.values():

        data = item["data"]

        chunk_type = data["chunk_type"]

        procedure = data["procedure"]

        # -----------------------------------------------------
        # Intent score
        # -----------------------------------------------------

        intent_score = get_chunk_type_score(
            detected_intent,
            chunk_type,
        )

        # -----------------------------------------------------
        # Procedure score
        # -----------------------------------------------------

        procedure_score = calculate_procedure_score(
            detected_procedure,
            procedure,
        )

        # -----------------------------------------------------
        # Bonus
        # -----------------------------------------------------

        
        
           # Intent is stronger than procedure
# once the procedure has already been identified.

        intent_bonus = (
            0.020 * intent_score
        )

        if detected_procedure:

            procedure_bonus = (
                0.012 * procedure_score
            )

            if procedure_score < 0.80:
             procedure_penalty = 0.015
            else:
                procedure_penalty = 0.0

        else:

            procedure_bonus = 0.0
            procedure_penalty = 0.0

        final_rrf_score = (
            item["base_rrf_score"]
            + intent_bonus
            + procedure_bonus
            - procedure_penalty
)
        item["intent_score"] = (
            intent_score
        )

        item["procedure_score"] = (
            procedure_score
        )

        item["intent_bonus"] = (
            intent_bonus
        )

        item["procedure_bonus"] = (
            procedure_bonus
        )

        item["rrf_score"] = (
            final_rrf_score
        )

        results.append(
            item
        )

    # =========================================================
    # 4. SORT
    # =========================================================

    results.sort(
        key=lambda item: (
            item["rrf_score"],
            item["intent_score"],
            item["procedure_score"],
        ),
        reverse=True,
    )

    return results[:limit]

def test_rrf_fusion(
    client: QdrantClient,
    model: SentenceTransformer,
) -> None:

    queries = [
        "đăng ký hộ kinh doanh",
        "khai sinh cần giấy tờ gì",
        "giấy phép xây dựng cần những gì",
        "cấp lại đăng ký hộ kinh doanh",
    ]

    print_header(
        "RRF FUSION TEST"
    )

    for query in queries:

        print()
        print_separator("=")
        print(
            f"Query: {query}"
        )
        print_separator("=")

        # -----------------------------------------------------
        # 1. Detect intent
        # -----------------------------------------------------

        detected_intent, _ = detect_intent(
            query
        )

        # -----------------------------------------------------
        # 2. Detect procedure
        # -----------------------------------------------------

        procedure_names = load_procedure_names(
            client
        )

        detected_procedure, _ = detect_procedure(
            query,
            procedure_names,
            detected_intent,
        )

        print(
            f"Detected intent    : "
            f"{detected_intent}"
        )

        print(
            f"Detected procedure : "
            f"{detected_procedure}"
        )

        # -----------------------------------------------------
        # 3. Vector retrieval
        # -----------------------------------------------------

        query_vector = encode_query(
            model,
            query,
        )

        vector_results = qdrant_search(
            client,
            query_vector,
            limit=QDRANT_TOP_K,
        )

        # -----------------------------------------------------
        # 4. Lexical retrieval
        # -----------------------------------------------------

        lexical_results = lexical_search(
            client,
            query,
            limit=QDRANT_TOP_K,
        )

        # -----------------------------------------------------
        # 5. Intent-aware RRF
        # -----------------------------------------------------

        fused_results = rrf_fusion(
            vector_results,
            lexical_results,
            detected_intent,
            detected_procedure,
            k=60,
            limit=5,
        )

        # -----------------------------------------------------
        # 6. Display
        # -----------------------------------------------------

        for rank, item in enumerate(
            fused_results,
            start=1,
        ):

            data = item["data"]

            print(
                f"{rank}. "
                f"RRF={item['rrf_score']:.6f} | "
                f"VRank={item['vector_rank']} | "
                f"LRank={item['lexical_rank']} | "
                f"Intent={item['intent_score']:.2f} | "
                f"Procedure={item['procedure_score']:.2f} | "
                f"{data['procedure']} | "
                f"{data['chunk_type']}"
            )
            
def test_lexical_search(
    client: QdrantClient,
) -> None:

    queries = [
        "đăng ký hộ kinh doanh",
        "cấp lại đăng ký hộ kinh doanh",
        "khai sinh cần giấy tờ gì",
        "giấy phép xây dựng cần những gì",
    ]

    print_header(
        "LEXICAL SEARCH TEST"
    )

    for query in queries:

        print()
        print_separator("=")
        print(
            f"Query: {query}"
        )
        print_separator("=")

        results = lexical_search(
            client,
            query,
            limit=5,
        )

        for index, item in enumerate(
            results,
            start=1,
        ):

            data = item["data"]

            print(
                f"{index}. "
                f"{item['lexical_score']:.4f} | "
                f"{data['procedure']} | "
                f"{data['chunk_type']}"
            )


# =============================================================================
# SCORE HELPERS
# =============================================================================

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def calculate_procedure_score(
    detected_procedure: Optional[str],
    candidate_procedure: str,
) -> float:
    """
    Calculate procedure similarity with protection against
    partial-name false positives.

    Exact procedure = 1.0.
    A shorter procedure name must not automatically match
    a longer, more specific procedure at 1.0.
    """

    if not detected_procedure or not candidate_procedure:
        return 0.0

    detected = normalize_without_accents(
        detected_procedure
    ).strip()

    candidate = normalize_without_accents(
        candidate_procedure
    ).strip()

    if not detected or not candidate:
        return 0.0

    # =========================================================
    # 1. Exact match
    # =========================================================
    if detected == candidate:
        return 1.0

    detected_tokens = set(detected.split())
    candidate_tokens = set(candidate.split())

    if not detected_tokens or not candidate_tokens:
        return 0.0

    # =========================================================
    # 2. Remove generic words
    # =========================================================
    generic_words = {
        "thu",
        "tuc",
        "dang",
        "ky",
        "giai",
        "quyet",
        "thuc",
        "hien",
        "cap",
    }

    detected_specific = detected_tokens - generic_words
    candidate_specific = candidate_tokens - generic_words

    if not detected_specific:
        detected_specific = detected_tokens

    if not candidate_specific:
        candidate_specific = candidate_tokens

    # =========================================================
    # 3. Exact token overlap
    # =========================================================
    overlap = detected_specific & candidate_specific

    detected_coverage = (
        len(overlap) / len(detected_specific)
    )

    candidate_coverage = (
        len(overlap) / len(candidate_specific)
    )

    # =========================================================
    # 4. Penalize extra specific candidate information
    #
    # Example:
    # "ket hon"
    # vs
    # "ket hon co yeu to nuoc ngoai"
    #
    # detected_coverage = 1.0
    # candidate_coverage < 1.0
    # =========================================================

    balanced_overlap = min(
        detected_coverage,
        candidate_coverage,
    )

    # =========================================================
    # 5. Fuzzy similarity
    # =========================================================
    from rapidfuzz import fuzz

    fuzzy_score = (
        fuzz.token_set_ratio(
            detected,
            candidate,
        ) / 100.0
    )

    # =========================================================
    # 6. Conflicting / distinguishing phrases
    # =========================================================
    distinguishing_terms = {
        "nuoc ngoai",
        "quoc tich",
        "co yeu to nuoc ngoai",
        "luu dong",
        "lan dau",
        "cap lai",
        "cap doi",
        "thay doi",
        "tam ngung",
        "cham dut",
        "tiep tuc",
    }

    detected_lower = set(detected.split())
    candidate_lower = set(candidate.split())

    # If candidate contains additional specific terms
    # that are completely absent from the detected procedure,
    # don't allow a perfect score.
    extra_terms = candidate_specific - detected_specific

    has_extra_specific = bool(extra_terms)

    # =========================================================
    # 7. Final score
    # =========================================================
    score = max(
        balanced_overlap,
        0.70 * fuzzy_score,
    )

    # Do not let a shorter procedure get score 1.0
    # against a more specific procedure.
    if has_extra_specific and candidate != detected:
        score = min(score, 0.89)

    return clamp(score)

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

    # ---------------------------------------------------------
    # 1. Prefer same procedure
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. Prefer exact chunk type for the detected intent
    # ---------------------------------------------------------
    exact_chunk = [
        r
        for r in results
        if get_chunk_type_score(
            detected_intent,
            r.chunk_type,
        ) >= 1.0
    ]

    # Only replace the result pool when an exact chunk
    # exists. Otherwise keep all candidates.
    if exact_chunk:
        results = exact_chunk

    # ---------------------------------------------------------
    # 3. Remove very weak vector matches
    # ---------------------------------------------------------
    strong = [
        r
        for r in results
        if r.vector_score >= MIN_VECTOR_SCORE
    ]

    if strong:
        results = strong

    # ---------------------------------------------------------
    # 4. Final top-k
    # ---------------------------------------------------------
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


# =============================================================================
# RETRIEVAL PIPELINE
# =============================================================================

def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    procedures: List[str],
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
        query,
        procedures,
        detected_intent,
    )

    print()
    print(
        f"Detected intent    : "
        f"{detected_intent}"
    )

    print(
        f"Detected procedure : "
        f"{detected_procedure}"
    )

    # -------------------------------------------------------------------------
    # STOP WHEN PROCEDURE CANNOT BE DETERMINED
    # -------------------------------------------------------------------------

    if detected_procedure is None:

        print()
        print(
            "[WARNING] Cannot determine administrative procedure."
        )

        print(
            "Không đủ thông tin để xác định thủ tục hành chính."
        )

        print(
            "Vui lòng cho biết tên hoặc nội dung cụ thể của thủ tục."
        )

        print()

        return []

    # -------------------------------------------------------------------------
    # STEP 3: EMBEDDING
    # -------------------------------------------------------------------------

    query_vector = encode_query(
        model,
        query,
    )

    # -------------------------------------------------------------------------
    # STEP 4: VECTOR RETRIEVAL
    # -------------------------------------------------------------------------

    vector_results = qdrant_search(
        client,
        query_vector,
        limit=top_k,
    )

    print()
    print(
        f"Vector candidates: "
        f"{len(vector_results)}"
    )

    if not vector_results:
        return []

    # -------------------------------------------------------------------------
    # STEP 5: LEXICAL RETRIEVAL
    # -------------------------------------------------------------------------

    lexical_results = lexical_search(
        client,
        query,
        limit=top_k,
    )

    print(
        f"Lexical candidates: "
        f"{len(lexical_results)}"
    )

    # -------------------------------------------------------------------------
    # STEP 6: RRF FUSION
    # -------------------------------------------------------------------------

    fused_results = rrf_fusion(
        vector_results,
        lexical_results,
        detected_intent,
        detected_procedure,
        k=60,
        limit=top_k,
    )

    print(
        f"Hybrid candidates: "
        f"{len(fused_results)}"
    )

    if not fused_results:
        return []

    # -------------------------------------------------------------------------
    # STEP 7: CONVERT FUSED RESULTS BACK TO POINTS
    #
    # RRF decides which documents survive into the final candidate pool.
    # The existing candidate selection + reranking logic then performs
    # the detailed legal ranking.
    # -------------------------------------------------------------------------

    fused_points = [
        item["point"]
        for item in fused_results
    ]

    # -------------------------------------------------------------------------
    # STEP 8: CANDIDATE SELECTION
    # -------------------------------------------------------------------------

    candidates = select_candidates(
        fused_points,
        query,
        detected_intent,
        detected_procedure,
    )

    # -------------------------------------------------------------------------
    # STEP 9: RERANK
    # -------------------------------------------------------------------------

    reranked = rerank_candidates(
        candidates,
        query,
        detected_intent,
        detected_procedure,
    )

    # -------------------------------------------------------------------------
    # STEP 10: FINAL FILTER
    # -------------------------------------------------------------------------

    final_results = filter_final_results(
        reranked,
        detected_procedure,
        detected_intent,
    )

    # -------------------------------------------------------------------------
    # STEP 11: FINAL TOP-K
    # -------------------------------------------------------------------------

    final_results = final_results[
        :final_top_k
    ]

    return final_results


# =============================================================================
# SINGLE QUERY TEST
# =============================================================================

def run_query(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    procedures: List[str],
) -> List[RetrievalResult]:

    results = retrieve(
        query=query,
        model=model,
        client=client,
        procedures=procedures,
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
    procedures: List[str],
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
            procedures=procedures,
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
    procedures: List[str],
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
                procedures,
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

    # -------------------------------------------------------------------------
    # Build dynamic procedure index from Qdrant
    # -------------------------------------------------------------------------
    procedures = load_procedure_names(client)

    print()
    print(f"Procedure index: {len(procedures)} unique procedures")

    for index, procedure in enumerate(procedures, start=1):
        print(f"{index:02d}. {procedure}")

    if not procedures:
        raise RuntimeError(
            "No procedure names were found in Qdrant payloads."
        )

    try:

        # ---------------------------------------------------------------------
        # TEST SUITE
        # ---------------------------------------------------------------------

        run_test_suite(
            model=model,
            client=client,
            procedures=procedures,
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
                procedures=procedures,
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

        client = connect_qdrant()

        model = load_model()

        procedures = load_procedure_names(
            client
        )

        run_test_suite(
            model,
            client,
            procedures,
        )

        client.close()

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
