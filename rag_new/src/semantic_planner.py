from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "procedures_canonical.csv"
)

DOMAIN_PROFILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "domain_profile.json"
)


# =============================================================================
# INTENTS
# =============================================================================

INTENTS = {
    "required_documents": [
        "hồ sơ",
        "giấy tờ",
        "giấy tờ cần",
        "cần những gì",
        "cần gì",
        "chuẩn bị gì",
        "chuẩn bị những gì",
        "thành phần hồ sơ",
        "hồ sơ gồm",
        "hồ sơ bao gồm",
        "cần chuẩn bị",
        "giấy tờ gồm",
    ],

    "fee": [
        "phí",
        "lệ phí",
        "bao nhiêu tiền",
        "mất bao nhiêu",
        "hết bao nhiêu",
        "tốn bao nhiêu",
        "giá bao nhiêu",
        "chi phí",
        "thu phí",
    ],

    "location": [
        "ở đâu",
        "nộp ở đâu",
        "nộp hồ sơ ở đâu",
        "đến đâu",
        "địa điểm",
        "nơi nộp",
        "nộp tại đâu",
        "làm ở đâu",
        "tiếp nhận ở đâu",
    ],

    "processing_time": [
        "bao lâu",
        "mất bao lâu",
        "thời gian",
        "thời gian giải quyết",
        "giải quyết trong bao lâu",
        "khi nào có",
        "bao giờ có",
    ],

    "legal_basis": [
        "căn cứ pháp lý",
        "căn cứ",
        "theo luật nào",
        "theo nghị định nào",
        "văn bản pháp lý",
        "quy định nào",
        "cơ sở pháp lý",
    ],

    "forms": [
        "biểu mẫu",
        "mẫu đơn",
        "mẫu tờ khai",
        "tờ khai",
        "tải mẫu",
        "mẫu giấy",
    ],

    "general_information": [
        "là gì",
        "thế nào",
        "như thế nào",
        "thủ tục",
        "thông tin",
        "quy trình",
        "trình tự",
        "đăng ký",
    ],
}


# =============================================================================
# ABBREVIATIONS
# =============================================================================

ABBREVIATIONS = {
    "đk": "đăng ký",
    "dk": "đăng ký",
    "dky": "đăng ký",
    "hs": "hồ sơ",
    "tphs": "thành phần hồ sơ",
}


# =============================================================================
# GENERIC WORDS
# =============================================================================

GENERIC_WORDS = {
    "thủ",
    "tục",
    "hành",
    "chính",
    "cần",
    "những",
    "giấy",
    "tờ",
    "hồ",
    "sơ",
    "bao",
    "gồm",
    "chuẩn",
    "bị",
    "gì",
    "nào",
    "như",
    "thế",
    "làm",
    "ở",
    "đâu",
    "nộp",
    "tại",
    "đến",
    "cho",
    "tôi",
    "mình",
    "muốn",
    "hỏi",
    "xin",
    "về",
    "có",
    "không",
    "mất",
    "lâu",
    "tiền",
    "phí",
    "lệ",
    "thời",
    "gian",
    "quy",
    "định",
}


# =============================================================================
# PROCEDURE SEMANTIC ALIASES
# =============================================================================
#
# Đây là phần quan trọng.
#
# Mục đích:
# "làm giấy khai sinh cho con"
#       -> "đăng ký khai sinh"
#
# "kết hôn với người nước ngoài"
#       -> "đăng ký kết hôn có yếu tố nước ngoài"
#
# Không cần LLM/embedding ở bước này.
# =============================================================================

PROCEDURE_ALIASES = {
    "khai sinh": [
        "khai sinh",
        "làm giấy khai sinh",
        "làm khai sinh",
        "đăng ký khai sinh",
        "giấy khai sinh",
        "khai sinh cho con",
        "làm giấy cho con",
    ],

    "khai tử": [
        "khai tử",
        "đăng ký khai tử",
        "giấy khai tử",
        "báo tử",
    ],

    "kết hôn": [
        "kết hôn",
        "đăng ký kết hôn",
        "đăng ký kết hôn trong nước",
    ],

    "kết hôn có yếu tố nước ngoài": [
        "kết hôn với người nước ngoài",
        "lấy chồng người nước ngoài",
        "lấy vợ người nước ngoài",
        "kết hôn người nước ngoài",
        "đăng ký kết hôn với người nước ngoài",
        "đăng ký kết hôn có yếu tố nước ngoài",
        "kết hôn có yếu tố nước ngoài",
    ],

    "cấp lại giấy khai sinh": [
        "cấp lại giấy khai sinh",
        "xin cấp lại giấy khai sinh",
        "làm lại giấy khai sinh",
        "cấp lại khai sinh",
        "xin lại giấy khai sinh",
    ],

    "đăng ký thường trú": [
        "đăng ký thường trú",
        "đăng ký hộ khẩu",
        "làm hộ khẩu",
        "nhập hộ khẩu",
        "đăng ký thường trú tại",
    ],
}


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text để procedure / intent detection
    không phụ thuộc quá nhiều vào viết hoa,
    dấu câu hoặc viết tắt.
    """

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Chuẩn hóa viết tắt
    for short, full in ABBREVIATIONS.items():
        pattern = rf"\b{re.escape(short)}\b"
        text = re.sub(pattern, full, text)

    # Chuẩn hóa một số cách viết thường gặp
    replacements = {
        "đăng kí": "đăng ký",
        "dang ky": "đăng ký",
        "khai sanh": "khai sinh",
        "khai tu": "khai tử",
        "ket hon": "kết hôn",
        "cap lai": "cấp lại",
        "thuong tru": "thường trú",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Loại punctuation
    text = re.sub(
        r"[^\w\sÀ-ỹ]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Gom whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


def meaningful_tokens(text: str) -> list[str]:
    tokens = tokenize(text)

    return [
        token
        for token in tokens
        if token not in GENERIC_WORDS
        and len(token) >= 2
    ]


# =============================================================================
# FUZZY SIMILARITY
# =============================================================================

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def token_match(a: str, b: str) -> bool:
    """
    So khớp token an toàn cho tiếng Việt.
    Âm tiết ngắn (<4 ký tự) chỉ được khớp chính xác, vì fuzzy sẽ
    gây nhầm như "khi" ~ "khai" (similarity 0.857).
    """
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return similarity(a, b) >= 0.85


# =============================================================================
# ALIAS SEMANTIC MATCH
# =============================================================================

def alias_score(
    query: str,
    procedure_name: str,
) -> float:
    """
    Tính điểm dựa trên các cách diễn đạt tương đương.

    Ví dụ:
        query = "làm giấy khai sinh cho con"
        procedure = "Thủ tục đăng ký khai sinh"

    -> nhận diện alias "làm giấy khai sinh"
    -> điểm cao
    """

    query_normalized = normalize_text(query)
    procedure_normalized = normalize_text(procedure_name)

    if not query_normalized or not procedure_normalized:
        return 0.0

    best_score = 0.0

    for canonical, aliases in PROCEDURE_ALIASES.items():

        canonical_normalized = normalize_text(canonical)

        # Procedure có chứa canonical concept
        procedure_has_canonical = (
            canonical_normalized in procedure_normalized
        )

        if not procedure_has_canonical:
            continue

        for alias in aliases:

            alias_normalized = normalize_text(alias)

            if not alias_normalized:
                continue

            # Alias xuất hiện nguyên trong query
            if alias_normalized in query_normalized:
                # Alias càng dài/cụ thể càng đáng tin:
                # "kết hôn với người nước ngoài" > "kết hôn"
                specificity = len(meaningful_tokens(alias_normalized))
                best_score = max(
                    best_score,
                    0.90 + 0.01 * min(specificity, 9),
                )
                continue

            # Fuzzy comparison giữa alias và query
            score = similarity(
                alias_normalized,
                query_normalized,
            )

            best_score = max(
                best_score,
                score * 0.90,
            )

            # Token overlap của alias với query
            alias_tokens = set(
                meaningful_tokens(alias_normalized)
            )

            query_tokens = set(
                meaningful_tokens(query_normalized)
            )

            if alias_tokens and query_tokens:

                overlap = len(
                    alias_tokens & query_tokens
                ) / len(alias_tokens)

                best_score = max(
                    best_score,
                    overlap * 0.88,
                )

    return min(1.0, best_score)


# =============================================================================
# LOAD PROCEDURES
# =============================================================================

def load_procedures() -> pd.DataFrame:

    if not CANONICAL_FILE.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n"
            f"{CANONICAL_FILE}"
        )

    df = pd.read_csv(
        CANONICAL_FILE,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "procedure_id",
        "procedure_name",
        "domain",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Canonical dataset missing columns: "
            f"{missing}"
        )

    return df


# =============================================================================
# LOAD DOMAIN PROFILE
# =============================================================================

def load_domain_profile() -> dict:

    if not DOMAIN_PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Domain profile not found:\n"
            f"{DOMAIN_PROFILE_FILE}"
        )

    with open(
        DOMAIN_PROFILE_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# =============================================================================
# INTENT DETECTION
# =============================================================================

def detect_intent(
    query: str,
) -> tuple[str, float]:

    normalized_query = normalize_text(query)

    if not normalized_query:
        return (
            "general_information",
            0.0,
        )

    scores = {}

    for intent, keywords in INTENTS.items():

        score = 0.0

        for keyword in keywords:

            keyword_normalized = normalize_text(keyword)

            # Exact phrase
            if keyword_normalized in normalized_query:

                token_count = len(
                    keyword_normalized.split()
                )

                score = max(
                    score,
                    min(
                        1.0,
                        0.70 + token_count * 0.10,
                    ),
                )

            else:

                # Token-level fuzzy fallback
                query_tokens = meaningful_tokens(
                    normalized_query
                )

                keyword_tokens = meaningful_tokens(
                    keyword_normalized
                )

                if not query_tokens:
                    continue

                # Keyword chỉ còn 1 token sau khi bỏ generic words
                # (vd "tờ khai" -> "khai") quá yếu để khớp mờ.
                if len(keyword_tokens) < 2:
                    continue

                matched = 0

                for kt in keyword_tokens:

                    if any(
                        token_match(kt, qt)
                        for qt in query_tokens
                    ):
                        matched += 1

                overlap = (
                    matched
                    / len(keyword_tokens)
                )

                score = max(
                    score,
                    overlap * 0.70,
                )

        scores[intent] = score

    best_intent = max(
        scores,
        key=scores.get,
    )

    best_score = scores[
        best_intent
    ]

    if best_score < 0.40:
        return (
            "general_information",
            best_score,
        )

    return (
        best_intent,
        best_score,
    )


# =============================================================================
# PROCEDURE SCORING
# =============================================================================

def score_procedure(query: str, procedure_name: str) -> float:
    """
    Score mức độ phù hợp giữa query và procedure.

    Strategy:
    1. Normalize query + procedure
    2. Kiểm tra exact procedure phrase
    3. Kiểm tra semantic aliases
    4. So sánh meaningful tokens
    5. Fuzzy matching
    6. Ưu tiên procedure cụ thể hơn khi query chứa alias tương ứng
    """

    query_normalized = normalize_text(query)
    procedure_normalized = normalize_text(procedure_name)

    if not query_normalized or not procedure_normalized:
        return 0.0

    # ============================================================
    # 1. EXACT PROCEDURE PHRASE
    # ============================================================

    # Bỏ các tiền tố mang tính mô tả của dataset
    procedure_core = procedure_normalized

    if procedure_core.startswith("thủ tục "):
        procedure_core = procedure_core[len("thủ tục "):]

    # Exact phrase xuất hiện trong query
    if procedure_core in query_normalized:
        # Procedure càng cụ thể thì càng có lợi thế.
        core_tokens = meaningful_tokens(procedure_core)

        if len(core_tokens) >= 5:
            return 1.0

        if len(core_tokens) >= 3:
            return 0.95

        return 0.90

    # ============================================================
    # 2. SEMANTIC ALIAS
    # ============================================================

    semantic_score = alias_score(
        query_normalized,
        procedure_normalized
    )

    # ============================================================
    # 3. TOKEN OVERLAP
    # ============================================================

    query_tokens = meaningful_tokens(query_normalized)
    procedure_tokens = meaningful_tokens(procedure_normalized)

    if not procedure_tokens:
        lexical_score = 0.0
    else:
        shared_tokens = set(query_tokens) & set(procedure_tokens)

        exact_overlap = len(shared_tokens) / len(procedure_tokens)

        # ========================================================
        # 4. FUZZY TOKEN MATCHING
        # ========================================================

        fuzzy_matches = 0

        for p_token in procedure_tokens:
            if any(
                token_match(p_token, q_token)
                for q_token in query_tokens
            ):
                fuzzy_matches += 1

        fuzzy_overlap = fuzzy_matches / len(procedure_tokens)

        lexical_score = (
            exact_overlap * 0.70
            + fuzzy_overlap * 0.30
        )

    # ============================================================
    # 5. FINAL SCORE
    # ============================================================

    score = max(
        semantic_score,
        lexical_score
    )

    return min(1.0, score)
# =============================================================================
# PROCEDURE DETECTION
# =============================================================================

def detect_procedure(
    query: str,
    procedures: pd.DataFrame
):
    """
    Detect procedure phù hợp nhất với query.

    Ưu tiên:
    1. Score cao
    2. Procedure cụ thể hơn
    3. Margin giữa top-1 và top-2
    """

    candidates = []

    for _, row in procedures.iterrows():

        procedure_name = str(row["procedure_name"])

        score = score_procedure(
            query,
            procedure_name
        )

        core_name = normalize_text(procedure_name)

        if core_name.startswith("thủ tục "):
            core_name = core_name[len("thủ tục "):]

        proc_tokens = set(meaningful_tokens(core_name))
        query_tokens = set(meaningful_tokens(query))

        # Token của thủ tục mà câu hỏi KHÔNG nhắc tới.
        # Ví dụ: query "đăng ký kết hôn" không nhắc "nước ngoài"
        # -> thủ tục "...có yếu tố nước ngoài" bị trừ điểm khi xếp hạng.
        uncovered = len(proc_tokens - query_tokens)

        candidates.append({
            "procedure_id": row["procedure_id"],
            "procedure_name": procedure_name,
            "domain": row["domain"],
            "score": score,
            "uncovered": uncovered,
            "specificity": len(proc_tokens),
        })

    # ============================================================
    # SORT
    # ============================================================

    candidates.sort(
        key=lambda x: (
            round(x["score"], 3),
            -x["uncovered"],
        ),
        reverse=True,
    )

    if not candidates:
        return None

    best = candidates[0]
    best["top_candidates"] = [
        {
            "procedure_id": c["procedure_id"],
            "procedure_name": c["procedure_name"],
            "score": round(c["score"], 4),
        }
        for c in candidates[:3]
        if c["score"] > 0
    ]

    second_score = (
        candidates[1]["score"]
        if len(candidates) > 1
        else 0.0
    )

    margin = best["score"] - second_score

    # Hai thủ tục cùng điểm và cùng mức "thừa token" -> mơ hồ thật sự
    # (ví dụ dataset có 2 thủ tục trùng tên).
    best["ambiguous"] = (
        len(candidates) > 1
        and best["score"] >= 0.85
        and candidates[1]["score"] >= best["score"] - 0.02
        and candidates[1]["uncovered"] == best["uncovered"]
    )

    # ============================================================
    # STRONG MATCH
    # ============================================================

    if best["score"] >= 0.85:
        return best

    # ============================================================
    # MEDIUM MATCH
    # ============================================================

    if (
        best["score"] >= 0.60
        and margin >= 0.10
    ):
        return best

    # ============================================================
    # NO RELIABLE PROCEDURE
    # ============================================================

    return None

# =============================================================================
# SEMANTIC PLAN
# =============================================================================

def build_plan(
    query: str,
    procedures_df: pd.DataFrame,
) -> dict:

    normalized_query = normalize_text(
        query
    )

    intent, intent_score = detect_intent(
        normalized_query
    )

    procedure = detect_procedure(
        normalized_query,
        procedures_df,
    )

    plan = {
        "query": query,
        "normalized_query": normalized_query,

        "intent": intent,
        "intent_score": round(
            intent_score,
            4,
        ),

        "procedure": (
            procedure["procedure_name"]
            if procedure
            else None
        ),

        "procedure_id": (
            procedure["procedure_id"]
            if procedure
            else None
        ),

        "procedure_score": (
            round(
                procedure["score"],
                4,
            )
            if procedure
            else 0.0
        ),

        "domain": (
            procedure["domain"]
            if procedure
            else None
        ),

        "candidates": (
            procedure.get("top_candidates", [])
            if procedure
            else []
        ),

        "ambiguous": (
            bool(procedure.get("ambiguous"))
            if procedure
            else False
        ),

        "should_retrieve": (
            procedure is not None
        ),

        "status": (
            "procedure_not_detected"
            if procedure is None
            else "ambiguous_procedure"
            if procedure.get("ambiguous")
            else "ready"
        ),
    }

    return plan


# =============================================================================
# PRINT PLAN
# =============================================================================

def print_plan(
    plan: dict,
) -> None:

    print()
    print("-" * 80)

    print(
        f"Query              : "
        f"{plan['query']}"
    )

    print(
        f"Normalized         : "
        f"{plan['normalized_query']}"
    )

    print(
        f"Intent             : "
        f"{plan['intent']}"
    )

    print(
        f"Intent score       : "
        f"{plan['intent_score']:.4f}"
    )

    print(
        f"Procedure          : "
        f"{plan['procedure']}"
    )

    print(
        f"Procedure ID       : "
        f"{plan['procedure_id']}"
    )

    print(
        f"Procedure score    : "
        f"{plan['procedure_score']:.4f}"
    )

    print(
        f"Domain             : "
        f"{plan['domain']}"
    )

    print(
        f"Should retrieve    : "
        f"{plan['should_retrieve']}"
    )

    print(
        f"Status             : "
        f"{plan['status']}"
    )


# =============================================================================
# TEST QUERIES
# =============================================================================

TEST_QUERIES = [
    "Đăng ký khai sinh cần những giấy tờ gì?",
    "Đk kết hôn cần chuẩn bị hồ sơ gì?",
    "Đăng ký kết hôn không?",
    "khai sinh ở đâu",
    "đăng kí khai sinh nộp ở đâu",
    "làm giấy khai sinh mất bao lâu?",
    "đăng ký khai sinh hết bao nhiêu tiền?",
    "căn cứ pháp lý của thủ tục đăng ký khai sinh",
    "hồ sơ gồm những gì",
    "ở đâu",
    "Đăng ký khai tử cần giấy tờ gì?",
    "đăng ký kết hôn có yếu tố nước ngoài cần hồ sơ gì?",
    "thủ tục xác nhận tình trạng hôn nhân là gì?",
    "đăng ký khai sinh",
    "Đk khai sinh cần gì?",

    # Extra semantic tests
    "làm giấy khai sinh cho con",
    "kết hôn với người nước ngoài",
    "xin cấp lại giấy khai sinh",
    "đăng ký hộ khẩu",
    "nhập hộ khẩu",
    "làm lại giấy khai sinh",
]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - SEMANTIC PLANNER / PHASE 5")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print()
    print("Loading canonical dataset...")

    procedures_df = load_procedures()

    print(
        f"Procedures loaded: "
        f"{len(procedures_df)}"
    )

    print(
        f"Unique procedure names: "
        f"{procedures_df['procedure_name'].nunique()}"
    )

    # -------------------------------------------------------------------------
    # Load domain profile
    # -------------------------------------------------------------------------

    profile = load_domain_profile()

    print(
        f"Domain profile loaded: "
        f"{profile.get('domain', 'unknown')}"
    )

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SEMANTIC PLANNER TEST")
    print("=" * 80)

    for query in TEST_QUERIES:

        plan = build_plan(
            query,
            procedures_df,
        )

        print_plan(plan)

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 5 TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()