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

PLANNER_VERSION = "v5.1-2026-10-05 (multi-intent + variant-guard + diacritics + context flags)"

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
        "mang theo gì",
        "cần mang gì",
        "mang những gì",
        "mang theo những gì",
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
        "mất tiền",
        "tốn tiền",
        "có mất phí",
        "miễn phí",
        "có thu phí",
        "phải trả bao nhiêu",
        "nộp bao nhiêu tiền",
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
        "cơ quan nào",
        "nơi nào",
        "chỗ nào",
        "đi đâu",
        "nộp ở cơ quan nào",
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
        "mấy ngày",
        "bao nhiêu ngày",
        "mất bao nhiêu ngày",
        "bao lâu thì xong",
        "khi nào xong",
        "bao giờ xong",
        "nhanh không",
        "bao lâu",
        "mất bao lâu",
        "thời gian",
        "thời gian giải quyết",
        "giải quyết trong bao lâu",
        "khi nào có",
        "bao giờ có",
    ],

    "legal_basis": [
        "luật nào",
        "nghị định nào",
        "thông tư nào",
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
                        0.60 + token_count * 0.10,
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
# DIACRITIC RESTORATION (câu gõ không dấu)
# =============================================================================
#
# "dang ky khai sinh het bao nhieu tien" -> "đăng ký khai sinh hết bao nhiêu tiền"
#
# Chỉ áp dụng khi cả câu KHÔNG có dấu nào. Từ điển lấy từ chính dữ liệu:
# tên thủ tục (ưu tiên cao) rồi từ khóa intent, alias, từ chung.
# Từ nào có nhiều cách khôi phục không rõ ràng thì giữ nguyên.
# =============================================================================

def fold_accents(text: str) -> str:

    text = str(text).replace("đ", "d").replace("Đ", "D")

    decomposed = unicodedata.normalize("NFD", text)

    return "".join(
        char for char in decomposed
        if unicodedata.category(char) != "Mn"
    )


def is_unaccented(text: str) -> bool:
    """Câu thuần ASCII (có chữ cái) = gõ không dấu."""

    text = str(text or "")

    return (
        any(char.isalpha() for char in text)
        and all(ord(char) < 128 for char in text)
    )


# Từ chức năng hay gặp nhưng không nằm trong tên thủ tục/từ khóa
EXTRA_WORDS = [
    "và", "còn", "với", "cần", "gì", "giấy", "tờ", "nào", "bao", "lâu",
    "đâu", "ở", "cho", "không", "có", "được", "phải", "hết", "mất", "mấy",
    "ngày", "tiền", "phí", "lệ", "hồ", "sơ", "nộp", "thủ", "tục", "làm",
    "xin", "của", "là", "những", "các", "một", "đồng", "thời", "cũng", "như",
]

# Từ đơn mơ hồ: chọn nghĩa phổ biến nhất khi đứng riêng
PREFERRED_WORDS = {
    "can": "cần", "va": "và", "gi": "gì", "o": "ở", "la": "là",
    "co": "có", "khong": "không", "giay": "giấy", "to": "tờ",
}


def _build_accent_vocabulary(
    procedures_df: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """
    Trả về (từ trong tên thủ tục, từ khác, cụm từ đã biết).
    Cụm từ: khóa là dạng bỏ dấu, giá trị là dạng có dấu.
    """

    primary: dict[str, set[str]] = {}
    secondary: dict[str, set[str]] = {}
    phrases: dict[str, str] = {}

    def words_of(text: str) -> list[str]:
        return re.findall(
            r"\w+",
            unicodedata.normalize("NFC", str(text).lower()),
        )

    def add_words(target: dict[str, set[str]], text: str) -> None:
        for word in words_of(text):
            target.setdefault(fold_accents(word), set()).add(word)

    def add_phrase(text: str) -> None:
        words = words_of(text)
        if len(words) >= 2:
            key = " ".join(fold_accents(w) for w in words)
            phrases.setdefault(key, " ".join(words))

    for name in procedures_df["procedure_name"]:
        name = str(name)
        add_words(primary, name)
        add_phrase(name)
        lowered = name.lower()
        if lowered.startswith("thủ tục "):
            add_phrase(lowered[len("thủ tục "):])

    # Cụm 2-4 từ cắt từ tên thủ tục: cho ngữ cảnh khi người dùng chỉ gõ một
    # phần tên ("yeu to nuoc ngoai" -> "yếu tố nước ngoài")
    ngrams: dict[str, set[str]] = {}

    for name in procedures_df["procedure_name"]:

        words = words_of(str(name))

        for size in (2, 3, 4):
            for start in range(len(words) - size + 1):
                segment = words[start:start + size]
                key = " ".join(fold_accents(w) for w in segment)
                ngrams.setdefault(key, set()).add(" ".join(segment))

    for key, originals in ngrams.items():
        if len(originals) == 1:
            phrases.setdefault(key, next(iter(originals)))

    sources: list[str] = []

    for keywords in INTENTS.values():
        sources.extend(keywords)

    for canonical, aliases in PROCEDURE_ALIASES.items():
        sources.append(canonical)
        sources.extend(aliases)

    sources.extend(GENERIC_WORDS)
    sources.extend(FILLER_WORDS)
    sources.extend(QUALIFIER_PHRASES)
    sources.extend(ABBREVIATIONS.values())
    sources.extend(EXTRA_WORDS)

    for text in sources:
        add_words(secondary, text)
        add_phrase(text)

    return primary, secondary, phrases


def restore_diacritics(
    query: str,
    procedures_df: pd.DataFrame,
) -> str:
    """
    Khôi phục dấu cho câu gõ không dấu. Dấu câu được giữ nguyên.
    1) khớp cụm từ đã biết (dài trước), 2) khớp từng từ không mơ hồ.
    """

    if not is_unaccented(query):
        return query

    primary, secondary, phrases = _build_accent_vocabulary(procedures_df)

    parts = re.findall(r"\w+|\W+", str(query).lower())

    word_positions = [
        index for index, part in enumerate(parts)
        if re.match(r"\w", part)
    ]

    max_phrase_len = max(
        (len(key.split()) for key in phrases),
        default=2,
    )

    cursor = 0

    while cursor < len(word_positions):

        restored = False

        longest = min(max_phrase_len, len(word_positions) - cursor)

        for length in range(longest, 1, -1):

            positions = word_positions[cursor:cursor + length]

            # Chỉ ghép các từ cách nhau bằng khoảng trắng (không qua dấu phẩy)
            if any(
                parts[positions[k] + 1].strip()
                for k in range(length - 1)
            ):
                continue

            key = " ".join(
                fold_accents(parts[position]) for position in positions
            )

            replacement = phrases.get(key)

            if replacement:

                for position, word in zip(positions, replacement.split()):
                    parts[position] = word

                cursor += length
                restored = True
                break

        if restored:
            continue

        position = word_positions[cursor]
        key = parts[position]

        in_names = primary.get(key)

        if in_names and len(in_names) == 1:
            parts[position] = next(iter(in_names))

        elif key in PREFERRED_WORDS:
            parts[position] = PREFERRED_WORDS[key]

        elif not in_names:
            others = secondary.get(key)

            if others and len(others) == 1:
                parts[position] = next(iter(others))

        cursor += 1

    return "".join(parts)


# =============================================================================
# MULTI-INTENT: CLAUSE SPLITTING
# =============================================================================
#
# "Đăng ký khai sinh cần giấy tờ gì và kết hôn có mất lệ phí không?"
#   -> clause 1: "đăng ký khai sinh cần giấy tờ gì"
#   -> clause 2: "kết hôn có mất lệ phí không"
#
# Mỗi clause tự nhận diện thủ tục + intent. Clause thiếu thủ tục hoặc thiếu
# intent sẽ mượn từ clause gần nhất.
# =============================================================================

CLAUSE_PUNCTUATION = re.compile(r"[,;?!\n]+")
CONJUNCTIONS = re.compile(r"\b(?:và|còn|với lại|đồng thời|cũng như)\b")


def split_clauses(
    query: str,
    procedures_df: pd.DataFrame,
) -> list[str]:
    """
    Tách câu hỏi thành các mệnh đề.
    Tên thủ tục có dấu phẩy hoặc chữ "và" bên trong được giữ nguyên.
    """

    text = unicodedata.normalize("NFC", str(query or "")).lower()

    # Bảo vệ tên thủ tục chứa dấu phân tách
    placeholders: dict[str, str] = {}

    for index, name in enumerate(procedures_df["procedure_name"]):

        core = unicodedata.normalize("NFC", str(name)).lower().strip()

        if core.startswith("thủ tục "):
            core = core[len("thủ tục "):]

        if not re.search(r"[,;]| và ", core):
            continue

        if core in text:
            key = f"zzprot{index}zz"
            placeholders[key] = core
            text = text.replace(core, key)

    pieces: list[str] = []

    for part in CLAUSE_PUNCTUATION.split(text):
        pieces.extend(CONJUNCTIONS.split(part))

    clauses = []

    for piece in pieces:

        for key, core in placeholders.items():
            piece = piece.replace(key, core)

        piece = piece.strip()

        if len(normalize_text(piece)) >= 2:
            clauses.append(piece)

    return clauses


# =============================================================================
# MULTI-INTENT: DETECT ALL INTENTS IN ONE CLAUSE
# =============================================================================

# "hồ sơ" sau các động từ này là HÀNH ĐỘNG ("nộp hồ sơ ở đâu"),
# không phải câu hỏi về thành phần hồ sơ.
WEAK_DOCUMENT_KEYWORDS = {"hồ sơ"}
ACTION_VERBS_BEFORE_DOCUMENT = {"nộp", "nhận", "tiếp", "gửi", "trả"}


def detect_intents(clause_normalized: str) -> list[tuple[str, float]]:
    return _detect_intents_full(clause_normalized)[0]


def _detect_intents_full(
    clause_normalized: str,
) -> tuple[list[tuple[str, float]], set[str]]:
    """
    Trả về MỌI intent xuất hiện trong clause (không gồm general_information),
    theo thứ tự xuất hiện.

    Cụm dài thắng cụm ngắn khi chồng lấn, ví dụ:
        "nộp hồ sơ ở đâu"  -> location (không tính thêm required_documents)
        "mất bao nhiêu ngày" -> processing_time (không tính thêm fee)
    """

    padded = f" {clause_normalized} "

    matches = []

    for intent, keywords in INTENTS.items():

        if intent == "general_information":
            continue

        for keyword in keywords:

            kw = normalize_text(keyword)

            if not kw:
                continue

            needle = f" {kw} "
            position = padded.find(needle)

            while position != -1:

                start = position + 1
                end = start + len(kw)

                if (
                    intent == "required_documents"
                    and kw in WEAK_DOCUMENT_KEYWORDS
                ):

                    previous = padded[:start].split()[-1:]

                    if previous and previous[0] in ACTION_VERBS_BEFORE_DOCUMENT:
                        position = padded.find(needle, position + 1)
                        continue

                matches.append(
                    (
                        intent,
                        start,
                        end,
                        len(kw.split()),
                    )
                )

                position = padded.find(needle, position + 1)

    # Cụm nhiều token trước, rồi cụm dài ký tự trước
    matches.sort(
        key=lambda m: (m[3], m[2] - m[1]),
        reverse=True,
    )

    accepted: list[tuple[str, int, int, int]] = []

    for intent, start, end, tokens in matches:

        overlaps_other_intent = any(
            other_intent != intent
            and not (end <= other_start or start >= other_end)
            for other_intent, other_start, other_end, _ in accepted
        )

        if overlaps_other_intent:
            continue

        accepted.append((intent, start, end, tokens))

    scores: dict[str, float] = {}
    first_position: dict[str, int] = {}

    for intent, start, _, tokens in accepted:

        scores[intent] = max(
            scores.get(intent, 0.0),
            min(1.0, 0.60 + 0.10 * tokens),
        )

        first_position[intent] = min(
            first_position.get(intent, start),
            start,
        )

    ordered = sorted(
        scores,
        key=lambda i: first_position[i],
    )

    consumed: set[str] = set()

    for _, start, end, _ in accepted:
        consumed.update(padded[start:end].split())

    return (
        [(intent, scores[intent]) for intent in ordered],
        consumed,
    )


# =============================================================================
# MULTI-INTENT: BUILD TASKS
# =============================================================================

# Từ đệm/xã giao, không phải chủ đề của câu hỏi
FILLER_WORDS = {
    "nhé", "nha", "ạ", "dạ", "vâng", "ơi", "bạn", "em", "anh", "chị",
    "giúp", "nhờ", "với", "thì", "vậy", "là", "cảm", "ơn", "ok",
    "nhiêu", "mấy", "lắm", "chưa", "rồi",
    # từ chỉ đồng tham chiếu / dẫn dắt ("thủ tục này", "còn ... thì sao")
    "này", "đó", "ấy", "nó", "kia", "nãy", "trên", "vừa", "sao", "nhỉ",
}


def subject_tokens(normalized: str, consumed: set[str]) -> set[str]:
    """
    Token mang nội dung riêng của clause (không phải từ khóa intent,
    không phải từ chung hay xã giao). Clause có token này là đang nói về
    một chủ đề cụ thể, nên không được mượn thủ tục của clause bên cạnh.
    """
    return {
        token
        for token in meaningful_tokens(normalized)
        if token not in consumed and token not in FILLER_WORDS
    }


def _nearest_index(
    infos: list[dict],
    index: int,
    predicate,
    prefer_next: bool,
) -> int | None:

    best_index = None
    best_key = None

    for other, info in enumerate(infos):

        if other == index or not predicate(info):
            continue

        is_next = other > index

        key = (
            abs(other - index),
            0 if is_next == prefer_next else 1,
        )

        if best_key is None or key < best_key:
            best_key = key
            best_index = other

    return best_index


# Từ chỉ BIẾN THỂ của thủ tục. Câu hỏi có từ này mà tên thủ tục không có
# nghĩa là dataset không có đúng thủ tục người dùng hỏi
# (vd hỏi "cấp lại giấy khai sinh" nhưng chỉ có "đăng ký khai sinh").
QUALIFIER_PHRASES = [
    "cấp lại",
    "làm lại",
    "đăng ký lại",
    "cải chính",
    "đính chính",
    "thay đổi",
    "bổ sung",
    "điều chỉnh",
    "trích lục",
    "gia hạn",
    "thu hồi",
]


def find_qualifier_mismatch(
    clause_text: str,
    procedure_name: str | None,
) -> list[str]:

    if not procedure_name:
        return []

    clause = f" {normalize_text(clause_text)} "
    name = f" {normalize_text(procedure_name)} "

    return [
        phrase
        for phrase in QUALIFIER_PHRASES
        if f" {normalize_text(phrase)} " in clause
        and f" {normalize_text(phrase)} " not in name
    ]


def _make_task(
    clause_text: str,
    intent: str,
    intent_score: float,
    procedure: dict | None,
    intent_explicit: bool = True,
    has_subject: bool = True,
) -> dict:

    name = procedure["procedure_name"] if procedure else None

    # Clause mượn thủ tục thì ghép tên vào để tìm ngữ nghĩa có ngữ cảnh
    core_name = normalize_text(name).removeprefix("thủ tục ") if name else ""

    query_text = (
        f"{name}. {clause_text}"
        if name and core_name not in normalize_text(clause_text)
        else clause_text
    )

    mismatch = find_qualifier_mismatch(clause_text, name)

    if procedure is None:
        status = "procedure_not_detected"
    elif procedure.get("ambiguous"):
        status = "ambiguous_procedure"
    elif mismatch:
        status = "variant_mismatch"
    else:
        status = "ready"

    return {
        "query": query_text,
        "clause": clause_text,
        "normalized_query": normalize_text(query_text),
        "intent": intent,
        "intent_score": round(intent_score, 4),
        "procedure": name,
        "procedure_id": procedure["procedure_id"] if procedure else None,
        "procedure_score": (
            round(procedure["score"], 4) if procedure else 0.0
        ),
        "domain": procedure["domain"] if procedure else None,
        "candidates": (
            procedure.get("top_candidates", []) if procedure else []
        ),
        "ambiguous": (
            bool(procedure.get("ambiguous")) if procedure else False
        ),
        "qualifier_mismatch": mismatch,
        "intent_explicit": intent_explicit,
        "has_subject": has_subject,
        "should_retrieve": procedure is not None,
        "status": status,
    }


def build_tasks(
    query: str,
    procedures_df: pd.DataFrame,
) -> list[dict]:

    query = restore_diacritics(query, procedures_df)

    clauses = split_clauses(query, procedures_df)

    if not clauses:
        clauses = [str(query or "")]

    infos = []

    for clause in clauses:

        normalized = normalize_text(clause)

        intents, consumed = _detect_intents_full(normalized)

        # Không có cụm khớp chính xác: dùng bộ nhận diện mờ như cũ
        if not intents:

            fuzzy_intent, fuzzy_score = detect_intent(normalized)

            if fuzzy_intent != "general_information":
                intents = [(fuzzy_intent, fuzzy_score)]

        infos.append(
            {
                "text": clause,
                "normalized": normalized,
                "intents": intents,
                "procedure": detect_procedure(normalized, procedures_df),
                "general_score": detect_intent(normalized)[1],
                "has_subject": bool(subject_tokens(normalized, consumed)),
            }
        )

    # Bỏ clause chẳng có thủ tục lẫn intent ("cho mình hỏi", "nhé"...)
    if len(infos) > 1:

        kept = [
            info for info in infos
            if info["intents"] or info["procedure"] or info["has_subject"]
        ]

        infos = kept or infos[:1]

    # Không clause nào nhận ra thủ tục: thử lại trên cả câu
    if all(info["procedure"] is None for info in infos):

        whole = detect_procedure(
            normalize_text(query),
            procedures_df,
        )

        if whole is not None:
            for info in infos:
                if not info["has_subject"] or len(infos) == 1:
                    info["procedure"] = whole

    tasks: list[dict] = []
    seen: set[tuple] = set()

    for index, info in enumerate(infos):

        # ---- intent ----
        intents = info["intents"]
        explicit = True

        if not intents:

            donor = _nearest_index(
                infos,
                index,
                lambda x: bool(x["intents"]),
                prefer_next=True,
            )

            if donor is not None:
                intents = infos[donor]["intents"]
            else:
                intents = [
                    ("general_information", info["general_score"])
                ]
                explicit = False

        # ---- procedure ----
        procedure = info["procedure"]

        if procedure is None and not info["has_subject"]:

            donor = _nearest_index(
                infos,
                index,
                lambda x: x["procedure"] is not None,
                prefer_next=False,
            )

            if donor is not None:
                procedure = infos[donor]["procedure"]

        for intent, score in intents:

            key = (
                procedure["procedure_id"] if procedure else None,
                intent,
            )

            if key in seen:
                continue

            seen.add(key)

            tasks.append(
                _make_task(
                    info["text"],
                    intent,
                    score,
                    procedure,
                    intent_explicit=explicit,
                    has_subject=info["has_subject"],
                )
            )

    return tasks


# =============================================================================
# SEMANTIC PLAN
# =============================================================================

def build_plan(
    query: str,
    procedures_df: pd.DataFrame,
) -> dict:
    """
    Trả về plan của task đầu tiên (giữ tương thích với code cũ),
    kèm "tasks" là danh sách đầy đủ và "is_multi".
    """

    tasks = build_tasks(query, procedures_df)

    plan = dict(tasks[0])

    plan["query"] = query
    plan["normalized_query"] = normalize_text(query)
    plan["tasks"] = tasks
    plan["is_multi"] = len(tasks) > 1

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

    if plan.get("is_multi"):

        print(
            f"Tasks              : "
            f"{len(plan['tasks'])} (multi-intent)"
        )

        for number, task in enumerate(plan["tasks"], start=1):

            print(
                f"  Task {number}: "
                f"{task['intent']:<20} "
                f"{task['procedure']} "
                f"({task['procedure_id']}) "
                f"[{task['status']}]"
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

    # Multi-intent
    "Đăng ký khai sinh cần giấy tờ gì và kết hôn có mất lệ phí không?",
    "khai sinh cần giấy tờ gì và mất bao nhiêu tiền",
    "lệ phí và thời gian đăng ký khai tử",
    "khai sinh và kết hôn cần giấy tờ gì",
    "nộp hồ sơ khai sinh ở đâu",
    "cho mình hỏi, đăng ký khai sinh mất bao lâu",
]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - SEMANTIC PLANNER / PHASE 5")
    print(f"Version: {PLANNER_VERSION}")
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