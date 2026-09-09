from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_URL = str(PROJECT_ROOT / "qdrant_data")

COLLECTION_NAME = "vietnamese_administrative_procedures"

# Số lượng candidate lấy từ Qdrant
RETRIEVAL_K = 20

# Số lượng kết quả cuối cùng đưa sang LLM
FINAL_K = 5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 80)
    print("LOADING BKAI EMBEDDING MODEL")
    print("=" * 80)

    print(f"Model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("[OK] BKAI model loaded")

    # get_embedding_dimension() thay cho
    # get_sentence_embedding_dimension()
    # để tránh FutureWarning

    print(
        f"Embedding dimension: "
        f"{model.get_embedding_dimension()}"
    )

    return model


# ============================================================
# CONNECT QDRANT
# ============================================================

def connect_qdrant():

    print()
    print("=" * 80)
    print("CONNECTING TO QDRANT")
    print("=" * 80)

    client = QdrantClient(path=QDRANT_URL)

    print("[OK] Qdrant connected")

    collections = client.get_collections()

    collection_names = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME not in collection_names:

        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found."
        )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    return client


# ============================================================
# EMBED QUERY
# ============================================================

def embed_query(
    model,
    query
):

    vector = model.encode(
        query,
        normalize_embeddings=True
    )

    return vector.tolist()


# ============================================================
# GLOBAL SEARCH
# ============================================================

def search_qdrant(
    client,
    query_vector,
    top_k=RETRIEVAL_K
):

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=top_k,

        with_payload=True

    )

    return results.points


# ============================================================
# QUERY INTENT
# ============================================================

def detect_intent(query):

    query_lower = query.lower().strip()


    # ========================================================
    # LOCATION
    # ========================================================
    #
    # LOCATION phải kiểm tra trước required_documents
    #
    # Ví dụ:
    #
    # "nộp hồ sơ ở đâu?"
    #
    # chứa cả "hồ sơ" và "ở đâu".
    #
    # Intent đúng là LOCATION.
    # ========================================================

    location_keywords = [

        "nộp hồ sơ ở đâu",

        "nộp ở đâu",

        "ở đâu",

        "địa điểm",

        "địa chỉ",

        "cơ quan nào",

        "nơi nào",

        "nơi tiếp nhận",

        "tiếp nhận hồ sơ",

        "cơ quan tiếp nhận",

    ]

    if any(
        keyword in query_lower
        for keyword in location_keywords
    ):

        return "location"


    # ========================================================
    # PROCESSING TIME
    # ========================================================

    processing_time_keywords = [

        "mất bao lâu",

        "bao lâu",

        "thời gian",

        "thời hạn",

        "bao nhiêu ngày",

        "trong bao nhiêu ngày",

        "khi nào có kết quả",

        "bao giờ có kết quả",

    ]

    if any(
        keyword in query_lower
        for keyword in processing_time_keywords
    ):

        return "processing_time"


    # ========================================================
    # FEE
    # ========================================================

    fee_keywords = [

        "lệ phí",

        "phí",

        "mất phí",

        "có mất tiền",

        "bao nhiêu tiền",

        "chi phí",

        "thu phí",

        "không thu phí",

    ]

    if any(
        keyword in query_lower
        for keyword in fee_keywords
    ):

        return "fee"


    # ========================================================
    # REQUIRED DOCUMENTS
    # ========================================================

    required_documents_keywords = [

        "giấy tờ",

        "hồ sơ",

        "chuẩn bị",

        "thành phần hồ sơ",

        "cần những gì",

        "cần giấy tờ gì",

        "hồ sơ gồm",

        "hồ sơ cần",

        "cần chuẩn bị",

    ]

    if any(
        keyword in query_lower
        for keyword in required_documents_keywords
    ):

        return "required_documents"


    # ========================================================
    # PROCEDURE
    # ========================================================

    procedure_keywords = [

        "trình tự",

        "các bước",

        "thực hiện như thế nào",

        "quy trình",

        "cách thực hiện",

        "cách thức thực hiện",

    ]

    if any(
        keyword in query_lower
        for keyword in procedure_keywords
    ):

        return "procedure"


    # ========================================================
    # GENERAL
    # ========================================================

    return "general"


# ============================================================
# PROCEDURE DETECTION
# ============================================================

def detect_procedure(query):

    query_lower = query.lower().strip()


    # --------------------------------------------------------
    # Các procedure hiện có trong dataset
    # --------------------------------------------------------

    procedures = {

        "khai sinh":
            "Thủ tục đăng ký khai sinh",

        "đăng ký khai sinh":
            "Thủ tục đăng ký khai sinh",

        "khai tử":
            "Thủ tục đăng ký khai tử",

        "đăng ký khai tử":
            "Thủ tục đăng ký khai tử",

        "kết hôn":
            "Thủ tục đăng ký kết hôn",

        "đăng ký kết hôn":
            "Thủ tục đăng ký kết hôn",

        "tình trạng hôn nhân":
            "Thủ tục xác nhận tình trạng hôn nhân",

        "xác nhận tình trạng hôn nhân":
            "Thủ tục xác nhận tình trạng hôn nhân",
    }


    # Match cụm dài trước
    # để tránh match sai

    sorted_procedures = sorted(
        procedures.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )


    for keyword, procedure_name in sorted_procedures:

        if keyword in query_lower:

            return procedure_name


    return None


# ============================================================
# TEXT INTENT CUES
# ============================================================
#
# Không dùng các từ chung như:
#
#   "giấy tờ"
#   "hồ sơ"
#   "hộ chiếu"
#   "căn cước"
#
# vì chúng xuất hiện ở rất nhiều procedure.
#
# Thay vào đó dùng section label đặc trưng.
# ============================================================

INTENT_TEXT_CUES = {

    "location": [

        (
            "địa điểm tiếp nhận hồ sơ",
            1.00
        ),

        (
            "địa điểm tiếp nhận",
            0.95
        ),

        (
            "nơi tiếp nhận hồ sơ",
            0.90
        ),

        (
            "cơ quan tiếp nhận",
            0.90
        ),

        (
            "tiếp nhận hồ sơ",
            0.85
        ),

    ],


    "processing_time": [

        (
            "thời gian giải quyết:",
            1.00
        ),

        (
            "thời gian giải quyết",
            0.95
        ),

        (
            "thời hạn giải quyết:",
            0.95
        ),

        (
            "thời hạn giải quyết",
            0.90
        ),

        (
            "ngày làm việc",
            0.70
        ),

    ],


    "fee": [

        (
            "lệ phí:",
            1.00
        ),

        (
            "lệ phí",
            0.95
        ),

        (
            "không thu phí",
            0.90
        ),

        (
            "thu phí",
            0.80
        ),

        (
            "đồng/bản",
            0.75
        ),

    ],


    "required_documents": [

        (
            "thành phần hồ sơ:",
            1.00
        ),

        (
            "thành phần hồ sơ",
            0.95
        ),

        (
            "hồ sơ gồm:",
            0.90
        ),

        (
            "hồ sơ gồm",
            0.85
        ),

    ],


    "procedure": [

        (
            "trình tự thực hiện:",
            1.00
        ),

        (
            "trình tự thực hiện",
            0.95
        ),

        (
            "các bước thực hiện",
            0.90
        ),

        (
            "quy trình thực hiện",
            0.90
        ),

        (
            "cách thức thực hiện",
            0.85
        ),

    ]

}


# ============================================================
# TEXT INTENT SCORE
# ============================================================

def calculate_text_intent_score(
    text,
    intent
):

    if not text:

        return 0.0


    if intent not in INTENT_TEXT_CUES:

        return 0.0


    text_lower = text.lower()


    matched_scores = []


    for cue, weight in INTENT_TEXT_CUES[intent]:

        if cue in text_lower:

            matched_scores.append(
                weight
            )


    if not matched_scores:

        return 0.0


    # Lấy cue mạnh nhất
    #
    # Không cộng tất cả cue để tránh
    # một chunk được cộng điểm quá mức.

    return max(
        matched_scores
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def calculate_keyword_score(
    query,
    text
):

    if not text:

        return 0.0


    query_lower = query.lower()

    text_lower = text.lower()


    query_words = set(
        query_lower.split()
    )


    if not query_words:

        return 0.0


    matched_words = sum(

        1

        for word in query_words

        if len(word) > 2
        and word in text_lower

    )


    return (
        matched_words
        /
        max(len(query_words), 1)
    )


# ============================================================
# PROCEDURE SCORE
# ============================================================

def calculate_procedure_score(
    target_procedure,
    procedure_name
):

    if not target_procedure:

        return 0.0


    if not procedure_name:

        return 0.0


    target = target_procedure.lower()

    current = procedure_name.lower()


    # Exact match

    if current == target:

        return 1.0


    # Substring match

    if target in current:

        return 0.80


    if current in target:

        return 0.70


    return 0.0


# ============================================================
# CHUNK SCORE
# ============================================================

def calculate_chunk_score(
    intent,
    chunk_type
):

    if not chunk_type:

        return 0.0


    # --------------------------------------------------------
    # Required documents
    # --------------------------------------------------------

    if intent == "required_documents":

        if chunk_type == "required_documents":

            return 1.0

        return 0.0


    # --------------------------------------------------------
    # General information
    # --------------------------------------------------------

    if intent in [

        "processing_time",

        "fee",

        "location"

    ]:

        if chunk_type == "general_information":

            return 1.0

        return 0.0


    # --------------------------------------------------------
    # Procedure
    # --------------------------------------------------------

    if intent == "procedure":

        if chunk_type == "procedure":

            return 1.0

        if chunk_type == "general_information":

            return 0.80

        return 0.0


    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    return 0.0


# ============================================================
# RERANK
# ============================================================

def rerank_results(
    query,
    results,
    final_k=FINAL_K
):

    intent = detect_intent(
        query
    )

    target_procedure = detect_procedure(
        query
    )


    print()
    print("=" * 80)
    print("RERANKING INFORMATION")
    print("=" * 80)

    print(
        f"Detected intent     : {intent}"
    )

    print(
        f"Detected procedure  : "
        f"{target_procedure}"
    )


    scored_results = []


    for result in results:

        payload = result.payload or {}


        text = payload.get(
            "text",
            ""
        )


        procedure_name = payload.get(
            "procedure_name",
            ""
        )


        chunk_type = payload.get(
            "chunk_type",
            ""
        )


        # ====================================================
        # 1. VECTOR SCORE
        # ====================================================

        vector_score = float(
            result.score
        )


        # ====================================================
        # 2. PROCEDURE SCORE
        # ====================================================

        procedure_score = calculate_procedure_score(

            target_procedure,

            procedure_name

        )


        # ====================================================
        # 3. CHUNK SCORE
        # ====================================================

        chunk_score = calculate_chunk_score(

            intent,

            chunk_type

        )


        # ====================================================
        # 4. KEYWORD SCORE
        # ====================================================

        keyword_score = calculate_keyword_score(

            query,

            text

        )


        # ====================================================
        # 5. TEXT INTENT SCORE
        # ====================================================

        text_intent_score = calculate_text_intent_score(

            text,

            intent

        )


        # ====================================================
        # 6. FINAL SCORE
        # ====================================================
        #
        # Vector       = 50%
        # Procedure    = 25%
        # Chunk type   = 15%
        # Keyword      = 5%
        # Text intent  = 5%
        #
        # ====================================================

        final_score = (

            0.50 * vector_score

            +

            0.25 * procedure_score

            +

            0.15 * chunk_score

            +

            0.05 * keyword_score

            +

            0.05 * text_intent_score

        )


        scored_results.append(

            {

                "result": result,

                "vector_score":
                    vector_score,

                "procedure_score":
                    procedure_score,

                "chunk_score":
                    chunk_score,

                "keyword_score":
                    keyword_score,

                "text_intent_score":
                    text_intent_score,

                "final_score":
                    final_score

            }

        )


    # ========================================================
    # SORT
    # ========================================================

    scored_results.sort(

        key=lambda item:
            item["final_score"],

        reverse=True

    )


    return scored_results[:final_k]


# ============================================================
# PROCEDURE FILTER
# ============================================================
#
# Đây là phần mới quan trọng.
#
# Nếu query đã xác định được procedure:
#
# "Đăng ký khai sinh..."
#
# thì ưu tiên CHỈ những chunk của:
#
# "Thủ tục đăng ký khai sinh"
#
# ============================================================

def filter_by_procedure(
    results,
    target_procedure
):

    if not target_procedure:

        return results


    target = target_procedure.lower()


    procedure_results = []


    for result in results:

        payload = result.payload or {}


        procedure_name = payload.get(
            "procedure_name",
            ""
        )


        if not procedure_name:

            continue


        current = procedure_name.lower()


        if current == target:

            procedure_results.append(
                result
            )


    return procedure_results


# ============================================================
# FILTER BY INTENT
# ============================================================
#
# Sau procedure filter, tiếp tục ưu tiên
# chunk phù hợp với câu hỏi.
#
# ============================================================

def filter_by_intent(
    results,
    intent
):

    if not results:

        return results


    if intent == "required_documents":

        filtered = [

            result

            for result in results

            if (
                result.payload or {}
            ).get(
                "chunk_type"
            )
            ==
            "required_documents"

        ]

        return filtered


    if intent in [

        "processing_time",

        "fee",

        "location"

    ]:

        filtered = [

            result

            for result in results

            if (
                result.payload or {}
            ).get(
                "chunk_type"
            )
            ==
            "general_information"

        ]

        return filtered


    if intent == "procedure":

        filtered = [

            result

            for result in results

            if (
                result.payload or {}
            ).get(
                "chunk_type"
            )
            in [

                "procedure",

                "general_information"

            ]

        ]

        return filtered


    return results


# ============================================================
# SMART CANDIDATE SELECTION
# ============================================================

def select_candidates(
    candidates,
    query,
    final_k=FINAL_K
):

    intent = detect_intent(
        query
    )

    target_procedure = detect_procedure(
        query
    )


    # ========================================================
    # CASE 1:
    # Có procedure + intent
    # ========================================================

    if target_procedure:

        # ----------------------------------------------------
        # Bước 1: Procedure filter
        # ----------------------------------------------------

        procedure_candidates = filter_by_procedure(

            candidates,

            target_procedure

        )


        # ----------------------------------------------------
        # Bước 2: Intent filter
        # ----------------------------------------------------

        intent_candidates = filter_by_intent(

            procedure_candidates,

            intent

        )


        # ----------------------------------------------------
        # Nếu đủ candidate
        # ----------------------------------------------------

        if len(intent_candidates) >= final_k:

            return intent_candidates


        # ----------------------------------------------------
        # Nếu intent filter quá chặt,
        # quay lại procedure candidates
        # ----------------------------------------------------

        if len(procedure_candidates) >= final_k:

            return procedure_candidates


        # ----------------------------------------------------
        # Nếu vẫn không đủ,
        # dùng candidate ban đầu làm fallback.
        # ----------------------------------------------------

        return procedure_candidates or candidates


    # ========================================================
    # CASE 2:
    # Không detect được procedure
    # ========================================================

    intent_candidates = filter_by_intent(

        candidates,

        intent

    )


    if len(intent_candidates) >= final_k:

        return intent_candidates


    return candidates


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 80)

    print()

    print("Query:")

    print(query)

    print()


    print(
        f"Top {len(results)} results:"
    )


    for rank, item in enumerate(

        results,

        start=1

    ):

        result = item["result"]

        payload = result.payload or {}


        print()

        print("-" * 80)


        print(
            f"Rank             : {rank}"
        )


        print(
            f"Final score      : "
            f"{item['final_score']:.4f}"
        )


        print(
            f"Vector score     : "
            f"{item['vector_score']:.4f}"
        )


        print(
            f"Procedure score  : "
            f"{item['procedure_score']:.4f}"
        )


        print(
            f"Chunk score      : "
            f"{item['chunk_score']:.4f}"
        )


        print(
            f"Keyword score    : "
            f"{item['keyword_score']:.4f}"
        )


        print(
            f"Text intent      : "
            f"{item['text_intent_score']:.4f}"
        )


        print(
            f"Chunk ID         : "
            f"{payload.get('chunk_id', '')}"
        )


        print(
            f"Document ID      : "
            f"{payload.get('document_id', '')}"
        )


        print(
            f"Procedure        : "
            f"{payload.get('procedure_name', '')}"
        )


        print(
            f"Field            : "
            f"{payload.get('field', '')}"
        )


        print(
            f"Submission       : "
            f"{payload.get('submission_method', '')}"
        )


        print(
            f"Chunk type       : "
            f"{payload.get('chunk_type', '')}"
        )


        print()

        print("TEXT:")

        print(
            payload.get(
                "text",
                ""
            )
        )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query,
    model,
    client,
    retrieval_k=RETRIEVAL_K,
    final_k=FINAL_K
):

    # ========================================================
    # 1. Embed query
    # ========================================================

    query_vector = embed_query(

        model,

        query

    )


    # ========================================================
    # 2. Qdrant Global Retrieval
    #
    # Lấy Top-20 candidate
    # ========================================================

    candidates = search_qdrant(

        client,

        query_vector,

        retrieval_k

    )


    # ========================================================
    # 3. Candidate Filtering
    # ========================================================

    selected_candidates = select_candidates(

        candidates,

        query,

        final_k

    )


    # ========================================================
    # 4. Reranking
    # ========================================================

    final_results = rerank_results(

        query,

        selected_candidates,

        final_k

    )


    return final_results


# ============================================================
# TEST DATA
# ============================================================

TEST_QUESTIONS = [

    # --------------------------------------------------------
    # Required documents
    # --------------------------------------------------------

    "Đăng ký khai sinh cần những giấy tờ gì?",

    "Đăng ký kết hôn cần chuẩn bị hồ sơ gì?",


    # --------------------------------------------------------
    # Processing time
    # --------------------------------------------------------

    "Thủ tục xác nhận tình trạng hôn nhân mất bao lâu?",


    # --------------------------------------------------------
    # Fee
    # --------------------------------------------------------

    "Đăng ký khai tử có mất lệ phí không?",


    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    "Tôi muốn đăng ký khai sinh thì nộp hồ sơ ở đâu?",


    # --------------------------------------------------------
    # Additional tests
    # --------------------------------------------------------

    "Đăng ký khai tử nộp hồ sơ ở đâu?",

    "Đăng ký kết hôn mất bao lâu?",

    "Đăng ký khai sinh có mất lệ phí không?",

    "Xác nhận tình trạng hôn nhân cần giấy tờ gì?",

    "Đăng ký kết hôn thực hiện như thế nào?",

]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VIETNAMESE ADMINISTRATIVE PROCEDURES"
    )

    print(
        "LEGAL RAG RETRIEVER"
    )

    print("=" * 80)


    # ========================================================
    # Load model
    # ========================================================

    model = load_model()


    # ========================================================
    # Connect Qdrant
    # ========================================================

    client = connect_qdrant()


    # ========================================================
    # Run tests
    # ========================================================

    for question in TEST_QUESTIONS:

        results = retrieve(

            question,

            model,

            client,

            RETRIEVAL_K,

            FINAL_K

        )


        display_results(

            question,

            results

        )


    # ========================================================
    # DONE
    # ========================================================

    print()

    print("=" * 80)

    print(
        "RETRIEVAL TEST COMPLETED"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
