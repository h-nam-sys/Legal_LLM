from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

QDRANT_URL = "http://localhost:6333"

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

    print(f"\nModel: {EMBEDDING_MODEL}")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("[OK] BKAI model loaded")

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
    print("CONNECTING TO QDRANT SERVER")
    print("=" * 80)

    print(f"\nQdrant URL: {QDRANT_URL}")

    client = QdrantClient(
        url=QDRANT_URL
    )

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

    # Kiểm tra collection có đúng 768 chiều không
    collection_info = client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print(
        f"Vectors: "
        f"{collection_info.points_count}"
    )

    return client


# ============================================================
# EMBED QUERY
# ============================================================

def embed_query(model, query):

    query = str(query).strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

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
    # LOCATION phải kiểm tra trước required_documents.
    #
    # Ví dụ:
    #
    # "Tôi muốn đăng ký khai sinh thì nộp hồ sơ ở đâu?"
    #
    # Query có cả "hồ sơ" và "ở đâu".
    # Intent đúng phải là LOCATION.
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

    # ========================================================
    # Các procedure hiện có trong dataset
    # ========================================================

    procedures = {

        "xác nhận tình trạng hôn nhân":
            "Thủ tục xác nhận tình trạng hôn nhân",

        "tình trạng hôn nhân":
            "Thủ tục xác nhận tình trạng hôn nhân",

        "đăng ký khai sinh":
            "Thủ tục đăng ký khai sinh",

        "khai sinh":
            "Thủ tục đăng ký khai sinh",

        "đăng ký khai tử":
            "Thủ tục đăng ký khai tử",

        "khai tử":
            "Thủ tục đăng ký khai tử",

        "đăng ký kết hôn":
            "Thủ tục đăng ký kết hôn",

        "kết hôn":
            "Thủ tục đăng ký kết hôn",

    }

    # Match cụm dài trước
    # để tránh match sai.

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

    # Chỉ lấy cue mạnh nhất
    # tránh cộng điểm quá mức.

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

    query_lower = query.lower().strip()

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

    # ========================================================
    # Required documents
    # ========================================================

    if intent == "required_documents":

        if chunk_type == "required_documents":
            return 1.0

        return 0.0


    # ========================================================
    # Processing time
    # ========================================================

    if intent == "processing_time":

        if chunk_type == "processing_time":
            return 1.0

        # general_information vẫn là fallback
        if chunk_type == "general_information":
            return 0.80

        return 0.0


    # ========================================================
    # Fee
    # ========================================================

    if intent == "fee":

        if chunk_type == "fee":
            return 1.0

        # general_information vẫn là fallback
        if chunk_type == "general_information":
            return 0.80

        return 0.0


    # ========================================================
    # Location
    # ========================================================

    if intent == "location":

        if chunk_type == "location":
            return 1.0

        # general_information vẫn là fallback
        if chunk_type == "general_information":
            return 0.80

        return 0.0


    # ========================================================
    # Procedure
    # ========================================================

    if intent == "procedure":

        if chunk_type == "procedure":
            return 1.0

        if chunk_type == "general_information":
            return 0.80

        return 0.0


    # ========================================================
    # General
    # ========================================================

    return 0.0


# ============================================================
# PROCEDURE FILTER
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

def filter_by_intent(
    results,
    intent
):

    if not results:
        return results

    # ========================================================
    # Mapping intent → chunk_type
    #
    # Đây là phần quan trọng đã sửa.
    #
    # Dataset mới có:
    #
    # required_documents
    # processing_time
    # fee
    # location
    # notes
    # form_link
    # general_information
    # ========================================================

    intent_to_chunk_type = {

        "required_documents":
            "required_documents",

        "processing_time":
            "processing_time",

        "fee":
            "fee",

        "location":
            "location",

    }

    # ========================================================
    # Intent có chunk_type cụ thể
    # ========================================================

    if intent in intent_to_chunk_type:

        target_chunk_type = (
            intent_to_chunk_type[intent]
        )

        filtered = [

            result

            for result in results

            if (
                result.payload or {}
            ).get("chunk_type")
            == target_chunk_type

        ]

        return filtered


    # ========================================================
    # Procedure
    #
    # Dataset hiện tại chưa tạo procedure chunk riêng.
    #
    # Vì vậy dùng general_information làm fallback.
    # ========================================================

    if intent == "procedure":

        filtered = [

            result

            for result in results

            if (
                result.payload or {}
            ).get("chunk_type")
            in [
                "procedure",
                "general_information"
            ]

        ]

        return filtered


    # ========================================================
    # General
    # ========================================================

    return results


# ============================================================
# SMART CANDIDATE SELECTION
# ============================================================

def select_candidates(
    candidates,
    query,
    final_k=FINAL_K
):

    intent = detect_intent(query)

    target_procedure = detect_procedure(query)

    print()
    print("=" * 80)
    print("CANDIDATE SELECTION")
    print("=" * 80)

    print(
        f"Detected intent    : {intent}"
    )

    print(
        f"Detected procedure : "
        f"{target_procedure}"
    )

    # ========================================================
    # CASE 1:
    # Có procedure
    # ========================================================

    if target_procedure:

        # ----------------------------------------------------
        # Bước 1: Procedure filter
        # ----------------------------------------------------

        procedure_candidates = filter_by_procedure(

            candidates,

            target_procedure

        )

        print(
            f"Procedure candidates: "
            f"{len(procedure_candidates)}"
        )

        # ----------------------------------------------------
        # Bước 2: Intent filter
        # ----------------------------------------------------

        intent_candidates = filter_by_intent(

            procedure_candidates,

            intent

        )

        print(
            f"Intent candidates    : "
            f"{len(intent_candidates)}"
        )

        # ----------------------------------------------------
        # Nếu tìm thấy ít nhất 1 candidate đúng intent
        #
        # KHÔNG yêu cầu phải đủ FINAL_K.
        #
        # Ví dụ:
        #
        # "Đăng ký khai sinh mất bao lâu?"
        #
        # chỉ có 1 processing_time chunk.
        #
        # Vẫn phải lấy chunk đó.
        # ----------------------------------------------------

        if intent_candidates:

            return intent_candidates[:final_k]

        # ----------------------------------------------------
        # Nếu không tìm thấy đúng intent
        # fallback về procedure
        # ----------------------------------------------------

        if procedure_candidates:

            return procedure_candidates[:final_k]

        # ----------------------------------------------------
        # Fallback cuối cùng
        # ----------------------------------------------------

        return candidates[:final_k]


    # ========================================================
    # CASE 2:
    # Không detect được procedure
    # ========================================================

    intent_candidates = filter_by_intent(

        candidates,

        intent

    )

    print(
        f"Intent candidates: "
        f"{len(intent_candidates)}"
    )

    if intent_candidates:

        return intent_candidates[:final_k]

    return candidates[:final_k]


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
        f"Detected intent     : "
        f"{intent}"
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

        procedure_score = (
            calculate_procedure_score(

                target_procedure,

                procedure_name

            )
        )

        # ====================================================
        # 3. CHUNK SCORE
        # ====================================================

        chunk_score = (
            calculate_chunk_score(

                intent,

                chunk_type

            )
        )

        # ====================================================
        # 4. KEYWORD SCORE
        # ====================================================

        keyword_score = (
            calculate_keyword_score(

                query,

                text

            )
        )

        # ====================================================
        # 5. TEXT INTENT SCORE
        # ====================================================

        text_intent_score = (
            calculate_text_intent_score(

                text,

                intent

            )
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
    # Lấy Top-20 candidate.
    # ========================================================

    candidates = search_qdrant(

        client,

        query_vector,

        retrieval_k

    )

    print(
        f"\nQdrant candidates: "
        f"{len(candidates)}"
    )

    # ========================================================
    # 3. Candidate Filtering
    # ========================================================

    selected_candidates = select_candidates(

        candidates,

        query,

        final_k

    )

    print(
        f"Selected candidates: "
        f"{len(selected_candidates)}"
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

    model = None
    client = None

    try:

        # ====================================================
        # 1. Load model
        # ====================================================

        model = load_model()

        # ====================================================
        # 2. Connect Qdrant
        # ====================================================

        client = connect_qdrant()

        # ====================================================
        # 3. Run tests
        # ====================================================

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

        # ====================================================
        # DONE
        # ====================================================

        print()

        print("=" * 80)

        print(
            "RETRIEVAL TEST COMPLETED"
        )

        print("=" * 80)

    finally:

        # ====================================================
        # Close Qdrant client
        # ====================================================

        if client is not None:

            client.close()

            print(
                "\n[OK] Qdrant client closed"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()