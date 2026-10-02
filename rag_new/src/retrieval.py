"""
LEGAL RAG - RETRIEVAL / PHASE 6

Tương ứng sơ đồ kiến trúc:
    2.5 Query Routing        -> route_query()
    2.6 Retrieval & Rerank   -> direct_lookup(), semantic_search(), rerank()
    2.7 Evidence Bundle      -> build_evidence_bundle()

Đầu vào : plan từ semantic_planner.build_plan()
Đầu ra  : evidence bundle (dict) để đưa cho bước 2.8 LLM Generation.
Nguyên tắc: không tự bịa nội dung, chỉ trả lại text chunk gốc (verbatim).
"""
from __future__ import annotations

import atexit
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from semantic_planner import (
    build_plan,
    load_domain_profile,
    load_procedures,
    meaningful_tokens,
    score_procedure,
)

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_storage"
COLLECTION_NAME = "vietnamese_administrative_procedures"
EMBEDDING_META_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings" / "embedding_metadata.json"
)

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "  # E5: query dùng "query: ", chunk đã dùng "passage: "

# Routing
DIRECT_MIN_PROCEDURE_SCORE = 0.85   # >= : exact procedure -> direct lookup
SEMANTIC_TOP_K = 8
FINAL_TOP_K = 4

# E5 cosine thường dồn trong khoảng 0.7-0.9, cần hiệu chỉnh bằng test thật.
MIN_SEMANTIC_SCORE = 0.78

# Rerank (rule-based). Có thể bật cross-encoder bên dưới.
INTENT_MATCH_BONUS = 0.30
PROCEDURE_MATCH_BONUS = 0.15
SUPPORT_INTENT_BONUS = 0.05

USE_CROSS_ENCODER = False
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

# Chunk phụ trợ đi kèm intent chính (chunk "forms" chỉ có khi dataset có cột biểu mẫu)
SUPPORTING_INTENTS = {
    "required_documents": ["forms"],
}

# Câu hỏi chung ("thủ tục X là gì?"): chunk general_information chỉ có
# tên/lĩnh vực/hình thức nộp, quá nghèo, nên gom thêm các chunk chính
# để LLM có đủ thông tin tổng quan.
OVERVIEW_INTENTS = [
    "general_information",
    "required_documents",
    "processing_time",
    "fee",
    "location",
]

# Đường semantic (không xác định được thủ tục) chỉ tin kết quả khi tên thủ tục
# có ít nhất một từ đặc trưng trùng với câu hỏi. Từ xuất hiện trong hơn
# COMMON_TOKEN_RATIO số thủ tục (vd "đăng", "ký") không được tính.
REQUIRE_LEXICAL_ANCHOR = True
COMMON_TOKEN_RATIO = 0.20

# Gợi ý thủ tục khi hỏi lại: chỉ lấy thủ tục có tín hiệu từ khóa thật
CANDIDATE_MIN_SCORE = 0.50


# =============================================================================
# RESOURCES (lazy singletons)
# =============================================================================

class Resources:
    _client: QdrantClient | None = None
    _embedder = None
    _reranker = None
    _procedures = None
    _profile: dict | None = None
    _common_tokens: set | None = None
    last_scores: list = []   # điểm semantic top-5 của lần retrieve gần nhất (để debug)

    @classmethod
    def client(cls) -> QdrantClient:
        if cls._client is None:
            cls._client = QdrantClient(path=str(QDRANT_PATH))
            atexit.register(cls._client.close)
        return cls._client

    @classmethod
    def embedder(cls):
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_name = DEFAULT_EMBEDDING_MODEL
            if EMBEDDING_META_FILE.exists():
                meta = json.loads(EMBEDDING_META_FILE.read_text(encoding="utf-8"))
                model_name = meta.get("model_name", model_name)
            cls._embedder = SentenceTransformer(model_name)
        return cls._embedder

    @classmethod
    def reranker(cls):
        if cls._reranker is None:
            from sentence_transformers import CrossEncoder

            cls._reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return cls._reranker

    @classmethod
    def procedures(cls):
        if cls._procedures is None:
            cls._procedures = load_procedures()
        return cls._procedures

    @classmethod
    def profile(cls) -> dict:
        if cls._profile is None:
            cls._profile = load_domain_profile()
        return cls._profile


# =============================================================================
# HELPERS
# =============================================================================

def procedure_required_intents() -> set[str]:
    """
    Intent cần biết thủ tục cụ thể mới trả lời được.
    Mọi intent cụ thể (hồ sơ, phí, địa điểm, thời gian, pháp lý, biểu mẫu)
    đều gắn với MỘT thủ tục; "hồ sơ gồm những gì" mà không rõ thủ tục
    thì phải hỏi lại, không được lấy bừa hồ sơ của thủ tục khác.
    Chỉ general_information được phép tìm ngữ nghĩa toàn collection.
    """
    return {
        "required_documents",
        "fee",
        "location",
        "processing_time",
        "legal_basis",
        "forms",
    }


def common_tokens() -> set[str]:
    """Từ xuất hiện trong quá nhiều tên thủ tục nên không đủ để phân biệt."""
    if Resources._common_tokens is None:
        df = Resources.procedures()
        counts: dict[str, int] = {}
        for name in df["procedure_name"]:
            for tok in set(meaningful_tokens(str(name))):
                counts[tok] = counts.get(tok, 0) + 1
        limit = COMMON_TOKEN_RATIO * len(df)
        Resources._common_tokens = {t for t, c in counts.items() if c > limit}
    return Resources._common_tokens


def has_lexical_anchor(query: str, procedure_name: str) -> bool:
    shared = set(meaningful_tokens(query)) & set(meaningful_tokens(procedure_name))
    return bool(shared - common_tokens())


def embed_query(query: str) -> list[float]:
    vector = Resources.embedder().encode(
        QUERY_PREFIX + query.strip(),
        normalize_embeddings=True,
    )
    return np.asarray(vector, dtype=np.float32).tolist()


def make_filter(
    procedure_id: str | None = None,
    intents: list[str] | None = None,
) -> Filter | None:
    must = []
    if procedure_id:
        must.append(
            FieldCondition(key="procedure_id", match=MatchValue(value=procedure_id))
        )
    if intents:
        if len(intents) == 1:
            must.append(
                FieldCondition(key="intent", match=MatchValue(value=intents[0]))
            )
        else:
            from qdrant_client.models import MatchAny

            must.append(FieldCondition(key="intent", match=MatchAny(any=intents)))
    return Filter(must=must) if must else None


def point_to_hit(point: Any, source: str, score: float | None = None) -> dict:
    payload = dict(point.payload or {})
    payload["score"] = float(score if score is not None else getattr(point, "score", 0.0))
    payload["source"] = source
    return payload


# =============================================================================
# 2.6a  DIRECT LOOKUP  (tra cứu trực tiếp theo procedure_id + intent)
# =============================================================================

def direct_lookup(procedure_id: str, intents: list[str]) -> list[dict]:
    points, _ = Resources.client().scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=make_filter(procedure_id, intents),
        limit=len(intents) * 4,
        with_payload=True,
        with_vectors=False,
    )
    # score=1.0: khớp chính xác theo khóa, không phải điểm tương đồng
    return [point_to_hit(p, "direct", 1.0) for p in points]


# =============================================================================
# 2.6b  SEMANTIC SEARCH  (Qdrant vector search, có thể kèm filter)
# =============================================================================

def semantic_search(
    query: str,
    procedure_id: str | None = None,
    intents: list[str] | None = None,
    top_k: int = SEMANTIC_TOP_K,
) -> list[dict]:
    result = Resources.client().query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(query),
        query_filter=make_filter(procedure_id, intents),
        limit=top_k,
        with_payload=True,
    )
    return [point_to_hit(p, "semantic") for p in result.points]


# =============================================================================
# 2.6c  RERANK
# =============================================================================

def merge_hits(*hit_lists: list[dict]) -> list[dict]:
    """Gộp theo chunk_id; direct thắng semantic, giữ điểm semantic nếu có."""
    merged: dict[str, dict] = {}
    for hits in hit_lists:
        for hit in hits:
            existing = merged.get(hit["chunk_id"])
            if existing is None:
                merged[hit["chunk_id"]] = hit
                continue
            sources = {existing["source"], hit["source"]}
            semantic_score = max(
                (h["score"] for h in (existing, hit) if h["source"] == "semantic"),
                default=0.0,
            )
            existing["source"] = "direct+semantic" if len(sources) > 1 else existing["source"]
            existing["semantic_score"] = semantic_score
    return list(merged.values())


def rerank(query: str, hits: list[dict], plan: dict) -> list[dict]:
    intent = plan["intent"]
    procedure_id = plan.get("procedure_id")
    support = set(SUPPORTING_INTENTS.get(intent, []))

    if USE_CROSS_ENCODER and hits:
        pairs = [(query, h["text"]) for h in hits]
        ce_scores = Resources.reranker().predict(pairs)
        for hit, ce in zip(hits, ce_scores):
            hit["rerank_score"] = float(ce)
    else:
        for hit in hits:
            base = hit.get("semantic_score", hit["score"])
            score = base
            if hit["intent"] == intent:
                score += INTENT_MATCH_BONUS
            elif hit["intent"] in support:
                score += SUPPORT_INTENT_BONUS
            if procedure_id and hit["procedure_id"] == procedure_id:
                score += PROCEDURE_MATCH_BONUS
            hit["rerank_score"] = round(score, 4)

    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)


# =============================================================================
# 2.5  QUERY ROUTING
# =============================================================================

def route_query(plan: dict) -> str:
    """
    direct   : biết chắc thủ tục -> tra thẳng (procedure_id, intent)
    hybrid   : biết thủ tục nhưng chưa chắc tuyệt đối -> direct + semantic
    semantic : không xác định được thủ tục -> tìm ngữ nghĩa toàn collection
    """
    if not plan.get("procedure_id"):
        return "semantic"
    if plan["procedure_score"] >= DIRECT_MIN_PROCEDURE_SCORE:
        return "direct"
    return "hybrid"


# =============================================================================
# 2.7  EVIDENCE BUNDLE
# =============================================================================

def to_evidence(hit: dict) -> dict:
    return {
        "chunk_id": hit["chunk_id"],
        "procedure_id": hit["procedure_id"],
        "procedure_name": hit["procedure_name"],
        "domain": hit.get("domain", ""),
        "intent": hit["intent"],
        "text": hit["text"],  # verbatim, không tóm tắt
        "score": round(hit.get("rerank_score", hit["score"]), 4),
        "source": hit["source"],
    }


def build_evidence_bundle(
    query: str,
    plan: dict,
    route: str,
    status: str,
    hits: list[dict],
    message: str = "",
    candidates: list[dict] | None = None,
) -> dict:
    return {
        "query": query,
        "intent": plan["intent"],
        "procedure_id": plan.get("procedure_id"),
        "procedure": plan.get("procedure"),
        "route": route,
        # ok | missing_field | needs_clarification | no_evidence
        "status": status,
        "message": message,
        "candidates": candidates or [],
        "evidence": [to_evidence(h) for h in hits],
    }


def lexical_candidates(query: str, limit: int = 3) -> list[dict]:
    """
    Gợi ý thủ tục dựa trên tín hiệu từ khóa của planner.
    Câu như "ở đâu" hay "hồ sơ gồm những gì" không có tín hiệu nào
    nên trả về rỗng, thay vì gợi ý ngẫu nhiên từ vector search.
    """
    scored = []
    for _, row in Resources.procedures().iterrows():
        score = score_procedure(query, str(row["procedure_name"]))
        if score >= CANDIDATE_MIN_SCORE:
            scored.append((score, row["procedure_id"], row["procedure_name"]))
    scored.sort(reverse=True)
    return [
        {"procedure_id": pid, "procedure_name": name}
        for _, pid, name in scored[:limit]
    ]


def candidate_procedures(hits: list[dict], limit: int = 3) -> list[dict]:
    seen, out = set(), []
    for hit in hits:
        pid = hit["procedure_id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"procedure_id": pid, "procedure_name": hit["procedure_name"]})
        if len(out) >= limit:
            break
    return out


# =============================================================================
# MAIN ENTRY
# =============================================================================

def retrieve(query: str) -> dict:
    Resources.last_scores = []
    plan = build_plan(query, Resources.procedures())
    route = route_query(plan)
    intent = plan["intent"]
    overview = intent == "general_information"
    wanted = (
        OVERVIEW_INTENTS
        if overview
        else [intent] + SUPPORTING_INTENTS.get(intent, [])
    )

    # ---- Planner báo mơ hồ (2 thủ tục ngang điểm): hỏi lại ----
    if plan.get("ambiguous"):
        return build_evidence_bundle(
            query, plan, route, "needs_clarification", [],
            message="Có nhiều thủ tục phù hợp. Bạn muốn hỏi thủ tục nào?",
            candidates=plan.get("candidates", []),
        )

    # ---- Có thủ tục: direct (+ semantic nếu hybrid) ----
    if route in ("direct", "hybrid"):
        pid = plan["procedure_id"]
        direct_hits = direct_lookup(pid, wanted)

        semantic_hits: list[dict] = []
        if route == "hybrid" or not direct_hits:
            semantic_hits = semantic_search(query, procedure_id=pid, intents=wanted)

        hits = rerank(query, merge_hits(direct_hits, semantic_hits), plan)

        if overview:
            hits.sort(
                key=lambda h: OVERVIEW_INTENTS.index(h["intent"])
                if h["intent"] in OVERVIEW_INTENTS else 99
            )
        top_k = len(OVERVIEW_INTENTS) if overview else FINAL_TOP_K

        # Chunk intent chính không tồn tại (field rỗng trong dataset)
        has_main = any(h["intent"] == intent for h in hits)
        if not has_main and not overview:
            # Evidence rỗng: không đưa chunk intent khác cho LLM để tránh
            # trả lời sai (vd hỏi địa điểm nhưng lại nhận chunk thời gian).
            return build_evidence_bundle(
                query, plan, route, "missing_field", [],
                message=(
                    f"Thủ tục '{plan['procedure']}' chưa có dữ liệu "
                    f"cho '{intent}' trong dataset."
                ),
            )

        if not hits:
            return build_evidence_bundle(
                query, plan, route, "no_evidence", [],
                message="Không tìm thấy bằng chứng cho thủ tục này.",
            )

        return build_evidence_bundle(query, plan, route, "ok", hits[:top_k])

    # ---- Không xác định thủ tục ----
    # Intent cụ thể (hồ sơ, phí, địa điểm...) bắt buộc phải biết thủ tục.
    if intent in procedure_required_intents():
        cands = lexical_candidates(query)
        message = (
            "Có thể bạn đang hỏi một trong các thủ tục sau. Bạn muốn hỏi thủ tục nào?"
            if cands
            else "Bạn vui lòng cho biết tên thủ tục cụ thể bạn muốn hỏi."
        )
        return build_evidence_bundle(
            query, plan, route, "needs_clarification", [],
            message=message, candidates=cands,
        )

    # general_information: tìm ngữ nghĩa toàn collection
    hits = rerank(query, semantic_search(query), plan)
    Resources.last_scores = [
        (h["chunk_id"], round(h["score"], 4)) for h in hits[:5]
    ]
    if REQUIRE_LEXICAL_ANCHOR:
        hits = [h for h in hits if has_lexical_anchor(query, h["procedure_name"])]
    cands = candidate_procedures(hits)

    if not hits or hits[0]["score"] < MIN_SEMANTIC_SCORE:
        return build_evidence_bundle(
            query, plan, route, "no_evidence", [],
            message="Không đủ bằng chứng liên quan trong dữ liệu.",
        )

    # Top-1 và top-2 thuộc hai thủ tục khác nhau, điểm sát nhau -> hỏi lại
    if len({h["procedure_id"] for h in hits[:2]}) > 1:
        gap = hits[0]["score"] - hits[1]["score"]
        if gap < 0.02:
            return build_evidence_bundle(
                query, plan, route, "needs_clarification", [],
                message="Có nhiều thủ tục phù hợp. Bạn muốn hỏi thủ tục nào?",
                candidates=cands,
            )

    return build_evidence_bundle(
        query, plan, route, "ok", hits[:FINAL_TOP_K], candidates=cands
    )


# =============================================================================
# TEST
# =============================================================================

TEST_QUERIES = [
    "Đăng ký khai sinh cần những giấy tờ gì?",
    "khai sinh ở đâu",
    "làm giấy khai sinh mất bao lâu?",
    "đăng ký khai sinh hết bao nhiêu tiền?",
    "căn cứ pháp lý của thủ tục đăng ký khai sinh",
    "kết hôn với người nước ngoài cần hồ sơ gì",
    "ở đâu",
    "hồ sơ gồm những gì",
    "thủ tục xác nhận tình trạng hôn nhân là gì?",
    "đăng ký hộ khẩu",
    "đăng ký kết hôn nộp ở đâu",
    "đăng ký khai sinh",
]

# Câu hỏi mẫu hiện ra khi chạy chế độ tương tác (gõ số để chọn)
SAMPLE_QUESTIONS = [
    "Đăng ký khai sinh cần những giấy tờ gì?",
    "đăng kí khai sinh có mất tiền không",
    "làm giấy khai sinh mất bao lâu?",
    "khai sinh nộp ở đâu",
    "căn cứ pháp lý của thủ tục đăng ký khai sinh",
    "Đk kết hôn cần chuẩn bị hồ sơ gì?",
    "kết hôn với người nước ngoài cần hồ sơ gì",
    "thủ tục xác nhận tình trạng hôn nhân là gì?",
    "hồ sơ gồm những gì",     # thiếu thủ tục -> hỏi lại
    "đăng ký hộ khẩu",        # ngoài dữ liệu
]

STATUS_LABEL = {
    "ok": "OK (đủ bằng chứng)",
    "missing_field": "THIẾU DỮ LIỆU (thủ tục không có field này)",
    "needs_clarification": "CẦN HỎI LẠI",
    "no_evidence": "KHÔNG CÓ BẰNG CHỨNG",
}


def print_bundle(bundle: dict, full: bool = False) -> None:
    print("\n" + "-" * 80)
    print(f"Câu hỏi   : {bundle['query']}")
    print(f"Intent    : {bundle['intent']}")
    print(f"Thủ tục   : {bundle['procedure']} ({bundle['procedure_id']})")
    print(f"Route     : {bundle['route']}")
    print(f"Trạng thái: {STATUS_LABEL.get(bundle['status'], bundle['status'])}")
    if bundle["message"]:
        print(f"Thông báo : {bundle['message']}")
    for c in bundle["candidates"]:
        print(f"  gợi ý   : {c['procedure_id']} - {c['procedure_name']}")
    for i, ev in enumerate(bundle["evidence"], 1):
        print(f"\n  [{i}] {ev['chunk_id']}  (intent={ev['intent']}, "
              f"score={ev['score']}, src={ev['source']})")
        text = ev["text"].strip()
        if full:
            for line in text.splitlines():
                print(f"      {line}")
        else:
            print(f"      {text.replace(chr(10), ' ')[:110]}...")


def print_samples() -> None:
    print("\nCÂU HỎI MẪU (gõ số để chọn, hoặc tự gõ câu hỏi của bạn):")
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"  {i:>2}. {q}")
    print("\nLệnh: 'm' xem lại gợi ý  |  't 0.85' đổi ngưỡng semantic  |  'exit' hoặc 'q' để thoát")


def interactive() -> None:
    print("Đang tải model và kết nối Qdrant...")
    Resources.embedder()
    Resources.client()
    print("Sẵn sàng.")
    print_samples()

    while True:
        try:
            raw = input("\nCâu hỏi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt.")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            print("Tạm biệt.")
            break
        if raw.lower() == "m":
            print_samples()
            continue
        if raw.lower().startswith("t "):
            global MIN_SEMANTIC_SCORE
            try:
                MIN_SEMANTIC_SCORE = float(raw.split()[1])
                print(f"MIN_SEMANTIC_SCORE = {MIN_SEMANTIC_SCORE}")
            except ValueError:
                print("Dùng: t 0.85")
            continue

        query = raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(SAMPLE_QUESTIONS):
                query = SAMPLE_QUESTIONS[idx - 1]
                print(f"→ {query}")
            else:
                print(f"Chọn số từ 1 đến {len(SAMPLE_QUESTIONS)}.")
                continue

        try:
            print_bundle(retrieve(query), full=True)
            if Resources.last_scores:
                print(f"\n  [debug] ngưỡng semantic = {MIN_SEMANTIC_SCORE}")
                for chunk_id, score in Resources.last_scores:
                    print(f"  [debug] {score:.4f}  {chunk_id}")
        except Exception as exc:  # không để một câu lỗi làm thoát cả chương trình
            print(f"Lỗi khi xử lý câu hỏi: {exc}")


def run_tests() -> None:
    print("=" * 80)
    print("LEGAL RAG - RETRIEVAL / PHASE 6 (TEST)")
    print("=" * 80)
    for q in TEST_QUERIES:
        print_bundle(retrieve(q))
    print("\n" + "=" * 80)
    print("PHASE 6 TEST COMPLETED")
    print("=" * 80)


def main() -> None:
    print("=" * 80)
    print("LEGAL RAG - RETRIEVAL / PHASE 6")
    print("=" * 80)
    if "--test" in sys.argv:
        run_tests()
    else:
        interactive()


if __name__ == "__main__":
    main()